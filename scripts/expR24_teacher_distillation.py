#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R24: Teacher distillation — train BGE retriever using LambdaRank as teacher.

Uses graded teacher scores (not binary hard negatives) to distill
LambdaRank ranking knowledge into a BGE bi-encoder. Combines standard
contrastive loss (global retrieval) with listwise KL distillation (ranking).

Fold-0 smoke test first, then 5-fold if gates pass.

Usage:
    uv run python3 scripts/expR24_teacher_distillation.py --phase teacher --fold 0
    uv run python3 scripts/expR24_teacher_distillation.py --phase train --fold 0
    uv run python3 scripts/expR24_teacher_distillation.py --phase eval --fold 0
    uv run python3 scripts/expR24_teacher_distillation.py --phase all --fold 0
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# NOTE: imports from scripts.expS2_lambdarank etc. are deferred to inside
# functions that need them. Importing them at module level pulls in `datasets`
# which creates a loky multiprocessing context that segfaults torch on macOS.

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO_ROOT / "cache" / "r21_production" / "dev_r21_oof_lists.json"
CACHE_DIR = REPO_ROOT / "cache" / "r24"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

POOL_K = 300
RRF_K = 20
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
SOURCES = ["A", "B", "C", "D", "F", "ALS", "R21"]

FEATURE_NAMES = [
    "rrf_rank_inv", "same_artist", "tag_jaccard",
    "artist_tok_overlap", "title_tok_overlap", "meta_tok_overlap",
    "already_played", "recency_weighted_meta",
    *[f"rank_{s}" for s in SOURCES],
    *[f"in_{s}" for s in SOURCES],
    "n_sources", "als_score", "n_hist",
]

# Distillation hyperparameters
DISTILL_BATCH = 16
DISTILL_EPOCHS = 3
DISTILL_LR = 2e-5
DISTILL_ALPHA = 0.5
CONTRASTIVE_TEMP = 0.05
SYMMETRIC_CONTRASTIVE = False
MARGIN_PAIRS_PER_QUERY = 1


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def build_track_text(tid, meta):
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    album = m.get("album_name", [])
    tags = m.get("tag_list", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = ", ".join(artists) if isinstance(artists, list) else str(artists)
    alb = album[0] if isinstance(album, list) and album else str(album)
    tag_str = ", ".join(str(t) for t in tags[:10]) if isinstance(tags, list) else str(tags)
    return f"{name} by {artist}. Album: {alb}. Tags: {tag_str}"


def build_query_text(case):
    parts = []
    for h in case["history"]:
        if h["role"] == "user":
            parts.append(str(h["content"]))
    parts.append(case["user_query"])
    return " ".join(parts[-3:])


def load_catalog():
    import pyarrow as pa
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    if not matches:
        raise FileNotFoundError("all_tracks arrow not found")
    with pa.memory_map(str(matches[-1]), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    cols = {col: table.column(col).to_pylist() for col in table.column_names}
    meta = {}
    track_ids = []
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        track_ids.append(tid)
        meta[tid] = {k: cols[k][i] for k in cols}
    return meta, track_ids


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1: Build teacher
# ──────────────────────────────────────────────────────────────────────────────

def build_pools_and_features(payload, r21_oof_lists, als_source, als_session_vecs,
                             als_factors, als_track_to_idx):
    """Build expanded pools (V3+ALS+R21) and LambdaRank features."""
    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.tune_postrank_v23 import tokens
    cases = payload["cases"]
    n = len(cases)
    n_feat = len(FEATURE_NAMES)
    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    all_pools = []

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_oof_lists[i][:200],
        }
        pool = weighted_rrf(src_lists, SOURCE_WEIGHTS, topk=POOL_K, k=RRF_K)
        all_pools.append(pool)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank = {}
        for sname, slist in src_lists.items():
            src_rank[sname] = {tid: rank + 1 for rank, tid in enumerate(slist)}

        user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                     + [c["user_query"]])
        played = c["music_turns"]
        n_hist = len(played)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]
        sv = als_session_vecs[i]

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]

            row[0] = 1.0 / rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags:
                row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(tat.get(tid, set()) & now_tok))
            row[4] = float(len(ttl.get(tid, set()) & now_tok))
            row[5] = float(len(tmt.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec

            for fi, sname in enumerate(SOURCES):
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0

            for fi, sname in enumerate(SOURCES):
                row[8 + len(SOURCES) + fi] = 1.0 if tid in src_rank[sname] else 0.0

            row[8 + 2 * len(SOURCES)] = sum(1 for sname in SOURCES if tid in src_rank[sname])

            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[8 + 2 * len(SOURCES) + 1] = float(np.dot(sv, als_factors[aidx]))

            row[8 + 2 * len(SOURCES) + 2] = float(n_hist)

    return X, gt_idx, sizes, all_pools


def train_teacher(X, gt_idx, sizes, sessions, train_idx, val_idx):
    """Train LambdaRank teacher on train cases, return model + scores for all cases."""
    import lightgbm as lgb
    n = len(sessions)
    n_feat = len(FEATURE_NAMES)
    X_flat = X.reshape(-1, n_feat)
    labels = np.zeros(n * POOL_K, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels[i * POOL_K + gt_idx[i]] = 1.0

    train_flat = []
    for j in train_idx:
        for k in range(int(sizes[j])):
            train_flat.append(j * POOL_K + k)
    val_flat = []
    for j in val_idx:
        for k in range(int(sizes[j])):
            val_flat.append(j * POOL_K + k)

    g_train = np.array([int(sizes[j]) for j in train_idx], dtype=np.int32)
    g_val = np.array([int(sizes[j]) for j in val_idx], dtype=np.int32)

    dtrain = lgb.Dataset(X_flat[train_flat], labels[train_flat],
                         group=g_train, feature_name=FEATURE_NAMES, free_raw_data=False)
    dval = lgb.Dataset(X_flat[val_flat], labels[val_flat],
                       group=g_val, reference=dtrain, free_raw_data=False)

    lgb_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "verbose": -1, "random_state": 42, "force_col_wise": True,
    }

    model = lgb.train(
        lgb_params, dtrain, num_boost_round=300,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    all_scores = model.predict(X_flat)
    teacher_scores = all_scores.reshape(n, POOL_K)

    imp = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1])
    print("  Top teacher features:")
    for fname, fimp in feat_imp[:8]:
        print(f"    {fname:25s} {fimp:10.1f}")

    val_ndcg = compute_ndcg(teacher_scores, gt_idx, sizes, val_idx)
    print(f"  Teacher val nDCG@20: {val_ndcg:.4f}")

    return model, teacher_scores


def compute_ndcg(scores, gt_idx, sizes, case_idx, k=20):
    vals = []
    for i in case_idx:
        gt = gt_idx[i]
        if gt < 0:
            vals.append(0.0)
            continue
        s = scores[i, :int(sizes[i])]
        gt_score = s[gt]
        rank0 = int(np.sum(s > gt_score) + np.sum((s == gt_score) & (np.arange(len(s)) < gt)))
        vals.append(1.0 / np.log2(rank0 + 2) if rank0 < k else 0.0)
    return float(np.mean(vals))


def phase_teacher(fold_id):
    """Build teacher scores for the specified fold."""
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds

    t0 = time.time()
    fold_dir = CACHE_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    print(f"{ts()} R24 Phase 1: Build teacher (fold {fold_id})")
    print(f"{'='*60}")

    print(f"\n{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    with open(R21_OOF) as f:
        r21_oof_lists = json.load(f)
    print(f"  {n} cases, {len(set(sessions))} sessions")

    print(f"\n{ts()} Building ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
    als_session_vecs = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_session_vecs.append(sv)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    print(f"\n{ts()} Building pools and features (pool_k={POOL_K}, {len(FEATURE_NAMES)} features)...",
          flush=True)
    X, gt_idx, sizes, all_pools = build_pools_and_features(
        payload, r21_oof_lists, als_source, als_session_vecs,
        als_factors, als_track_to_idx)
    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

    folds = grouped_session_folds(sessions, seed=0)
    val_idx = folds[fold_id].tolist()
    train_idx = [j for j in range(n) if j not in set(val_idx)]
    print(f"  Fold {fold_id}: train={len(train_idx)}, val={len(val_idx)}")

    print(f"\n{ts()} Training LambdaRank teacher...", flush=True)
    teacher_model, teacher_scores = train_teacher(
        X, gt_idx, sizes, sessions, train_idx, val_idx)

    np.save(fold_dir / "teacher_scores.npy", teacher_scores)
    np.save(fold_dir / "gt_idx.npy", gt_idx)
    np.save(fold_dir / "sizes.npy", sizes)
    with open(fold_dir / "pools.json", "w") as f:
        json.dump(all_pools, f)
    with open(fold_dir / "fold_split.json", "w") as f:
        json.dump({"train_idx": train_idx, "val_idx": val_idx}, f)

    teacher_model.save_model(str(fold_dir / "teacher.lgb"))

    elapsed = time.time() - t0
    print(f"\n{ts()} Phase 1 complete. Elapsed: {elapsed:.1f}s")
    print(f"  Saved to: {fold_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2: Train BGE with distillation
# ──────────────────────────────────────────────────────────────────────────────

def sample_margin_pairs(all_pools, teacher_scores, cases, meta, train_idx, rng):
    """Sample (query, track_hi, track_lo, margin) pairs from teacher-scored pools.

    For each train case, sample MARGIN_PAIRS_PER_QUERY pairs where the teacher
    score gap is meaningful (>= 0.5). Graded margins, not binary labels.
    """
    queries, hi_texts, lo_texts, margins = [], [], [], []
    for i in train_idx:
        pool = all_pools[i]
        scores = teacher_scores[i, :len(pool)]
        if len(pool) < 10:
            continue

        sorted_idx = np.argsort(-scores)
        n_pool = len(pool)

        for _ in range(MARGIN_PAIRS_PER_QUERY):
            # Sample hi from top quartile, lo from bottom quartile
            hi_pos = rng.randint(0, max(1, n_pool // 4))
            lo_pos = rng.randint(n_pool * 3 // 4, n_pool)
            hi_j = sorted_idx[hi_pos]
            lo_j = sorted_idx[lo_pos]
            margin = float(scores[hi_j] - scores[lo_j])
            if margin < 0.1:
                continue
            queries.append(build_query_text(cases[i]))
            hi_texts.append(build_track_text(pool[hi_j], meta))
            lo_texts.append(build_track_text(pool[lo_j], meta))
            margins.append(margin)

    return queries, hi_texts, lo_texts, np.array(margins, dtype=np.float32)


def phase_train(fold_id, run_tag="base", init_model=None, epochs=None):
    """Train BGE with contrastive + MarginMSE distillation loss.

    Memory-efficient: encodes 4B texts per step (same as R21's 2B).
    Contrastive: (query, GT) with in-batch negatives — global retrieval signal.
    MarginMSE: (query, track_hi, track_lo, margin) — teacher ranking signal.
    """
    t0 = time.time()
    fold_dir = CACHE_DIR / f"fold_{fold_id}"

    print(f"{ts()} R24 Phase 2: Train BGE with distillation (fold {fold_id})")
    print(f"{'='*60}")

    import torch
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer

    print(f"\n{ts()} Loading teacher data...", flush=True)
    teacher_scores = np.load(fold_dir / "teacher_scores.npy")
    with open(fold_dir / "pools.json") as f:
        all_pools = json.load(f)
    with open(fold_dir / "fold_split.json") as f:
        split = json.load(f)
    train_idx = split["train_idx"]

    print(f"\n{ts()} Loading dev cases and catalog...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    meta, _ = load_catalog()

    # Build contrastive examples (query, GT) — same as R21
    contr_queries = []
    contr_gt_texts = []
    for i in train_idx:
        gt_tid = cases[i]["gt"]
        if gt_tid not in meta:
            continue
        contr_queries.append(build_query_text(cases[i]))
        contr_gt_texts.append(build_track_text(gt_tid, meta))
    n_contr = len(contr_queries)
    print(f"  Contrastive examples: {n_contr}")

    # Build MarginMSE pairs from teacher (skip if alpha=0)
    if DISTILL_ALPHA > 0:
        rng = np.random.RandomState(42)
        mq, mhi, mlo, mmargins = sample_margin_pairs(
            all_pools, teacher_scores, cases, meta, train_idx, rng)
        n_margin = len(mq)
        print(f"  MarginMSE pairs: {n_margin}")
    else:
        mq, mhi, mlo, mmargins = [], [], [], np.array([], dtype=np.float32)
        n_margin = 0
        print("  MarginMSE: SKIPPED (alpha=0)")

    # Free large objects before loading model
    import gc
    del payload, meta, all_pools, teacher_scores, cases
    gc.collect()

    model_src = init_model if init_model else MODEL_NAME
    print(f"\n{ts()} Initializing from {model_src}...", flush=True)
    model = SentenceTransformer(model_src, device="cpu")
    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=DISTILL_LR, weight_decay=1e-4)

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256,
                            return_tensors="pt")
        out = model.forward(encoded)
        return F_t.normalize(out["sentence_embedding"], dim=-1)

    B = DISTILL_BATCH
    n_epochs = epochs if epochs is not None else DISTILL_EPOCHS
    n_steps = (n_contr + B - 1) // B

    sym_str = " symmetric" if SYMMETRIC_CONTRASTIVE else ""
    print(f"\n{ts()} Training: {n_epochs} epochs, batch={B}, "
          f"alpha={DISTILL_ALPHA},{sym_str} run_tag={run_tag}", flush=True)

    model.train()
    for epoch in range(n_epochs):
        perm_c = np.random.permutation(n_contr)
        perm_m = np.random.permutation(n_margin)
        epoch_contrastive = 0.0
        epoch_distill = 0.0
        epoch_total = 0.0
        n_batches = 0

        for step in range(n_steps):
            bp_c = perm_c[step * B: (step + 1) * B]
            if len(bp_c) < 2:
                continue
            b = len(bp_c)

            # ── Contrastive: encode queries + GT tracks (2B texts) ──
            q_texts = [contr_queries[j] for j in bp_c]
            gt_texts = [contr_gt_texts[j] for j in bp_c]
            q_emb = encode_with_grad(q_texts)
            gt_emb = encode_with_grad(gt_texts)
            sim = q_emb @ gt_emb.T / CONTRASTIVE_TEMP
            labels = torch.arange(b, device=sim.device)
            if SYMMETRIC_CONTRASTIVE:
                contrastive_loss = 0.5 * (F_t.cross_entropy(sim, labels) +
                                          F_t.cross_entropy(sim.T, labels))
            else:
                contrastive_loss = F_t.cross_entropy(sim, labels)

            # ── MarginMSE: encode query + hi + lo (3B texts) ──
            if DISTILL_ALPHA > 0:
                bp_m = perm_m[step * B: (step + 1) * B]
                if len(bp_m) >= 2:
                    mq_texts = [mq[j] for j in bp_m]
                    mhi_texts = [mhi[j] for j in bp_m]
                    mlo_texts = [mlo[j] for j in bp_m]
                    mq_emb = encode_with_grad(mq_texts)
                    mhi_emb = encode_with_grad(mhi_texts)
                    mlo_emb = encode_with_grad(mlo_texts)

                    student_margin = (mq_emb * mhi_emb).sum(-1) - (mq_emb * mlo_emb).sum(-1)
                    teacher_margin = torch.tensor(mmargins[bp_m], dtype=torch.float32)
                    teacher_margin = teacher_margin / (teacher_margin.abs().max() + 1e-8)
                    distill_loss = F_t.mse_loss(student_margin, teacher_margin)
                else:
                    distill_loss = torch.tensor(0.0)
            else:
                distill_loss = torch.tensor(0.0)

            loss = contrastive_loss + DISTILL_ALPHA * distill_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_contrastive += contrastive_loss.item()
            epoch_distill += distill_loss.item()
            epoch_total += loss.item()
            n_batches += 1

            if n_batches % 25 == 0:
                print(f"    [{ts()}] step {n_batches}/{n_steps} "
                      f"loss={epoch_total/n_batches:.4f} "
                      f"(contr={epoch_contrastive/n_batches:.4f} "
                      f"distill={epoch_distill/n_batches:.4f})", flush=True)

        print(f"  Epoch {epoch}: total={epoch_total/max(n_batches,1):.4f} "
              f"contrastive={epoch_contrastive/max(n_batches,1):.4f} "
              f"distill={epoch_distill/max(n_batches,1):.4f}", flush=True)

    model_dir = fold_dir / f"model_{run_tag}"
    model.save(str(model_dir))
    print(f"\n{ts()} Model saved to {model_dir}")

    elapsed = time.time() - t0
    print(f"{ts()} Phase 2 complete. Elapsed: {elapsed:.1f}s ({elapsed/3600:.1f}h)")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Evaluate
# ──────────────────────────────────────────────────────────────────────────────

def phase_eval(fold_id, run_tag="base"):
    """Evaluate distilled BGE on fold val set."""
    t0 = time.time()
    fold_dir = CACHE_DIR / f"fold_{fold_id}"

    print(f"{ts()} R24 Phase 3: Evaluate (fold {fold_id})")
    print(f"{'='*60}")

    from sentence_transformers import SentenceTransformer

    with open(fold_dir / "fold_split.json") as f:
        split = json.load(f)
    val_idx = split["val_idx"]

    print(f"\n{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    meta, all_track_ids = load_catalog()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]

    with open(R21_OOF) as f:
        r21_oof_lists = json.load(f)

    print(f"\n{ts()} Loading distilled model ({run_tag})...", flush=True)
    model = SentenceTransformer(str(fold_dir / f"model_{run_tag}"), device="cpu")

    print(f"\n{ts()} Encoding {len(all_track_ids)} catalog tracks...", flush=True)
    track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    val_cases = [cases[i] for i in val_idx]
    val_queries = [build_query_text(c) for c in val_cases]
    print(f"\n{ts()} Encoding {len(val_queries)} val queries...", flush=True)
    query_embs = model.encode(val_queries, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    print(f"\n{ts()} Retrieving top-300...", flush=True)
    r24_lists = []
    for qi in range(len(val_cases)):
        scores = track_embs @ query_embs[qi]
        played_set = set(val_cases[qi]["music_turns"])
        for idx, tid in enumerate(all_track_ids):
            if tid in played_set:
                scores[idx] = -np.inf
        top_idx = np.argpartition(-scores, 300)[:300]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        r24_lists.append([all_track_ids[j] for j in top_idx])

    # Load train track set for unseen analysis
    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_tracks = set()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                train_tracks.add(str(c["content"]).strip())

    # ── Compute metrics ──
    n_val = len(val_idx)
    r21_val_lists = [r21_oof_lists[i] for i in val_idx]

    r24_hit = {}
    r21_hit = {}
    r24_gt_ranks = []
    r21_gt_ranks = []
    for depth in [20, 50, 100, 200, 300]:
        r24_hit[depth] = sum(1 for qi in range(n_val)
                             if val_cases[qi]["gt"] in r24_lists[qi][:depth])
        r21_hit[depth] = sum(1 for qi in range(n_val)
                             if val_cases[qi]["gt"] in r21_val_lists[qi][:depth])
    for qi in range(n_val):
        gt = val_cases[qi]["gt"]
        if gt in r24_lists[qi]:
            r24_gt_ranks.append(r24_lists[qi].index(gt) + 1)
        if gt in r21_val_lists[qi]:
            r21_gt_ranks.append(r21_val_lists[qi].index(gt) + 1)

    unique_vs_r21 = sum(1 for qi in range(n_val)
                        if val_cases[qi]["gt"] in r24_lists[qi][:200]
                        and val_cases[qi]["gt"] not in r21_val_lists[qi][:200])
    lost_vs_r21 = sum(1 for qi in range(n_val)
                      if val_cases[qi]["gt"] not in r24_lists[qi][:200]
                      and val_cases[qi]["gt"] in r21_val_lists[qi][:200])

    unseen_total = sum(1 for qi in range(n_val)
                       if val_cases[qi]["gt"] not in train_tracks)
    unseen_hit = sum(1 for qi in range(n_val)
                     if val_cases[qi]["gt"] not in train_tracks
                     and val_cases[qi]["gt"] in r24_lists[qi][:200])
    r21_unseen = sum(1 for qi in range(n_val)
                     if val_cases[qi]["gt"] not in train_tracks
                     and val_cases[qi]["gt"] in r21_val_lists[qi][:200])

    # Pool hit simulation: V3 + ALS + R24 at pool_k=300
    # Import here (not at function top) to avoid loky/torch segfault
    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector

    print(f"\n{ts()} Building pool hit simulation...", flush=True)
    als_factors, als_track_ids_als, als_track_to_idx = build_als()
    als_source = []
    for c in val_cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            sc = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-sc, 200)[:200]
            top_idx = top_idx[np.argsort(-sc[top_idx])]
            als_source.append([als_track_ids_als[j] for j in top_idx])
        else:
            als_source.append([])

    sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}
    sw_r24 = {**sw, "R24": 1.0}
    sw_r21 = {**sw, "R21": 1.0}

    pool_hit_r24 = 0
    pool_hit_r21 = 0
    for qi in range(n_val):
        gi = val_idx[qi]
        gt = val_cases[qi]["gt"]
        base_src = {
            "A": payload["src_a"][gi], "B": payload["src_b"][gi],
            "C": payload["src_c"][gi], "D": payload["src_d"][gi],
            "F": payload["src_f"][gi], "ALS": als_source[qi],
        }
        src_r24 = {**base_src, "R24": r24_lists[qi][:200]}
        src_r21 = {**base_src, "R21": r21_val_lists[qi][:200]}

        if gt in set(weighted_rrf(src_r24, sw_r24, topk=300, k=20)):
            pool_hit_r24 += 1
        if gt in set(weighted_rrf(src_r21, sw_r21, topk=300, k=20)):
            pool_hit_r21 += 1

    # ── Report ──
    print(f"\n{'='*60}")
    print(f"R24 FOLD-{fold_id} RESULTS")
    print(f"{'='*60}")

    print("\nStandalone retrieval (R24 vs R21 OOF):")
    print("  %-10s %8s %8s %8s %8s %8s" % ("Depth", "R24", "R24%", "R21", "R21%", "Delta"))
    print("  " + "-" * 50)
    for d in [20, 50, 100, 200, 300]:
        print(f"  hit@{d:<5} {r24_hit[d]:>8} {r24_hit[d]/n_val:>7.1%} "
              f"{r21_hit[d]:>8} {r21_hit[d]/n_val:>7.1%} {r24_hit[d]-r21_hit[d]:>+8d}")

    r24_median = float(np.median(r24_gt_ranks)) if r24_gt_ranks else float("inf")
    r21_median = float(np.median(r21_gt_ranks)) if r21_gt_ranks else float("inf")
    print("\nMedian GT rank (among found GTs):")
    print(f"  R24: {r24_median:.0f} (n={len(r24_gt_ranks)})")
    print(f"  R21: {r21_median:.0f} (n={len(r21_gt_ranks)})")

    print("\nUnique analysis:")
    print(f"  Unique GTs vs R21 (R24 finds, R21 misses): {unique_vs_r21}")
    print(f"  Lost GTs vs R21 (R21 finds, R24 misses):   {lost_vs_r21}")
    print(f"  Net: {unique_vs_r21 - lost_vs_r21:+d}")

    print("\nUnseen analysis:")
    print(f"  R24 unseen hit@200: {unseen_hit}/{unseen_total} ({unseen_hit/max(unseen_total,1):.1%})")
    print(f"  R21 unseen hit@200: {r21_unseen}/{unseen_total} ({r21_unseen/max(unseen_total,1):.1%})")

    ph_r24 = pool_hit_r24 / n_val
    ph_r21 = pool_hit_r21 / n_val
    print("\nPool hit@300:")
    print(f"  R24: {ph_r24:.4f}")
    print(f"  R21: {ph_r21:.4f}")
    print(f"  Delta: {ph_r24 - ph_r21:+.4f}")

    # Gates
    r21_fold_hit200 = r21_hit[200]
    gate_hit = r24_hit[200] >= r21_fold_hit200
    gate_unique = unique_vs_r21 > lost_vs_r21
    gate_unseen = unseen_hit >= r21_unseen - 10
    gate_pool = ph_r24 >= ph_r21

    print("\nGates:")
    print(f"  hit@200 >= R21 ({r21_fold_hit200}): {'PASS' if gate_hit else 'FAIL'} ({r24_hit[200]})")
    print(f"  unique > lost: {'PASS' if gate_unique else 'FAIL'} ({unique_vs_r21} > {lost_vs_r21})")
    print(f"  unseen no regression: {'PASS' if gate_unseen else 'FAIL'} ({unseen_hit} vs {r21_unseen})")
    print(f"  pool_hit improves: {'PASS' if gate_pool else 'FAIL'} ({ph_r24:.4f} vs {ph_r21:.4f})")

    all_pass = gate_hit and gate_unique and gate_unseen and gate_pool
    print(f"\n  ALL GATES: {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        print("\n>>> Proceed to 5-fold evaluation <<<")
    else:
        print("\n>>> Fold-0 smoke test failed — iterate on approach <<<")

    # Save results
    results = {
        "fold": fold_id, "run_tag": run_tag,
        "r24_hit": {str(d): r24_hit[d] for d in [20, 50, 100, 200, 300]},
        "r21_hit": {str(d): r21_hit[d] for d in [20, 50, 100, 200, 300]},
        "median_gt_rank_r24": r24_median, "median_gt_rank_r21": r21_median,
        "unique_vs_r21": unique_vs_r21, "lost_vs_r21": lost_vs_r21,
        "unseen_hit": unseen_hit, "unseen_total": unseen_total,
        "r21_unseen": r21_unseen,
        "pool_hit_r24": ph_r24, "pool_hit_r21": ph_r21,
        "gates": {"hit": gate_hit, "unique": gate_unique,
                  "unseen": gate_unseen, "pool": gate_pool, "all": all_pass},
        "hyperparams": {
            "pool_k": POOL_K,
            "distill_epochs": DISTILL_EPOCHS, "distill_lr": DISTILL_LR,
            "distill_alpha": DISTILL_ALPHA,
            "contrastive_temp": CONTRASTIVE_TEMP,
            "symmetric_contrastive": SYMMETRIC_CONTRASTIVE,
            "batch_size": DISTILL_BATCH,
            "margin_pairs_per_query": MARGIN_PAIRS_PER_QUERY,
        },
        "created_at": datetime.now().isoformat(),
    }
    with open(fold_dir / f"results_{run_tag}.json", "w") as f:
        json.dump(results, f, indent=2)

    out_path = REPO_ROOT / "exp" / "eval" / f"expR24_{run_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")

    elapsed = time.time() - t0
    print(f"\n{ts()} Phase 3 complete. Elapsed: {elapsed:.1f}s ({elapsed/3600:.1f}h)")


def main():
    parser = argparse.ArgumentParser(description="R24: Teacher distillation")
    parser.add_argument("--phase", choices=["teacher", "train", "eval", "all"], default="all")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--run_tag", type=str, default="base",
                        help="Tag for model/results isolation (e.g. 'sym1', 'alpha2')")
    parser.add_argument("--symmetric_contrastive", action="store_true",
                        help="Use bidirectional CLIP-style contrastive loss")
    parser.add_argument("--distill_alpha", type=float, default=None,
                        help="Override DISTILL_ALPHA (0=pure contrastive)")
    parser.add_argument("--init_model", type=str, default=None,
                        help="Path to model for initialization (default: pretrained BGE)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    args = parser.parse_args()

    global SYMMETRIC_CONTRASTIVE, DISTILL_ALPHA
    if args.symmetric_contrastive:
        SYMMETRIC_CONTRASTIVE = True
    if args.distill_alpha is not None:
        DISTILL_ALPHA = args.distill_alpha

    if args.phase in ("teacher", "all"):
        phase_teacher(args.fold)
    if args.phase in ("train", "all"):
        phase_train(args.fold, args.run_tag, init_model=args.init_model, epochs=args.epochs)
    if args.phase in ("eval", "all"):
        phase_eval(args.fold, args.run_tag)


if __name__ == "__main__":
    main()
