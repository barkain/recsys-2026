#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R33c: Neural MLP pool reranker on hist_7.

Scores each candidate in pool@300 using:
- frozen R21 query embedding (session text → 768d)
- frozen R21 candidate embedding (768d)
- existing LambdaRank feature vector (29d)
- MLP scorer → relevance score

Trains on fold-0 hist_7 cases where GT is in pool.
CPU-only. No cross-encoder initially.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank_grouped import grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R21_EMBS = REPO / "cache" / "r21_production" / "track_embeddings.npy"
R21_IDS = REPO / "cache" / "r21_production" / "track_ids.json"
R21_MODEL = REPO / "cache" / "r21_production" / "model"
R33_DIR = REPO / "cache" / "r33c"
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
FEAT_NAMES = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
N_LR_FEAT = len(FEAT_NAMES)


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PoolRerankerMLP(nn.Module):
    def __init__(self, emb_dim=768, lr_feat_dim=29, hidden=256, dropout=0.2):
        super().__init__()
        input_dim = emb_dim + emb_dim + lr_feat_dim + 3
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, query_emb, cand_emb, lr_features, interaction_features):
        """
        query_emb: (B, 768) — frozen R21 query embedding
        cand_emb: (B, 768) — frozen R21 candidate embedding
        lr_features: (B, 29) — LambdaRank features
        interaction_features: (B, 4) — cosine sim, dot product, norm diff, etc.
        Returns: (B,) scores
        """
        x = torch.cat([query_emb, cand_emb, lr_features, interaction_features], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Data building
# ---------------------------------------------------------------------------

def build_query_text(case):
    parts = [str(r["content"]) for r in case["history"] if r["role"] == "user"]
    parts.append(case["user_query"])
    return " ".join(parts[-3:])


def encode_queries(cases, model_path):
    """Encode query texts using frozen R21 model."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(model_path), device="cpu")
    queries = [build_query_text(c) for c in cases]
    embs = model.encode(queries, batch_size=64, normalize_embeddings=True,
                        show_progress_bar=True).astype(np.float32)
    del model
    gc.collect()
    return embs


def build_pool_dataset(cases, payload, als_source, als_vecs, als_factors,
                       als_track_to_idx, track_pop, r21_source, r21_embs, r21_id_to_idx,
                       query_embs):
    """Build per-candidate feature rows for pool@300."""
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1
    r21_embs_norm = r21_embs / (np.linalg.norm(r21_embs, axis=1, keepdims=True) + 1e-8)

    dataset = []

    for ci, c in enumerate(cases):
        i = c["_global_idx"]
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        gt = c["gt"]
        if gt not in pool:
            continue

        src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                    for sn, sl in src_lists.items()}
        user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
        played = c["music_turns"]
        n_hist = len(played)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]
        sv = als_vecs[i]
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}

        q_emb = query_embs[ci]
        gt_pool_idx = pool.index(gt)

        # Build negatives: top-20 wrong + same-artist wrong + random wrong
        neg_indices = set()
        for j in range(min(20, len(pool))):
            if j != gt_pool_idx:
                neg_indices.add(j)
        gt_artist = ta.get(gt, "")
        for j, tid in enumerate(pool[:POOL_K]):
            if j != gt_pool_idx and ta.get(tid, "") == gt_artist and gt_artist:
                neg_indices.add(j)
        rng = np.random.RandomState(ci)
        while len(neg_indices) < 24 and len(neg_indices) < len(pool) - 1:
            j = rng.randint(0, len(pool))
            if j != gt_pool_idx:
                neg_indices.add(j)

        candidates = [gt_pool_idx] + sorted(neg_indices)
        labels = [1] + [0] * len(neg_indices)

        for cand_idx, label in zip(candidates, labels):
            tid = pool[cand_idx]
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())

            lr_feat = np.zeros(N_LR_FEAT, dtype=np.float32)
            rank = cand_idx + 1
            lr_feat[0] = 1.0 / rank
            lr_feat[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags:
                lr_feat[2] = len(ct & l_tags) / len(ct | l_tags)
            lr_feat[3] = float(len(tat.get(tid, set()) & now_tok))
            lr_feat[4] = float(len(ttl.get(tid, set()) & now_tok))
            lr_feat[5] = float(len(tmt.get(tid, set()) & all_tok))
            lr_feat[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            lr_feat[7] = rec
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                lr_feat[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                lr_feat[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
            lr_feat[20] = sum(1 for sn in ["A", "B", "C", "D", "F", "ALS"]
                             if tid in src_rank.get(sn, {}))
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    lr_feat[21] = float(np.dot(sv, als_factors[aidx]))
            lr_feat[22] = float(n_hist)
            lr_feat[23] = track_pop.get(tid, 0) / max_pop
            lr_feat[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            lr_feat[25] = float(artist_counts.get(ca, 0)) if ca else 0
            lr_feat[26] = lr_feat[20]
            lr_feat[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            lr_feat[28] = 1.0 if tid in r21_rank_map else 0.0

            cand_r21_idx = r21_id_to_idx.get(tid)
            if cand_r21_idx is not None:
                c_emb = r21_embs_norm[cand_r21_idx]
            else:
                c_emb = np.zeros(768, dtype=np.float32)

            cosine = float(np.dot(q_emb, c_emb))
            dot_raw = float(np.dot(q_emb, r21_embs[cand_r21_idx])) if cand_r21_idx is not None else 0.0
            interact = np.array([cosine, dot_raw, float(rank), float(label)], dtype=np.float32)

            dataset.append({
                "case_idx": ci,
                "q_emb": q_emb,
                "c_emb": c_emb,
                "lr_feat": lr_feat,
                "interact": interact[:3],
                "label": label,
                "pool_rank": rank,
                "tid": tid,
            })

    return dataset


def main():
    t0 = time.time()
    print(f"{ts()} R33c: Neural MLP Pool Reranker")
    print("=" * 70)

    R33_DIR.mkdir(parents=True, exist_ok=True)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    r21_ids = json.loads(Path(R21_IDS).read_text())
    r21_id_to_idx = {tid: i for i, tid in enumerate(r21_ids)}
    r21_embs = np.load(R21_EMBS).astype(np.float32)

    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            sc = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc[als_track_to_idx[t]] = -np.inf
            top = np.argpartition(-sc, 200)[:200]
            top = top[np.argsort(-sc[top])]
            als_source.append([als_track_ids[j] for j in top])
            als_vecs.append(sv)
        else:
            als_source.append([])
            als_vecs.append(None)

    track_pop = build_popularity_stats()
    ta = payload["track_artist"]

    # Fold-0 split
    folds = grouped_session_folds(sessions, seed=0)
    fold0_val = set(folds[0].tolist())

    # Focus on hist_7
    h7_train = [i for i in range(len(cases))
                if cases[i]["n_prior_music"] == 7 and i not in fold0_val]
    h7_val = [i for i in range(len(cases))
              if cases[i]["n_prior_music"] == 7 and i in fold0_val]
    # Also train on hist_5-6 for more data
    h56_train = [i for i in range(len(cases))
                 if cases[i]["n_prior_music"] >= 5 and i not in fold0_val]

    print(f"  hist_7 train: {len(h7_train)}, hist_7 val: {len(h7_val)}")
    print(f"  hist_5+ train: {len(h56_train)}")

    # Encode queries
    query_cache = R33_DIR / "query_embs.npy"
    if query_cache.exists():
        print(f"{ts()} Loading cached query embeddings...")
        all_query_embs = np.load(query_cache)
    else:
        print(f"{ts()} Encoding queries with R21 model...")
        all_query_embs = encode_queries(cases, R21_MODEL)
        np.save(query_cache, all_query_embs)
        print(f"  Saved {all_query_embs.shape}")

    # Build train dataset (hist_5+ for more data)
    train_cases = []
    for i in h56_train:
        c = dict(cases[i])
        c["_global_idx"] = i
        train_cases.append(c)
    train_q_embs = all_query_embs[[c["_global_idx"] for c in train_cases]]

    print(f"\n{ts()} Building train dataset...")
    train_data = build_pool_dataset(
        train_cases, payload, als_source, als_vecs, als_factors,
        als_track_to_idx, track_pop, r21_source, r21_embs, r21_id_to_idx,
        train_q_embs)
    n_pos = sum(1 for d in train_data if d["label"] == 1)
    print(f"  {len(train_data)} rows ({n_pos} positive)")

    # Build val dataset (hist_7 fold-0 only)
    val_cases_list = []
    for i in h7_val:
        c = dict(cases[i])
        c["_global_idx"] = i
        val_cases_list.append(c)
    val_q_embs = all_query_embs[[c["_global_idx"] for c in val_cases_list]]

    print(f"{ts()} Building val dataset...")
    val_data = build_pool_dataset(
        val_cases_list, payload, als_source, als_vecs, als_factors,
        als_track_to_idx, track_pop, r21_source, r21_embs, r21_id_to_idx,
        val_q_embs)
    n_val_pos = sum(1 for d in val_data if d["label"] == 1)
    print(f"  {len(val_data)} rows ({n_val_pos} positive)")

    # Train MLP
    print(f"\n{ts()} Training MLP reranker...")
    model = PoolRerankerMLP(emb_dim=768, lr_feat_dim=N_LR_FEAT, hidden=256, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params/1e6:.2f}M params")

    # Group by case for listwise loss
    from collections import defaultdict
    train_groups: dict[int, list] = defaultdict(list)
    for d in train_data:
        train_groups[d["case_idx"]].append(d)

    model.train()
    for epoch in range(20):
        group_keys = list(train_groups.keys())
        np.random.shuffle(group_keys)
        epoch_loss = 0.0
        n_batches = 0

        for gk in group_keys:
            group = train_groups[gk]
            q = torch.from_numpy(np.stack([d["q_emb"] for d in group]))
            c_e = torch.from_numpy(np.stack([d["c_emb"] for d in group]))
            lr = torch.from_numpy(np.stack([d["lr_feat"] for d in group]))
            inter = torch.from_numpy(np.stack([d["interact"] for d in group]))
            labels = torch.tensor([d["label"] for d in group], dtype=torch.float32)

            scores = model(q, c_e, lr, inter)
            pos_mask = labels == 1
            if pos_mask.sum() == 0:
                continue
            pos_score = scores[pos_mask].mean()
            neg_scores = scores[~pos_mask]
            if len(neg_scores) == 0:
                continue
            loss = -torch.log(torch.sigmoid(pos_score - neg_scores) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Eval on val
        model.training = False
        val_groups: dict[int, list] = defaultdict(list)
        for d in val_data:
            val_groups[d["case_idx"]].append(d)

        ndcg_sum = 0.0
        n_val_cases = 0
        with torch.no_grad():
            for vk in val_groups:
                vg = val_groups[vk]
                q = torch.from_numpy(np.stack([d["q_emb"] for d in vg]))
                c_e = torch.from_numpy(np.stack([d["c_emb"] for d in vg]))
                lr = torch.from_numpy(np.stack([d["lr_feat"] for d in vg]))
                inter = torch.from_numpy(np.stack([d["interact"] for d in vg]))
                labels = [d["label"] for d in vg]
                scores_v = model(q, c_e, lr, inter).numpy()
                ranked = np.argsort(-scores_v)
                for pos, idx in enumerate(ranked[:20]):
                    if labels[idx] == 1:
                        ndcg_sum += 1.0 / np.log2(pos + 2)
                        break
                n_val_cases += 1

        val_ndcg = ndcg_sum / max(n_val_cases, 1)
        model.train()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}  val_h7_ndcg={val_ndcg:.5f}")

    # Final eval
    print(f"\n{ts()} Final evaluation on fold-0 hist_7...")
    model.training = False

    val_groups_final: dict[int, list] = defaultdict(list)
    for d in val_data:
        val_groups_final[d["case_idx"]].append(d)

    # Compare with LambdaRank baseline on same cases
    baseline_ndcg = 0.0
    mlp_ndcg = 0.0
    n_cases = 0
    same_base = 0.0
    same_mlp = 0.0
    diff_base = 0.0
    diff_mlp = 0.0
    n_same = 0
    n_diff = 0

    with torch.no_grad():
        for vk in val_groups_final:
            vg = val_groups_final[vk]
            ci = vg[0]["case_idx"]
            case = val_cases_list[ci]
            gt_artist = ta.get(case["gt"], "")
            is_same = gt_artist and gt_artist in {ta.get(t, "") for t in case["music_turns"]}

            q = torch.from_numpy(np.stack([d["q_emb"] for d in vg]))
            c_e = torch.from_numpy(np.stack([d["c_emb"] for d in vg]))
            lr_f = torch.from_numpy(np.stack([d["lr_feat"] for d in vg]))
            inter = torch.from_numpy(np.stack([d["interact"] for d in vg]))
            labels_v = [d["label"] for d in vg]
            pool_ranks = [d["pool_rank"] for d in vg]

            mlp_scores = model(q, c_e, lr_f, inter).numpy()
            mlp_ranked = np.argsort(-mlp_scores)
            base_ranked = np.argsort(pool_ranks)

            for pos, idx in enumerate(mlp_ranked[:20]):
                if labels_v[idx] == 1:
                    v = 1.0 / np.log2(pos + 2)
                    mlp_ndcg += v
                    if is_same:
                        same_mlp += v
                    else:
                        diff_mlp += v
                    break

            for pos, idx in enumerate(base_ranked[:20]):
                if labels_v[idx] == 1:
                    v = 1.0 / np.log2(pos + 2)
                    baseline_ndcg += v
                    if is_same:
                        same_base += v
                    else:
                        diff_base += v
                    break

            n_cases += 1
            if is_same:
                n_same += 1
            else:
                n_diff += 1

    b_ndcg = baseline_ndcg / max(n_cases, 1)
    m_ndcg = mlp_ndcg / max(n_cases, 1)
    print(f"\n  Pool-rank baseline h7 nDCG: {b_ndcg:.5f}")
    print(f"  MLP reranker h7 nDCG:       {m_ndcg:.5f} ({m_ndcg-b_ndcg:+.5f})")
    if n_same > 0:
        print(f"  same_artist: base={same_base/n_same:.5f} mlp={same_mlp/n_same:.5f}")
    if n_diff > 0:
        print(f"  diff_artist: base={diff_base/n_diff:.5f} mlp={diff_mlp/n_diff:.5f}")

    # Gate
    sep = "=" * 70
    print(f"\n{sep}")
    print("GATE CHECK")
    dh7 = m_ndcg - b_ndcg
    g = dh7 >= 0.005
    print(f"  Δh7 nDCG: {dh7:+.5f} {'PASS' if g else 'FAIL'}")

    out_path = REPO / "exp" / "eval" / "expR33c_neural_reranker.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"baseline_h7": b_ndcg, "mlp_h7": m_ndcg, "delta": dh7,
                    "n_cases": n_cases, "n_same": n_same, "n_diff": n_diff}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
