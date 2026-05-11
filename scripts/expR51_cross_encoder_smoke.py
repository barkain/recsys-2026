#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301,S311
"""R51: Cross-encoder smoke test.

Fine-tune a cross-encoder (ms-marco-MiniLM-L-6-v2) to score (conversation, candidate)
text pairs. Blend CE scores with LR scores and measure on fold-0 hist_7.

TWO SEPARATE PHASES (to avoid torch+lightgbm segfault):
  --phase build:      lightgbm, NO torch/transformers
  --phase train_eval: torch/transformers, NO lightgbm

Run:
  .venv/bin/python scripts/expR51_cross_encoder_smoke.py --phase build
  .venv/bin/python scripts/expR51_cross_encoder_smoke.py --phase train_eval
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


# ═══════════════════════════════════════════════════════════════════
# PHASE: build
# ═══════════════════════════════════════════════════════════════════

def phase_build():
    """Build R39 baseline, training data, and fold-0 scoring data.
    Imports lightgbm. Does NOT import torch/transformers.
    """
    t0 = time.time()
    print(f"{ts()} R51 Phase: build")
    print("=" * 70)

    os.environ["OMP_NUM_THREADS"] = "4"

    import lightgbm as lgb

    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
    from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
    from scripts.tune_postrank_v23 import tokens

    FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
    FEAT_ALBUM = [
        "same_album_last1", "same_album_last3", "same_album_any",
        "album_history_count", "pool_same_album_count",
    ]
    FEAT_ALL = FEAT_BASE + FEAT_ALBUM
    n_feat_base = len(FEAT_BASE)
    n_feat = len(FEAT_ALL)

    # ── Load data ──────────────────────────────────────────────────
    print(f"{ts()} Loading payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    # ── Track metadata ─────────────────────────────────────────────
    print(f"{ts()} Loading track metadata...")
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_meta = {}
    for item in ds:
        tid = str(item["track_id"])
        tn_raw = item.get("track_name", [])
        tn = str(tn_raw[0]) if isinstance(tn_raw, list) and tn_raw else str(tn_raw) if tn_raw else ""

        an_raw = item.get("artist_name", [])
        an = ", ".join(str(a) for a in an_raw) if isinstance(an_raw, list) else str(an_raw) if an_raw else ""

        aln_raw = item.get("album_name", [])
        aln = str(aln_raw[0]) if isinstance(aln_raw, list) and aln_raw else str(aln_raw) if aln_raw else ""

        alid_raw = item.get("album_id", [])
        alid = str(alid_raw[0]) if isinstance(alid_raw, list) and alid_raw else ""

        tl_raw = item.get("tag_list", [])
        tags = [str(t) for t in tl_raw[:5]] if isinstance(tl_raw, list) else []

        track_meta[tid] = {
            "track_name": tn, "artist_name": an, "album_name": aln,
            "album_id": alid, "tags": tags,
        }

    # Album mapping
    track_album = {}
    for tid, m in track_meta.items():
        alid = m.get("album_id", "")
        track_album[tid] = alid if alid else m.get("album_name", "")
    non_empty = sum(1 for v in track_album.values() if v)
    print(f"  Track metadata: {len(track_meta)} tracks, {non_empty} with album")

    # ── ALS ────────────────────────────────────────────────────────
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

    folds = grouped_session_folds(sessions, seed=0)

    # ── Build feature matrix + pools ───────────────────────────────
    print(f"{ts()} Building R39 feature matrix ({n_feat} features)...")

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = []

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        pools.append(pool)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                    for sn, sl in src_lists.items()}
        user_msgs = [str(r["content"]) for r in c["history"]
                     if r["role"] == "user"] + [c["user_query"]]
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

        # Album precomputation
        last1_album = track_album.get(played[-1], "") if played else ""
        last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
        all_albums = [track_album.get(t, "") for t in played]
        album_hist_counts = Counter(a for a in all_albums if a)

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]

            # Base 29 features (EXACT R39)
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
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
            row[20] = sum(1 for sn in ["A", "B", "C", "D", "F", "ALS"]
                         if tid in src_rank.get(sn, {}))
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            row[28] = 1.0 if tid in r21_rank_map else 0.0

            # Album features (EXACT R39)
            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for tid2 in pool[:POOL_K]
                                   if track_album.get(tid2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

    # ── CV5 LambdaRank ─────────────────────────────────────────────
    print(f"{ts()} Running CV5 LambdaRank (R39 config)...")
    case_ndcg = np.zeros(n)
    lr_scores_all = np.full((n, POOL_K), -np.inf, dtype=np.float64)
    lr_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
        "verbose": -1, "seed": 0,
    }

    for fi in range(5):
        val_set = set(folds[fi].tolist())
        tr = [j for j in range(n) if j not in val_set]
        va = sorted(val_set)
        X_tr, y_tr, g_tr = [], [], []
        X_va, y_va, g_va = [], [], []
        for idx in tr:
            s = int(sizes[idx])
            for k in range(s):
                X_tr.append(X[idx, k])
                y_tr.append(1.0 if k == gt_idx[idx] else 0.0)
            g_tr.append(s)
        for idx in va:
            s = int(sizes[idx])
            for k in range(s):
                X_va.append(X[idx, k])
                y_va.append(1.0 if k == gt_idx[idx] else 0.0)
            g_va.append(s)
        ds_tr = lgb.Dataset(np.array(X_tr), label=np.array(y_tr),
                            group=g_tr, feature_name=list(FEAT_ALL))
        ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                            group=g_va, reference=ds_tr)
        model = lgb.train(lr_params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
        preds = model.predict(np.array(X_va))
        offset = 0
        for idx in va:
            s = int(sizes[idx])
            sc = preds[offset:offset + s]
            lr_scores_all[idx, :s] = sc
            offset += s
            if gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg[idx] = 1.0 / np.log2(gt_pos[0] + 2)

    # Metrics
    h7_indices = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_ndcg = float(np.mean([case_ndcg[i] for i in h7_indices]))
    cv5 = float(np.mean(case_ndcg))
    print(f"  R39 reproduction: h7={h7_ndcg:.5f}  cv5={cv5:.5f}")

    # HARD CHECK
    expected_h7 = 0.24298
    if abs(h7_ndcg - expected_h7) > 0.001:
        print("\n  *** HARD CHECK FAILED ***")
        print(f"  Expected h7 ~{expected_h7}, got {h7_ndcg:.5f}")
        print(f"  Difference: {abs(h7_ndcg - expected_h7):.5f} > 0.001")
        print("  ABORTING.")
        sys.exit(1)
    print(f"  HARD CHECK PASSED: h7={h7_ndcg:.5f} within +/-0.001 of {expected_h7}")

    # ── Compute baseline_top100_only metrics ───────────────────────
    print(f"\n{ts()} Computing baseline_top100_only metrics...")

    fold0_set = set(folds[0].tolist())
    fold0_h7 = [i for i in h7_indices if i in fold0_set]

    # For fold-0 h7 cases, compute h7 using only top-100 LR-ranked candidates
    top100_ndcg_all = []  # all fold-0 h7
    top100_ndcg_gt_in = []  # fold-0 h7 where GT is in top-100

    for i in fold0_h7:
        ps = int(sizes[i])
        sc = lr_scores_all[i, :ps]
        lr_ranked = np.argsort(-sc)
        top100_pool_indices = lr_ranked[:min(100, ps)]

        gt_in_top100 = False
        ndcg_val = 0.0
        if gt_idx[i] >= 0 and gt_idx[i] in top100_pool_indices:
            gt_in_top100 = True
            # Re-rank within top-100
            top100_scores = sc[top100_pool_indices]
            re_ranked = np.argsort(-top100_scores)
            gt_local = np.where(top100_pool_indices[re_ranked] == gt_idx[i])[0]
            if len(gt_local) > 0 and gt_local[0] < 20:
                ndcg_val = 1.0 / np.log2(gt_local[0] + 2)

        top100_ndcg_all.append(ndcg_val)
        if gt_in_top100:
            top100_ndcg_gt_in.append(ndcg_val)

    top100_h7 = float(np.mean(top100_ndcg_all)) if top100_ndcg_all else 0.0
    top100_h7_gt_in = float(np.mean(top100_ndcg_gt_in)) if top100_ndcg_gt_in else 0.0

    print(f"  fold-0 h7 cases: {len(fold0_h7)}")
    print(f"  top100 h7 (all): {top100_h7:.5f}")
    print(f"  top100 h7 (GT in top100): {top100_h7_gt_in:.5f} ({len(top100_ndcg_gt_in)} cases)")

    # ── Build query/candidate text strings ─────────────────────────
    print(f"\n{ts()} Building query and candidate text strings...")

    def build_query_text(case):
        """Build query text from conversation history and recent tracks."""
        # Last 3-5 turns of user/assistant
        conv_entries = []
        for h in case["history"]:
            role = h["role"]
            if role in ("user", "assistant"):
                conv_entries.append((role, str(h["content"])))
        # Take last 5 (or 3 if fewer)
        conv_entries = conv_entries[-5:]
        # Add current user query
        conv_entries.append(("user", case["user_query"]))

        lines = ["Conversation:"]
        for role, content in conv_entries:
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")

        # Recent tracks (last 8)
        played = case["music_turns"]
        recent = played[-8:]
        if recent:
            lines.append("")
            lines.append("Recent tracks:")
            for idx, tid in enumerate(recent, 1):
                m = track_meta.get(tid, {})
                tname = m.get("track_name", "unknown")
                aname = m.get("artist_name", "unknown")
                lines.append(f"{idx}. {tname} | {aname}")

        return "\n".join(lines)

    def build_candidate_text(tid):
        """Build candidate text string."""
        m = track_meta.get(tid, {})
        tname = m.get("track_name", "unknown")
        aname = m.get("artist_name", "unknown")
        alname = m.get("album_name", "")
        tags = m.get("tags", [])
        tag_str = ", ".join(tags) if tags else ""
        return f"{tname} | {aname} | {alname} | {tag_str}"

    # ── Training data: hist_5+ from folds 1-4 ─────────────────────
    print(f"\n{ts()} Constructing training data...")

    train_folds = set()
    for fi in range(1, 5):
        train_folds.update(folds[fi].tolist())

    train_examples = []
    n_skipped_no_gt = 0
    n_skipped_gt_not_top100 = 0

    for i in sorted(train_folds):
        c = cases[i]
        if c["n_prior_music"] < 5:
            continue
        if gt_idx[i] < 0:
            n_skipped_no_gt += 1
            continue

        ps = int(sizes[i])
        sc = lr_scores_all[i, :ps]
        lr_ranked = np.argsort(-sc)

        # Find GT rank in LR ordering
        gt_pool_idx = gt_idx[i]
        gt_lr_rank_arr = np.where(lr_ranked == gt_pool_idx)[0]
        if len(gt_lr_rank_arr) == 0:
            n_skipped_no_gt += 1
            continue
        gt_lr_rank = int(gt_lr_rank_arr[0])  # 0-indexed

        if gt_lr_rank >= 100:
            n_skipped_gt_not_top100 += 1
            continue

        pool = pools[i]
        gt_tid = c["gt"]

        # Build negatives
        negatives = []

        # 5 from LR top-20 wrong candidates
        top5_wrong = []
        for j in range(min(20, ps)):
            pidx = int(lr_ranked[j])
            if pidx != gt_pool_idx:
                top5_wrong.append(pidx)
            if len(top5_wrong) >= 5:
                break
        # If we need more, continue deeper in the top-100.
        if len(top5_wrong) < 5:
            for j in range(20, min(100, ps)):
                pidx = int(lr_ranked[j])
                if pidx != gt_pool_idx:
                    top5_wrong.append(pidx)
                if len(top5_wrong) >= 5:
                    break
        negatives.extend(top5_wrong[:5])

        # 5 from candidates ranked near GT (GT_rank +/- 3)
        near_gt = []
        for offset in [-3, -2, -1, 1, 2, 3]:
            r = gt_lr_rank + offset
            if 0 <= r < min(100, ps):
                pidx = int(lr_ranked[r])
                if pidx != gt_pool_idx and pidx not in negatives:
                    near_gt.append(pidx)
        negatives.extend(near_gt[:5])

        # 5 sampled from remaining top-100 (deterministic per case)
        already_selected = set(negatives) | {gt_pool_idx}
        remaining = np.array([int(lr_ranked[j]) for j in range(min(100, ps))
                              if int(lr_ranked[j]) not in already_selected])
        case_rng = np.random.RandomState(i)
        case_rng.shuffle(remaining)
        negatives.extend(remaining[:5].tolist())

        # Pad if we don't have enough (shouldn't happen with pool_k=300)
        while len(negatives) < 15:
            for j in range(min(100, ps)):
                pidx = int(lr_ranked[j])
                if pidx != gt_pool_idx and pidx not in set(negatives):
                    negatives.append(pidx)
                if len(negatives) >= 15:
                    break
            break

        negatives = negatives[:15]

        # Build text strings
        query_text = build_query_text(c)
        candidate_texts = [build_candidate_text(gt_tid)]  # GT at index 0
        for neg_pidx in negatives:
            candidate_texts.append(build_candidate_text(pool[neg_pidx]))

        train_examples.append({
            "query_text": query_text,
            "candidate_texts": candidate_texts,
            "gt_index": 0,
        })

    print(f"  n_train_cases: {len(train_examples)}")
    print(f"  n_train_pairs: {len(train_examples) * 16}")
    print(f"  skipped (no GT in pool): {n_skipped_no_gt}")
    print(f"  skipped (GT not in top-100): {n_skipped_gt_not_top100}")

    # ── Fold-0 scoring data ────────────────────────────────────────
    print(f"\n{ts()} Building fold-0 scoring data...")

    fold0_scoring_cases = []
    fold0_query_texts = []
    fold0_candidate_texts_per_case = []

    for i in fold0_h7:
        c = cases[i]
        ps = int(sizes[i])
        sc = lr_scores_all[i, :ps]
        lr_ranked = np.argsort(-sc)
        top100_pool_indices = lr_ranked[:min(100, ps)]

        top100_tids = [pools[i][int(idx)] for idx in top100_pool_indices]
        top100_lr_scores = [float(sc[int(idx)]) for idx in top100_pool_indices]

        gt_tid = c["gt"]
        gt_in_top100 = gt_tid in top100_tids

        fold0_scoring_cases.append({
            "case_idx": i,
            "pool": top100_tids,
            "lr_scores": top100_lr_scores,
            "gt_tid": gt_tid,
            "gt_in_top100": gt_in_top100,
        })

        fold0_query_texts.append(build_query_text(c))
        fold0_candidate_texts_per_case.append(
            [build_candidate_text(tid) for tid in top100_tids]
        )

    n_fold0_gt_in = sum(1 for ec in fold0_scoring_cases if ec["gt_in_top100"])
    print(f"  n_fold0_hist7: {len(fold0_h7)}")
    print(f"  n_fold0_gt_in_top100: {n_fold0_gt_in}")

    # ── Save ───────────────────────────────────────────────────────
    print(f"\n{ts()} Saving artifacts...")

    cache_dir = REPO / "cache"
    cache_dir.mkdir(exist_ok=True)

    with open(cache_dir / "r51_train_examples.pkl", "wb") as f:
        pickle.dump(train_examples, f)
    print(f"  Saved r51_train_examples.pkl ({len(train_examples)} examples)")

    fold0_data = {
        "cases": fold0_scoring_cases,
        "query_texts": fold0_query_texts,
        "candidate_texts_per_case": fold0_candidate_texts_per_case,
    }
    with open(cache_dir / "r51_fold0_eval.pkl", "wb") as f:
        pickle.dump(fold0_data, f)
    print(f"  Saved r51_fold0_eval.pkl ({len(fold0_scoring_cases)} cases)")

    pool300_h7_fold0 = float(np.mean([case_ndcg[i] for i in fold0_h7])) if fold0_h7 else 0.0
    baselines = {
        "pool300_h7_all": h7_ndcg,
        "pool300_h7_fold0": pool300_h7_fold0,
        "top100_h7": top100_h7,
        "top100_h7_gt_in_top100": top100_h7_gt_in,
    }
    with open(cache_dir / "r51_baselines.json", "w") as f:
        json.dump(baselines, f, indent=2)
    print(f"  Saved r51_baselines.json: {baselines}")

    elapsed = time.time() - t0
    print(f"\n{ts()} Build phase complete. Elapsed: {elapsed:.1f}s ({elapsed/60:.1f}min)")


# ═══════════════════════════════════════════════════════════════════
# PHASE: train_eval
# ═══════════════════════════════════════════════════════════════════

def phase_train_eval():
    """Train cross-encoder and measure on fold-0.
    Imports torch/transformers. Does NOT import lightgbm.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    t0 = time.time()
    print(f"{ts()} R51 Phase: train_eval")
    print("=" * 70)

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_dir = REPO / "cache"

    # ── Load training data ─────────────────────────────────────────
    print(f"{ts()} Loading training data...")
    with open(cache_dir / "r51_train_examples.pkl", "rb") as f:
        train_examples = pickle.load(f)
    print(f"  {len(train_examples)} training examples")

    # Shuffle deterministically with numpy
    rng = np.random.RandomState(42)
    indices = np.arange(len(train_examples))
    rng.shuffle(indices)
    train_examples = [train_examples[j] for j in indices]

    # ── Load model ─────────────────────────────────────────────────
    print(f"{ts()} Loading cross-encoder model...")
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    model.train()

    grad_accum = 4
    n_steps = len(train_examples)
    n_optimizer_steps = (n_steps + grad_accum - 1) // grad_accum
    warmup_steps = int(0.1 * n_optimizer_steps)
    print(f"  Total examples: {n_steps}, optimizer steps: {n_optimizer_steps}, warmup: {warmup_steps}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Linear warmup scheduler
    def get_lr_scale(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_scale)

    # ── Training loop ──────────────────────────────────────────────
    print(f"\n{ts()} Starting training (1 epoch, grad_accum={grad_accum})...")
    running_loss = 0.0
    log_interval = 100

    optimizer.zero_grad()

    for step, example in enumerate(train_examples):
        logit_list = []
        for ct in example["candidate_texts"]:
            inp = tokenizer(
                example["query_text"], ct,
                padding=True, truncation=True, max_length=256,
                return_tensors="pt"
            )
            logit_list.append(model(**inp).logits.view(-1)[0])
        logits = torch.stack(logit_list)  # [16]
        loss = F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long))
        loss = loss / grad_accum
        loss.backward()

        if step == 0:
            grad_norm = sum(p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None)
            print(f"  DEBUG step=0 grad_norm={grad_norm:.6f} loss={loss.item():.4f} logits={logits.detach().tolist()[:4]}")

        running_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            lr_now = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            eta_min = elapsed / (step + 1) * (n_steps - step - 1) / 60
            print(f"  {ts()} step {step+1}/{n_steps}  loss={avg_loss:.4f}  "
                  f"lr={lr_now:.2e}  eta={eta_min:.1f}min")
            running_loss = 0.0

    # Final optimizer step for remaining grads
    if n_steps % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad()

    train_elapsed = time.time() - t0
    print(f"\n{ts()} Training complete. Elapsed: {train_elapsed:.1f}s ({train_elapsed/60:.1f}min)")

    # ── Save model ─────────────────────────────────────────────────
    save_dir = cache_dir / "r51_ce_model"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"  Saved model to {save_dir}")

    # ── Scoring on fold-0 ─────────────────────────────────────────
    print(f"\n{ts()} Loading fold-0 scoring data...")
    with open(cache_dir / "r51_fold0_eval.pkl", "rb") as f:
        fold0_data = pickle.load(f)

    scoring_cases = fold0_data["cases"]
    query_texts = fold0_data["query_texts"]
    candidate_texts_per_case = fold0_data["candidate_texts_per_case"]

    with open(cache_dir / "r51_baselines.json") as f:
        baselines = json.load(f)

    print(f"  {len(scoring_cases)} fold-0 hist_7 cases")
    print(f"  Baselines: {baselines}")

    # Score all candidates
    print(f"\n{ts()} Scoring fold-0 candidates with cross-encoder...")
    # Switch model to inference mode
    _set_inference = getattr(model, "eval")
    _set_inference()

    ce_scores_per_case = []
    for ci in range(len(scoring_cases)):
        query = query_texts[ci]
        cands = candidate_texts_per_case[ci]

        # Score one at a time to avoid OOM
        all_logits = []
        for ct in cands:
            inp = tokenizer(
                query, ct,
                padding=True, truncation=True, max_length=256,
                return_tensors="pt"
            )
            with torch.no_grad():
                logit = model(**inp).logits.squeeze().item()
            all_logits.append(logit)

        ce_scores = np.array(all_logits)
        ce_scores_per_case.append(ce_scores)

        if (ci + 1) % 50 == 0:
            print(f"  {ts()} Scored {ci+1}/{len(scoring_cases)} cases")

    print(f"  {ts()} Scoring complete.")

    # ── Sweep beta ─────────────────────────────────────────────────
    print(f"\n{ts()} Sweeping beta values...")
    print("=" * 70)

    betas = [0.05, 0.1, 0.2, 0.5, 1.0]
    sweep_results = {}

    # Load track_artist for same/diff analysis
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    ta = payload["track_artist"]
    cases_all = payload["cases"]

    for beta in betas:
        ndcg_all = []
        ndcg_gt_in = []
        recovered = 0
        lost = 0
        top20_overlaps = []
        top1_changed = 0

        for ci, ec in enumerate(scoring_cases):
            lr_sc = np.array(ec["lr_scores"])
            ce_sc = ce_scores_per_case[ci]
            pool_tids = ec["pool"]
            gt_tid = ec["gt_tid"]
            gt_in = ec["gt_in_top100"]

            # Z-score normalize
            lr_z = (lr_sc - lr_sc.mean()) / (lr_sc.std() + 1e-8)
            ce_z = (ce_sc - ce_sc.mean()) / (ce_sc.std() + 1e-8)

            # Blend
            final = lr_z + beta * ce_z

            # Rank
            ranked_indices = np.argsort(-final)
            top20_tids = [pool_tids[j] for j in ranked_indices[:20]]

            # nDCG@20
            ndcg_val = 0.0
            if gt_tid in top20_tids:
                gt_pos = top20_tids.index(gt_tid)
                ndcg_val = 1.0 / np.log2(gt_pos + 2)

            ndcg_all.append(ndcg_val)
            if gt_in:
                ndcg_gt_in.append(ndcg_val)

            # Baseline top-100 rank for recovered/lost
            lr_ranked = np.argsort(-lr_sc)
            baseline_top20_tids = [pool_tids[j] for j in lr_ranked[:20]]

            baseline_hit = gt_tid in baseline_top20_tids
            blend_hit = gt_tid in top20_tids

            top20_overlaps.append(
                len(set(baseline_top20_tids) & set(top20_tids)) / max(len(baseline_top20_tids), 1)
            )
            if baseline_top20_tids and top20_tids and baseline_top20_tids[0] != top20_tids[0]:
                top1_changed += 1

            if not baseline_hit and blend_hit:
                recovered += 1
            elif baseline_hit and not blend_hit:
                lost += 1

        h7_all = float(np.mean(ndcg_all)) if ndcg_all else 0.0
        h7_gt_in = float(np.mean(ndcg_gt_in)) if ndcg_gt_in else 0.0

        # Same/diff artist analysis
        same_ndcg = []
        diff_ndcg = []
        for ci, ec in enumerate(scoring_cases):
            case_idx = ec["case_idx"]
            c = cases_all[case_idx]
            gt_tid_local = ec["gt_tid"]
            gt_artist = ta.get(gt_tid_local, "")
            played_artists = {ta.get(t, "") for t in c["music_turns"]} - {""}
            is_same = gt_artist and gt_artist in played_artists

            if is_same:
                same_ndcg.append(ndcg_all[ci])
            else:
                diff_ndcg.append(ndcg_all[ci])

        same_val = float(np.mean(same_ndcg)) if same_ndcg else 0.0
        diff_val = float(np.mean(diff_ndcg)) if diff_ndcg else 0.0

        net = recovered - lost

        sweep_results[beta] = {
            "h7_all": h7_all,
            "h7_gt_in_top100": h7_gt_in,
            "same": same_val,
            "diff": diff_val,
            "recovered": recovered,
            "lost": lost,
            "net": net,
            "top20_mean_overlap": float(np.mean(top20_overlaps)) if top20_overlaps else 0.0,
            "top1_changed": top1_changed,
        }

        delta_all = h7_all - baselines["top100_h7"]
        delta_gt_in = h7_gt_in - baselines["top100_h7_gt_in_top100"]

        print(f"  beta={beta:.2f}  h7_all={h7_all:.5f} (d={delta_all:+.5f})  "
              f"h7_gt_in={h7_gt_in:.5f} (d={delta_gt_in:+.5f})  "
              f"same={same_val:.5f}  diff={diff_val:.5f}  "
              f"rec={recovered} lost={lost} net={net:+d}  "
              f"top20_overlap={np.mean(top20_overlaps):.3f} top1_changed={top1_changed}")

    # ── Hard stop check ────────────────────────────────────────────
    best_gt_in = max(r["h7_gt_in_top100"] for r in sweep_results.values())
    baseline_gt_in = baselines["top100_h7_gt_in_top100"]
    recommendation = "investigate"

    if best_gt_in <= baseline_gt_in:
        print("\n  HARD STOP: CE does not improve ranking")
        print(f"  Best h7_gt_in_top100: {best_gt_in:.5f} vs baseline {baseline_gt_in:.5f}")
        recommendation = "stop"
    else:
        best_beta = max(sweep_results, key=lambda b: sweep_results[b]["h7_gt_in_top100"])
        improvement = best_gt_in - baseline_gt_in
        print(f"\n  Best beta: {best_beta} -> h7_gt_in_top100={best_gt_in:.5f} "
              f"(+{improvement:.5f} vs baseline)")
        # Check if improvement is meaningful
        if improvement < 0.005:
            recommendation = "marginal"
        else:
            recommendation = "promising"

    # ── Summary ────────────────────────────────────────────────────
    sep = "=" * 70
    print(f"\n{sep}")
    print("R51 CROSS-ENCODER SMOKE TEST SUMMARY")
    print(sep)
    print("  Baselines:")
    print(f"    pool300_h7_all:          {baselines['pool300_h7_all']:.5f}")
    print(f"    pool300_h7_fold0:        {baselines['pool300_h7_fold0']:.5f}")
    print(f"    top100_h7:               {baselines['top100_h7']:.5f}")
    print(f"    top100_h7_gt_in_top100:  {baselines['top100_h7_gt_in_top100']:.5f}")
    print(f"  Recommendation: {recommendation}")

    # ── Save results ───────────────────────────────────────────────
    output = {
        "experiment": "R51_cross_encoder_smoke",
        "timestamp": datetime.now().isoformat(),
        "baselines": baselines,
        "sweep": {str(b): v for b, v in sweep_results.items()},
        "recommendation": recommendation,
        "training": {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "n_train_examples": len(train_examples),
            "epochs": 1,
            "lr": 2e-5,
            "grad_accum": 4,
            "max_length": 512,
            "warmup_pct": 0.1,
            "device": "cpu",
        },
        "scoring": {
            "n_fold0_hist7": len(scoring_cases),
            "n_fold0_gt_in_top100": sum(1 for ec in scoring_cases if ec["gt_in_top100"]),
        },
    }

    out_path = REPO / "exp" / "eval" / "expR51_cross_encoder_smoke.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n{ts()} Saved to {out_path}")

    total_elapsed = time.time() - t0
    print(f"Total elapsed: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def phase_lr_sweep():
    """Mini-overfit test: 200 examples at 3 LR values. Verifies gradient path and finds working LR."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    cache_dir = REPO / "cache"
    with open(cache_dir / "r51_train_examples.pkl", "rb") as f:
        train_examples = pickle.load(f)

    rng = np.random.RandomState(42)
    indices = np.arange(len(train_examples))
    rng.shuffle(indices)
    subset = [train_examples[j] for j in indices[:200]]

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for lr in [1e-5, 5e-5, 1e-4]:
        print(f"\n{'='*60}")
        print(f"LR = {lr}")
        print(f"{'='*60}")

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        losses = []
        for step, example in enumerate(subset):
            logit_list = []
            for ct in example["candidate_texts"]:
                inp = tokenizer(
                    example["query_text"], ct,
                    padding=True, truncation=True, max_length=256,
                    return_tensors="pt"
                )
                logit_list.append(model(**inp).logits.view(-1)[0])
            logits = torch.stack(logit_list)
            loss = F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long))
            loss.backward()

            if step == 0:
                grad_norm = sum(p.grad.detach().norm().item() for p in model.parameters() if p.grad is not None)
                print(f"  grad_norm={grad_norm:.6f}  logits[:4]={logits.detach().tolist()[:4]}")

            if (step + 1) % 4 == 0:
                optimizer.step()
                optimizer.zero_grad()

            losses.append(loss.item())
            if (step + 1) % 50 == 0:
                avg = np.mean(losses[-50:])
                print(f"  step {step+1}/200  avg_loss={avg:.4f}")

        final_avg = np.mean(losses[-50:])
        below_random = final_avg < 2.77
        print(f"  FINAL avg_loss={final_avg:.4f}  below_random(2.77)={below_random}")
        del model, optimizer

    print("\nLR sweep complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="R51: Cross-encoder smoke test")
    parser.add_argument("--phase", required=True, choices=["build", "train_eval", "lr_sweep"],
                        help="build: lightgbm, train_eval: torch, lr_sweep: quick LR test")
    args = parser.parse_args()

    if args.phase == "build":
        phase_build()
    elif args.phase == "train_eval":
        phase_train_eval()
    elif args.phase == "lr_sweep":
        phase_lr_sweep()
