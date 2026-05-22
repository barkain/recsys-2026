#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R76 Phase 0A — Build top-30 OOF training dataset for neural residual ranker.

No model training in this script. Just produces the dataset and verifies
sanity stats:
- GT-in-top30 rate (fold-0 all, h7)
- baseline OOF R54c nDCG reproduced
- per-fold feature value distributions
- same/diff artist split

For fold-0:
1. Train R54c-style sibling LR on folds 1-4 → OOF_R54c_for_fold0 (same as R71).
2. Score fold-0 cases with R54-stacked RRF top-300 pool, take OOF top-30
   per case.
3. For each of those candidates, compute:
   - oof_r54c_score (raw LightGBM output)
   - oof_r54c_rank (1..30, normalized)
   - 37 LR features (FEAT_R39_ALL + FEAT_R54 — what R54c uses)
   - 3 R68 features (r68_rank_inv, r68_presence, r68_cosine — fold-0 OOF R68 lists)
   - 5 semantic features:
       query_candidate_cosine (BGE-large dot product)
       max_sim_to_played       (max cosine to any played track)
       mean_sim_to_played      (mean cosine to all played tracks)
       max_artist_centroid_cos (cosine to artist-centroid of played catalog)
       num_history             (count of played tracks)
   - label (1.0 if GT, 0.0 otherwise)

Output: cache/r76/top30_fold0_dataset.parquet (or .pkl if pyarrow unavailable)
Eval: docs/r76_phase0a_dataset_audit.md
"""
from __future__ import annotations
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expR54_phase3_blind_submission import (  # noqa: E402
    FEAT_R39_ALL, FEAT_R54, FEAT_ALL,
    _featurize_row,
)
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
POOL_K = 300
RRF_K = 20
TOP_30 = 30
TOP_K = 20
FOLD = 0

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R68_DIR = REPO / "cache" / "r68" / "phase0_fold0"
R68_DEV_LISTS = R68_DIR / "oof_r68_lists_fold0.json"
R68_TRACK_EMBS = R68_DIR / "track_embeddings.npy"
R68_TRACK_IDS = R68_DIR / "track_ids.json"
R68_QUERY_EMBS = R68_DIR / "query_embeddings_dev.npy"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"

OUT_DIR = REPO / "cache" / "r76"
OUT_DATA = OUT_DIR / "top30_fold0_dataset.pkl"
OUT_JSON = REPO / "exp" / "eval" / "expR76_phase0a_dataset_stats.json"
OUT_DOC = REPO / "docs" / "r76_phase0a_dataset_audit.md"


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R76 Phase 0A — Build top-30 OOF dataset (fold-0)")
    print("=" * 70)

    print(f"{ts()} Loading payloads ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    track_artist = maps["track_artist"]

    print(f"{ts()} Loading W0 fold map ...", flush=True)
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold0_idx = [i for i in range(n) if case_fold[i] == FOLD]
    train_idx = [i for i in range(n) if case_fold[i] != FOLD]
    print(f"  fold-0: {len(fold0_idx)}  train: {len(train_idx)}")

    print(f"{ts()} Loading R68 fold-0 lists + embeddings ...", flush=True)
    with open(R68_DEV_LISTS) as f:
        r68_data = json.load(f)
    r68_val_idx = r68_data.get("val_idx") or r68_data["manifest"]["val_idx"]
    r68_lists_by_case = {
        int(case_idx): [(str(t), float(s)) for t, s in r68_data["lists"][k_pos]]
        for k_pos, case_idx in enumerate(r68_val_idx)
    }
    r68_track_embs = np.load(R68_TRACK_EMBS)
    r68_track_ids = json.load(open(R68_TRACK_IDS))
    r68_query_embs = np.load(R68_QUERY_EMBS)
    case_to_qemb_row = {int(case_idx): k_pos
                       for k_pos, case_idx in enumerate(r68_val_idx)}
    r68_track_to_idx = {tid: j for j, tid in enumerate(r68_track_ids)}
    print(f"  R68 lists: {len(r68_lists_by_case)}  query_embs: {r68_query_embs.shape}  "
          f"track_embs: {r68_track_embs.shape}")

    print(f"{ts()} Building case index ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    def featurize37(case_idx, src_lists, pool):
        case = cases[case_idx]
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}
        return _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[case_idx],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][case_idx],
            track_pop, max_pop, track_album,
        )

    # ---- Step 1: Train OOF R54c-style LR on folds 1-4 ----
    print(f"\n{ts()} === Step 1: Train OOF R54c on folds 1-4 ({len(train_idx)} cases) ===")
    X_tr, y_tr, g_tr = [], [], []
    t_feat = time.time()
    for ki, i in enumerate(train_idx):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gi = pool.index(gt) if gt in pool else -1
        feats = featurize37(i, src_lists, pool)
        for k_row in range(len(pool)):
            X_tr.append(feats[k_row])
            y_tr.append(1.0 if k_row == gi else 0.0)
        g_tr.append(len(pool))
        if (ki + 1) % 1000 == 0:
            print(f"    train feats {ki + 1}/{len(train_idx)} ({time.time() - t_feat:.0f}s)",
                  flush=True)
    t_lr = time.time()
    ds = lgb.Dataset(np.array(X_tr, dtype=np.float64),
                     label=np.array(y_tr, dtype=np.float64),
                     group=g_tr, feature_name=list(FEAT_ALL))
    oof_r54c = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    oof_r54c_path = REPO / "cache" / "r76" / "oof_r54c_fold0.txt"
    oof_r54c_path.parent.mkdir(parents=True, exist_ok=True)
    oof_r54c.save_model(str(oof_r54c_path))
    print(f"  OOF R54c trained in {time.time() - t_lr:.0f}s, saved {oof_r54c_path}",
          flush=True)
    del X_tr, y_tr, g_tr, ds

    # ---- Step 2: Score fold-0, take top-30, compute features ----
    print(f"\n{ts()} === Step 2: Score fold-0, build top-30 candidates with features ===")
    # Precompute query history artist centroids per fold-0 case
    rows_out = []  # one dict per (case, candidate)
    case_summaries = []  # per-case stats

    # baseline OOF R54c top-20 nDCG accumulator for sanity check
    baseline_ndcg_at_20_list = []
    baseline_h7_ndcg_list = []
    baseline_same_ndcg_list = []
    baseline_diff_ndcg_list = []

    gt_in_top30_count = 0
    h7_gt_in_top30_count = 0
    h7_idx = [i for i in fold0_idx if int(cases[i]["n_prior_music"]) == 7]
    same_idx = [i for i in fold0_idx if same_artist_case(cases[i], track_artist)]

    t_eval = time.time()
    for ki, i in enumerate(fold0_idx):
        case = cases[i]
        is_h7 = int(case["n_prior_music"]) == 7
        is_same = same_artist_case(case, track_artist)

        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        feats = featurize37(i, src_lists, pool)
        scores = oof_r54c.predict(feats)
        order = np.argsort(-scores, kind="mergesort")
        gt = case["gt"]
        gt_in_top30 = False
        gt_rank_in_pool = -1
        if gt in pool:
            gt_rank_in_pool = pool.index(gt)
        # baseline top-20 nDCG for sanity (OOF R54c)
        b_gt_rank = -1
        if gt_rank_in_pool >= 0:
            pos = np.where(order == gt_rank_in_pool)[0]
            if len(pos):
                b_gt_rank = int(pos[0]) + 1
        b_ndcg = ndcg_at_k(b_gt_rank, TOP_K)
        baseline_ndcg_at_20_list.append(b_ndcg)
        if is_h7:
            baseline_h7_ndcg_list.append(b_ndcg)
        if is_same:
            baseline_same_ndcg_list.append(b_ndcg)
        else:
            baseline_diff_ndcg_list.append(b_ndcg)

        # Take top-30
        top30_pool_pos = order[:TOP_30]
        top30_tids = [pool[int(p)] for p in top30_pool_pos]
        top30_scores = [float(scores[int(p)]) for p in top30_pool_pos]
        top30_features = feats[top30_pool_pos]  # shape (30, 37)

        # R68 features
        r68_list = r68_lists_by_case.get(i, [])
        r68_rank_map = {str(t): r + 1 for r, (t, _) in enumerate(r68_list[:POOL_K])}
        r68_score_map = {str(t): float(s) for t, s in r68_list}

        # Semantic features: query embedding for this case
        q_row = case_to_qemb_row.get(i)
        if q_row is not None:
            qemb = r68_query_embs[q_row]
        else:
            qemb = None
        # Played track embeddings
        played = case["music_turns"]
        played_track_embs = []
        for pt in played:
            pidx = r68_track_to_idx.get(str(pt))
            if pidx is not None:
                played_track_embs.append(r68_track_embs[pidx])
        if played_track_embs:
            played_arr = np.stack(played_track_embs)  # shape (n_played, D)
            # artist centroid for played tracks (artist-level pooling)
            played_artists = [track_artist.get(pt, "") for pt in played]
            artist_to_embs = defaultdict(list)
            for a, e in zip(played_artists, played_track_embs):
                if a:
                    artist_to_embs[a].append(e)
            artist_centroids = {a: np.mean(np.stack(embs), axis=0)
                                for a, embs in artist_to_embs.items()}
        else:
            played_arr = None
            artist_centroids = {}

        if gt in top30_tids:
            gt_in_top30 = True
            gt_in_top30_count += 1
            if is_h7:
                h7_gt_in_top30_count += 1

        for k_row, tid in enumerate(top30_tids):
            r54c_score = top30_scores[k_row]
            r54c_rank = k_row + 1
            r54c_rank_inv = 1.0 / r54c_rank
            feat_vec = top30_features[k_row]  # 37 dim

            r68_r_inv = 1.0 / r68_rank_map.get(tid, POOL_K + 1) if tid in r68_rank_map else 0.0
            r68_pres = 1.0 if tid in r68_rank_map else 0.0
            r68_cos = r68_score_map.get(tid, 0.0)

            # Semantic features
            tidx = r68_track_to_idx.get(str(tid))
            t_emb = r68_track_embs[tidx] if tidx is not None else None
            if qemb is not None and t_emb is not None:
                qc_cos = float(np.dot(qemb, t_emb))
            else:
                qc_cos = 0.0
            if played_arr is not None and t_emb is not None:
                sims = played_arr @ t_emb
                max_sim_played = float(np.max(sims))
                mean_sim_played = float(np.mean(sims))
            else:
                max_sim_played = 0.0
                mean_sim_played = 0.0
            cand_artist = track_artist.get(tid, "")
            if cand_artist and cand_artist in artist_centroids and t_emb is not None:
                max_artist_cos = float(np.dot(artist_centroids[cand_artist], t_emb))
            else:
                max_artist_cos = 0.0
            n_history = len(played)

            row = {
                "case_idx": i,
                "session_id": case["session_id"],
                "turn_number": int(case.get("turn_number", 0)),
                "candidate_track_id": tid,
                "is_h7": is_h7,
                "is_same_artist": is_same,
                "label": 1.0 if tid == gt else 0.0,
                "oof_r54c_score": r54c_score,
                "oof_r54c_rank": r54c_rank,
                "oof_r54c_rank_inv": r54c_rank_inv,
                "r68_rank_inv": r68_r_inv,
                "r68_presence": r68_pres,
                "r68_cosine": r68_cos,
                "sem_qc_cos": qc_cos,
                "sem_max_sim_played": max_sim_played,
                "sem_mean_sim_played": mean_sim_played,
                "sem_max_artist_cos": max_artist_cos,
                "sem_n_history": float(n_history),
            }
            # 37 LR features as separate columns
            for fname, fval in zip(FEAT_ALL, feat_vec):
                row[f"lr_{fname}"] = float(fval)
            rows_out.append(row)

        case_summaries.append({
            "case_idx": i,
            "is_h7": is_h7,
            "is_same_artist": is_same,
            "gt_in_top30": gt_in_top30,
            "gt_in_pool": gt_rank_in_pool >= 0,
            "b_gt_rank_top20": b_gt_rank,
            "b_ndcg_at_20": b_ndcg,
        })
        if (ki + 1) % 200 == 0:
            print(f"    fold-0 cases {ki + 1}/{len(fold0_idx)} ({time.time() - t_eval:.0f}s)",
                  flush=True)

    print(f"  built {len(rows_out)} candidate rows for {len(case_summaries)} cases")

    # ---- Step 3: Sanity stats ----
    print(f"\n{ts()} === Step 3: Sanity stats ===")
    gt_top30_rate = gt_in_top30_count / len(fold0_idx)
    h7_top30_rate = h7_gt_in_top30_count / len(h7_idx) if h7_idx else 0
    baseline_all_ndcg = float(np.mean(baseline_ndcg_at_20_list))
    baseline_h7_ndcg = float(np.mean(baseline_h7_ndcg_list))
    baseline_same_ndcg = float(np.mean(baseline_same_ndcg_list)) if baseline_same_ndcg_list else 0
    baseline_diff_ndcg = float(np.mean(baseline_diff_ndcg_list)) if baseline_diff_ndcg_list else 0

    print(f"  GT in top-30 (fold-0):    {gt_in_top30_count}/{len(fold0_idx)} = {gt_top30_rate:.4f}")
    print(f"  GT in top-30 (h7):        {h7_gt_in_top30_count}/{len(h7_idx)} = {h7_top30_rate:.4f}")
    print(f"  Baseline OOF R54c top-20 nDCG:")
    print(f"    all_fold0 ({len(fold0_idx)}):     {baseline_all_ndcg:.4f}")
    print(f"    h7        ({len(h7_idx)}):       {baseline_h7_ndcg:.4f}")
    print(f"    same_art  ({len(same_idx)}):     {baseline_same_ndcg:.4f}")
    print(f"    diff_art  ({len(fold0_idx) - len(same_idx)}):     {baseline_diff_ndcg:.4f}")

    # Feature distribution sanity
    print(f"\n  Feature value distributions (top-30 candidates only):")
    for fkey in ["oof_r54c_score", "r68_cosine", "sem_qc_cos",
                 "sem_max_sim_played", "sem_mean_sim_played", "sem_max_artist_cos"]:
        vals = [r[fkey] for r in rows_out]
        print(f"    {fkey:24} min={min(vals):.4f} mean={float(np.mean(vals)):.4f} "
              f"max={max(vals):.4f}")

    # Save dataset
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DATA, "wb") as f:
        pickle.dump({
            "rows": rows_out,
            "case_summaries": case_summaries,
            "feature_names": [k for k in rows_out[0].keys()
                             if k not in {"case_idx", "session_id", "turn_number",
                                          "candidate_track_id", "is_h7",
                                          "is_same_artist", "label"}],
            "metadata": {
                "fold": FOLD,
                "n_fold0_cases": len(fold0_idx),
                "n_h7_cases": len(h7_idx),
                "n_same_cases": len(same_idx),
                "top_k": TOP_30,
                "lr_feature_count": len(FEAT_ALL),
                "r68_feature_count": 3,
                "semantic_feature_count": 5,
                "total_feature_count": len(FEAT_ALL) + 3 + 5 + 1,  # +1 for r54c_rank_inv (separate)
            },
        }, f)
    print(f"\n{ts()} Saved dataset to {OUT_DATA}")

    # Save stats JSON
    stats = {
        "experiment": "R76 Phase 0A — top-30 OOF dataset",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "fold": FOLD,
        "n_cases_fold0": len(fold0_idx),
        "n_cases_h7": len(h7_idx),
        "n_cases_same": len(same_idx),
        "n_candidate_rows": len(rows_out),
        "gt_in_top30": {
            "all_fold0": {"count": gt_in_top30_count, "rate": gt_top30_rate},
            "h7": {"count": h7_gt_in_top30_count, "rate": h7_top30_rate},
        },
        "baseline_oof_r54c_top20_ndcg": {
            "all_fold0": baseline_all_ndcg,
            "h7": baseline_h7_ndcg,
            "same_artist": baseline_same_ndcg,
            "diff_artist": baseline_diff_ndcg,
        },
        "dataset_path": str(OUT_DATA),
        "model_path": str(oof_r54c_path),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(f"{ts()} Saved stats to {OUT_JSON}")

    # Doc
    md = [
        "# R76 Phase 0A — Top-30 OOF dataset for neural residual ranker",
        "",
        f"Elapsed: {stats['elapsed_s']:.0f}s",
        f"Fold: {FOLD}",
        "",
        "## Dataset shape",
        "",
        f"- Cases (fold-0): **{stats['n_cases_fold0']}**",
        f"- h7 cases: {stats['n_cases_h7']}",
        f"- same-artist cases: {stats['n_cases_same']}",
        f"- Candidate rows (cases × top-30): **{stats['n_candidate_rows']}**",
        f"- Features per row: **{stats['n_candidate_rows'] and 37 + 3 + 5 + 1 + 1}** (37 LR + 3 R68 + 5 semantic + 1 R54c score + 1 R54c rank_inv)",
        "",
        "## GT-in-top-30 (the candidate ceiling)",
        "",
        f"- all_fold0: **{gt_in_top30_count}/{stats['n_cases_fold0']} = {gt_top30_rate:.4f}**",
        f"- h7: **{h7_gt_in_top30_count}/{stats['n_cases_h7']} = {h7_top30_rate:.4f}**",
        "",
        "If h7 GT-in-top-30 < 0.50, the ceiling for a residual reranker is hard.",
        "",
        "## Baseline OOF R54c top-20 nDCG (reproduces well from R71)",
        "",
        "| Subset | n | nDCG@20 |",
        "|---|---:|---:|",
        f"| all_fold0 | {stats['n_cases_fold0']} | {baseline_all_ndcg:.4f} |",
        f"| h7 | {stats['n_cases_h7']} | {baseline_h7_ndcg:.4f} |",
        f"| same_artist | {stats['n_cases_same']} | {baseline_same_ndcg:.4f} |",
        f"| diff_artist | {stats['n_cases_fold0'] - stats['n_cases_same']} | {baseline_diff_ndcg:.4f} |",
        "",
        "## Files",
        "",
        f"- Dataset: `{OUT_DATA}`",
        f"- OOF R54c model: `{oof_r54c_path}`",
        f"- Stats JSON: `{OUT_JSON}`",
        "",
        "## Next: Phase 0B",
        "",
        "Train a small MLP residual model on this dataset.",
        "Score = zscore(oof_r54c_score) + beta * neural_delta(features).",
        "Listwise CE or pairwise softplus loss. Fold-0 CV-within.",
        "Gate: h7 Δ ≥ +0.005 vs baseline, same-artist Δ ≥ -0.002, recovered > lost.",
    ]
    OUT_DOC.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved {OUT_DOC}")


if __name__ == "__main__":
    main()
