#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R80 Phase 0A — Build top-300 listwise dataset for fold-0 (no GPU).

For each fold-0 case:
- Train OOF R54c sibling LR on folds 1-4 (or load if already trained)
- Score fold-0 case to get top-300 candidates + R54c scores
- For each candidate, compute:
  - 37 LR features (R39 + R54)
  - oof_r54c_score (raw)
  - oof_r54c_rank_norm (rank / 300)
  - r68_rank_inv, r68_presence, r68_cosine (fold-0 OOF R68)
  - bge_track_emb (1024-dim, from R68 cache)
  - bge_query_emb (1024-dim, broadcast per case)
  - 5 semantic scalars (query-cand cosine, max/mean sim to played, artist centroid)

Output:
- cache/r80/listwise_dataset_fold0.pkl: dict with per-case arrays
- cache/r80/eval_baseline.json: OOF R54c top-20 nDCG (R71 baseline)

~5-10 min on Mac CPU.
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
TOP_20 = 20
FOLD = 0

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

R68_DIR = REPO / "cache" / "r68" / "phase0_fold0"
R68_LISTS = R68_DIR / "oof_r68_lists_fold0.json"
R68_TRACK_EMBS = R68_DIR / "track_embeddings.npy"
R68_TRACK_IDS = R68_DIR / "track_ids.json"
R68_QUERY_EMBS = R68_DIR / "query_embeddings_dev.npy"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
OOF_R54C_PATH = REPO / "cache" / "r76" / "oof_r54c_fold0.txt"  # reuse from R76

OUT_DIR = REPO / "cache" / "r80"
OUT_DATA = OUT_DIR / "listwise_dataset_fold0.pkl"
OUT_BASELINE = OUT_DIR / "eval_baseline.json"
OUT_STATS = REPO / "exp" / "eval" / "expR80_phase0a_stats.json"
OUT_DOC = REPO / "docs" / "r80_phase0a_audit.md"


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_k(gt_rank: int, k: int) -> float:
    if gt_rank <= 0 or gt_rank > k:
        return 0.0
    return 1.0 / math.log2(gt_rank + 1)


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R80 Phase 0A — top-300 listwise dataset")
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
    h7_fold0 = [i for i in fold0_idx if int(cases[i]["n_prior_music"]) == 7]
    print(f"  fold-0: {len(fold0_idx)}  train: {len(train_idx)}  h7: {len(h7_fold0)}")

    print(f"{ts()} Loading R68 fold-0 artifacts ...", flush=True)
    with open(R68_LISTS) as f:
        r68_data = json.load(f)
    r68_val_idx = r68_data.get("val_idx") or r68_data["manifest"]["val_idx"]
    r68_lists_by_case = {
        int(case_idx): [(str(t), float(s)) for t, s in r68_data["lists"][k_pos]]
        for k_pos, case_idx in enumerate(r68_val_idx)
    }
    r68_track_embs = np.load(R68_TRACK_EMBS)  # (47K, 1024)
    r68_track_ids = json.load(open(R68_TRACK_IDS))
    r68_query_embs = np.load(R68_QUERY_EMBS)  # (1600, 1024)
    case_to_qemb = {int(case_idx): k_pos
                    for k_pos, case_idx in enumerate(r68_val_idx)}
    r68_track_to_idx = {tid: j for j, tid in enumerate(r68_track_ids)}
    print(f"  R68: {r68_track_embs.shape} catalog, {r68_query_embs.shape} queries")

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

    # ---- Load or train OOF R54c LR on folds 1-4 ----
    if OOF_R54C_PATH.exists():
        print(f"\n{ts()} Loading cached OOF R54c LR from {OOF_R54C_PATH}", flush=True)
        oof_r54c = lgb.Booster(model_file=str(OOF_R54C_PATH))
        print(f"  loaded ({oof_r54c.num_feature()} features)", flush=True)
    else:
        print(f"\n{ts()} === Training OOF R54c LR on folds 1-4 ({len(train_idx)} cases) ===")
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
                print(f"    feats {ki + 1}/{len(train_idx)} ({time.time() - t_feat:.0f}s)",
                      flush=True)
        ds = lgb.Dataset(np.array(X_tr, dtype=np.float64),
                         label=np.array(y_tr, dtype=np.float64),
                         group=g_tr, feature_name=list(FEAT_ALL))
        oof_r54c = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
        OOF_R54C_PATH.parent.mkdir(parents=True, exist_ok=True)
        oof_r54c.save_model(str(OOF_R54C_PATH))
        print(f"  trained & saved", flush=True)
        del X_tr, y_tr, g_tr, ds

    # ---- Build listwise dataset for fold-0 ----
    print(f"\n{ts()} === Building top-300 listwise features for fold-0 ===")
    # Each case becomes:
    #   pool: list[str], len 300
    #   features: np.ndarray, (300, 47)  # 37 LR + 2 r54c (score, rank_norm) + 3 R68 + 5 sem scalars
    #   bge_track: np.ndarray, (300, 1024)
    #   bge_query: np.ndarray, (1024,)
    #   gt_in_pool: bool
    #   gt_pool_idx: int (-1 if not in pool)
    cases_out = []
    t_build = time.time()
    n_gt_in_pool = 0
    n_h7_gt_in_pool = 0

    for ki, i in enumerate(fold0_idx):
        case = cases[i]
        is_h7 = int(case["n_prior_music"]) == 7
        is_same_artist = same_artist_case(case, track_artist)

        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        feats37 = featurize37(i, src_lists, pool)  # (300, 37)
        scores_r54c = oof_r54c.predict(feats37)  # (300,)
        order = np.argsort(-scores_r54c, kind="mergesort")

        # Reorder pool by R54c score so rank_norm = position in this list
        pool_sorted = [pool[int(j)] for j in order]
        feats37_sorted = feats37[order]
        scores_sorted = scores_r54c[order]

        gt = case["gt"]
        gt_idx = -1
        for r, tid in enumerate(pool_sorted):
            if tid == gt:
                gt_idx = r
                break
        if gt_idx >= 0:
            n_gt_in_pool += 1
            if is_h7:
                n_h7_gt_in_pool += 1

        # R68 features per candidate
        r68_list = r68_lists_by_case.get(i, [])
        r68_rank_map = {str(t): r + 1 for r, (t, _) in enumerate(r68_list[:POOL_K])}
        r68_score_map = {str(t): float(s) for t, s in r68_list}

        # BGE query embedding for this case
        qemb_row = case_to_qemb.get(i)
        bge_query = r68_query_embs[qemb_row] if qemb_row is not None else np.zeros(1024, dtype=np.float32)

        # BGE played track embeddings + artist centroids
        played = case["music_turns"]
        played_embs = []
        played_artists = []
        for pt in played:
            pidx = r68_track_to_idx.get(str(pt))
            if pidx is not None:
                played_embs.append(r68_track_embs[pidx])
                played_artists.append(track_artist.get(pt, ""))
        played_arr = np.stack(played_embs) if played_embs else None
        artist_centroids = defaultdict(list)
        if played_embs:
            for a, e in zip(played_artists, played_embs):
                if a:
                    artist_centroids[a].append(e)
        artist_centroids = {a: np.mean(np.stack(es), axis=0)
                            for a, es in artist_centroids.items()}

        # Build per-candidate feature matrix (47 dim numeric)
        # Track embeddings are referenced by track_id at training time (not duplicated here)
        # numeric part columns:
        #   0..36: 37 LR features
        #   37: r54c_score
        #   38: r54c_rank_norm (rank+1)/300
        #   39: r68_rank_inv
        #   40: r68_presence
        #   41: r68_cosine
        #   42: bge_query_cand_cos
        #   43: bge_max_sim_played
        #   44: bge_mean_sim_played
        #   45: bge_max_artist_centroid_cos
        #   46: n_history (normalized by 10)
        numeric = np.zeros((POOL_K, 47), dtype=np.float32)
        # Track BGE indices (catalog rows) — Colab dereferences via catalog cache
        track_emb_idx = np.full(POOL_K, -1, dtype=np.int32)

        for k_row, tid in enumerate(pool_sorted):
            numeric[k_row, :37] = feats37_sorted[k_row]
            numeric[k_row, 37] = scores_sorted[k_row]
            numeric[k_row, 38] = (k_row + 1) / POOL_K
            numeric[k_row, 39] = 1.0 / r68_rank_map.get(tid, POOL_K + 1) if tid in r68_rank_map else 0.0
            numeric[k_row, 40] = 1.0 if tid in r68_rank_map else 0.0
            numeric[k_row, 41] = r68_score_map.get(tid, 0.0)

            tidx = r68_track_to_idx.get(str(tid))
            if tidx is not None:
                track_emb_idx[k_row] = tidx
                t_emb = r68_track_embs[tidx]
                numeric[k_row, 42] = float(np.dot(bge_query, t_emb))
                if played_arr is not None:
                    sims = played_arr @ t_emb
                    numeric[k_row, 43] = float(sims.max())
                    numeric[k_row, 44] = float(sims.mean())
                ca = track_artist.get(tid, "")
                if ca and ca in artist_centroids:
                    numeric[k_row, 45] = float(np.dot(artist_centroids[ca], t_emb))
            numeric[k_row, 46] = min(len(played) / 10.0, 1.0)

        cases_out.append({
            "case_idx": i,
            "session_id": case["session_id"],
            "turn_number": int(case.get("turn_number", 0)),
            "is_h7": is_h7,
            "is_same_artist": is_same_artist,
            "n_prior_music": int(case["n_prior_music"]),
            "pool": pool_sorted,
            "gt": gt,
            "gt_pool_idx": gt_idx,
            "gt_in_pool": gt_idx >= 0,
            "numeric_features": numeric,                    # (300, 47)
            "track_emb_idx": track_emb_idx,                 # (300,) int32 → catalog
            "bge_query_emb": bge_query.astype(np.float32),  # (1024,)
            "r54c_top20_baseline": pool_sorted[:TOP_20],
            "r54c_b_ndcg_at_20": ndcg_at_k(gt_idx + 1, TOP_20) if 0 <= gt_idx < TOP_20 else 0.0,
        })

        if (ki + 1) % 200 == 0:
            print(f"    fold-0 cases {ki + 1}/{len(fold0_idx)} ({time.time() - t_build:.0f}s)",
                  flush=True)

    print(f"  built {len(cases_out)} cases")
    print(f"  GT in top-300 pool: {n_gt_in_pool}/{len(fold0_idx)} = {n_gt_in_pool/len(fold0_idx):.4f}")
    print(f"  GT in top-300 (h7): {n_h7_gt_in_pool}/{len(h7_fold0)} = {n_h7_gt_in_pool/len(h7_fold0):.4f}")

    # ---- Baseline metrics ----
    def avg(rows, key, where=None):
        if where is not None:
            rows = [r for r in rows if where(r)]
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    baseline_metrics = {}
    for label, where in [
        ("all_fold0", lambda r: True),
        ("h7", lambda r: r["is_h7"]),
        ("same_artist", lambda r: r["is_same_artist"]),
        ("diff_artist", lambda r: not r["is_same_artist"]),
        ("h7_same", lambda r: r["is_h7"] and r["is_same_artist"]),
        ("h7_diff", lambda r: r["is_h7"] and not r["is_same_artist"]),
    ]:
        rows = [r for r in cases_out if where(r)]
        baseline_metrics[label] = {
            "n": len(rows),
            "ndcg_at_20": avg(rows, "r54c_b_ndcg_at_20"),
        }

    print(f"\n  Baseline OOF R54c top-20 nDCG (fold-0):")
    for label, m in baseline_metrics.items():
        print(f"    {label:14}  n={m['n']:5d}  nDCG@20={m['ndcg_at_20']:.4f}")

    # ---- Save ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save catalog as fp16 for compact shipping (96 MB instead of 192 MB)
    catalog_path = OUT_DIR / "catalog_track_embs_fp16.npy"
    catalog_ids_path = OUT_DIR / "catalog_track_ids.json"
    np.save(catalog_path, r68_track_embs.astype(np.float16))
    catalog_ids_path.write_text(json.dumps(r68_track_ids))
    print(f"\n{ts()} Saved catalog (fp16) → {catalog_path} "
          f"({catalog_path.stat().st_size/1e6:.0f} MB)", flush=True)

    print(f"\n{ts()} Saving compact dataset (~{sum(c['numeric_features'].nbytes for c in cases_out)/1e6:.0f} MB) ...",
          flush=True)
    with open(OUT_DATA, "wb") as f:
        pickle.dump({
            "cases": cases_out,
            "metadata": {
                "fold": FOLD,
                "n_cases": len(cases_out),
                "pool_k": POOL_K,
                "n_numeric_features": 47,
                "track_emb_dim": 1024,
                "query_emb_dim": 1024,
                "feature_names": [
                    *list(FEAT_ALL),  # 37 LR
                    "oof_r54c_score", "oof_r54c_rank_norm",  # 2
                    "r68_rank_inv", "r68_presence", "r68_cosine",  # 3
                    "bge_qc_cos", "bge_max_sim_played",
                    "bge_mean_sim_played", "bge_max_artist_cos",
                    "n_history_norm",  # 5
                ],
            },
        }, f)
    print(f"{ts()} Saved → {OUT_DATA} ({OUT_DATA.stat().st_size/1e9:.2f} GB)")

    with open(OUT_BASELINE, "w") as f:
        json.dump({
            "experiment": "R80 Phase 0A — OOF R54c fold-0 baseline",
            "created_at": datetime.now().isoformat(),
            "baseline_metrics": baseline_metrics,
        }, f, indent=2)
    print(f"{ts()} Saved baseline → {OUT_BASELINE}")

    stats = {
        "experiment": "R80 Phase 0A — listwise dataset",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "fold": FOLD,
        "n_cases": len(cases_out),
        "n_h7": len(h7_fold0),
        "pool_k": POOL_K,
        "gt_in_pool_count": n_gt_in_pool,
        "gt_in_pool_rate": n_gt_in_pool / len(fold0_idx),
        "h7_gt_in_pool_count": n_h7_gt_in_pool,
        "h7_gt_in_pool_rate": n_h7_gt_in_pool / max(len(h7_fold0), 1),
        "baseline_fold0": baseline_metrics,
        "files": {
            "dataset": str(OUT_DATA),
            "dataset_size_gb": OUT_DATA.stat().st_size / 1e9,
            "baseline": str(OUT_BASELINE),
        },
    }
    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"{ts()} Saved stats → {OUT_STATS}")

    md = [
        "# R80 Phase 0A — top-300 listwise dataset audit",
        "",
        f"Elapsed: {stats['elapsed_s']:.0f}s",
        f"Cases (fold-0): {len(cases_out)}",
        f"h7 cases: {len(h7_fold0)}",
        "",
        "## Pool coverage",
        "",
        f"- GT in top-300: **{n_gt_in_pool}/{len(fold0_idx)} = {n_gt_in_pool/len(fold0_idx):.4f}**",
        f"- GT in top-300 (h7): **{n_h7_gt_in_pool}/{len(h7_fold0)} = {n_h7_gt_in_pool/len(h7_fold0):.4f}**",
        "",
        "## Baseline (OOF R54c top-20)",
        "",
        "| Subset | n | nDCG@20 |",
        "|---|---:|---:|",
    ]
    for label, m in baseline_metrics.items():
        md.append(f"| {label} | {m['n']} | {m['ndcg_at_20']:.4f} |")
    md += [
        "",
        "## Phase 0B gates",
        "",
        "- h7 nDCG Δ ≥ +0.005 vs baseline above",
        "- same-artist Δ ≥ -0.002",
        "- recovered > lost on h7",
        "- top-1 churn /80 ≤ 25",
        "- top-20 overlap ≥ 14/20",
        "",
        "## Per-candidate feature schema",
        "",
        "47 numeric + 1024 BGE-large track embedding + 1024 BGE-large query embedding (broadcast)",
        "= 2095 dim per candidate. Project to 256 in model.",
        "",
        "## Files",
        "",
        f"- Dataset: `{OUT_DATA}` ({stats['files']['dataset_size_gb']:.2f} GB)",
        f"- Baseline: `{OUT_BASELINE}`",
    ]
    OUT_DOC.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved doc → {OUT_DOC}")


if __name__ == "__main__":
    main()
