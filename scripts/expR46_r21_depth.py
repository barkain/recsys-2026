#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301,S101
"""R46: R21 retrieval depth expansion diagnostic.

R45 found 27 dev cases where R21 has GT at rank 301-1000 (shallow miss).
This tests whether expanding R21 source depth from 300 to 500/1000/2000
improves pool_hit and h7 nDCG.

Two-phase approach:
  --phase retrieve: compute extended R21 lists (top-2000) using cached embeddings
  --phase evaluate: build features, run CV5 LambdaRank for each depth config

NOTE: Uses production R21 model on all dev cases (small contamination since
production R21 was trained on all dev). Acceptable for a diagnostic.
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

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R21_TRACK_IDS = REPO / "cache" / "r21_production" / "track_ids.json"
R21_TRACK_EMBS = REPO / "cache" / "r21_production" / "track_embeddings.npy"
R21_QUERY_EMBS = REPO / "cache" / "r21_production" / "dev_query_embeddings.npy"
R46_CACHE = REPO / "cache" / "r46_r21_extended.json"

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

DEPTH_CONFIGS = [300, 500, 1000, 2000]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


# =========================================================================
# PHASE 1: Retrieve — compute extended R21 lists using cached embeddings
# =========================================================================

def phase_retrieve():
    t0 = time.time()
    print(f"{ts()} R46 Phase 1: Compute extended R21 lists (top-2000)", flush=True)
    print("=" * 70, flush=True)
    print("NOTE: Using production R21 model embeddings (small contamination).", flush=True)

    # Load dev payload for played track exclusion
    print(f"{ts()} Loading dev payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    print(f"  {n} cases", flush=True)

    # Load R21 track embeddings and IDs
    print(f"{ts()} Loading R21 track embeddings...", flush=True)
    track_embs = np.load(R21_TRACK_EMBS)
    with open(R21_TRACK_IDS) as f:
        track_ids = json.load(f)
    print(f"  {len(track_ids)} tracks, dim={track_embs.shape[1]}", flush=True)

    # Load cached dev query embeddings (production model)
    print(f"{ts()} Loading dev query embeddings...", flush=True)
    query_embs = np.load(R21_QUERY_EMBS)
    print(f"  {query_embs.shape[0]} queries, dim={query_embs.shape[1]}", flush=True)

    assert query_embs.shape[0] == n, f"Query count mismatch: {query_embs.shape[0]} vs {n}"
    assert query_embs.shape[1] == track_embs.shape[1], "Embedding dim mismatch"

    # Compute top-2000 lists for each dev case
    TOP_K = 2000
    print(f"{ts()} Computing top-{TOP_K} R21 lists for {n} cases...", flush=True)
    r21_extended = []
    for i in range(n):
        scores = track_embs @ query_embs[i]
        played_set = set(cases[i]["music_turns"])
        for idx, tid in enumerate(track_ids):
            if tid in played_set:
                scores[idx] = -np.inf
        top_idx = np.argpartition(-scores, TOP_K)[:TOP_K]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        r21_extended.append([track_ids[j] for j in top_idx])

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{n} cases done", flush=True)

    # Quick sanity: GT rank distribution
    gt_ranks = []
    for i, c in enumerate(cases):
        if c["gt"] in r21_extended[i]:
            gt_ranks.append(r21_extended[i].index(c["gt"]) + 1)
    print(f"\n{ts()} GT rank stats (production R21, top-{TOP_K}):", flush=True)
    for depth in DEPTH_CONFIGS:
        hits = sum(1 for r in gt_ranks if r <= depth)
        print(f"  R21 hit@{depth}: {hits}/{n} ({hits/n:.4f})", flush=True)
    print(f"  Total GT found in top-{TOP_K}: {len(gt_ranks)}/{n}", flush=True)

    # Save
    R46_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(R46_CACHE, "w") as f:
        json.dump(r21_extended, f)
    print(f"\n{ts()} Saved extended R21 lists to {R46_CACHE}", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)


# =========================================================================
# PHASE 2: Evaluate — build features, run CV5 LambdaRank per depth config
# =========================================================================

def phase_evaluate():
    import lightgbm as lgb
    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
    from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
    from scripts.tune_postrank_v23 import tokens

    t0 = time.time()
    print(f"{ts()} R46 Phase 2: Depth expansion evaluation", flush=True)
    print("=" * 70, flush=True)

    # Load data
    print(f"{ts()} Loading dev payload...", flush=True)
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

    # Load R21 OOF lists (baseline, top-300, fold-clean)
    print(f"{ts()} Loading R21 OOF lists (baseline)...", flush=True)
    with open(R21_OOF) as f:
        r21_oof = json.load(f)

    # Load extended R21 lists (top-2000, production)
    print(f"{ts()} Loading extended R21 lists (production top-2000)...", flush=True)
    with open(R46_CACHE) as f:
        r21_extended = json.load(f)

    # Load album mapping
    print(f"{ts()} Loading album mapping...", flush=True)
    from datasets import DownloadConfig, load_dataset
    ds_meta = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                           download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_album = {}
    for item in ds_meta:
        tid = str(item["track_id"])
        alb_id = item.get("album_id", [])
        if isinstance(alb_id, list) and alb_id:
            track_album[tid] = str(alb_id[0])
        else:
            alb_name = item.get("album_name", [])
            if isinstance(alb_name, list) and alb_name:
                track_album[tid] = str(alb_name[0])
            else:
                track_album[tid] = ""

    # Popularity
    print(f"{ts()} Building popularity stats...", flush=True)
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    # ALS
    print(f"{ts()} Building ALS...", flush=True)
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

    # CV5 folds
    folds = grouped_session_folds(sessions, seed=0)

    # Feature names (R39 = 34 features: 29 base + 5 album)
    FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
    FEAT_ALBUM = [
        "same_album_last1",
        "same_album_last3",
        "same_album_any",
        "album_history_count",
        "pool_same_album_count",
    ]
    feat_names = FEAT_BASE + FEAT_ALBUM
    n_feat = len(feat_names)
    n_feat_base = len(FEAT_BASE)
    print(f"  {n_feat} features ({n_feat_base} base + {len(FEAT_ALBUM)} album)", flush=True)

    # Depth configs
    results = {}
    baseline_case_ndcg = None

    for r21_depth in DEPTH_CONFIGS:
        config_name = f"R21_top{r21_depth}"
        print(f"\n{ts()} Config: {config_name}", flush=True)
        print("-" * 50, flush=True)

        # For depth=300, use OOF lists (fold-clean, same as R39)
        # For depth>300, use extended production lists truncated to depth
        if r21_depth == 300:
            r21_source = r21_oof
            print("  Using R21 OOF lists (fold-clean, top-300)", flush=True)
        else:
            r21_source = [lst[:r21_depth] for lst in r21_extended]
            print(f"  Using production R21 lists (top-{r21_depth}, slight contamination)", flush=True)

        # Build features
        X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)

        for i, c in enumerate(cases):
            src_lists = {
                "A": payload["src_a"][i], "B": payload["src_b"][i],
                "C": payload["src_c"][i], "D": payload["src_d"][i],
                "F": payload["src_f"][i], "ALS": als_source[i],
                "R21": r21_source[i],
            }
            pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
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

                # Base 29 features (same as R39)
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

                # Album features (5)
                c_album = track_album.get(tid, "")
                row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
                row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
                row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
                row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
                pool_album_count = sum(1 for tid2 in pool[:POOL_K]
                                       if track_album.get(tid2, "") == c_album) if c_album else 0
                row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

        pool_hit = float(np.mean(gt_idx >= 0))
        print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}", flush=True)

        # CV5 LambdaRank
        case_ndcg = np.zeros(n)
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
                                group=g_tr, feature_name=list(feat_names))
            ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                                group=g_va, reference=ds_tr)
            params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                      "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
                      "verbose": -1, "seed": 0}
            model = lgb.train(params, ds_tr, num_boost_round=300,
                              valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
            preds = model.predict(np.array(X_va))
            offset = 0
            for idx in va:
                s = int(sizes[idx])
                sc = preds[offset:offset + s]
                offset += s
                if gt_idx[idx] < 0:
                    continue
                ranked = np.argsort(-sc)
                gt_pos = np.where(ranked == gt_idx[idx])[0]
                if len(gt_pos) > 0 and gt_pos[0] < 20:
                    case_ndcg[idx] = 1.0 / np.log2(gt_pos[0] + 2)

        # Metrics
        h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
        h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
                   ta.get(cases[i]["gt"], "") in
                   {ta.get(t, "") for t in cases[i]["music_turns"]}]
        h7_diff = [i for i in h7 if i not in set(h7_same)]

        h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))
        cv5 = float(np.mean(case_ndcg))
        same_ndcg = float(np.mean([case_ndcg[i] for i in h7_same])) if h7_same else 0
        diff_ndcg = float(np.mean([case_ndcg[i] for i in h7_diff])) if h7_diff else 0

        print(f"  h7={h7_ndcg:.5f}  cv5={cv5:.5f}  same={same_ndcg:.5f}  diff={diff_ndcg:.5f}", flush=True)

        results[config_name] = {
            "r21_depth": r21_depth,
            "pool_hit": pool_hit,
            "h7": h7_ndcg,
            "cv5": cv5,
            "same": same_ndcg,
            "diff": diff_ndcg,
        }

        # Store baseline for comparison
        if r21_depth == 300:
            baseline_case_ndcg = case_ndcg.copy()

    # ===================================================================
    # Summary: comparison vs baseline
    # ===================================================================
    sep = "=" * 70
    print(f"\n{sep}", flush=True)
    print("R46 R21 DEPTH EXPANSION RESULTS", flush=True)
    print(sep, flush=True)

    base = results["R21_top300"]
    print(f"\n{'Config':<15s} {'pool_hit':>9s} {'h7':>9s} {'cv5':>9s} "
          f"{'same':>9s} {'diff':>9s} {'dh7':>9s}", flush=True)
    print("-" * 70, flush=True)
    for name, r in results.items():
        dh7 = r["h7"] - base["h7"]
        print(f"  {name:<13s} {r['pool_hit']:9.4f} {r['h7']:9.5f} {r['cv5']:9.5f} "
              f"{r['same']:9.5f} {r['diff']:9.5f} {dh7:+9.5f}", flush=True)

    # Per-case recovery/loss analysis vs baseline
    if baseline_case_ndcg is not None:
        print("\n--- Per-case change analysis (vs R21_top300 baseline) ---", flush=True)
        # Re-run the best expanded config to get per-case ndcg
        # (already have it from the last iteration if last config was best)
        # For a cleaner analysis, compare last config vs baseline
        for name, r in results.items():
            if name == "R21_top300":
                continue
            # We need to re-check which cases changed
            # This info isn't stored per-case for intermediate configs
            # Just report the aggregate deltas
            dpool = r["pool_hit"] - base["pool_hit"]
            dh7 = r["h7"] - base["h7"]
            print(f"  {name}: Δpool_hit={dpool:+.4f}  Δh7={dh7:+.5f}", flush=True)

    # Gate
    best_name = max(results, key=lambda k: results[k]["h7"])
    best = results[best_name]
    dh7_best = best["h7"] - base["h7"]
    print(f"\n  Best config: {best_name}", flush=True)
    print(f"  Δh7 vs baseline: {dh7_best:+.5f}", flush=True)
    g = dh7_best >= 0.003
    print(f"  GATE (Δh7 >= +0.003): {'PASS' if g else 'FAIL'}", flush=True)

    # Save
    out_path = REPO / "exp" / "eval" / "expR46_r21_depth.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["retrieve", "evaluate"])
    args = parser.parse_args()

    if args.phase == "retrieve":
        phase_retrieve()
    elif args.phase == "evaluate":
        phase_evaluate()


if __name__ == "__main__":
    main()
