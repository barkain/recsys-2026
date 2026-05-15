#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54 Phase 2b: Fold-0 R39 ranking integration test.

Integrates Phase 2 fold-0 retrieval lists into the R39 pipeline as an extra RRF source.
LambdaRank is trained ONCE on folds 1-4 with R39 baseline pool (no R54).
Then applied to fold-0 with multiple pool configurations.

This tests whether adding R54 candidates to the pool — without changing features —
yields better fold-0 ranking. Pool composition is the only variable.

Why pool-only: We have R54 OOF only for fold-0. Adding r54_rank/r54_presence as
LambdaRank features would require R54 OOF on training folds, which doesn't exist.
That belongs to a future phase (5-fold OOF training).

Configs:
  baseline           : R39 7 sources (A,B,C,D,F,ALS,R21)
  +R54 w=0.5         : R39 + R54 at weight 0.5
  +R54 w=1.0         : R39 + R54 at weight 1.0
  +R54 w=1.5         : R39 + R54 at weight 1.5
  replace_R21        : R39 sources with R21 replaced by R54 (R21 features in row still from R21)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
PHASE2_LISTS = REPO / "cache" / "r54" / "phase2" / "fold0_r54p2_lists.json"
R21_FOLD_INDICES = REPO / "cache" / "r21_production" / "fold_indices.json"

RRF_K = 20
POOL_K = 300

# R39 baseline weights
SW_BASE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
FEAT_ALL = FEAT_BASE + FEAT_ALBUM


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_albums():
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_album = {}
    for item in ds:
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
    return track_album


def build_features(pool, case, payload_row, src_lists, r21_rank_map_full, als_factors,
                   als_track_to_idx, als_vec, track_pop, max_pop, ta, tt, ttl, tat, tmt,
                   track_album):
    """Build R39 album-aware feature matrix for one case's pool. 34 features."""
    n_pool = len(pool)
    n_feat = len(FEAT_ALL)
    X = np.zeros((n_pool, n_feat), dtype=np.float64)

    user_msgs = [str(r["content"]) for r in case["history"] if r["role"] == "user"] + [case["user_query"]]
    played = case["music_turns"]
    n_hist = len(played)
    now_tok = tokens(user_msgs[-1]) if user_msgs else set()
    all_tok = tokens(" ".join(user_msgs))
    played_set = set(played)
    l_artist = ta.get(played[-1], "") if played else ""
    l_tags = tt.get(played[-1], set()) if played else set()
    prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
             for j, t in enumerate(reversed(played))]
    pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
    artist_counts = Counter(a for a in pool_artists if a)

    src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                for sn, sl in src_lists.items()}

    last1_album = track_album.get(played[-1], "") if played else ""
    last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
    all_albums = [track_album.get(t, "") for t in played]
    album_hist_counts = Counter(a for a in all_albums if a)

    n_feat_base = len(FEAT_BASE)

    for rank, tid in enumerate(pool[:POOL_K], start=1):
        ca = ta.get(tid, "")
        ct = tt.get(tid, set())
        row = X[rank - 1]

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
        row[20] = sum(1 for sn in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sn, {}))
        if als_vec is not None:
            aidx = als_track_to_idx.get(tid)
            if aidx is not None:
                row[21] = float(np.dot(als_vec, als_factors[aidx]))
        row[22] = float(n_hist)
        row[23] = track_pop.get(tid, 0) / max_pop
        row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
        row[25] = float(artist_counts.get(ca, 0)) if ca else 0
        row[26] = row[20]
        row[27] = 1.0 / r21_rank_map_full[tid] if tid in r21_rank_map_full else 0.0
        row[28] = 1.0 if tid in r21_rank_map_full else 0.0

        c_album = track_album.get(tid, "")
        row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
        row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
        row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
        row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
        pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
        row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

    return X


def build_case_pool_and_features(case_idx, src_lists_for_case, weights, cases, payload,
                                  r21_source_full, als_factors, als_track_to_idx, als_vec,
                                  track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album):
    """Build pool@POOL_K via weighted RRF, then features."""
    pool = weighted_rrf(src_lists_for_case, weights, topk=POOL_K, k=RRF_K)
    # R21 feature row always uses production R21 OOF list, not whatever's in pool RRF
    r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source_full[case_idx][:300])}
    X = build_features(pool, cases[case_idx], None, src_lists_for_case, r21_rank_map,
                       als_factors, als_track_to_idx, als_vec, track_pop, max_pop,
                       ta, tt, ttl, tat, tmt, track_album)
    return pool, X


def compute_metrics(name, fold0_idx, cases, pools, scores, ta, baseline_top20=None):
    """Compute h7 nDCG, pool_hit, same/diff, top-20 overlap vs baseline."""
    n = len(fold0_idx)
    pool_hit = 0
    case_ndcg = {}
    case_top20 = {}
    h7_indices = []
    h7_same_indices = []
    h7_diff_indices = []

    for i, ci in enumerate(fold0_idx):
        c = cases[ci]
        gt = c["gt"]
        pool = pools[i]
        sc = scores[i]

        if gt in pool:
            pool_hit += 1

        if len(sc) == 0 or len(pool) == 0:
            case_ndcg[ci] = 0.0
            case_top20[ci] = []
            continue

        ranked = np.argsort(-sc)
        case_top20[ci] = [pool[j] for j in ranked[:20]]

        gt_in_pool = gt in pool
        if gt_in_pool:
            gt_idx = pool.index(gt)
            gt_pos = np.where(ranked == gt_idx)[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg[ci] = 1.0 / np.log2(gt_pos[0] + 2)
            else:
                case_ndcg[ci] = 0.0
        else:
            case_ndcg[ci] = 0.0

        if c.get("n_prior_music") == 7:
            h7_indices.append(ci)
            gt_artist = ta.get(gt, "")
            history_artists = {ta.get(t, "") for t in c["music_turns"]}
            if gt_artist and gt_artist in history_artists:
                h7_same_indices.append(ci)
            else:
                h7_diff_indices.append(ci)

    pool_hit_rate = pool_hit / n
    h7 = float(np.mean([case_ndcg[i] for i in h7_indices])) if h7_indices else 0.0
    cv5 = float(np.mean(list(case_ndcg.values()))) if case_ndcg else 0.0
    same = float(np.mean([case_ndcg[i] for i in h7_same_indices])) if h7_same_indices else 0.0
    diff = float(np.mean([case_ndcg[i] for i in h7_diff_indices])) if h7_diff_indices else 0.0

    # Top-20 churn vs baseline
    recovered = lost = unchanged = 0
    if baseline_top20:
        for ci in fold0_idx:
            base = set(baseline_top20.get(ci, []))
            now = set(case_top20.get(ci, []))
            gt = cases[ci]["gt"]
            in_base = gt in base
            in_now = gt in now
            if in_now and not in_base:
                recovered += 1
            elif in_base and not in_now:
                lost += 1
            elif in_base and in_now:
                unchanged += 1

    return {
        "name": name,
        "n": n,
        "pool_hit": pool_hit, "pool_hit_rate": pool_hit_rate,
        "h7_ndcg": h7, "cv5_ndcg": cv5,
        "h7_same_ndcg": same, "h7_diff_ndcg": diff,
        "n_h7": len(h7_indices), "n_h7_same": len(h7_same_indices), "n_h7_diff": len(h7_diff_indices),
        "recovered": recovered, "lost": lost, "unchanged": unchanged,
        "net": recovered - lost,
    }, case_top20


def main():
    t0 = time.time()
    print(f"{ts()} R54 Phase 2b: fold-0 R39 ranking integration")
    print("=" * 70)

    print(f"{ts()} Loading payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    with open(R21_OOF) as f:
        r21_source_full = json.load(f)

    print(f"{ts()} Loading Phase 2 fold-0 lists...")
    p2_data = json.load(open(PHASE2_LISTS))
    p2_lists = p2_data["lists"]
    print(f"  Phase 2: {len(p2_lists)} lists, top-0 first 3: {p2_lists[0][:3]}")

    print(f"{ts()} Loading album mapping...")
    track_album = load_track_albums()

    print(f"{ts()} Loading popularity...")
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

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
    fold0_idx = sorted(folds[0].tolist())
    train_idx = sorted(set(range(n)) - set(fold0_idx))
    print(f"  Fold-0: {len(fold0_idx)} val, {len(train_idx)} train")

    # ============================================================
    # Build TRAINING features (folds 1-4) once with R39 baseline pool
    # ============================================================
    print(f"\n{ts()} Building training features (folds 1-4, R39 baseline pool)...")
    X_tr, y_tr, g_tr = [], [], []
    tr_start = time.time()
    for ti, idx in enumerate(train_idx):
        src_lists = {
            "A": payload["src_a"][idx], "B": payload["src_b"][idx],
            "C": payload["src_c"][idx], "D": payload["src_d"][idx],
            "F": payload["src_f"][idx], "ALS": als_source[idx],
            "R21": r21_source_full[idx],
        }
        pool, X = build_case_pool_and_features(
            idx, src_lists, SW_BASE, cases, payload, r21_source_full,
            als_factors, als_track_to_idx, als_vecs[idx],
            track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album)
        gt = cases[idx]["gt"]
        s = len(pool)
        if s == 0:
            continue
        gi = pool.index(gt) if gt in pool else -1
        for k in range(s):
            X_tr.append(X[k])
            y_tr.append(1.0 if k == gi else 0.0)
        g_tr.append(s)
        if (ti + 1) % 1000 == 0:
            elapsed = time.time() - tr_start
            print(f"  {ti + 1}/{len(train_idx)} ({elapsed:.0f}s)", flush=True)

    print(f"  Training features built in {time.time() - tr_start:.0f}s ({len(g_tr)} groups)")

    # ============================================================
    # Train LambdaRank
    # ============================================================
    print(f"\n{ts()} Training LambdaRank...")
    X_tr_arr = np.array(X_tr)
    y_tr_arr = np.array(y_tr)
    ds_tr = lgb.Dataset(X_tr_arr, label=y_tr_arr, group=g_tr, feature_name=list(FEAT_ALL))
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
              "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
              "verbose": -1, "seed": 0}
    model = lgb.train(params, ds_tr, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])
    print(f"  Trained on {len(g_tr)} groups, {len(y_tr)} candidates, {len(FEAT_ALL)} features")
    print(f"  Total time: {time.time() - t0:.0f}s")

    # ============================================================
    # Evaluate fold-0 under each pool config
    # ============================================================
    configs = [
        ("baseline", {}, None),
        ("+R54_w0.5", {"R54": 0.5}, p2_lists),
        ("+R54_w1.0", {"R54": 1.0}, p2_lists),
        ("+R54_w1.5", {"R54": 1.5}, p2_lists),
        ("replace_R21_w_R54_w1.0", {"_replace_R21_with_R54": True}, p2_lists),
    ]

    all_results = {}
    baseline_top20 = None

    for cfg_name, extras, r54_lists in configs:
        print(f"\n{ts()} Config: {cfg_name}")

        pools = []
        scores = []
        for vi, idx in enumerate(fold0_idx):
            src_lists = {
                "A": payload["src_a"][idx], "B": payload["src_b"][idx],
                "C": payload["src_c"][idx], "D": payload["src_d"][idx],
                "F": payload["src_f"][idx], "ALS": als_source[idx],
                "R21": r21_source_full[idx],
            }
            weights = dict(SW_BASE)

            if extras.get("_replace_R21_with_R54"):
                src_lists["R21"] = r54_lists[vi]
                # R21 weight retained (1.0)
            else:
                for k, v in extras.items():
                    if k == "R54":
                        src_lists["R54"] = r54_lists[vi]
                        weights["R54"] = v

            pool, X = build_case_pool_and_features(
                idx, src_lists, weights, cases, payload, r21_source_full,
                als_factors, als_track_to_idx, als_vecs[idx],
                track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album)

            pools.append(pool)
            if len(pool) > 0:
                scores.append(model.predict(X[:len(pool)]))
            else:
                scores.append(np.array([]))

        metrics, top20 = compute_metrics(cfg_name, fold0_idx, cases, pools, scores, ta,
                                         baseline_top20=baseline_top20)
        if cfg_name == "baseline":
            baseline_top20 = top20

        all_results[cfg_name] = metrics
        print(f"  pool_hit@{POOL_K}: {metrics['pool_hit']}/{metrics['n']} ({metrics['pool_hit_rate']:.4f})")
        print(f"  h7 nDCG: {metrics['h7_ndcg']:.5f}  cv5: {metrics['cv5_ndcg']:.5f}  "
              f"same: {metrics['h7_same_ndcg']:.5f}  diff: {metrics['h7_diff_ndcg']:.5f}")
        if cfg_name != "baseline":
            print(f"  vs baseline top-20: recovered={metrics['recovered']}  lost={metrics['lost']}  "
                  f"net={metrics['net']:+d}")

    # ============================================================
    # Summary table
    # ============================================================
    print(f"\n{ts()} === SUMMARY ===")
    print(f"  {'Config':<25} {'pool_hit':>9} {'h7':>8} {'cv5':>8} {'same':>8} {'diff':>8} {'rec':>5} {'lost':>5} {'net':>5}")
    base = all_results["baseline"]
    for cfg_name, _, _ in configs:
        r = all_results[cfg_name]
        dh7 = r["h7_ndcg"] - base["h7_ndcg"]
        dph = r["pool_hit_rate"] - base["pool_hit_rate"]
        marker = ""
        if cfg_name != "baseline":
            marker = f"  Δh7={dh7:+.4f}  Δpool={dph:+.4f}"
        print(f"  {cfg_name:<25} {r['pool_hit']:>9} {r['h7_ndcg']:>8.4f} {r['cv5_ndcg']:>8.4f} "
              f"{r['h7_same_ndcg']:>8.4f} {r['h7_diff_ndcg']:>8.4f} "
              f"{r['recovered']:>5} {r['lost']:>5} {r['net']:>+5}{marker}")

    # ============================================================
    # Gate check
    # ============================================================
    print(f"\n{ts()} === GATE CHECK ===")
    print(f"  Baseline h7: {base['h7_ndcg']:.5f}")
    print(f"  Baseline pool_hit: {base['pool_hit_rate']:.4f}")
    best_h7_cfg = max((c for c in all_results if c != "baseline"),
                       key=lambda c: all_results[c]["h7_ndcg"])
    best_h7 = all_results[best_h7_cfg]["h7_ndcg"]
    best_dh7 = best_h7 - base["h7_ndcg"]
    print(f"  Best non-baseline: {best_h7_cfg} h7={best_h7:.5f}  Δh7={best_dh7:+.5f}")

    h7_gate = best_dh7 >= 0.010
    pool_strong = any(
        all_results[c]["pool_hit_rate"] - base["pool_hit_rate"] >= 0.02
        and all_results[c]["h7_ndcg"] >= base["h7_ndcg"]
        for c in all_results if c != "baseline"
    )
    print(f"  Gate A (h7 Δ >= +0.010): {'PASS' if h7_gate else 'FAIL'}")
    print(f"  Gate B (pool Δ >= +0.02 AND h7 non-negative): {'PASS' if pool_strong else 'FAIL'}")

    decision = "PROCEED to full 5-fold" if (h7_gate or pool_strong) else "STOP — integration insufficient"
    print(f"\n  Decision: {decision}")

    # ============================================================
    # Save
    # ============================================================
    out = {
        "configs": list(all_results.values()),
        "results": all_results,
        "best_non_baseline": best_h7_cfg,
        "best_dh7": best_dh7,
        "gate_h7": h7_gate,
        "gate_pool": pool_strong,
        "decision": decision,
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    out_path = REPO / "exp" / "eval" / "expR54_phase2b_fold0_r39_integration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{ts()} Phase 2b complete. Elapsed: {time.time() - t0:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
