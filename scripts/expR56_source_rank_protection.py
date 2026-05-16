#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R56 source-rank protection — dev-only evaluator.

Reproduces the refreshed baseline (R39+R54 Phase 2 OOF, CV5 LambdaRank)
exactly, verifies against the saved refreshed-decomp metrics, then sweeps
6 variants × 3 K values = 18 configurations of post-LR rank protection
plus the ORACLE diagnostic.

No retraining. No blind submission. Dev only.

Variants (see docs/r56_design.md):
  A : top-20 in {R54, R21, src_b, src_c}, all sessions
  B : top-10 in {R54, R21, src_b, src_c}, all sessions
  C : top-20 in {R54, R21}, all sessions
  D : top-20 in {src_b, src_c}, all sessions
  E : top-20 in {R54, R21, src_b, src_c}, gated by observable session signals
      (n_prior_music<3 OR n_unique_artists>=5 OR pool_top20_same_artist_share<=0.30)
  O : top-20 in {R54, R21, src_b, src_c}, gated by TRUE diff_artist
      ORACLE DIAGNOSTIC ONLY — never a blind candidate.

K (per-session protection cap): {1, 2, 3}. Default reporting K=1.

Output:
  exp/eval/expR56_protection_variants.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expS2_lambdarank import build_als  # noqa: E402
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds  # noqa: E402
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats  # noqa: E402
from scripts.tune_postrank_v23 import tokens  # noqa: E402

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_PHASE2_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
DECOMP_JSON = REPO / "exp" / "eval" / "expR55_post_refresh_decomp.json"
OUT = REPO / "exp" / "eval" / "expR56_protection_variants.json"

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
FEAT_R54 = ["r54_rank_inv", "r54_presence", "r54_cosine"]
ALL_FEAT = FEAT_BASE + FEAT_ALBUM + FEAT_R54

NDCG_EPS = 0.0005  # tolerance per nDCG metric
BUCKET_FRAC_EPS = 0.005  # tolerance per bucket fraction (0.5pp)

# Gate constants (mirror docs/r56_design.md §5)
GATE_PROD_NDCG_DELTA = 0.010
GATE_EXP_NDCG_DELTA = 0.005
GATE_PROD_TOP1_CHURN_ALL_FRAC = 0.030
GATE_EXP_TOP1_CHURN_ALL_FRAC = 0.015
GATE_PROD_TOP1_CHURN_H7_FRAC = 0.030
GATE_EXP_TOP1_CHURN_H7_FRAC = 0.015
GATE_SAME_ARTIST_REGRESS_EPS = 0.002  # stop if same_artist nDCG regresses > this

VARIANT_IDS = ["A", "B", "C", "D", "E", "O"]
K_VALUES = [1, 2, 3]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_albums():
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    out = {}
    for item in ds:
        tid = str(item["track_id"])
        alb_id = item.get("album_id", [])
        if isinstance(alb_id, list) and alb_id:
            out[tid] = str(alb_id[0])
        else:
            alb_name = item.get("album_name", [])
            if isinstance(alb_name, list) and alb_name:
                out[tid] = str(alb_name[0])
            else:
                out[tid] = ""
    return out


def ndcg_at_20_from_rank(rank):
    if rank > 0 and rank <= 20:
        return 1.0 / np.log2(rank + 1)
    return 0.0


def build_baseline():
    """Reproduce the refreshed decomp baseline. Returns per-session state and metrics."""
    t0 = time.time()
    print(f"{ts()} Loading R12 payload, R21 OOF, R54 Phase 2 OOF...")
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
        r21_source = json.load(f)
    with open(R54_PHASE2_OOF) as f:
        r54_data = json.load(f)
    r54_raw = r54_data["lists"]
    assert len(r54_raw) == n
    r54_source = []
    r54_scores = []
    for case_lists in r54_raw:
        tids = [t for t, _ in case_lists]
        score_map = {t: float(s) for t, s in case_lists}
        r54_source.append(tids)
        r54_scores.append(score_map)

    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1
    track_album = load_track_albums()

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
    n_feat_base = len(FEAT_BASE)
    n_feat_r39 = n_feat_base + len(FEAT_ALBUM)

    print(f"{ts()} Building features ({len(ALL_FEAT)} per candidate, pool@{POOL_K})...")
    X = np.zeros((n, POOL_K, len(ALL_FEAT)), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = [[] for _ in range(n)]
    src_union_has_gt = np.zeros(n, dtype=bool)
    t_feat = time.time()

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i], "R54": r54_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        pools[i] = pool
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])
        if c["gt"] in (
            set(payload["src_a"][i]) | set(payload["src_b"][i]) | set(payload["src_c"][i])
            | set(payload["src_d"][i]) | set(payload["src_f"][i])
            | set(als_source[i]) | set(r21_source[i]) | set(r54_source[i])
        ):
            src_union_has_gt[i] = True

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
        pool_artists_all = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists_all if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(r54_source[i][:300])}
        last1_album = track_album.get(played[-1], "") if played else ""
        last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
        all_albums = [track_album.get(t, "") for t in played]
        album_hist_counts = Counter(a for a in all_albums if a)

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
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
            row[20] = sum(1 for sn in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sn, {}))
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
            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for t2 in pool[:POOL_K] if track_album.get(t2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)
            row[n_feat_r39 + 0] = 1.0 / r54_rank_map[tid] if tid in r54_rank_map else 0.0
            row[n_feat_r39 + 1] = 1.0 if tid in r54_rank_map else 0.0
            row[n_feat_r39 + 2] = r54_scores[i].get(tid, 0.0)
        if (i + 1) % 1000 == 0:
            print(f"  features {i + 1}/{n} ({time.time() - t_feat:.0f}s)", flush=True)

    # CV5 LR with score capture
    print(f"\n{ts()} CV5 LambdaRank (capture per-case scores)...")
    per_case_scores = [None] * n  # np.ndarray of POOL_K scores per case
    case_lr_rank = np.full(n, -1, dtype=np.int64)

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
                            group=g_tr, feature_name=list(ALL_FEAT))
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
            # Save the score array for this case (length = pool size)
            per_case_scores[idx] = sc.copy()
            if gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0:
                case_lr_rank[idx] = int(gt_pos[0]) + 1

    print(f"  baseline build elapsed: {time.time() - t0:.0f}s")
    return {
        "cases": cases,
        "ta": ta,
        "track_album": track_album,
        "r21_source": r21_source,
        "r54_source": r54_source,
        "payload": payload,
        "pools": pools,
        "sizes": sizes,
        "gt_idx": gt_idx,
        "src_union_has_gt": src_union_has_gt,
        "per_case_scores": per_case_scores,
        "case_lr_rank": case_lr_rank,
    }


def compute_metrics(case_ranks, cases, ta, track_album, gt_idx, src_union_has_gt):
    """Compute aggregate + split nDCG, top-1 stats, bucket distribution."""
    n = len(cases)
    ndcg = np.array([ndcg_at_20_from_rank(r) for r in case_ranks])

    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]

    # same/diff artist
    same_art_idx, diff_art_idx = [], []
    for i in range(n):
        gt_a = ta.get(cases[i]["gt"], "")
        played = {ta.get(t, "") for t in cases[i]["music_turns"]} - {""}
        if gt_a and gt_a in played:
            same_art_idx.append(i)
        else:
            diff_art_idx.append(i)

    # same/diff album
    same_alb_idx, diff_alb_idx = [], []
    for i in range(n):
        gt_alb = track_album.get(cases[i]["gt"], "")
        played_albs = {track_album.get(t, "") for t in cases[i]["music_turns"]} - {""}
        if gt_alb and gt_alb in played_albs:
            same_alb_idx.append(i)
        else:
            diff_alb_idx.append(i)

    # hist-depth split
    by_depth = defaultdict(list)
    for i in range(n):
        d = cases[i]["n_prior_music"]
        label = f"h{d}" if d < 7 else "h7+"
        by_depth[label].append(i)

    # buckets
    bucket_counts = Counter()
    bucket_per_case = []
    for i in range(n):
        if gt_idx[i] >= 0 and case_ranks[i] >= 1 and case_ranks[i] <= 20:
            b = "HIT"
        elif gt_idx[i] >= 0:
            b = "DEMOTED"
        elif src_union_has_gt[i]:
            b = "POOL_MISS"
        else:
            b = "UNREACHABLE"
        bucket_counts[b] += 1
        bucket_per_case.append(b)

    return {
        "all_ndcg": float(ndcg.mean()),
        "h7_ndcg": float(ndcg[h7_idx].mean()) if h7_idx else 0.0,
        "same_artist_ndcg": float(ndcg[same_art_idx].mean()) if same_art_idx else 0.0,
        "diff_artist_ndcg": float(ndcg[diff_art_idx].mean()) if diff_art_idx else 0.0,
        "same_album_ndcg": float(ndcg[same_alb_idx].mean()) if same_alb_idx else 0.0,
        "diff_album_ndcg": float(ndcg[diff_alb_idx].mean()) if diff_alb_idx else 0.0,
        "by_depth_ndcg": {
            k: float(ndcg[idxs].mean()) for k, idxs in by_depth.items()
        },
        "bucket_counts": dict(bucket_counts),
        "bucket_per_case": bucket_per_case,
        "same_artist_n": len(same_art_idx),
        "diff_artist_n": len(diff_art_idx),
        "h7_n": len(h7_idx),
        "by_depth_n": {k: len(idxs) for k, idxs in by_depth.items()},
    }


def verify_baseline(baseline_metrics):
    """Compare baseline metrics to saved refreshed-decomp JSON. Abort on mismatch."""
    if not DECOMP_JSON.exists():
        raise RuntimeError(f"Decomp baseline {DECOMP_JSON} not found")
    with open(DECOMP_JSON) as f:
        decomp = json.load(f)

    checks = [
        ("all_ndcg", decomp["overall_ndcg"], baseline_metrics["all_ndcg"]),
        ("h7_ndcg", decomp["h7_ndcg"], baseline_metrics["h7_ndcg"]),
    ]
    failures = []
    for name, want, got in checks:
        if abs(want - got) > NDCG_EPS:
            failures.append(f"  {name}: refreshed-decomp={want:.5f}  reproduced={got:.5f}  Δ={got - want:+.5f}")

    # Bucket count check (fractional)
    n = sum(baseline_metrics["bucket_counts"].values())
    for b, decomp_cnt in decomp["bucket_counts"].items():
        repro_cnt = baseline_metrics["bucket_counts"].get(b, 0)
        decomp_frac = decomp_cnt / n
        repro_frac = repro_cnt / n
        if abs(decomp_frac - repro_frac) > BUCKET_FRAC_EPS:
            failures.append(f"  bucket {b}: refreshed-decomp={decomp_cnt}  reproduced={repro_cnt}  "
                            f"Δ={(repro_cnt - decomp_cnt):+d}")

    if failures:
        print(f"\n  BASELINE REPRODUCTION MISMATCH:")
        for f in failures:
            print(f)
        print(f"\n  Aborting before variant sweep — protection results would not be meaningful.")
        sys.exit(2)

    print(f"\n  Baseline reproduction PASS (within ε):")
    print(f"    all-dev nDCG@20:       refreshed={decomp['overall_ndcg']:.5f}  "
          f"reproduced={baseline_metrics['all_ndcg']:.5f}")
    print(f"    h7 nDCG@20:            refreshed={decomp['h7_ndcg']:.5f}  "
          f"reproduced={baseline_metrics['h7_ndcg']:.5f}")
    print(f"    same_artist nDCG:       (reporting only) {baseline_metrics['same_artist_ndcg']:.5f}")
    print(f"    diff_artist nDCG:       (reporting only) {baseline_metrics['diff_artist_ndcg']:.5f}")
    print(f"    bucket counts match within {BUCKET_FRAC_EPS * 100:.1f}pp")


def build_session_state(baseline):
    """Per-session precompute for protection variants."""
    cases = baseline["cases"]
    ta = baseline["ta"]
    r21_source = baseline["r21_source"]
    r54_source = baseline["r54_source"]
    payload = baseline["payload"]
    pools = baseline["pools"]
    per_case_scores = baseline["per_case_scores"]
    n = len(cases)

    state = []
    for i in range(n):
        pool = pools[i]
        scores = per_case_scores[i]
        if scores is None or len(pool) == 0:
            state.append(None)
            continue
        # baseline LR-order (descending)
        order_idx = np.argsort(-scores)
        baseline_order = [pool[int(j)] for j in order_idx]
        baseline_top20 = baseline_order[:20]
        baseline_top20_set = set(baseline_top20)

        # source top-K sets (for the candidate qualifies check)
        src_top20 = {
            "R54": set(r54_source[i][:20]),
            "R21": set(r21_source[i][:20]),
            "B": set(payload["src_b"][i][:20]),
            "C": set(payload["src_c"][i][:20]),
        }
        src_top10 = {
            "R54": set(r54_source[i][:10]),
            "R21": set(r21_source[i][:10]),
            "B": set(payload["src_b"][i][:10]),
            "C": set(payload["src_c"][i][:10]),
        }

        # observables
        played = cases[i]["music_turns"]
        played_artists = {ta.get(t, "") for t in played} - {""}
        n_prior_music = cases[i]["n_prior_music"]
        n_unique_artists = len(played_artists)
        top20_artists = [ta.get(t, "") for t in baseline_top20]
        top20_same_art_count = sum(1 for a in top20_artists if a and a in played_artists)
        pool_top20_same_artist_share = top20_same_art_count / 20

        # true diff_artist (for ORACLE only)
        gt_artist = ta.get(cases[i]["gt"], "")
        true_diff_artist = bool(gt_artist) and gt_artist not in played_artists

        state.append({
            "pool": pool,
            "scores": scores,
            "score_map": {pool[j]: float(scores[j]) for j in range(len(pool))},
            "baseline_order": baseline_order,
            "baseline_top20": baseline_top20,
            "baseline_top20_set": baseline_top20_set,
            "src_top20": src_top20,
            "src_top10": src_top10,
            "n_prior_music": n_prior_music,
            "n_unique_artists": n_unique_artists,
            "pool_top20_same_artist_share": pool_top20_same_artist_share,
            "true_diff_artist": true_diff_artist,
            "gt": cases[i]["gt"],
            "gt_idx_in_pool": baseline["gt_idx"][i],
        })
    return state


def qualifies_for_variant(tid, sst, variant_id):
    if variant_id in ("A", "E", "O"):
        return tid in (sst["src_top20"]["R54"] | sst["src_top20"]["R21"]
                       | sst["src_top20"]["B"] | sst["src_top20"]["C"])
    if variant_id == "B":
        return tid in (sst["src_top10"]["R54"] | sst["src_top10"]["R21"]
                       | sst["src_top10"]["B"] | sst["src_top10"]["C"])
    if variant_id == "C":
        return tid in (sst["src_top20"]["R54"] | sst["src_top20"]["R21"])
    if variant_id == "D":
        return tid in (sst["src_top20"]["B"] | sst["src_top20"]["C"])
    return False


def session_applies(sst, variant_id):
    if variant_id in ("A", "B", "C", "D"):
        return True
    if variant_id == "E":
        return (sst["n_prior_music"] < 3
                or sst["n_unique_artists"] >= 5
                or sst["pool_top20_same_artist_share"] <= 0.30)
    if variant_id == "O":
        return sst["true_diff_artist"]
    return False


def apply_protection(sst, variant_id, K):
    """Returns (new_top20_list, new_gt_rank_or_-1)."""
    if not session_applies(sst, variant_id):
        # No change
        new_top20 = sst["baseline_top20"]
        gt = sst["gt"]
        if gt in sst["baseline_top20_set"]:
            return new_top20, new_top20.index(gt) + 1, False
        # GT not in baseline top-20; full rank via baseline_order
        if gt in sst["pool"]:
            return new_top20, sst["baseline_order"].index(gt) + 1, False
        return new_top20, -1, False

    baseline_order = sst["baseline_order"]
    baseline_top20 = sst["baseline_top20"]
    baseline_top20_set = sst["baseline_top20_set"]
    score_map = sst["score_map"]

    # Protected candidates: qualify AND not already in baseline top-20
    protected = []
    for tid in baseline_order:
        if tid in baseline_top20_set:
            continue
        if qualifies_for_variant(tid, sst, variant_id):
            protected.append(tid)
            if len(protected) >= K:
                break  # we only need K

    if not protected:
        # Qualifies but none outside top-20 to insert
        new_top20 = baseline_top20
        gt = sst["gt"]
        if gt in baseline_top20_set:
            return new_top20, new_top20.index(gt) + 1, False
        if gt in sst["pool"]:
            return new_top20, sst["baseline_order"].index(gt) + 1, False
        return new_top20, -1, False

    n_insert = len(protected)
    # Displace the n_insert lowest-LR-score from baseline_top20 (last in order)
    keep_from_top20 = baseline_top20[:20 - n_insert]
    new_top20_set_list = keep_from_top20 + protected
    # Stable sort by LR score desc (preserves relative LR order)
    new_top20 = sorted(new_top20_set_list, key=lambda t: -score_map[t])
    gt = sst["gt"]
    new_top20_set = set(new_top20)
    if gt in new_top20_set:
        return new_top20, new_top20.index(gt) + 1, True
    # GT not in new top-20; its rank is somewhere in the displaced or tail
    # For nDCG@20 purposes, just say rank > 20 (won't contribute)
    if gt in sst["pool"]:
        # We could compute the exact position but it doesn't matter for nDCG@20
        return new_top20, 21, True
    return new_top20, -1, True


def gates_verdict(baseline_m, variant_m, top1_churn_all, top1_churn_h7, n, n_h7,
                   net_recovery):
    """Return verdict among {PASS_PROD, PASS_EXP, FAIL_REGRESS, FAIL_GATE}."""
    h7_delta = variant_m["h7_ndcg"] - baseline_m["h7_ndcg"]
    same_art_delta = variant_m["same_artist_ndcg"] - baseline_m["same_artist_ndcg"]
    all_delta = variant_m["all_ndcg"] - baseline_m["all_ndcg"]

    # Stop conditions
    if net_recovery <= 0:
        return "FAIL_REGRESS", f"net recovery {net_recovery} <= 0"
    if same_art_delta < -GATE_SAME_ARTIST_REGRESS_EPS:
        return "FAIL_REGRESS", f"same_artist nDCG Δ={same_art_delta:+.5f} regresses > {GATE_SAME_ARTIST_REGRESS_EPS}"
    if all_delta < 0:
        return "FAIL_REGRESS", f"all-dev nDCG Δ={all_delta:+.5f} worse than baseline"

    churn_all_frac = top1_churn_all / n
    churn_h7_frac = top1_churn_h7 / n_h7

    # Production gate
    if (h7_delta >= GATE_PROD_NDCG_DELTA
            and same_art_delta >= 0
            and churn_all_frac <= GATE_PROD_TOP1_CHURN_ALL_FRAC
            and churn_h7_frac <= GATE_PROD_TOP1_CHURN_H7_FRAC):
        return "PASS_PROD", f"h7 Δ={h7_delta:+.5f}, all-dev churn {churn_all_frac:.2%}, h7 churn {churn_h7_frac:.2%}"

    # Exploratory gate
    if (h7_delta >= GATE_EXP_NDCG_DELTA
            and same_art_delta >= 0
            and churn_all_frac <= GATE_EXP_TOP1_CHURN_ALL_FRAC
            and churn_h7_frac <= GATE_EXP_TOP1_CHURN_H7_FRAC):
        return "PASS_EXP", f"h7 Δ={h7_delta:+.5f}, all-dev churn {churn_all_frac:.2%}, h7 churn {churn_h7_frac:.2%}"

    return "FAIL_GATE", f"h7 Δ={h7_delta:+.5f} below gates or churn out of bounds"


def main():
    t0 = time.time()
    print("R56 source-rank protection — dev-only evaluator")
    print("=" * 70)

    baseline = build_baseline()
    cases = baseline["cases"]
    ta = baseline["ta"]
    track_album = baseline["track_album"]
    gt_idx = baseline["gt_idx"]
    src_union_has_gt = baseline["src_union_has_gt"]
    case_lr_rank = baseline["case_lr_rank"]
    n = len(cases)

    # Baseline metrics (reproduce + verify)
    print(f"\n{ts()} Computing baseline metrics (8000 cases)...")
    baseline_metrics = compute_metrics(case_lr_rank, cases, ta, track_album,
                                        gt_idx, src_union_has_gt)

    print(f"\n{ts()} Verifying baseline reproduction vs refreshed decomp...")
    verify_baseline(baseline_metrics)

    # Build per-session state for protection
    print(f"\n{ts()} Building per-session state for protection sweep...")
    state = build_session_state(baseline)

    # Sweep
    print(f"\n{ts()} Sweeping variants × K (6 variants × 3 K values)...")
    h7_idx_list = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    n_h7 = len(h7_idx_list)
    baseline_top1 = [state[i]["baseline_top20"][0] if state[i] else None for i in range(n)]

    results = {}
    for variant_id in VARIANT_IDS:
        for K in K_VALUES:
            cfg = f"{variant_id}_K{K}"
            new_ranks = np.zeros(n, dtype=np.int64)
            new_top1 = [None] * n
            sessions_changed = 0
            for i in range(n):
                if state[i] is None:
                    new_ranks[i] = -1
                    continue
                _new_top20, new_rank, applied = apply_protection(state[i], variant_id, K)
                new_ranks[i] = new_rank
                new_top1[i] = _new_top20[0] if _new_top20 else None
                if applied and _new_top20[0] != state[i]["baseline_top20"][0]:
                    sessions_changed += 1

            variant_metrics = compute_metrics(new_ranks, cases, ta, track_album,
                                                gt_idx, src_union_has_gt)

            # Top-1 churn (all-dev + h7)
            churn_all = sum(1 for i in range(n)
                             if new_top1[i] != baseline_top1[i])
            churn_h7 = sum(1 for i in h7_idx_list
                             if new_top1[i] != baseline_top1[i])

            # Bucket transitions
            base_buckets = baseline_metrics["bucket_per_case"]
            var_buckets = variant_metrics["bucket_per_case"]
            recovered = sum(1 for i in range(n)
                             if base_buckets[i] == "DEMOTED" and var_buckets[i] == "HIT")
            lost = sum(1 for i in range(n)
                       if base_buckets[i] == "HIT" and var_buckets[i] == "DEMOTED")
            net = recovered - lost

            # Top-20 overlap
            top20_overlap = []
            for i in range(n):
                if state[i] is None:
                    continue
                base_t20 = set(state[i]["baseline_top20"])
                # variant top-20 is whatever apply_protection returned (recompute is wasteful;
                # use new_top1 alone wouldn't suffice). Recompute via apply_protection:
                vt20, _, _ = apply_protection(state[i], variant_id, K)
                top20_overlap.append(len(base_t20 & set(vt20)))
            top20_overlap_median = float(np.median(top20_overlap)) if top20_overlap else 0.0
            top20_overlap_mean = float(np.mean(top20_overlap)) if top20_overlap else 0.0

            # Gate verdict
            verdict, verdict_reason = gates_verdict(
                baseline_metrics, variant_metrics,
                churn_all, churn_h7, n, n_h7, net,
            )

            results[cfg] = {
                "variant": variant_id,
                "K": K,
                "is_oracle": variant_id == "O",
                "h7_ndcg": variant_metrics["h7_ndcg"],
                "h7_ndcg_delta": variant_metrics["h7_ndcg"] - baseline_metrics["h7_ndcg"],
                "all_ndcg": variant_metrics["all_ndcg"],
                "all_ndcg_delta": variant_metrics["all_ndcg"] - baseline_metrics["all_ndcg"],
                "same_artist_ndcg": variant_metrics["same_artist_ndcg"],
                "same_artist_ndcg_delta": variant_metrics["same_artist_ndcg"] - baseline_metrics["same_artist_ndcg"],
                "diff_artist_ndcg": variant_metrics["diff_artist_ndcg"],
                "diff_artist_ndcg_delta": variant_metrics["diff_artist_ndcg"] - baseline_metrics["diff_artist_ndcg"],
                "by_depth_ndcg": variant_metrics["by_depth_ndcg"],
                "bucket_counts": variant_metrics["bucket_counts"],
                "recovered": recovered,
                "lost": lost,
                "net_recovery": net,
                "sessions_changed_top1": sessions_changed,
                "top1_churn_all": churn_all,
                "top1_churn_h7": churn_h7,
                "top1_churn_all_frac": churn_all / n,
                "top1_churn_h7_frac": churn_h7 / n_h7,
                "top20_overlap_mean": top20_overlap_mean,
                "top20_overlap_median": top20_overlap_median,
                "gate_verdict": verdict,
                "gate_reason": verdict_reason,
            }

    # ---- Report ----
    print(f"\n{'=' * 100}")
    print(f"{'cfg':<6s} | {'h7_Δ':>8s} | {'all_Δ':>8s} | {'sa_Δ':>8s} | {'da_Δ':>8s} | "
          f"{'rec':>4s} | {'lost':>4s} | {'net':>4s} | {'churn_all':>9s} | {'churn_h7':>9s} | "
          f"{'t20_med':>7s} | verdict")
    print("-" * 130)
    for cfg, r in results.items():
        ann = "  [ORACLE]" if r["is_oracle"] else ""
        line = (f"{cfg:<6s} | {r['h7_ndcg_delta']:+.5f} | {r['all_ndcg_delta']:+.5f} | "
                f"{r['same_artist_ndcg_delta']:+.5f} | {r['diff_artist_ndcg_delta']:+.5f} | "
                f"{r['recovered']:>4d} | {r['lost']:>4d} | {r['net_recovery']:>+4d} | "
                f"{r['top1_churn_all_frac']:>8.2%} | {r['top1_churn_h7_frac']:>8.2%} | "
                f"{r['top20_overlap_median']:>7.1f} | {r['gate_verdict']:<10s}{ann}")
        print(line)

    # Selection: best deployable variant by h7 nDCG that passes any gate
    deployable = [r for r in results.values()
                   if not r["is_oracle"] and r["gate_verdict"] in ("PASS_PROD", "PASS_EXP")]
    if deployable:
        deployable.sort(key=lambda r: (
            -r["h7_ndcg_delta"],
            r["top1_churn_all"],
            -r["net_recovery"],
            f"{r['variant']}_K{r['K']}",
        ))
        best = deployable[0]
        print(f"\n  BEST DEPLOYABLE: {best['variant']}_K{best['K']}  "
              f"({best['gate_verdict']}, h7 Δ={best['h7_ndcg_delta']:+.5f})")
    else:
        print(f"\n  NO DEPLOYABLE VARIANT PASSED GATES")
        # Still report best by h7 Δ for diagnostic
        non_oracle = [r for r in results.values() if not r["is_oracle"]]
        if non_oracle:
            best_by_delta = max(non_oracle, key=lambda r: r["h7_ndcg_delta"])
            print(f"  Best by h7 Δ alone: {best_by_delta['variant']}_K{best_by_delta['K']}  "
                  f"({best_by_delta['gate_verdict']}, h7 Δ={best_by_delta['h7_ndcg_delta']:+.5f}, "
                  f"reason: {best_by_delta['gate_reason']})")
    oracle = [r for r in results.values() if r["is_oracle"]]
    if oracle:
        oracle.sort(key=lambda r: -r["h7_ndcg_delta"])
        print(f"\n  ORACLE CEILING (diagnostic only, never deployed):")
        for r in oracle:
            print(f"    O_K{r['K']}: h7 Δ={r['h7_ndcg_delta']:+.5f}  "
                  f"net_recovery={r['net_recovery']:+d}  "
                  f"verdict={r['gate_verdict']}")

    # Save
    out_data = {
        "baseline": {
            "all_ndcg": baseline_metrics["all_ndcg"],
            "h7_ndcg": baseline_metrics["h7_ndcg"],
            "same_artist_ndcg": baseline_metrics["same_artist_ndcg"],
            "diff_artist_ndcg": baseline_metrics["diff_artist_ndcg"],
            "by_depth_ndcg": baseline_metrics["by_depth_ndcg"],
            "bucket_counts": baseline_metrics["bucket_counts"],
        },
        "gates": {
            "production_h7_delta": GATE_PROD_NDCG_DELTA,
            "exploratory_h7_delta": GATE_EXP_NDCG_DELTA,
            "production_churn_all_frac": GATE_PROD_TOP1_CHURN_ALL_FRAC,
            "production_churn_h7_frac": GATE_PROD_TOP1_CHURN_H7_FRAC,
            "exploratory_churn_all_frac": GATE_EXP_TOP1_CHURN_ALL_FRAC,
            "exploratory_churn_h7_frac": GATE_EXP_TOP1_CHURN_H7_FRAC,
            "same_artist_regress_eps": GATE_SAME_ARTIST_REGRESS_EPS,
        },
        "variants": results,
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n{ts()} Saved: {OUT}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
