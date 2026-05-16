#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R57b — minimal implementation of new categorical structural features.

Excludes duration and release-year features (R49C negative evidence).
Includes ONLY genuinely new categorical metadata features that R49C did
not implement:

  isrc_country_match    : candidate's ISRC country code appears in history
                          ISRC countries
  isrc_registrant_match : candidate's ISRC registrant code (chars 2-4)
                          appears in history
  artist_id_match       : candidate's artist_id appears in any played
                          track's artist_id (canonical UUID match,
                          orthogonal to existing artist-name feature)

Mandatory baseline reproduction before each experimental config (same
guarantee as R56). If baseline diverges from refreshed-decomp, abort.

Configs:
  base      : R39+R54 (37 features) — the baseline
  +isrc     : baseline + isrc_country_match + isrc_registrant_match (39)
  +artist   : baseline + artist_id_match (38)
  +both     : baseline + all three (40)

Per config: CV5 LambdaRank, report h7/all/same/diff nDCG, recovered/lost/
net (DEMOTED→HIT vs HIT→DEMOTED), top-1 churn (all-dev + h7), feature
importance for new features.

Gates (mirroring R56):
  Production: h7 Δ >= +0.010, same/diff no regress, churn <= 3.0%
  Exploratory: h7 Δ >= +0.005, churn <= 1.5%, no regress

Output:
  exp/eval/expR57b_structural_implementation.json
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
OUT = REPO / "exp" / "eval" / "expR57b_structural_implementation.json"

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
FEAT_R54 = ["r54_rank_inv", "r54_presence", "r54_cosine"]
FEAT_BASELINE = FEAT_BASE + FEAT_ALBUM + FEAT_R54

# New structural features (R57b)
FEAT_ISRC = ["isrc_country_match", "isrc_registrant_match"]
FEAT_ARTIST_ID = ["artist_id_match_history"]

CONFIGS = {
    "base":   FEAT_BASELINE,
    "+isrc":  FEAT_BASELINE + FEAT_ISRC,
    "+artist": FEAT_BASELINE + FEAT_ARTIST_ID,
    "+both":  FEAT_BASELINE + FEAT_ISRC + FEAT_ARTIST_ID,
}

# Gates
GATE_PROD_NDCG_DELTA = 0.010
GATE_EXP_NDCG_DELTA = 0.005
GATE_PROD_TOP1_CHURN_FRAC = 0.030
GATE_EXP_TOP1_CHURN_FRAC = 0.015
GATE_SAME_ARTIST_REGRESS_EPS = 0.002

NDCG_EPS = 0.0005
BUCKET_FRAC_EPS = 0.005


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_metadata_compact():
    """Load track_id -> {artist_id, isrc_country, isrc_registrant}."""
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    meta = {}
    for item in ds:
        tid = str(item["track_id"])
        # artist_id (list of strings)
        aid_list = item.get("artist_id", [])
        if isinstance(aid_list, list) and aid_list:
            aid = str(aid_list[0])
        else:
            aid = str(aid_list) if aid_list else None
        # ISRC (list of strings, uppercase)
        isrc_list = item.get("ISRC", [])
        if isinstance(isrc_list, list) and isrc_list:
            isrc = str(isrc_list[0]).strip().upper()
        else:
            isrc = str(isrc_list).strip().upper() if isrc_list else None
        country = isrc[:2] if isrc and len(isrc) >= 5 else None
        registrant = isrc[2:5] if isrc and len(isrc) >= 5 else None
        meta[tid] = {
            "artist_id": aid,
            "isrc_country": country,
            "isrc_registrant": registrant,
        }
    return meta


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


def ndcg_at_20(rank):
    if rank > 0 and rank <= 20:
        return 1.0 / np.log2(rank + 1)
    return 0.0


def build_features(payload, r21_source, r54_source, r54_scores, als_source, als_vecs,
                    als_factors, als_track_to_idx, track_pop, track_album, track_meta,
                    feat_names):
    """Build feature matrix for given feature set."""
    cases = payload["cases"]
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    n = len(cases)
    max_pop = max(track_pop.values()) if track_pop else 1
    n_feat = len(feat_names)
    n_feat_base = len(FEAT_BASE)
    n_feat_r39 = n_feat_base + len(FEAT_ALBUM)
    n_feat_baseline = len(FEAT_BASELINE)  # 36

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = [[] for _ in range(n)]
    src_union_has_gt = np.zeros(n, dtype=bool)

    # Determine offsets for the new feature blocks within this feat_names
    isrc_offset = -1
    artist_offset = -1
    if "isrc_country_match" in feat_names:
        isrc_offset = feat_names.index("isrc_country_match")
    if "artist_id_match_history" in feat_names:
        artist_offset = feat_names.index("artist_id_match_history")

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
        if c["gt"] in (set(payload["src_a"][i]) | set(payload["src_b"][i]) | set(payload["src_c"][i])
                       | set(payload["src_d"][i]) | set(payload["src_f"][i])
                       | set(als_source[i]) | set(r21_source[i]) | set(r54_source[i])):
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

        # NEW structural metadata precomputation
        hist_artist_ids = {track_meta.get(t, {}).get("artist_id") for t in played} - {None, ""}
        hist_isrc_country = {track_meta.get(t, {}).get("isrc_country") for t in played} - {None, ""}
        hist_isrc_registrant = {track_meta.get(t, {}).get("isrc_registrant") for t in played} - {None, ""}

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

            # NEW: structural features (only if this config includes them)
            tm_struct = track_meta.get(tid, {})
            if isrc_offset >= 0:
                t_country = tm_struct.get("isrc_country")
                t_reg = tm_struct.get("isrc_registrant")
                row[isrc_offset + 0] = 1.0 if (t_country and t_country in hist_isrc_country) else 0.0
                row[isrc_offset + 1] = 1.0 if (t_reg and t_reg in hist_isrc_registrant) else 0.0
            if artist_offset >= 0:
                t_aid = tm_struct.get("artist_id")
                row[artist_offset] = 1.0 if (t_aid and t_aid in hist_artist_ids) else 0.0

    return X, gt_idx, sizes, pools, src_union_has_gt


def run_cv5_lr(X, gt_idx, sizes, pools, folds, feat_names):
    """Run CV5 LambdaRank, return per-case rank + feature importances aggregated."""
    n = X.shape[0]
    case_lr_rank = np.full(n, -1, dtype=np.int64)
    case_top1 = [None] * n
    importances_sum = np.zeros(len(feat_names), dtype=np.float64)
    n_folds = 0
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
        # Importance (gain)
        imp = model.feature_importance(importance_type="gain")
        importances_sum += imp
        n_folds += 1
        offset = 0
        for idx in va:
            s = int(sizes[idx])
            sc = preds[offset:offset + s]
            offset += s
            ranked = np.argsort(-sc)
            case_top1[idx] = pools[idx][int(ranked[0])] if len(ranked) else None
            if gt_idx[idx] < 0:
                continue
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0:
                case_lr_rank[idx] = int(gt_pos[0]) + 1
    importances_avg = importances_sum / max(n_folds, 1)
    return case_lr_rank, case_top1, dict(zip(feat_names, importances_avg.tolist()))


def compute_metrics(case_ranks, cases, ta, gt_idx, src_union_has_gt):
    n = len(cases)
    ndcg = np.array([ndcg_at_20(r) for r in case_ranks])
    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    same_art_idx, diff_art_idx = [], []
    for i in range(n):
        gt_a = ta.get(cases[i]["gt"], "")
        played = {ta.get(t, "") for t in cases[i]["music_turns"]} - {""}
        if gt_a and gt_a in played:
            same_art_idx.append(i)
        else:
            diff_art_idx.append(i)
    buckets = []
    for i in range(n):
        if gt_idx[i] >= 0 and case_ranks[i] >= 1 and case_ranks[i] <= 20:
            buckets.append("HIT")
        elif gt_idx[i] >= 0:
            buckets.append("DEMOTED")
        elif src_union_has_gt[i]:
            buckets.append("POOL_MISS")
        else:
            buckets.append("UNREACHABLE")
    bucket_counts = Counter(buckets)
    return {
        "all_ndcg": float(ndcg.mean()),
        "h7_ndcg": float(ndcg[h7_idx].mean()) if h7_idx else 0.0,
        "same_artist_ndcg": float(ndcg[same_art_idx].mean()) if same_art_idx else 0.0,
        "diff_artist_ndcg": float(ndcg[diff_art_idx].mean()) if diff_art_idx else 0.0,
        "bucket_counts": dict(bucket_counts),
        "buckets_per_case": buckets,
        "h7_n": len(h7_idx),
        "h7_idx": h7_idx,
    }


def verify_baseline(metrics):
    if not DECOMP_JSON.exists():
        raise RuntimeError(f"Missing {DECOMP_JSON}")
    with open(DECOMP_JSON) as f:
        decomp = json.load(f)
    failures = []
    if abs(metrics["all_ndcg"] - decomp["overall_ndcg"]) > NDCG_EPS:
        failures.append(f"all_ndcg: refreshed={decomp['overall_ndcg']:.5f}  repro={metrics['all_ndcg']:.5f}")
    if abs(metrics["h7_ndcg"] - decomp["h7_ndcg"]) > NDCG_EPS:
        failures.append(f"h7_ndcg: refreshed={decomp['h7_ndcg']:.5f}  repro={metrics['h7_ndcg']:.5f}")
    n = sum(metrics["bucket_counts"].values())
    for b, dc in decomp["bucket_counts"].items():
        rc = metrics["bucket_counts"].get(b, 0)
        if abs(dc / n - rc / n) > BUCKET_FRAC_EPS:
            failures.append(f"bucket {b}: refreshed={dc} repro={rc}")
    if failures:
        print(f"\n  BASELINE REPRODUCTION MISMATCH:")
        for f in failures:
            print(f"    {f}")
        sys.exit(2)
    print(f"\n  Baseline reproduction PASS (within ε):")
    print(f"    all-dev nDCG: refreshed={decomp['overall_ndcg']:.5f}  repro={metrics['all_ndcg']:.5f}")
    print(f"    h7 nDCG:      refreshed={decomp['h7_ndcg']:.5f}  repro={metrics['h7_ndcg']:.5f}")


def gates_verdict(baseline_m, variant_m, churn_all, churn_h7, n, n_h7, net_recovery):
    h7_delta = variant_m["h7_ndcg"] - baseline_m["h7_ndcg"]
    same_delta = variant_m["same_artist_ndcg"] - baseline_m["same_artist_ndcg"]
    all_delta = variant_m["all_ndcg"] - baseline_m["all_ndcg"]
    if net_recovery <= 0:
        return "FAIL_REGRESS", f"net recovery {net_recovery} <= 0"
    if same_delta < -GATE_SAME_ARTIST_REGRESS_EPS:
        return "FAIL_REGRESS", f"same_artist Δ={same_delta:+.5f} regresses"
    if all_delta < 0:
        return "FAIL_REGRESS", f"all-dev Δ={all_delta:+.5f} worse than baseline"
    churn_all_frac = churn_all / n
    churn_h7_frac = churn_h7 / n_h7
    if (h7_delta >= GATE_PROD_NDCG_DELTA and same_delta >= 0
            and churn_all_frac <= GATE_PROD_TOP1_CHURN_FRAC
            and churn_h7_frac <= GATE_PROD_TOP1_CHURN_FRAC):
        return "PASS_PROD", f"h7 Δ={h7_delta:+.5f}"
    if (h7_delta >= GATE_EXP_NDCG_DELTA and same_delta >= 0
            and churn_all_frac <= GATE_EXP_TOP1_CHURN_FRAC
            and churn_h7_frac <= GATE_EXP_TOP1_CHURN_FRAC):
        return "PASS_EXP", f"h7 Δ={h7_delta:+.5f}"
    return "FAIL_GATE", f"h7 Δ={h7_delta:+.5f} below gate or churn out of bounds"


def main():
    t0 = time.time()
    print("R57b structural feature implementation — dev-only evaluator")
    print("=" * 70)

    print(f"\n{ts()} Loading R12 payload, R21 OOF, R54 Phase 2 OOF, metadata...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)
    ta = payload["track_artist"]
    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R54_PHASE2_OOF) as f:
        r54_data = json.load(f)
    r54_raw = r54_data["lists"]
    r54_source, r54_scores = [], []
    for cl in r54_raw:
        r54_source.append([t for t, _ in cl])
        r54_scores.append({t: float(s) for t, s in cl})
    track_pop = build_popularity_stats()
    track_album = load_track_albums()
    track_meta = load_track_metadata_compact()

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

    # Run each config
    results = {}
    baseline_metrics = None
    baseline_top1 = None
    n_h7 = sum(1 for c in cases if c["n_prior_music"] == 7)

    for cfg_name, feat_names in CONFIGS.items():
        print(f"\n{ts()} Config {cfg_name}: {len(feat_names)} features...")
        X, gt_idx, sizes, pools, src_union_has_gt = build_features(
            payload, r21_source, r54_source, r54_scores, als_source, als_vecs,
            als_factors, als_track_to_idx, track_pop, track_album, track_meta,
            feat_names,
        )
        case_ranks, case_top1, importances = run_cv5_lr(X, gt_idx, sizes, pools, folds, feat_names)
        metrics = compute_metrics(case_ranks, cases, ta, gt_idx, src_union_has_gt)

        if cfg_name == "base":
            baseline_metrics = metrics
            baseline_top1 = case_top1
            print(f"\n{ts()} Verifying baseline reproduction vs refreshed decomp...")
            verify_baseline(metrics)
            results[cfg_name] = {
                "h7_ndcg": metrics["h7_ndcg"],
                "all_ndcg": metrics["all_ndcg"],
                "same_artist_ndcg": metrics["same_artist_ndcg"],
                "diff_artist_ndcg": metrics["diff_artist_ndcg"],
                "bucket_counts": metrics["bucket_counts"],
                "feature_importance": importances,
                "n_features": len(feat_names),
            }
            continue

        # Compare to baseline
        base_buckets = baseline_metrics["buckets_per_case"]
        var_buckets = metrics["buckets_per_case"]
        recovered = sum(1 for i in range(n)
                         if base_buckets[i] == "DEMOTED" and var_buckets[i] == "HIT")
        lost = sum(1 for i in range(n)
                   if base_buckets[i] == "HIT" and var_buckets[i] == "DEMOTED")
        net = recovered - lost
        churn_all = sum(1 for i in range(n) if case_top1[i] != baseline_top1[i])
        churn_h7 = sum(1 for i in metrics["h7_idx"] if case_top1[i] != baseline_top1[i])
        verdict, reason = gates_verdict(
            baseline_metrics, metrics, churn_all, churn_h7, n, n_h7, net,
        )
        # New-feature importance subset
        new_feat_names = [f for f in feat_names if f not in FEAT_BASELINE]
        new_feat_importance = {f: importances[f] for f in new_feat_names}
        results[cfg_name] = {
            "h7_ndcg": metrics["h7_ndcg"],
            "h7_ndcg_delta": metrics["h7_ndcg"] - baseline_metrics["h7_ndcg"],
            "all_ndcg": metrics["all_ndcg"],
            "all_ndcg_delta": metrics["all_ndcg"] - baseline_metrics["all_ndcg"],
            "same_artist_ndcg": metrics["same_artist_ndcg"],
            "same_artist_ndcg_delta": metrics["same_artist_ndcg"] - baseline_metrics["same_artist_ndcg"],
            "diff_artist_ndcg": metrics["diff_artist_ndcg"],
            "diff_artist_ndcg_delta": metrics["diff_artist_ndcg"] - baseline_metrics["diff_artist_ndcg"],
            "bucket_counts": metrics["bucket_counts"],
            "recovered": recovered,
            "lost": lost,
            "net_recovery": net,
            "top1_churn_all": churn_all,
            "top1_churn_h7": churn_h7,
            "top1_churn_all_frac": churn_all / n,
            "top1_churn_h7_frac": churn_h7 / n_h7,
            "feature_importance": importances,
            "new_feature_importance": new_feat_importance,
            "gate_verdict": verdict,
            "gate_reason": reason,
            "n_features": len(feat_names),
        }

    # Report
    print(f"\n{'=' * 100}")
    print(f"{'cfg':<10s} | {'h7_Δ':>9s} | {'all_Δ':>9s} | {'sa_Δ':>9s} | {'da_Δ':>9s} | "
          f"{'rec':>4s} | {'lost':>4s} | {'net':>4s} | {'churn_all':>9s} | {'churn_h7':>9s} | verdict")
    print("-" * 130)
    for cfg, r in results.items():
        if cfg == "base":
            print(f"{cfg:<10s} | {'baseline (h7=' + f'{r['h7_ndcg']:.5f}' + ', all=' + f'{r['all_ndcg']:.5f}' + ')':>9s} ...")
            continue
        line = (f"{cfg:<10s} | {r['h7_ndcg_delta']:+.5f} | {r['all_ndcg_delta']:+.5f} | "
                f"{r['same_artist_ndcg_delta']:+.5f} | {r['diff_artist_ndcg_delta']:+.5f} | "
                f"{r['recovered']:>4d} | {r['lost']:>4d} | {r['net_recovery']:>+4d} | "
                f"{r['top1_churn_all_frac']:>8.2%} | {r['top1_churn_h7_frac']:>8.2%} | {r['gate_verdict']}")
        print(line)
        # Feature importance for new features
        if r.get("new_feature_importance"):
            print(f"           new-feature importance: ", end="")
            for f, imp in r["new_feature_importance"].items():
                print(f"{f}={imp:.0f}  ", end="")
            print()

    # Save
    out_data = {
        "baseline": {
            "h7_ndcg": baseline_metrics["h7_ndcg"],
            "all_ndcg": baseline_metrics["all_ndcg"],
            "same_artist_ndcg": baseline_metrics["same_artist_ndcg"],
            "diff_artist_ndcg": baseline_metrics["diff_artist_ndcg"],
            "bucket_counts": baseline_metrics["bucket_counts"],
            "feature_importance": results["base"]["feature_importance"],
        },
        "configs": {k: v for k, v in results.items() if k != "base"},
        "gates": {
            "production_h7_delta": GATE_PROD_NDCG_DELTA,
            "exploratory_h7_delta": GATE_EXP_NDCG_DELTA,
            "production_churn_frac": GATE_PROD_TOP1_CHURN_FRAC,
            "exploratory_churn_frac": GATE_EXP_TOP1_CHURN_FRAC,
            "same_artist_regress_eps": GATE_SAME_ARTIST_REGRESS_EPS,
        },
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n{ts()} Saved: {OUT}  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
