#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R58 inventory — no-train feasibility capture.

Per docs/r58_design.md §3.6, this script ONLY:

1. Verifies R58-required artifact existence and sizes.
2. Reproduces the R39+R54 dev baseline (CV5 LambdaRank) within ε.
3. Captures per-case top-50 LR rows with the 37 baseline features +
   per-source ranks + margin_to_20.
4. Writes an inventory report.

It does NOT:
- Train any specialist model
- Touch blind data
- Compute pairwise / cross-candidate aggregates
- Encode raw 768d embeddings
- Tune or retrain LR

Exits with code 0 (PROCEED-TO-ARCHITECTURE-PHASE) or 2 (BLOCK-AND-FREEZE).

Outputs:
  cache/r58/top50_dev.pkl          — per-case top-50 table
  exp/eval/expR58_inventory.json   — inventory report
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
R21_MODEL_DIR = REPO / "cache" / "r21_production" / "model"
R21_TRACK_EMBS = REPO / "cache" / "r21_production" / "track_embeddings.npy"
R54_PHASE2_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
LR_MODEL = REPO / "cache" / "r54_phase3_lr_model.txt"
ALS_CACHE = REPO / "cache" / "r54_phase3_als.npz"
MAPS_CACHE = REPO / "cache" / "r54_phase3_payload_maps.pkl"
POP_CACHE = REPO / "cache" / "r54_phase3_track_pop.json"
DECOMP_JSON = REPO / "exp" / "eval" / "expR55_post_refresh_decomp.json"
OUT_TABLE = REPO / "cache" / "r58" / "top50_dev.pkl"
OUT_REPORT = REPO / "exp" / "eval" / "expR58_inventory.json"

# R58 inventory targets — required-for-feasibility artifacts
REQUIRED = {
    "r12_payload": R12_CACHE,
    "r21_oof": R21_OOF,
    "r21_model_dir": R21_MODEL_DIR,
    "r21_track_embs": R21_TRACK_EMBS,
    "r54_phase2_oof": R54_PHASE2_OOF,
    "als_cache": ALS_CACHE,
    "maps_cache": MAPS_CACHE,
    "pop_cache": POP_CACHE,
    "decomp_baseline": DECOMP_JSON,
}

POOL_K = 300
RRF_K = 20
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
FEAT_R54 = ["r54_rank_inv", "r54_presence", "r54_cosine"]
ALL_FEAT = FEAT_BASE + FEAT_ALBUM + FEAT_R54

NDCG_EPS = 0.0005
BUCKET_FRAC_EPS = 0.005


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def file_size(p):
    if not p.exists():
        return None
    if p.is_dir():
        total = 0
        for child in p.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total
    return p.stat().st_size


def fmt_size(n):
    if n is None:
        return "MISSING"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def check_artifacts():
    report = {}
    halt = False
    print(f"{ts()} Artifact presence + size check")
    for name, path in REQUIRED.items():
        size = file_size(path)
        ok = size is not None
        report[name] = {"path": str(path), "exists": ok, "size_bytes": size,
                        "size_human": fmt_size(size)}
        marker = "OK " if ok else "MISS"
        print(f"  [{marker}] {name:<20s} {fmt_size(size) if size else 'missing':<10s}  {path}")
        if not ok:
            halt = True
    return report, halt


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
            out[tid] = str(alb_name[0]) if isinstance(alb_name, list) and alb_name else ""
    return out


def build_features_and_cv5():
    """Build R39+R54 features, run CV5 LR, return per-case data."""
    print(f"{ts()} Loading R12 + R21 OOF + R54 Phase 2 OOF...")
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
    r54_source, r54_scores = [], []
    for cl in r54_raw:
        r54_source.append([t for t, _ in cl])
        r54_scores.append({t: float(s) for t, s in cl})
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
    fold_of = np.zeros(n, dtype=np.int64)
    for fi in range(5):
        for idx in folds[fi]:
            fold_of[idx] = fi

    n_feat_base = len(FEAT_BASE)
    n_feat_r39 = n_feat_base + len(FEAT_ALBUM)
    X = np.zeros((n, POOL_K, len(ALL_FEAT)), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = [[] for _ in range(n)]
    src_union_has_gt = np.zeros(n, dtype=bool)

    # Per-source rank lookup tables (will be reused for the top-50 table)
    per_case_src_lists: list[dict] = []

    print(f"{ts()} Building features (37 per candidate, pool@{POOL_K})...")
    t_feat = time.time()
    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i], "R54": r54_source[i],
        }
        per_case_src_lists.append(src_lists)
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
        if (i + 1) % 2000 == 0:
            print(f"  features {i + 1}/{n} ({time.time() - t_feat:.0f}s)", flush=True)

    # CV5 LR, capture per-case scores for the entire pool (so we can take top-50)
    print(f"{ts()} CV5 LambdaRank (capture per-case scores over pool@{POOL_K})...")
    case_lr_rank = np.full(n, -1, dtype=np.int64)
    per_case_lr_scores: list[np.ndarray] = [None] * n
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
            per_case_lr_scores[idx] = sc.copy()
            if gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0:
                case_lr_rank[idx] = int(gt_pos[0]) + 1

    return {
        "cases": cases,
        "ta": ta,
        "X": X,
        "gt_idx": gt_idx,
        "sizes": sizes,
        "pools": pools,
        "src_union_has_gt": src_union_has_gt,
        "case_lr_rank": case_lr_rank,
        "per_case_lr_scores": per_case_lr_scores,
        "fold_of": fold_of,
        "per_case_src_lists": per_case_src_lists,
        "r54_scores": r54_scores,
        "track_album": track_album,
    }


def compute_baseline_metrics(state):
    cases = state["cases"]
    ta = state["ta"]
    gt_idx = state["gt_idx"]
    case_lr_rank = state["case_lr_rank"]
    src_union_has_gt = state["src_union_has_gt"]
    n = len(cases)

    ndcg = np.array([1.0 / np.log2(r + 1) if r > 0 and r <= 20 else 0.0
                      for r in case_lr_rank])
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
        if gt_idx[i] >= 0 and case_lr_rank[i] >= 1 and case_lr_rank[i] <= 20:
            buckets.append("HIT")
        elif gt_idx[i] >= 0:
            buckets.append("DEMOTED")
        elif src_union_has_gt[i]:
            buckets.append("POOL_MISS")
        else:
            buckets.append("UNREACHABLE")
    return {
        "all_ndcg": float(ndcg.mean()),
        "h7_ndcg": float(ndcg[h7_idx].mean()) if h7_idx else 0.0,
        "same_artist_ndcg": float(ndcg[same_art_idx].mean()) if same_art_idx else 0.0,
        "diff_artist_ndcg": float(ndcg[diff_art_idx].mean()) if diff_art_idx else 0.0,
        "bucket_counts": dict(Counter(buckets)),
    }


def verify_baseline(metrics):
    with open(DECOMP_JSON) as f:
        decomp = json.load(f)
    failures = []
    if abs(metrics["all_ndcg"] - decomp["overall_ndcg"]) > NDCG_EPS:
        failures.append(f"all_ndcg: refresh={decomp['overall_ndcg']:.5f}  here={metrics['all_ndcg']:.5f}")
    if abs(metrics["h7_ndcg"] - decomp["h7_ndcg"]) > NDCG_EPS:
        failures.append(f"h7_ndcg: refresh={decomp['h7_ndcg']:.5f}  here={metrics['h7_ndcg']:.5f}")
    n = sum(metrics["bucket_counts"].values())
    for b, dc in decomp["bucket_counts"].items():
        rc = metrics["bucket_counts"].get(b, 0)
        if abs(dc / n - rc / n) > BUCKET_FRAC_EPS:
            failures.append(f"bucket {b}: refresh={dc} here={rc}")
    return failures


def build_top50_table(state):
    """Per-case top-50 row table with the 37 baseline features + per-source ranks."""
    cases = state["cases"]
    X = state["X"]
    pools = state["pools"]
    sizes = state["sizes"]
    gt_idx = state["gt_idx"]
    per_case_lr_scores = state["per_case_lr_scores"]
    fold_of = state["fold_of"]
    per_case_src_lists = state["per_case_src_lists"]
    r54_scores = state["r54_scores"]
    n = len(cases)

    # Precompute per-source rank maps per case (300-deep)
    print(f"{ts()} Building per-case top-50 table...")
    rows = []
    for i in range(n):
        scores = per_case_lr_scores[i]
        if scores is None or sizes[i] == 0:
            continue
        # Rank pool candidates by LR score; index in X for the feature row is the
        # rank-position-in-pool (pool[k] -> X[i, k])
        order = np.argsort(-scores)  # indices into pool
        # margin_to_20 = LR_score(rank 1, i.e. top of order) - LR_score(rank 20)
        if len(order) >= 20:
            score_at_20 = float(scores[order[19]])
        else:
            score_at_20 = float(scores[order[-1]])
        top_score = float(scores[order[0]])
        margin_to_20 = top_score - score_at_20
        gt_in_pool = bool(gt_idx[i] >= 0)
        # GT LR rank (within pool, 1..N) — needed for analysis
        gt_lr_rank = -1
        if gt_in_pool:
            gt_pos = np.where(order == gt_idx[i])[0]
            if len(gt_pos):
                gt_lr_rank = int(gt_pos[0]) + 1

        src_lists = per_case_src_lists[i]
        src_rank_maps = {
            "A": {tid: r + 1 for r, tid in enumerate(src_lists["A"])},
            "B": {tid: r + 1 for r, tid in enumerate(src_lists["B"])},
            "C": {tid: r + 1 for r, tid in enumerate(src_lists["C"])},
            "D": {tid: r + 1 for r, tid in enumerate(src_lists["D"])},
            "F": {tid: r + 1 for r, tid in enumerate(src_lists["F"])},
            "ALS": {tid: r + 1 for r, tid in enumerate(src_lists["ALS"])},
            "R21": {tid: r + 1 for r, tid in enumerate(src_lists["R21"])},
            "R54": {tid: r + 1 for r, tid in enumerate(src_lists["R54"])},
        }
        r54_score_map = r54_scores[i]

        for r_idx in range(min(50, len(order))):
            k = int(order[r_idx])
            tid = pools[i][k]
            lr_score = float(scores[k])
            feats = X[i, k]  # 37-dim feature row
            row = {
                "case_idx": i,
                "session_id": cases[i]["session_id"],
                "fold_id": int(fold_of[i]),
                "candidate_rank": r_idx + 1,
                "candidate_track_id": tid,
                "lr_score": lr_score,
                "lr_score_minus_top": lr_score - top_score,
                "lr_score_minus_at20": lr_score - score_at_20,
                "margin_to_20_case": margin_to_20,
                "gt_flag": int(tid == cases[i]["gt"]),
                "gt_in_pool": int(gt_in_pool),
                "gt_lr_rank": int(gt_lr_rank),
                "r21_rank": int(src_rank_maps["R21"].get(tid, -1)),
                "r54_rank": int(src_rank_maps["R54"].get(tid, -1)),
                "r54_cosine": float(r54_score_map.get(tid, 0.0)),
                "a_rank": int(src_rank_maps["A"].get(tid, -1)),
                "b_rank": int(src_rank_maps["B"].get(tid, -1)),
                "c_rank": int(src_rank_maps["C"].get(tid, -1)),
                "d_rank": int(src_rank_maps["D"].get(tid, -1)),
                "f_rank": int(src_rank_maps["F"].get(tid, -1)),
                "als_rank": int(src_rank_maps["ALS"].get(tid, -1)),
                "features": feats.astype(np.float32).tolist(),
            }
            rows.append(row)
    return rows


FIELD_AVAILABILITY = {
    "lr_score": ("YES", "captured from CV5 LR within this script"),
    "lr_rank": ("YES", "derived from lr_score per case"),
    "margin_to_20_case": ("YES", "derived from top-1 minus rank-20 LR score"),
    "r54_rank": ("YES", "from R54 Phase 2 OOF"),
    "r54_cosine": ("YES", "from R54 Phase 2 OOF list scores"),
    "r21_rank": ("YES", "from R21 OOF"),
    "a_rank": ("YES", "from R12 payload src_a"),
    "b_rank": ("YES", "from R12 payload src_b"),
    "c_rank": ("YES", "from R12 payload src_c"),
    "d_rank": ("YES", "from R12 payload src_d"),
    "f_rank": ("YES", "from R12 payload src_f"),
    "als_rank": ("YES", "rebuilt from cached ALS factors"),
    "37 baseline features per candidate": ("YES", "already computed in CV5 feature loop"),
    "raw 768d embeddings": ("DEFERRED", "out of Phase 1 scope per design §3.8"),
    "pairwise / cross-candidate aggregates": ("DEFERRED", "Phase 2 architecture decision"),
}


def main():
    t0 = time.time()
    print("R58 inventory — no-train feasibility capture")
    print("=" * 70)

    artifact_report, halt = check_artifacts()
    if halt:
        print(f"\n  MISSING required artifact(s). BLOCK-AND-FREEZE.")
        report = {
            "verdict": "BLOCK_AND_FREEZE",
            "reason": "missing required artifacts",
            "artifacts": artifact_report,
        }
        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_REPORT, "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(2)

    state = build_features_and_cv5()
    baseline = compute_baseline_metrics(state)

    print(f"\n{ts()} Verifying baseline reproduction vs refreshed decomp...")
    failures = verify_baseline(baseline)
    if failures:
        print(f"  BASELINE REPRODUCTION MISMATCH:")
        for f in failures:
            print(f"    {f}")
        report = {
            "verdict": "BLOCK_AND_FREEZE",
            "reason": "baseline reproduction failure",
            "baseline": baseline,
            "failures": failures,
            "artifacts": artifact_report,
        }
        OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_REPORT, "w") as f:
            json.dump(report, f, indent=2)
        sys.exit(2)
    print(f"  PASS within ε")
    print(f"    all-dev nDCG: {baseline['all_ndcg']:.5f}")
    print(f"    h7 nDCG:      {baseline['h7_ndcg']:.5f}")
    print(f"    same_artist:  {baseline['same_artist_ndcg']:.5f}")
    print(f"    diff_artist:  {baseline['diff_artist_ndcg']:.5f}")
    print(f"    buckets:      {baseline['bucket_counts']}")

    rows = build_top50_table(state)
    OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TABLE, "wb") as f:
        pickle.dump(rows, f)
    table_size = OUT_TABLE.stat().st_size
    print(f"\n{ts()} Top-50 table: {len(rows)} rows  size {fmt_size(table_size)}  -> {OUT_TABLE}")

    # Sanity stats on the table
    n_cases = len({r["case_idx"] for r in rows})
    cases_with_gt_in_top50 = sum(1 for r in rows if r["gt_flag"]) > 0  # placeholder
    gt_top1 = sum(1 for r in rows if r["candidate_rank"] == 1 and r["gt_flag"])
    gt_in_top50 = len({r["case_idx"] for r in rows if r["gt_flag"]})
    avg_r54_rank_top50 = float(np.mean([r["r54_rank"] for r in rows if r["r54_rank"] > 0]))
    avg_r21_rank_top50 = float(np.mean([r["r21_rank"] for r in rows if r["r21_rank"] > 0]))
    print(f"\n  Table sanity:")
    print(f"    cases with at least one row:        {n_cases}")
    print(f"    GT is the candidate's top-1:        {gt_top1}/{n_cases}")
    print(f"    GT present somewhere in top-50:     {gt_in_top50}/{n_cases}")
    print(f"    avg R54 rank of top-50 candidates:  {avg_r54_rank_top50:.1f}")
    print(f"    avg R21 rank of top-50 candidates:  {avg_r21_rank_top50:.1f}")

    report = {
        "verdict": "PROCEED_TO_ARCHITECTURE_PHASE",
        "baseline": baseline,
        "artifacts": artifact_report,
        "top50_table": {
            "path": str(OUT_TABLE),
            "rows": len(rows),
            "cases": n_cases,
            "size_bytes": table_size,
            "size_human": fmt_size(table_size),
            "gt_top1": gt_top1,
            "gt_in_top50": gt_in_top50,
        },
        "field_availability": {k: {"verdict": v[0], "note": v[1]}
                                for k, v in FIELD_AVAILABILITY.items()},
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
        "phase": "R58 Phase 1 inventory — no training, no submission",
        "next_step_blocked_on": (
            "Architecture-choice doc review (Phase 2 of R58 plan). "
            "Do NOT proceed to specialist training until that doc is approved."
        ),
    }
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{ts()} Inventory report: {OUT_REPORT}")
    print(f"Elapsed: {time.time() - t0:.0f}s")
    print(f"\nVerdict: PROCEED_TO_ARCHITECTURE_PHASE")
    print(f"  Next step BLOCKED on: architecture-choice doc review (Phase 2).")
    print(f"  Do NOT proceed to specialist training until that doc is approved.")


if __name__ == "__main__":
    main()
