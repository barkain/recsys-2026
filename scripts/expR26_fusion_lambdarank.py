#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R26 Phase 2: Fusion + LambdaRank with intent retrieval sources.

Tests (controlled sequence per codex):
1. R21 baseline (29 features, 7 sources) — control
2. +Q3 intent dense (31 features, 8 sources)
3. +Q2+Q3 (33 features, 9 sources) — only if Q3 improves

Uses R22b build_features as template with Q3/Q2 added.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"

import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

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
Q3_LISTS = REPO / "cache" / "r26" / "q3_dense_results.json"
Q2_LISTS = None  # built inline from BM25
RRF_K = 20
POOL_K = 300

FEATURE_NAMES_R21 = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEATURE_NAMES_Q3 = FEATURE_NAMES_R21 + ["q3_rank_inv", "q3_presence"]
FEATURE_NAMES_Q2Q3 = FEATURE_NAMES_Q3 + ["q2_rank_inv", "q2_presence"]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_features(cases, payload, als_source, als_vecs, als_factors, als_track_to_idx,
                   track_pop, r21_source, extra_sources, source_weights, feature_names):
    n = len(cases)
    n_feat = len(feature_names)
    has_q3 = "q3_rank_inv" in feature_names
    has_q2 = "q2_rank_inv" in feature_names

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        src_lists: dict[str, list[str]] = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        if has_q3 and "Q3" in extra_sources:
            src_lists["Q3"] = extra_sources["Q3"][i]
        if has_q2 and "Q2" in extra_sources:
            src_lists["Q2"] = extra_sources["Q2"][i]

        pool = weighted_rrf(src_lists, source_weights, topk=POOL_K, k=RRF_K)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank: dict[str, dict[str, int]] = {}
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
        sv = als_vecs[i]

        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}
        q3_rank_map: dict[str, int] = {}
        q2_rank_map: dict[str, int] = {}
        if has_q3 and "Q3" in extra_sources:
            q3_rank_map = {tid: r + 1 for r, tid in enumerate(extra_sources["Q3"][i][:300])}
        if has_q2 and "Q2" in extra_sources:
            q2_rank_map = {tid: r + 1 for r, tid in enumerate(extra_sources["Q2"][i][:300])}

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
            row[20] = sum(1 for sname in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sname, {}))
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
            if has_q3:
                row[29] = 1.0 / q3_rank_map[tid] if tid in q3_rank_map else 0.0
                row[30] = 1.0 if tid in q3_rank_map else 0.0
            if has_q2:
                row[31] = 1.0 / q2_rank_map[tid] if tid in q2_rank_map else 0.0
                row[32] = 1.0 if tid in q2_rank_map else 0.0

    return X, gt_idx, sizes


def run_cv_sliced(X, gt_idx, sizes, cases, sessions, feature_names, seed=0):
    n = X.shape[0]
    folds = grouped_session_folds(sessions, seed=seed)
    case_ndcg = np.zeros(n)

    for fi in range(5):
        val_idx = set(folds[fi].tolist())
        train_idx = [j for j in range(n) if j not in val_idx]
        val_list = sorted(val_idx)

        X_flat_train, y_train, g_train = [], [], []
        X_flat_val, y_val, g_val = [], [], []

        for i in train_idx:
            s = int(sizes[i])
            for k in range(s):
                X_flat_train.append(X[i, k])
                y_train.append(1.0 if k == gt_idx[i] else 0.0)
            g_train.append(s)

        for i in val_list:
            s = int(sizes[i])
            for k in range(s):
                X_flat_val.append(X[i, k])
                y_val.append(1.0 if k == gt_idx[i] else 0.0)
            g_val.append(s)

        X_tr = np.array(X_flat_train)
        X_va = np.array(X_flat_val)
        ds_tr = lgb.Dataset(X_tr, label=np.array(y_train), group=g_train,
                            feature_name=list(feature_names))
        ds_va = lgb.Dataset(X_va, label=np.array(y_val), group=g_val, reference=ds_tr)

        params = {
            "objective": "lambdarank", "metric": "ndcg",
            "eval_at": [20], "num_leaves": 31, "learning_rate": 0.05,
            "min_data_in_leaf": 10, "verbose": -1, "seed": seed,
        }
        model = lgb.train(params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])

        preds_va = model.predict(X_va)
        offset = 0
        for i in val_list:
            s = int(sizes[i])
            scores = preds_va[offset:offset + s]
            offset += s
            if gt_idx[i] < 0:
                continue
            ranked = np.argsort(-scores)
            gt_pos = np.where(ranked == gt_idx[i])[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg[i] = 1.0 / np.log2(gt_pos[0] + 2)

    slices = {}
    for depth in range(8):
        idx = [i for i in range(n) if cases[i]["n_prior_music"] == depth]
        if idx:
            slices[f"hist_{depth}"] = float(np.mean([case_ndcg[i] for i in idx]))

    has_hist = [i for i in range(n) if len(cases[i]["music_turns"]) > 0]
    no_hist = [i for i in range(n) if len(cases[i]["music_turns"]) == 0]

    return {
        "cv5": float(np.mean(case_ndcg)),
        "last_turn": slices.get("hist_7", 0),
        "hist_0": slices.get("hist_0", 0),
        "hist_7": slices.get("hist_7", 0),
        "slices": slices,
        "has_hist_ndcg": float(np.mean([case_ndcg[i] for i in has_hist])) if has_hist else 0,
        "no_hist_ndcg": float(np.mean([case_ndcg[i] for i in no_hist])) if no_hist else 0,
    }


def build_q2_lists(cases, intents_path):
    """Build Q2 artist-boost BM25 lists."""
    from mcrs.retrieval_modules.bm25 import BM25Retriever

    with open(intents_path) as f:
        intents_list = json.load(f)
    intent_map = {(r["session_id"], r["turn_number"]): r.get("intent") for r in intents_list}

    queries = []
    for case in cases:
        intent = intent_map.get((case["session_id"], case["turn_number"]))
        if intent:
            parts = []
            for artist in intent.get("positive_artists", []):
                parts.extend([artist] * 3)
            for genre in intent.get("genres", []):
                parts.extend([genre] * 2)
            parts.extend(intent.get("moods", []))
            if intent.get("era"):
                parts.append(intent["era"])
            if intent.get("context"):
                parts.append(intent["context"])
            parts.extend(intent.get("similarity_anchors", []))
            queries.append(" ".join(parts) if parts else intent.get("summary", case["user_query"]))
        else:
            queries.append(case["user_query"])

    bm25 = BM25Retriever(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
        corpus_types=["track_name", "artist_name", "album_name", "tag_list"],
        cache_dir=str(REPO / "cache"),
    )
    return bm25.batch_text_to_item_retrieval(queries, topk=300)


def print_comparison(configs):
    """Print comparison table."""
    names = list(configs.keys())

    sep = "=" * 70
    print(f"\n{sep}")
    print("R26 PHASE 2: FUSION + LAMBDARANK")
    print(sep)

    header = f"  {'Metric':<25}"
    for name in names:
        header += f" {name:>12}"
    if len(names) > 1:
        for name in names[1:]:
            header += f" {'D_'+name:>12}"
    print(header)
    print(f"  {'-'*len(header)}")

    for metric in ["pool_hit", "cv5", "last_turn", "hist_0", "hist_7", "has_hist_ndcg", "no_hist_ndcg"]:
        row = f"  {metric:<25}"
        vals = []
        for name in names:
            v = configs[name].get(metric, 0)
            vals.append(v)
            fmt = f"{v:>12.5f}" if metric != "pool_hit" else f"{v:>12.4f}"
            row += fmt
        if len(names) > 1:
            for v in vals[1:]:
                d = v - vals[0]
                row += f" {d:>+12.5f}"
        print(row)


def main():
    t0 = time.time()
    print(f"{ts()} R26 Phase 2: Fusion + LambdaRank")
    print("=" * 70)

    print(f"\n{ts()} Loading data...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    print(f"  {n} cases, {len(set(sessions))} sessions")

    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(Q3_LISTS) as f:
        q3_source = json.load(f)
    print(f"  R21 OOF lists: {len(r21_source)}")
    print(f"  Q3 dense lists: {len(q3_source)}")

    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
    als_vecs = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
            als_vecs.append(sv)
        else:
            als_source.append([])
            als_vecs.append(None)

    track_pop = build_popularity_stats()

    configs: dict[str, dict] = {}

    # ---------------------------------------------------------------
    # Config 1: R21 baseline (control)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Config 1: R21 baseline (29 features, 7 sources)")
    sw_r21 = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
    X1, gt1, sz1 = build_features(cases, payload, als_source, als_vecs, als_factors,
                                   als_track_to_idx, track_pop, r21_source, {},
                                   sw_r21, FEATURE_NAMES_R21)
    pool_hit_r21 = float(np.mean(gt1 >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit_r21:.4f}")
    print("  Running CV5...")
    r1 = run_cv_sliced(X1, gt1, sz1, cases, sessions, FEATURE_NAMES_R21)
    r1["pool_hit"] = pool_hit_r21
    configs["R21"] = r1
    print(f"  CV5: {r1['cv5']:.5f}  last_turn: {r1['last_turn']:.5f}  hist_0: {r1['hist_0']:.5f}")

    # ---------------------------------------------------------------
    # Config 2: +Q3 intent dense (weight=0.5)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Config 2: +Q3 intent dense (31 features, 8 sources, w=0.5)")
    sw_q3 = {**sw_r21, "Q3": 0.5}
    X2, gt2, sz2 = build_features(cases, payload, als_source, als_vecs, als_factors,
                                   als_track_to_idx, track_pop, r21_source,
                                   {"Q3": q3_source}, sw_q3, FEATURE_NAMES_Q3)
    pool_hit_q3 = float(np.mean(gt2 >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit_q3:.4f}")
    print("  Running CV5...")
    r2 = run_cv_sliced(X2, gt2, sz2, cases, sessions, FEATURE_NAMES_Q3)
    r2["pool_hit"] = pool_hit_q3
    configs["+Q3_w05"] = r2
    print(f"  CV5: {r2['cv5']:.5f}  last_turn: {r2['last_turn']:.5f}  hist_0: {r2['hist_0']:.5f}")

    # Decision: if Q3 dilutes, try lower weight
    q3_pool_delta = pool_hit_q3 - pool_hit_r21
    q3_cv5_delta = r2["cv5"] - r1["cv5"]
    if q3_pool_delta >= 0.02 and q3_cv5_delta < -0.001:
        print("\n  Q3 w=0.5 raises pool but hurts CV5 -- trying w=0.25...")
        sw_q3_low = {**sw_r21, "Q3": 0.25}
        X2b, gt2b, sz2b = build_features(cases, payload, als_source, als_vecs, als_factors,
                                          als_track_to_idx, track_pop, r21_source,
                                          {"Q3": q3_source}, sw_q3_low, FEATURE_NAMES_Q3)
        pool_hit_q3b = float(np.mean(gt2b >= 0))
        print(f"  pool_hit@{POOL_K}: {pool_hit_q3b:.4f}")
        r2b = run_cv_sliced(X2b, gt2b, sz2b, cases, sessions, FEATURE_NAMES_Q3)
        r2b["pool_hit"] = pool_hit_q3b
        configs["+Q3_w025"] = r2b
        print(f"  CV5: {r2b['cv5']:.5f}  last_turn: {r2b['last_turn']:.5f}")

    # ---------------------------------------------------------------
    # Config 3: +Q2+Q3 (only if Q3 improves)
    # ---------------------------------------------------------------
    best_q3_key = "+Q3_w05"
    best_q3 = configs[best_q3_key]
    if "+Q3_w025" in configs and configs["+Q3_w025"]["cv5"] > configs["+Q3_w05"]["cv5"]:
        best_q3_key = "+Q3_w025"
        best_q3 = configs[best_q3_key]

    q3_improves = (best_q3["pool_hit"] - pool_hit_r21 >= 0.015) and (best_q3["cv5"] >= r1["cv5"] - 0.001)

    if q3_improves:
        print(f"\n{ts()} Config 3: +Q2+Q3 (33 features, 9 sources)")
        print("  Building Q2 BM25 lists...")
        q2_source = build_q2_lists(cases, REPO / "cache" / "r26" / "intents_dev.json")
        q3_weight = 0.5 if best_q3_key == "+Q3_w05" else 0.25
        sw_q2q3 = {**sw_r21, "Q3": q3_weight, "Q2": 0.25}
        X3, gt3, sz3 = build_features(cases, payload, als_source, als_vecs, als_factors,
                                       als_track_to_idx, track_pop, r21_source,
                                       {"Q3": q3_source, "Q2": q2_source},
                                       sw_q2q3, FEATURE_NAMES_Q2Q3)
        pool_hit_q2q3 = float(np.mean(gt3 >= 0))
        print(f"  pool_hit@{POOL_K}: {pool_hit_q2q3:.4f}")
        r3 = run_cv_sliced(X3, gt3, sz3, cases, sessions, FEATURE_NAMES_Q2Q3)
        r3["pool_hit"] = pool_hit_q2q3
        configs["+Q2Q3"] = r3
        print(f"  CV5: {r3['cv5']:.5f}  last_turn: {r3['last_turn']:.5f}  hist_0: {r3['hist_0']:.5f}")
    else:
        print("\n  Q3 did not improve enough -- skipping Q2+Q3 test.")
        print(f"    pool_hit delta: {best_q3['pool_hit']-pool_hit_r21:+.4f} (gate >= 0.015)")
        print(f"    CV5 delta: {best_q3['cv5']-r1['cv5']:+.5f} (gate >= -0.001)")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print_comparison(configs)

    # Gate check
    sep = "=" * 70
    print(f"\n{sep}")
    print("GATE CHECK")
    best_name = max(configs, key=lambda k: configs[k]["cv5"] if k != "R21" else -1)
    best = configs[best_name]
    pool_delta = best["pool_hit"] - pool_hit_r21
    cv5_delta = best["cv5"] - r1["cv5"]
    last_delta = best["last_turn"] - r1["last_turn"]
    hist7_delta = best.get("hist_7", 0) - r1.get("hist_7", 0)

    print(f"  Best config: {best_name}")
    g_pool = pool_delta >= 0.020
    g_cv5 = cv5_delta >= 0.005
    g_last = last_delta >= 0.005
    g_hist7 = hist7_delta >= -0.003
    print(f"  pool_hit delta >= +0.020: {'PASS' if g_pool else 'FAIL'} ({pool_delta:+.4f})")
    print(f"  CV5 delta >= +0.005:      {'PASS' if g_cv5 else 'FAIL'} ({cv5_delta:+.5f})")
    print(f"  last_turn delta >= +0.005:{'PASS' if g_last else 'FAIL'} ({last_delta:+.5f})")
    print(f"  hist_7 no regression:     {'PASS' if g_hist7 else 'FAIL'} ({hist7_delta:+.5f})")
    print(f"  ALL GATES: {'PASS' if (g_pool and g_cv5 and g_last and g_hist7) else 'FAIL'}")

    # Save
    out_path = REPO / "exp" / "eval" / "expR26_stage2_fusion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "configs": {k: v for k, v in configs.items()},
        "best": best_name,
        "gates": {
            "pool_hit_delta": pool_delta, "cv5_delta": cv5_delta,
            "last_turn_delta": last_delta, "hist7_delta": hist7_delta,
            "all_pass": g_pool and g_cv5 and g_last and g_hist7,
        },
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
