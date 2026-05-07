#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R28: Deep-history specialist — optimize last_turn (hist_7) directly.

Experiments:
1. Slice-specific LambdaRank (train only on hist_5..7)
2. Slice-weighted LambdaRank (weight hist_7 cases 3x-5x)
3. Deep-history feature set (behavioral continuity features)
4. Source weight sweep for hist_7

Primary gate: last_turn >= R21 +0.005. CV5 is secondary.
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
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEATURE_NAMES_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]

DEEP_FEATURES = [
    "recent_artist_match_any3",
    "recent_artist_match_any5",
    "recent_album_overlap",
    "recent_tag_jaccard_3",
    "recent_tag_jaccard_5",
    "source_agreement_top20",
    "r21_als_agreement",
    "r21_f_agreement",
    "als_f_agreement",
    "repeated_artist_concentration",
    "session_artist_diversity",
]
FEATURE_NAMES_DEEP = FEATURE_NAMES_BASE + DEEP_FEATURES


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_features(cases, payload, als_source, als_vecs, als_factors,
                   als_track_to_idx, track_pop, r21_source, feature_names):
    n = len(cases)
    n_feat = len(feature_names)
    has_deep = "recent_artist_match_any3" in feature_names

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    # Preload track album mapping if available
    track_album: dict[str, str] = payload.get("track_album", {})

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
        prior_list = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]
        sv = als_vecs[i]
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}

        # Deep history precomputation
        recent3_artists: set[str] = set()
        recent5_artists: set[str] = set()
        recent3_tags: set[str] = set()
        recent5_tags: set[str] = set()
        recent_albums: set[str] = set()
        r21_top20: set[str] = set()
        als_top20: set[str] = set()
        f_top20: set[str] = set()
        n_unique_artists = 0
        n_session = 1
        if has_deep:
            recent3_artists = {ta.get(t, "") for t in played[-3:]} - {""}
            recent5_artists = {ta.get(t, "") for t in played[-5:]} - {""}
            for t in played[-3:]:
                recent3_tags |= tt.get(t, set())
            for t in played[-5:]:
                recent5_tags |= tt.get(t, set())
            recent_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
            r21_top20 = set(r21_source[i][:20])
            als_top20 = set(als_source[i][:20])
            f_top20 = set(src_lists["F"][:20])
            session_artists = [ta.get(t, "") for t in played]
            n_unique_artists = len({a for a in session_artists if a})
            n_session = max(len(session_artists), 1)

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]

            # Base 29 features (same as R21 pipeline)
            row[0] = 1.0 / rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags:
                row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(tat.get(tid, set()) & now_tok))
            row[4] = float(len(ttl.get(tid, set()) & now_tok))
            row[5] = float(len(tmt.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior_list:
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

            if has_deep:
                base = len(FEATURE_NAMES_BASE)
                row[base + 0] = 1.0 if ca and ca in recent3_artists else 0.0
                row[base + 1] = 1.0 if ca and ca in recent5_artists else 0.0
                c_album = track_album.get(tid, "")
                row[base + 2] = 1.0 if c_album and c_album in recent_albums else 0.0
                if ct and recent3_tags:
                    row[base + 3] = len(ct & recent3_tags) / len(ct | recent3_tags)
                if ct and recent5_tags:
                    row[base + 4] = len(ct & recent5_tags) / len(ct | recent5_tags)
                agreement = sum(1 for s in ["R21", "ALS", "F", "A", "B", "C"]
                               if tid in set((src_lists.get(s) or [])[:20]))
                row[base + 5] = agreement / 6.0
                row[base + 6] = 1.0 if tid in r21_top20 and tid in als_top20 else 0.0
                row[base + 7] = 1.0 if tid in r21_top20 and tid in f_top20 else 0.0
                row[base + 8] = 1.0 if tid in als_top20 and tid in f_top20 else 0.0
                row[base + 9] = artist_counts.get(ca, 0) / n_session if ca else 0.0
                row[base + 10] = n_unique_artists / n_session

    return X, gt_idx, sizes


def run_cv_last_turn(X, gt_idx, sizes, cases, sessions, feature_names,
                     train_filter=None, sample_weight_fn=None, seed=0):
    """CV5 with optional training filter and sample weights.

    train_filter: if set, only train on cases where filter(case) is True
    sample_weight_fn: if set, returns weight for each training case
    """
    n = X.shape[0]
    folds = grouped_session_folds(sessions, seed=seed)
    case_ndcg = np.zeros(n)

    for fi in range(5):
        val_idx = set(folds[fi].tolist())
        all_train = [j for j in range(n) if j not in val_idx]

        if train_filter:
            train_list = [j for j in all_train if train_filter(cases[j])]
        else:
            train_list = all_train
        val_list = sorted(val_idx)

        X_flat_train, y_train, g_train, w_train = [], [], [], []
        X_flat_val, y_val, g_val = [], [], []

        for idx in train_list:
            s = int(sizes[idx])
            w = sample_weight_fn(cases[idx]) if sample_weight_fn else 1.0
            for k in range(s):
                X_flat_train.append(X[idx, k])
                y_train.append(1.0 if k == gt_idx[idx] else 0.0)
                w_train.append(w)
            g_train.append(s)

        for idx in val_list:
            s = int(sizes[idx])
            for k in range(s):
                X_flat_val.append(X[idx, k])
                y_val.append(1.0 if k == gt_idx[idx] else 0.0)
            g_val.append(s)

        if not X_flat_train:
            continue

        ds_tr = lgb.Dataset(np.array(X_flat_train), label=np.array(y_train),
                            group=g_train, weight=np.array(w_train),
                            feature_name=list(feature_names))
        ds_va = lgb.Dataset(np.array(X_flat_val), label=np.array(y_val),
                            group=g_val, reference=ds_tr)

        params = {
            "objective": "lambdarank", "metric": "ndcg",
            "eval_at": [20], "num_leaves": 31, "learning_rate": 0.05,
            "min_data_in_leaf": 10, "verbose": -1, "seed": seed,
        }
        model = lgb.train(params, ds_tr, num_boost_round=300,
                          valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])

        preds_va = model.predict(np.array(X_flat_val))
        offset = 0
        for idx in val_list:
            s = int(sizes[idx])
            sc = preds_va[offset:offset + s]
            offset += s
            if gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg[idx] = 1.0 / np.log2(gt_pos[0] + 2)

    slices = {}
    for depth in range(8):
        idx_list = [i for i in range(n) if cases[i]["n_prior_music"] == depth]
        if idx_list:
            slices[f"hist_{depth}"] = float(np.mean([case_ndcg[i] for i in idx_list]))

    hist57 = [i for i in range(n) if cases[i]["n_prior_music"] >= 5]

    return {
        "cv5": float(np.mean(case_ndcg)),
        "last_turn": slices.get("hist_7", 0),
        "hist_0": slices.get("hist_0", 0),
        "hist_7": slices.get("hist_7", 0),
        "hist_57": float(np.mean([case_ndcg[i] for i in hist57])) if hist57 else 0,
        "slices": slices,
    }


def main():
    t0 = time.time()
    print(f"{ts()} R28: Deep-History Specialist")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]

    # Check for track_album in payload
    has_album = "track_album" in payload
    if not has_album:
        print("  WARNING: track_album not in payload, album features will be zero")

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    print(f"\n{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
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
    # Exp 0: Baseline (global R21 ranker, 29 features)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Exp 0: R21 baseline (global, 29 features)")
    X_base, gt_base, sz_base = build_features(
        cases, payload, als_source, als_vecs, als_factors,
        als_track_to_idx, track_pop, r21_source, FEATURE_NAMES_BASE)
    pool_hit = float(np.mean(gt_base >= 0))
    print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")
    r0 = run_cv_last_turn(X_base, gt_base, sz_base, cases, sessions, FEATURE_NAMES_BASE)
    r0["pool_hit"] = pool_hit
    configs["baseline"] = r0
    print(f"  CV5={r0['cv5']:.5f}  last_turn={r0['last_turn']:.5f}  "
          f"hist_57={r0['hist_57']:.5f}  hist_0={r0['hist_0']:.5f}")

    # ---------------------------------------------------------------
    # Exp 1: Slice-specific (train only hist_5..7)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Exp 1: Slice-specific (train on hist_5..7 only)")
    r1 = run_cv_last_turn(
        X_base, gt_base, sz_base, cases, sessions, FEATURE_NAMES_BASE,
        train_filter=lambda c: c["n_prior_music"] >= 5)
    r1["pool_hit"] = pool_hit
    configs["slice_57"] = r1
    print(f"  CV5={r1['cv5']:.5f}  last_turn={r1['last_turn']:.5f}  "
          f"hist_57={r1['hist_57']:.5f}  hist_0={r1['hist_0']:.5f}")

    # ---------------------------------------------------------------
    # Exp 2a: Weighted (hist_7 x3)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Exp 2a: Weighted (hist_7 x3)")
    r2a = run_cv_last_turn(
        X_base, gt_base, sz_base, cases, sessions, FEATURE_NAMES_BASE,
        sample_weight_fn=lambda c: 3.0 if c["n_prior_music"] == 7 else 1.0)
    r2a["pool_hit"] = pool_hit
    configs["weighted_3x"] = r2a
    print(f"  CV5={r2a['cv5']:.5f}  last_turn={r2a['last_turn']:.5f}  "
          f"hist_57={r2a['hist_57']:.5f}  hist_0={r2a['hist_0']:.5f}")

    # ---------------------------------------------------------------
    # Exp 2b: Weighted (hist_7 x5)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Exp 2b: Weighted (hist_7 x5)")
    r2b = run_cv_last_turn(
        X_base, gt_base, sz_base, cases, sessions, FEATURE_NAMES_BASE,
        sample_weight_fn=lambda c: 5.0 if c["n_prior_music"] == 7 else 1.0)
    r2b["pool_hit"] = pool_hit
    configs["weighted_5x"] = r2b
    print(f"  CV5={r2b['cv5']:.5f}  last_turn={r2b['last_turn']:.5f}  "
          f"hist_57={r2b['hist_57']:.5f}  hist_0={r2b['hist_0']:.5f}")

    # ---------------------------------------------------------------
    # Exp 2c: Weighted (hist_5..7 x3)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Exp 2c: Weighted (hist_5..7 x3)")
    r2c = run_cv_last_turn(
        X_base, gt_base, sz_base, cases, sessions, FEATURE_NAMES_BASE,
        sample_weight_fn=lambda c: 3.0 if c["n_prior_music"] >= 5 else 1.0)
    r2c["pool_hit"] = pool_hit
    configs["weighted_57_3x"] = r2c
    print(f"  CV5={r2c['cv5']:.5f}  last_turn={r2c['last_turn']:.5f}  "
          f"hist_57={r2c['hist_57']:.5f}  hist_0={r2c['hist_0']:.5f}")

    # ---------------------------------------------------------------
    # Exp 3: Deep-history features (40 features)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Exp 3: Deep-history features ({len(FEATURE_NAMES_DEEP)} features)")
    X_deep, gt_deep, sz_deep = build_features(
        cases, payload, als_source, als_vecs, als_factors,
        als_track_to_idx, track_pop, r21_source, FEATURE_NAMES_DEEP)
    r3 = run_cv_last_turn(X_deep, gt_deep, sz_deep, cases, sessions, FEATURE_NAMES_DEEP)
    r3["pool_hit"] = pool_hit
    configs["deep_features"] = r3
    print(f"  CV5={r3['cv5']:.5f}  last_turn={r3['last_turn']:.5f}  "
          f"hist_57={r3['hist_57']:.5f}  hist_0={r3['hist_0']:.5f}")

    # ---------------------------------------------------------------
    # Exp 3b: Deep features + hist_7 x3 weighting
    # ---------------------------------------------------------------
    print(f"\n{ts()} Exp 3b: Deep features + hist_7 x3")
    r3b = run_cv_last_turn(
        X_deep, gt_deep, sz_deep, cases, sessions, FEATURE_NAMES_DEEP,
        sample_weight_fn=lambda c: 3.0 if c["n_prior_music"] == 7 else 1.0)
    r3b["pool_hit"] = pool_hit
    configs["deep_weighted_3x"] = r3b
    print(f"  CV5={r3b['cv5']:.5f}  last_turn={r3b['last_turn']:.5f}  "
          f"hist_57={r3b['hist_57']:.5f}  hist_0={r3b['hist_0']:.5f}")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("R28 DEEP-HISTORY SPECIALIST — RESULTS")
    print(sep)
    print(f"  {'Config':<22} {'last_turn':>10} {'hist_57':>10} {'CV5':>10} {'hist_0':>10} "
          f"{'D_last':>10} {'D_h57':>10}")
    print(f"  {'-'*82}")
    for name, r in configs.items():
        dl = r["last_turn"] - r0["last_turn"]
        d57 = r["hist_57"] - r0["hist_57"]
        print(f"  {name:<22} {r['last_turn']:>10.5f} {r['hist_57']:>10.5f} "
              f"{r['cv5']:>10.5f} {r['hist_0']:>10.5f} {dl:>+10.5f} {d57:>+10.5f}")

    # Gate check
    print(f"\n{sep}")
    print("GATE CHECK (primary: last_turn >= +0.005)")
    best_name = None
    best_last = r0["last_turn"]
    for name, r in configs.items():
        if name == "baseline":
            continue
        dl = r["last_turn"] - r0["last_turn"]
        d57 = r["hist_57"] - r0["hist_57"]
        g_last = dl >= 0.005
        g_57 = d57 >= 0.003
        g_pool = r["pool_hit"] >= pool_hit - 0.005
        status = "PASS" if (g_last and g_57 and g_pool) else "FAIL"
        marker = " ***" if g_last else ""
        print(f"  {name:<22} last={dl:+.5f} h57={d57:+.5f} -> {status}{marker}")
        if g_last and r["last_turn"] > best_last:
            best_last = r["last_turn"]
            best_name = name

    if best_name:
        print(f"\n  BEST: {best_name} (last_turn={best_last:.5f}, delta={best_last-r0['last_turn']:+.5f})")
    else:
        print("\n  NO CONFIG PASSES last_turn gate.")
        print("  If slice-specific and weighted both fail, pivot to cross-encoder reranker.")

    out_path = REPO / "exp" / "eval" / "expR28_deep_history.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"configs": configs, "best": best_name, "pool_hit": pool_hit}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
