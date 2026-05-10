#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R41a: Rare-tag overlap features on R39 baseline.

R40 forensics found rare_tag_overlap in 53/176 miss cases (30%).
Add IDF-weighted rare-tag features for last3 and all-history.
"""
from __future__ import annotations

import json
import math
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
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1",
    "same_album_last3",
    "same_album_any",
    "album_history_count",
    "pool_same_album_count",
]
FEAT_RARE_TAG = [
    "rare_tag_count_last3",
    "rare_tag_idf_last3",
    "rare_tag_max_idf_last3",
    "rare_tag_count_all",
    "rare_tag_idf_all",
    "rare_tag_max_idf_all",
    "rare_tag_frac_cand",
]

RARE_THRESHOLD = 500


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


def load_catalog_tags():
    """Load track_id -> tag_set and compute catalog-wide tag document frequencies."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_tags: dict[str, set[str]] = {}
    tag_df: Counter[str] = Counter()
    for item in ds:
        tid = str(item["track_id"])
        raw = item.get("tag_list", [])
        if isinstance(raw, list):
            tags = {str(t).lower().strip() for t in raw if str(t).strip()}
        else:
            tags = set()
        track_tags[tid] = tags
        for tag in tags:
            tag_df[tag] += 1
    n_docs = len(track_tags)
    tag_idf = {tag: math.log(n_docs / (df + 1)) for tag, df in tag_df.items()}
    rare_tags = {tag for tag, df in tag_df.items() if df < RARE_THRESHOLD}
    print(f"  Catalog: {n_docs} tracks, {len(tag_df)} unique tags, {len(rare_tags)} rare (df<{RARE_THRESHOLD})")
    return track_tags, tag_idf, rare_tags


def main():
    t0 = time.time()
    print(f"{ts()} R41a: Rare-Tag Features on R39 Baseline")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
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

    print(f"{ts()} Loading album mapping...")
    track_album = load_track_albums()

    print(f"{ts()} Loading catalog tags + IDF...")
    catalog_tags, tag_idf, rare_tags = load_catalog_tags()

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

    n = len(cases)
    folds = grouped_session_folds(sessions, seed=0)
    n_feat_base = len(FEAT_BASE)

    configs = [
        ("R39_baseline", FEAT_BASE + FEAT_ALBUM, False),
        ("R39+rare_tags", FEAT_BASE + FEAT_ALBUM + FEAT_RARE_TAG, True),
    ]

    results = {}
    config_ndcg = {}

    for config_name, feat_names, use_rare in configs:
        n_feat = len(feat_names)
        n_album_offset = n_feat_base
        n_rare_offset = n_feat_base + len(FEAT_ALBUM)
        print(f"\n{ts()} Config: {config_name} ({n_feat} features)")

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
            pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
            artist_counts = Counter(a for a in pool_artists if a)
            r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}

            # Album precomputation (always needed — part of R39 baseline)
            last1_album = track_album.get(played[-1], "") if played else ""
            last3_albums = {track_album.get(t, "") for t in played[-3:]} - {""}
            all_albums = [track_album.get(t, "") for t in played]
            album_hist_counts = Counter(a for a in all_albums if a)

            # Rare-tag precomputation
            hist_rare_last3: set[str] = set()
            hist_rare_all: set[str] = set()
            if use_rare:
                for t in played[-3:]:
                    hist_rare_last3 |= catalog_tags.get(t, set()) & rare_tags
                for t in played:
                    hist_rare_all |= catalog_tags.get(t, set()) & rare_tags

            for rank, tid in enumerate(pool[:POOL_K], start=1):
                ca = ta.get(tid, "")
                ct = tt.get(tid, set())
                row = X[i, rank - 1]

                # Base 29 features (exact R39a)
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

                # Album features (R39 baseline)
                c_album = track_album.get(tid, "")
                row[n_album_offset + 0] = 1.0 if c_album and c_album == last1_album else 0.0
                row[n_album_offset + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
                row[n_album_offset + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
                row[n_album_offset + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
                pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
                row[n_album_offset + 4] = pool_album_count / max(len(pool), 1)

                # Rare-tag features
                if use_rare:
                    cand_tags = catalog_tags.get(tid, set())
                    cand_rare = cand_tags & rare_tags

                    overlap_last3 = cand_rare & hist_rare_last3
                    row[n_rare_offset + 0] = float(len(overlap_last3))
                    row[n_rare_offset + 1] = sum(tag_idf.get(t, 0) for t in overlap_last3)
                    row[n_rare_offset + 2] = max((tag_idf.get(t, 0) for t in overlap_last3), default=0.0)

                    overlap_all = cand_rare & hist_rare_all
                    row[n_rare_offset + 3] = float(len(overlap_all))
                    row[n_rare_offset + 4] = sum(tag_idf.get(t, 0) for t in overlap_all)
                    row[n_rare_offset + 5] = max((tag_idf.get(t, 0) for t in overlap_all), default=0.0)

                    row[n_rare_offset + 6] = len(overlap_all) / max(len(cand_rare), 1)

        pool_hit = float(np.mean(gt_idx >= 0))
        print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

        # CV5 LambdaRank
        case_ndcg = np.zeros(n)
        mdl = None
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
            mdl = lgb.train(params, ds_tr, num_boost_round=300,
                            valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])
            preds = mdl.predict(np.array(X_va))
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

        config_ndcg[config_name] = case_ndcg.copy()

        # Feature importance (last fold model)
        if use_rare and mdl is not None:
            importance = mdl.feature_importance(importance_type="gain")
            feat_imp = sorted(zip(feat_names, importance), key=lambda x: -x[1])
            print("  Top features by gain (last fold):")
            for fname, imp in feat_imp[:20]:
                print(f"    {fname}: {imp:.1f}")

        h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
        h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
                   ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
        h7_diff = [i for i in h7 if i not in set(h7_same)]

        h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))
        cv5_ndcg = float(np.mean(case_ndcg))
        same_ndcg = float(np.mean([case_ndcg[i] for i in h7_same])) if h7_same else 0
        diff_ndcg = float(np.mean([case_ndcg[i] for i in h7_diff])) if h7_diff else 0

        print(f"  h7={h7_ndcg:.5f}  cv5={cv5_ndcg:.5f}  same={same_ndcg:.5f}  diff={diff_ndcg:.5f}")

        results[config_name] = {
            "pool_hit": pool_hit, "h7": h7_ndcg, "cv5": cv5_ndcg,
            "same": same_ndcg, "diff": diff_ndcg,
        }

    # ---------------------------------------------------------------
    # Gate + recovered/lost analysis
    # ---------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("R41a RARE-TAG FEATURES")
    print(sep)
    base = results["R39_baseline"]
    test = results["R39+rare_tags"]
    dh7 = test["h7"] - base["h7"]
    dsame = test["same"] - base["same"]
    ddiff = test["diff"] - base["diff"]
    print(f"  baseline:     h7={base['h7']:.5f}  same={base['same']:.5f}  diff={base['diff']:.5f}")
    print(f"  +rare_tags:   h7={test['h7']:.5f}  same={test['same']:.5f}  diff={test['diff']:.5f}")
    print(f"  Δh7={dh7:+.5f}  Δsame={dsame:+.5f}  Δdiff={ddiff:+.5f}")

    # Recovered/lost on h7
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    base_ndcg = config_ndcg["R39_baseline"]
    test_ndcg = config_ndcg["R39+rare_tags"]
    recovered = sum(1 for i in h7 if test_ndcg[i] > base_ndcg[i] + 1e-6)
    lost = sum(1 for i in h7 if test_ndcg[i] < base_ndcg[i] - 1e-6)
    unchanged = len(h7) - recovered - lost
    print(f"\n  Recovered: {recovered}  Lost: {lost}  Unchanged: {unchanged}")
    print(f"  Net: {recovered - lost:+d}")

    # Check forensic bucket recovery
    r40_path = REPO / "exp" / "eval" / "expR40_forensics.json"
    if r40_path.exists():
        with open(r40_path) as f:
            r40 = json.load(f)
        forensic_rare_cases = set()
        for ex in r40.get("examples", []):
            if "rare_tag_overlap" in ex.get("patterns", ex.get("reasons", [])):
                forensic_rare_cases.add(ex.get("case_idx", -1))
        if forensic_rare_cases:
            forensic_recovered = sum(1 for i in forensic_rare_cases
                                     if test_ndcg[i] > base_ndcg[i] + 1e-6)
            forensic_lost = sum(1 for i in forensic_rare_cases
                                if test_ndcg[i] < base_ndcg[i] - 1e-6)
            print(f"\n  Forensic rare_tag_overlap bucket ({len(forensic_rare_cases)} cases):")
            print(f"    Recovered: {forensic_recovered}  Lost: {forensic_lost}")

    # Gate
    g_h7 = dh7 >= 0.005
    g_rl = recovered > lost
    print(f"\n  GATE h7 >= +0.005:  {dh7:+.5f} {'PASS' if g_h7 else 'FAIL'}")
    print(f"  GATE recovered>lost: {recovered}>{lost} {'PASS' if g_rl else 'FAIL'}")
    overall = g_h7 and g_rl
    print(f"  OVERALL: {'PASS — proceed to blind' if overall else 'FAIL — move to R41b'}")

    out_path = REPO / "exp" / "eval" / "expR41a_rare_tag.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
