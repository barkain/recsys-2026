#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""Refreshed dev error decomposition AFTER R54 integration.

Uses the same CV5 LambdaRank methodology as R42 but adds R54 Phase 2 OOF as
the 8th source AND as features (rank_inv, presence, cosine). Phase 3 OOF is
gone (deleted during R55 disk cleanup); Phase 2 OOF is an OOF-correct proxy
that's architecturally identical (structured query + R21 positive text +
BGE-base, just trained on dev folds only without the 20k train-split).

Methodology constraints (user-mandated):
- NO new training
- Reproduce the current dev baseline (R39+R54) from existing artifacts
- Use Phase 2 OOF as proxy for R54 behaviour, not exact production truth
- Output bucket counts: hit, LR-demoted/in-pool, pool miss, unreachable
- Split by hist depth, same/diff artist, same album, short vs long history
- Identify largest remaining fixable bucket
- DO NOT implement R56 features yet

Buckets (per user spec):
  HIT             : LR-top@20 contains GT (= R42's A)
  DEMOTED         : GT in pool@300 but LR ranks GT outside top-20 (= R42's B+C)
  POOL_MISS       : GT in source union but RRF@300 drops it (= R42's D)
  UNREACHABLE     : GT in no source (= R42's E)

Output:
  exp/eval/expR55_post_refresh_decomp.json
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
OUT = REPO / "exp" / "eval" / "expR55_post_refresh_decomp.json"

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R54": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = [
    "same_album_last1", "same_album_last3", "same_album_any",
    "album_history_count", "pool_same_album_count",
]
FEAT_R54 = ["r54_rank_inv", "r54_presence", "r54_cosine"]
ALL_FEAT = FEAT_BASE + FEAT_ALBUM + FEAT_R54  # 28 + 5 + 3 = 36


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


def main():
    t0 = time.time()
    print(f"{ts()} Refreshed error decomposition (R39+R54 dev baseline)")
    print("=" * 70)

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

    print(f"  {n} dev cases")
    print(f"  Loading R21 OOF...")
    with open(R21_OOF) as f:
        r21_source = json.load(f)

    print(f"  Loading R54 Phase 2 OOF (proxy)...")
    with open(R54_PHASE2_OOF) as f:
        r54_data = json.load(f)
    r54_raw = r54_data["lists"]
    assert len(r54_raw) == n, f"R54 OOF length {len(r54_raw)} != {n}"
    r54_source: list[list[str]] = []
    r54_scores: list[dict[str, float]] = []
    for case_lists in r54_raw:
        tids = [t for t, _ in case_lists]
        score_map = {t: float(s) for t, s in case_lists}
        r54_source.append(tids)
        r54_scores.append(score_map)

    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    print(f"  Loading track-album map...")
    track_album = load_track_albums()

    print(f"  Building ALS...")
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

    # ---- Feature build with R54 ----
    print(f"\n{ts()} Building features ({len(ALL_FEAT)} per candidate, pool@{POOL_K})...")
    X = np.zeros((n, POOL_K, len(ALL_FEAT)), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = [[] for _ in range(n)]
    src_union_has_gt = np.zeros(n, dtype=bool)
    n_feat_base = len(FEAT_BASE)
    n_feat_r39 = n_feat_base + len(FEAT_ALBUM)

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
        # Pre-compute source union membership (for POOL_MISS vs UNREACHABLE)
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
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
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
            # Album
            c_album = track_album.get(tid, "")
            row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
            row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
            row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
            row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
            pool_album_count = sum(1 for t2 in pool[:POOL_K] if track_album.get(t2, "") == c_album) if c_album else 0
            row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)
            # R54
            row[n_feat_r39 + 0] = 1.0 / r54_rank_map[tid] if tid in r54_rank_map else 0.0
            row[n_feat_r39 + 1] = 1.0 if tid in r54_rank_map else 0.0
            row[n_feat_r39 + 2] = r54_scores[i].get(tid, 0.0)
        if (i + 1) % 1000 == 0:
            print(f"  features {i + 1}/{n} ({time.time() - t_feat:.0f}s)", flush=True)

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"\n  pool_hit@{POOL_K} (incl. R54): {pool_hit:.4f}")
    print(f"  src_union_hit:                 {float(np.mean(src_union_has_gt)):.4f}")

    # ---- CV5 LambdaRank ----
    print(f"\n{ts()} CV5 LambdaRank (R39+R54 config)...")
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
            if gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gt_idx[idx])[0]
            if len(gt_pos) > 0:
                case_lr_rank[idx] = int(gt_pos[0]) + 1

    # Sanity: h7 nDCG should be near 0.246 (R39 baseline 0.243 + Phase 2 R54 lift ~0.0034)
    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_ndcg_vals = []
    for i in h7_idx:
        if case_lr_rank[i] > 0 and case_lr_rank[i] <= 20:
            h7_ndcg_vals.append(1.0 / np.log2(case_lr_rank[i] + 1))
        else:
            h7_ndcg_vals.append(0.0)
    h7_ndcg = float(np.mean(h7_ndcg_vals)) if h7_ndcg_vals else 0.0
    print(f"\n  h7 nDCG@20 = {h7_ndcg:.5f}  (expected ~0.246; R39 baseline 0.24298 + R54 Phase 2 lift)")

    # Overall nDCG@20 across all 8000
    all_ndcg = []
    for i in range(n):
        if case_lr_rank[i] > 0 and case_lr_rank[i] <= 20:
            all_ndcg.append(1.0 / np.log2(case_lr_rank[i] + 1))
        else:
            all_ndcg.append(0.0)
    overall_ndcg = float(np.mean(all_ndcg))
    print(f"  all  nDCG@20 = {overall_ndcg:.5f}")

    # ---- Bucket assignment ----
    print(f"\n{ts()} Bucket classification (user spec: HIT / DEMOTED / POOL_MISS / UNREACHABLE)...")
    case_bucket = {}
    for i in range(n):
        if gt_idx[i] >= 0 and case_lr_rank[i] >= 1 and case_lr_rank[i] <= 20:
            case_bucket[i] = "HIT"
        elif gt_idx[i] >= 0:
            case_bucket[i] = "DEMOTED"
        elif src_union_has_gt[i]:
            case_bucket[i] = "POOL_MISS"
        else:
            case_bucket[i] = "UNREACHABLE"
    bucket_counts = Counter(case_bucket[i] for i in range(n))
    print(f"  Overall bucket distribution (all {n} cases):")
    for b in ["HIT", "DEMOTED", "POOL_MISS", "UNREACHABLE"]:
        c = bucket_counts[b]
        print(f"    {b:<13s}: {c:>5d} ({c / n * 100:.1f}%)")

    # ---- Splits ----
    def split_counter(idxs):
        c = Counter()
        for i in idxs:
            c[case_bucket[i]] += 1
        return c

    # by hist depth
    by_depth = defaultdict(list)
    for i in range(n):
        d = cases[i]["n_prior_music"]
        d_label = f"h{d}" if d < 7 else "h7+"
        by_depth[d_label].append(i)

    # short vs long history
    short_hist = [i for i in range(n) if cases[i]["n_prior_music"] < 3]
    long_hist = [i for i in range(n) if cases[i]["n_prior_music"] >= 3]

    # same/diff artist
    same_artist = []
    diff_artist = []
    for i in range(n):
        gt_artist = ta.get(cases[i]["gt"], "")
        played_artists = {ta.get(t, "") for t in cases[i]["music_turns"]} - {""}
        if gt_artist and gt_artist in played_artists:
            same_artist.append(i)
        else:
            diff_artist.append(i)

    # same album
    same_album = []
    diff_album = []
    for i in range(n):
        gt_album = track_album.get(cases[i]["gt"], "")
        played_albums = {track_album.get(t, "") for t in cases[i]["music_turns"]} - {""}
        if gt_album and gt_album in played_albums:
            same_album.append(i)
        else:
            diff_album.append(i)

    def print_split(label, groups: dict):
        print(f"\n  Split: {label}")
        header = f"    {'group':>10s} | {'n':>4s} | {'HIT':>5s} {'DEMOTED':>7s} {'POOL_MISS':>9s} {'UNREACH':>7s}  ndcg@20"
        print(header)
        print(f"    {'-' * (len(header) - 4)}")
        for k, idxs in groups.items():
            if not idxs:
                continue
            c = split_counter(idxs)
            ndcg_vals = []
            for i in idxs:
                if case_lr_rank[i] > 0 and case_lr_rank[i] <= 20:
                    ndcg_vals.append(1.0 / np.log2(case_lr_rank[i] + 1))
                else:
                    ndcg_vals.append(0.0)
            ndcg_avg = float(np.mean(ndcg_vals)) if ndcg_vals else 0.0
            tot = len(idxs)
            print(f"    {k:>10s} | {tot:>4d} | "
                  f"{c['HIT']:>5d} {c['DEMOTED']:>7d} {c['POOL_MISS']:>9d} {c['UNREACHABLE']:>7d}"
                  f"  {ndcg_avg:.4f}")

    print_split("hist depth", dict(sorted(by_depth.items(), key=lambda kv: kv[0])))
    print_split("history length (cut=3)", {"hist<3": short_hist, "hist>=3": long_hist})
    print_split("artist relationship", {"same_artist": same_artist, "diff_artist": diff_artist})
    print_split("album relationship", {"same_album": same_album, "diff_album": diff_album})

    # ---- Fixable bucket sizing ----
    # Of the DEMOTED bucket, how many are "easy fix" (R21 or BM25 had GT in top-20 but LR dropped it)?
    demoted = [i for i in range(n) if case_bucket[i] == "DEMOTED"]
    bm25b_top20_demoted = 0
    bm25c_top20_demoted = 0
    r21_top20_demoted = 0
    r54_top20_demoted = 0
    for i in demoted:
        gt = cases[i]["gt"]
        if gt in r21_source[i][:20]:
            r21_top20_demoted += 1
        if gt in r54_source[i][:20]:
            r54_top20_demoted += 1
        if gt in payload["src_b"][i][:20]:
            bm25b_top20_demoted += 1
        if gt in payload["src_c"][i][:20]:
            bm25c_top20_demoted += 1

    # POOL_MISS: which source had GT? Helps decide if pool expansion or RRF tuning is the fix.
    pool_miss = [i for i in range(n) if case_bucket[i] == "POOL_MISS"]
    pm_src_had_gt = Counter()
    for i in pool_miss:
        gt = cases[i]["gt"]
        for sname, key in [("A", "src_a"), ("B", "src_b"), ("C", "src_c"),
                            ("D", "src_d"), ("F", "src_f")]:
            if gt in payload[key][i]:
                pm_src_had_gt[sname] += 1
        if gt in als_source[i]:
            pm_src_had_gt["ALS"] += 1
        if gt in r21_source[i]:
            pm_src_had_gt["R21"] += 1
        if gt in r54_source[i]:
            pm_src_had_gt["R54"] += 1

    print(f"\n{ts()} Sub-analysis: DEMOTED bucket ({len(demoted)} cases)")
    print(f"  GT was top-20 in R21:     {r21_top20_demoted:>4d}")
    print(f"  GT was top-20 in R54:     {r54_top20_demoted:>4d}")
    print(f"  GT was top-20 in src_b:   {bm25b_top20_demoted:>4d}")
    print(f"  GT was top-20 in src_c:   {bm25c_top20_demoted:>4d}")
    print(f"\n{ts()} Sub-analysis: POOL_MISS bucket ({len(pool_miss)} cases)")
    print(f"  Which source had GT (cases may overlap):")
    for sname, ct in pm_src_had_gt.most_common():
        print(f"    {sname:>5s}: {ct}")

    # Priority ranking
    print(f"\n{'=' * 70}")
    print("LARGEST FIXABLE BUCKET — Priority")
    print("=" * 70)
    targets = [
        ("DEMOTED (LR-fixable)", bucket_counts["DEMOTED"],
         "GT is in pool@300 but LR ranks it >20. Can be addressed by ranker "
         "calibration (R56): better features, candidate protection, score "
         "rescaling. Bounded risk because retriever stays."),
        ("POOL_MISS (RRF / pool tuning)", bucket_counts["POOL_MISS"],
         "GT in source union but RRF@300 drops it. Can be addressed by RRF "
         "weight tuning, source weight per-bucket, or pool expansion. Cheap "
         "but bounded ceiling."),
        ("UNREACHABLE (new retriever)", bucket_counts["UNREACHABLE"],
         "GT in no source. Needs a fundamentally different retriever or "
         "structural feature. Highest risk, highest ceiling."),
    ]
    targets.sort(key=lambda x: -x[1])
    for ri, (name, cnt, note) in enumerate(targets, 1):
        share = cnt / n * 100
        print(f"  {ri}. {name:<35s} {cnt:>5d} ({share:.1f}% of all cases)")
        print(f"     {note}")

    # ---- Save ----
    out_data = {
        "n_cases": n,
        "h7_ndcg": h7_ndcg,
        "overall_ndcg": overall_ndcg,
        "pool_hit": pool_hit,
        "src_union_hit": float(np.mean(src_union_has_gt)),
        "bucket_counts": dict(bucket_counts),
        "bucket_share": {b: bucket_counts[b] / n for b in bucket_counts},
        "by_hist_depth": {
            k: dict(split_counter(idxs)) for k, idxs in by_depth.items()
        },
        "by_hist_length": {
            "hist<3": dict(split_counter(short_hist)),
            "hist>=3": dict(split_counter(long_hist)),
        },
        "by_artist": {
            "same_artist": dict(split_counter(same_artist)),
            "diff_artist": dict(split_counter(diff_artist)),
        },
        "by_album": {
            "same_album": dict(split_counter(same_album)),
            "diff_album": dict(split_counter(diff_album)),
        },
        "demoted_recoverability": {
            "r21_top20": r21_top20_demoted,
            "r54_top20": r54_top20_demoted,
            "bm25b_top20": bm25b_top20_demoted,
            "bm25c_top20": bm25c_top20_demoted,
        },
        "pool_miss_source_coverage": dict(pm_src_had_gt),
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
        "notes": (
            "Used R54 Phase 2 OOF as proxy for R54 production (Phase 3 OOF "
            "was deleted in pre-R55 disk cleanup). Phase 2 dev lift is ~0.001 "
            "lower than Phase 3, so bucket boundaries may be marginally noisier "
            "than the exact production state. Pattern-level conclusions are "
            "expected to transfer."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n{ts()} Saved: {OUT}")
    print(f"Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
