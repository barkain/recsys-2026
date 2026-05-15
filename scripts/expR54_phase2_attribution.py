#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54 Phase 2 integration attribution diagnostic.

For h7 cases newly admitted by R54 into the R39+R54 pool but not in R39 baseline pool:
- count
- their final LR rank distribution: 1-20, 21-50, 51-100, 101-300
- R54 retrieval rank distribution for these GTs
- R54 cosine for recovered vs not recovered
- nDCG gain weighted by recovered ranks vs lost ranks

Runs CV5 twice: baseline and best R54 config (w=1.5 +feats).
Captures per-case final LR rank of GT for both.
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
R54_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"

RRF_K = 20
POOL_K = 300
SW_BASE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = ["same_album_last1", "same_album_last3", "same_album_any",
              "album_history_count", "pool_same_album_count"]
FEAT_R54 = ["r54_rank_inv", "r54_presence", "r54_cosine"]
FEAT_R39_ALL = FEAT_BASE + FEAT_ALBUM
FEAT_ALL = FEAT_R39_ALL + FEAT_R54


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


def build_features(pool, case, src_lists, r21_rank_map, r54_rank_map, r54_score_map,
                   als_factors, als_track_to_idx, als_vec, track_pop, max_pop,
                   ta, tt, ttl, tat, tmt, track_album, feat_names):
    n_pool = len(pool)
    n_feat = len(feat_names)
    X = np.zeros((n_pool, n_feat), dtype=np.float64)
    use_r54 = "r54_rank_inv" in feat_names
    n_feat_base = len(FEAT_BASE)

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
        row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
        row[28] = 1.0 if tid in r21_rank_map else 0.0

        c_album = track_album.get(tid, "")
        row[n_feat_base + 0] = 1.0 if c_album and c_album == last1_album else 0.0
        row[n_feat_base + 1] = 1.0 if c_album and c_album in last3_albums else 0.0
        row[n_feat_base + 2] = 1.0 if c_album and c_album in album_hist_counts else 0.0
        row[n_feat_base + 3] = float(album_hist_counts.get(c_album, 0)) / max(n_hist, 1) if c_album else 0.0
        pool_album_count = sum(1 for tid2 in pool[:POOL_K] if track_album.get(tid2, "") == c_album) if c_album else 0
        row[n_feat_base + 4] = pool_album_count / max(len(pool), 1)

        if use_r54:
            r54_off = len(FEAT_R39_ALL)
            row[r54_off + 0] = 1.0 / r54_rank_map[tid] if tid in r54_rank_map else 0.0
            row[r54_off + 1] = 1.0 if tid in r54_rank_map else 0.0
            row[r54_off + 2] = r54_score_map.get(tid, 0.0)

    return X


def run_cv5_capture_gt_rank(cases, payload, r21_source, r54_source, r54_scores,
                              als_source, als_vecs, als_factors, als_track_to_idx,
                              track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album,
                              weights, feat_names, label):
    """Run CV5 LambdaRank, return per-case GT final LR rank (or -1 if not in pool)."""
    n = len(cases)
    n_feat = len(feat_names)
    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools = {}

    print(f"  Building features for {label}...", flush=True)
    t_feat = time.time()
    for i in range(n):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        if "R54" in weights:
            src_lists["R54"] = r54_source[i]
        pool = weighted_rrf(src_lists, weights, topk=POOL_K, k=RRF_K)
        sizes[i] = len(pool)
        pools[i] = pool
        if cases[i]["gt"] in pool:
            gt_idx[i] = pool.index(cases[i]["gt"])
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(r54_source[i][:300])}
        Xi = build_features(pool, cases[i], src_lists, r21_rank_map,
                             r54_rank_map, r54_scores[i],
                             als_factors, als_track_to_idx, als_vecs[i],
                             track_pop, max_pop, ta, tt, ttl, tat, tmt,
                             track_album, feat_names)
        X[i, :len(pool)] = Xi
    print(f"  Features built in {time.time() - t_feat:.0f}s")

    sessions = [c["session_id"] for c in cases]
    folds = grouped_session_folds(sessions, seed=0)

    gt_final_rank = np.full(n, -1, dtype=np.int64)
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
            if len(gt_pos) > 0:
                gt_final_rank[idx] = int(gt_pos[0])

    return gt_final_rank, gt_idx, sizes, pools


def main():
    t0 = time.time()
    print(f"{ts()} R54 Phase 2 attribution diagnostic")
    print("=" * 70)

    print(f"{ts()} Loading payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    r21_source = json.load(open(R21_OOF))
    r54_data = json.load(open(R54_OOF))
    r54_source = []
    r54_scores = []
    for case_lists in r54_data["lists"]:
        tids = [t for t, _ in case_lists]
        score_map = {t: float(s) for t, s in case_lists}
        r54_source.append(tids)
        r54_scores.append(score_map)

    print(f"{ts()} Loading album mapping...")
    track_album = load_track_albums()

    print(f"{ts()} Loading popularity & ALS...")
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1
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

    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    print(f"  h7 cases: {len(h7_idx)}")

    # Run both configs
    print(f"\n{ts()} Running BASELINE R39 (34 feats)...")
    base_rank, base_gt_idx, base_sizes, base_pools = run_cv5_capture_gt_rank(
        cases, payload, r21_source, r54_source, r54_scores,
        als_source, als_vecs, als_factors, als_track_to_idx,
        track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album,
        SW_BASE, FEAT_R39_ALL, "baseline")

    print(f"\n{ts()} Running R39+R54_w1.5_feats (37 feats)...")
    r54_weights = {**SW_BASE, "R54": 1.5}
    r54_rank, r54_gt_idx, r54_sizes, r54_pools = run_cv5_capture_gt_rank(
        cases, payload, r21_source, r54_source, r54_scores,
        als_source, als_vecs, als_factors, als_track_to_idx,
        track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album,
        r54_weights, FEAT_ALL, "R39+R54")

    # ============================================================
    # Attribution analysis
    # ============================================================
    print(f"\n{ts()} === ATTRIBUTION (on h7 cases) ===")

    # 1. New admissions to pool
    newly_admitted = []
    already_in_baseline = []
    not_in_either = []
    for i in h7_idx:
        in_base = base_gt_idx[i] >= 0
        in_r54 = r54_gt_idx[i] >= 0
        if in_r54 and not in_base:
            newly_admitted.append(i)
        elif in_base and in_r54:
            already_in_baseline.append(i)
        elif not in_base and not in_r54:
            not_in_either.append(i)

    print(f"  GTs newly admitted by R54: {len(newly_admitted)}")
    print(f"  GTs in both pools: {len(already_in_baseline)}")
    print(f"  GTs in neither: {len(not_in_either)}")

    # 2. Where do newly admitted end up?
    print(f"\n  Final LR rank distribution for newly-admitted h7 GTs:")
    bins = {"1-20": 0, "21-50": 0, "51-100": 0, "101-300": 0, "300+": 0}
    for i in newly_admitted:
        r = r54_rank[i]
        if r < 0:
            bins["300+"] += 1
        elif r < 20:
            bins["1-20"] += 1
        elif r < 50:
            bins["21-50"] += 1
        elif r < 100:
            bins["51-100"] += 1
        else:
            bins["101-300"] += 1
    for b, c in bins.items():
        pct = c / max(len(newly_admitted), 1) * 100
        print(f"    {b:<10}: {c:>4} ({pct:.1f}%)")

    # 3. R54 retrieval rank distribution for these GTs
    print(f"\n  R54 retrieval rank for newly-admitted h7 GTs:")
    rank_bins = {"1-20": 0, "21-50": 0, "51-100": 0, "101-300": 0}
    for i in newly_admitted:
        gt = cases[i]["gt"]
        r54_list = r54_source[i]
        if gt in r54_list:
            rank = r54_list.index(gt)
            if rank < 20:
                rank_bins["1-20"] += 1
            elif rank < 50:
                rank_bins["21-50"] += 1
            elif rank < 100:
                rank_bins["51-100"] += 1
            else:
                rank_bins["101-300"] += 1
    for b, c in rank_bins.items():
        pct = c / max(len(newly_admitted), 1) * 100
        print(f"    {b:<10}: {c:>4} ({pct:.1f}%)")

    # 4. R54 cosine for recovered (in top-20 LR) vs not
    print(f"\n  R54 cosine: recovered (top-20 LR) vs not (newly-admitted h7):")
    recovered_cosines = []
    not_recovered_cosines = []
    for i in newly_admitted:
        gt = cases[i]["gt"]
        cos = r54_scores[i].get(gt, 0.0)
        final_r = r54_rank[i]
        if 0 <= final_r < 20:
            recovered_cosines.append(cos)
        else:
            not_recovered_cosines.append(cos)
    if recovered_cosines:
        print(f"    recovered (n={len(recovered_cosines)}): mean={np.mean(recovered_cosines):.4f}  "
              f"median={np.median(recovered_cosines):.4f}  min={min(recovered_cosines):.4f}  max={max(recovered_cosines):.4f}")
    if not_recovered_cosines:
        print(f"    not_recovered (n={len(not_recovered_cosines)}): mean={np.mean(not_recovered_cosines):.4f}  "
              f"median={np.median(not_recovered_cosines):.4f}  min={min(not_recovered_cosines):.4f}  max={max(not_recovered_cosines):.4f}")

    # 5. nDCG gain/loss weighted by ranks (h7)
    print(f"\n  nDCG impact (h7):")
    ndcg_baseline = 0.0
    ndcg_r54 = 0.0
    for i in h7_idx:
        if 0 <= base_rank[i] < 20:
            ndcg_baseline += 1.0 / np.log2(base_rank[i] + 2)
        if 0 <= r54_rank[i] < 20:
            ndcg_r54 += 1.0 / np.log2(r54_rank[i] + 2)
    ndcg_baseline /= len(h7_idx)
    ndcg_r54 /= len(h7_idx)
    print(f"    baseline h7 nDCG: {ndcg_baseline:.5f}")
    print(f"    R54      h7 nDCG: {ndcg_r54:.5f}")
    print(f"    Δh7 nDCG: {ndcg_r54 - ndcg_baseline:+.5f}")

    # 6. Per-h7-case: rank changes
    moved_up = moved_down = stayed = newly_in_top20 = lost_from_top20 = 0
    for i in h7_idx:
        b = base_rank[i]
        r = r54_rank[i]
        b_in = 0 <= b < 20
        r_in = 0 <= r < 20
        if r_in and not b_in:
            newly_in_top20 += 1
        elif b_in and not r_in:
            lost_from_top20 += 1
        elif b_in and r_in:
            if r < b:
                moved_up += 1
            elif r > b:
                moved_down += 1
            else:
                stayed += 1

    print(f"\n  h7 top-20 movement:")
    print(f"    newly in top-20: {newly_in_top20}")
    print(f"    lost from top-20: {lost_from_top20}")
    print(f"    moved up in top-20: {moved_up}")
    print(f"    moved down in top-20: {moved_down}")
    print(f"    stayed: {stayed}")
    print(f"    net top-20: {newly_in_top20 - lost_from_top20:+d}")

    # 7. Newly admitted h7 cases: where are they NOT in top-20?
    not_in_top20_count = sum(1 for i in newly_admitted if not (0 <= r54_rank[i] < 20))
    print(f"\n  Of {len(newly_admitted)} newly-admitted h7 GTs:")
    print(f"    In LR top-20: {len(newly_admitted) - not_in_top20_count}")
    print(f"    NOT in LR top-20: {not_in_top20_count}  ← these are 'admitted but buried'")

    # Save
    out = {
        "newly_admitted": len(newly_admitted),
        "in_both_pools": len(already_in_baseline),
        "in_neither": len(not_in_either),
        "final_lr_rank_distribution": bins,
        "r54_retrieval_rank_distribution": rank_bins,
        "recovered_cosines_mean": float(np.mean(recovered_cosines)) if recovered_cosines else None,
        "recovered_cosines_n": len(recovered_cosines),
        "not_recovered_cosines_mean": float(np.mean(not_recovered_cosines)) if not_recovered_cosines else None,
        "not_recovered_cosines_n": len(not_recovered_cosines),
        "h7_ndcg_baseline": float(ndcg_baseline),
        "h7_ndcg_r54": float(ndcg_r54),
        "h7_top20_movement": {
            "newly_in_top20": newly_in_top20,
            "lost_from_top20": lost_from_top20,
            "moved_up": moved_up,
            "moved_down": moved_down,
            "stayed": stayed,
            "net": newly_in_top20 - lost_from_top20,
        },
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    out_path = REPO / "exp" / "eval" / "expR54_phase2_attribution.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Attribution complete. Elapsed: {time.time() - t0:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
