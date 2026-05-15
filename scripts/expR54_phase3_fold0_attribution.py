#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54 Phase 3 fold-0 attribution comparison.

Runs fold-0 R39 integration with three pool configs:
  baseline   : R39 7 sources
  +P2_w1.5   : R39 + Phase 2 R54 (structured query, dev only) at weight 1.5
  +P3_w1.5   : R39 + Phase 3 R54 (structured query, dev+train-split) at weight 1.5

LambdaRank trained ONCE on folds 1-4 with R39 baseline pool (no R54), 34 features.
Applied to fold-0 under each config with pool-only integration.

Reports admission/conversion attribution side-by-side:
  - h7 GTs newly admitted by R54 to pool
  - admitted -> top-20 conversion
  - admitted-but-buried count
  - R54 rank/cosine recovered vs buried
  - fold-0 h7 nDCG delta

Success conditions for Phase 3:
  1. admissions increase >= +30% vs Phase 2 fold-0
  2. admitted->top-20 conversion improves materially
  3. fold-0 h7 integration Δ >= +0.010 vs baseline
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
P2_FOLD0 = REPO / "cache" / "r54" / "phase2_full" / "fold_0" / "oof_lists.json"
P3_FOLD0 = REPO / "cache" / "r54" / "phase3_smoke" / "fold_0" / "oof_lists.json"

RRF_K = 20
POOL_K = 300
SW_BASE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_ALBUM = ["same_album_last1", "same_album_last3", "same_album_any",
              "album_history_count", "pool_same_album_count"]
FEAT_ALL = FEAT_BASE + FEAT_ALBUM  # 34 features, no R54


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


def load_r54_lists(path):
    """Load (lists, scores_per_case) from an R54 OOF lists file."""
    data = json.load(open(path))
    lists = []
    scores = []
    for case_lists in data["lists"]:
        assert case_lists is not None and len(case_lists) > 0
        tids = [t for t, _ in case_lists]
        score_map = {t: float(s) for t, s in case_lists}
        lists.append(tids)
        scores.append(score_map)
    return lists, scores


def build_features(pool, case, src_lists, r21_rank_map,
                   als_factors, als_track_to_idx, als_vec, track_pop, max_pop,
                   ta, tt, ttl, tat, tmt, track_album):
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
    n_feat_base = len(FEAT_BASE)

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

    return X


def eval_config(name, fold0_idx, cases, payload, r21_source, r54_lists, r54_scores,
                als_source, als_vecs, als_factors, als_track_to_idx,
                track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album, model, weights):
    """For a pool config, return per-case GT final rank and admissions metrics."""
    n = len(fold0_idx)
    gt_final_rank = {}
    pools = {}
    gt_idx = {}

    for vi, idx in enumerate(fold0_idx):
        src_lists = {
            "A": payload["src_a"][idx], "B": payload["src_b"][idx],
            "C": payload["src_c"][idx], "D": payload["src_d"][idx],
            "F": payload["src_f"][idx], "ALS": als_source[idx],
            "R21": r21_source[idx],
        }
        if "R54" in weights and r54_lists is not None:
            src_lists["R54"] = r54_lists[idx]
        pool = weighted_rrf(src_lists, weights, topk=POOL_K, k=RRF_K)
        pools[idx] = pool
        gt = cases[idx]["gt"]
        gt_idx[idx] = pool.index(gt) if gt in pool else -1

        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[idx][:300])}
        X = build_features(pool, cases[idx], src_lists, r21_rank_map,
                            als_factors, als_track_to_idx, als_vecs[idx],
                            track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album)
        if len(pool) > 0:
            sc = model.predict(X[:len(pool)])
            ranked = np.argsort(-sc)
            if gt_idx[idx] >= 0:
                gt_pos = np.where(ranked == gt_idx[idx])[0]
                gt_final_rank[idx] = int(gt_pos[0]) if len(gt_pos) > 0 else -1
            else:
                gt_final_rank[idx] = -1
        else:
            gt_final_rank[idx] = -1

    return gt_final_rank, gt_idx, pools


def attribution_table(label, h7_fold0, gt_baseline_pool, gt_r54_pool,
                       gt_baseline_rank, gt_r54_rank, r54_lists, r54_scores, cases):
    """Compute attribution metrics."""
    # Admissions: h7 GTs in R54-augmented pool but NOT in baseline pool
    newly_admitted = [i for i in h7_fold0 if gt_r54_pool[i] >= 0 and gt_baseline_pool[i] < 0]
    print(f"\n  === {label} attribution (fold-0 h7) ===")
    print(f"  Newly admitted h7 GTs: {len(newly_admitted)}")

    # Conversion: of admitted, how many end up in LR top-20?
    converted = [i for i in newly_admitted if 0 <= gt_r54_rank[i] < 20]
    buried = [i for i in newly_admitted if i not in set(converted)]
    print(f"  Admitted -> top-20: {len(converted)} ({len(converted) / max(len(newly_admitted), 1):.1%})")
    print(f"  Admitted but buried: {len(buried)}")

    # R54 retrieval rank distribution for newly-admitted
    r54_rank_bins = {"1-20": 0, "21-50": 0, "51-100": 0, "101-300": 0}
    for i in newly_admitted:
        gt = cases[i]["gt"]
        if gt in r54_lists[i]:
            rank = r54_lists[i].index(gt)
            if rank < 20:
                r54_rank_bins["1-20"] += 1
            elif rank < 50:
                r54_rank_bins["21-50"] += 1
            elif rank < 100:
                r54_rank_bins["51-100"] += 1
            else:
                r54_rank_bins["101-300"] += 1
    print(f"  R54 retrieval rank for admitted: {dict(r54_rank_bins)}")

    # Cosine: recovered vs buried
    recovered_cos = [r54_scores[i].get(cases[i]["gt"], 0.0) for i in converted]
    buried_cos = [r54_scores[i].get(cases[i]["gt"], 0.0) for i in buried]
    rec_mean = float(np.mean(recovered_cos)) if recovered_cos else None
    bur_mean = float(np.mean(buried_cos)) if buried_cos else None
    print(f"  R54 cosine — recovered (n={len(recovered_cos)}): "
          f"mean={rec_mean:.4f}" if rec_mean else f"  R54 cosine recovered: none")
    print(f"  R54 cosine — buried (n={len(buried_cos)}): "
          f"mean={bur_mean:.4f}" if bur_mean else f"  R54 cosine buried: none")

    return {
        "label": label,
        "newly_admitted": len(newly_admitted),
        "converted_top20": len(converted),
        "buried": len(buried),
        "conversion_rate": len(converted) / max(len(newly_admitted), 1),
        "r54_rank_bins": r54_rank_bins,
        "recovered_cosine_mean": rec_mean, "recovered_n": len(recovered_cos),
        "buried_cosine_mean": bur_mean, "buried_n": len(buried_cos),
    }


def main():
    t0 = time.time()
    print(f"{ts()} R54 Phase 3 fold-0 attribution comparison (P2 vs P3)")
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

    r21_source = json.load(open(R21_OOF))

    if not P2_FOLD0.exists():
        print(f"  ERROR: P2 fold-0 lists missing at {P2_FOLD0}")
        sys.exit(1)
    if not P3_FOLD0.exists():
        print(f"  ERROR: P3 fold-0 lists missing at {P3_FOLD0}")
        sys.exit(1)

    print(f"{ts()} Loading Phase 2 fold-0 lists...")
    p2_data = json.load(open(P2_FOLD0))
    p2_val_idx = p2_data.get("val_idx", None)
    p2_lists_raw = p2_data["lists"]

    print(f"{ts()} Loading Phase 3 fold-0 lists...")
    p3_data = json.load(open(P3_FOLD0))
    p3_lists_raw = p3_data["lists"]

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

    folds = grouped_session_folds(sessions, seed=0)
    fold0_idx = sorted(folds[0].tolist())
    train_idx = sorted(set(range(n)) - set(fold0_idx))
    print(f"  Fold-0: {len(fold0_idx)} val, {len(train_idx)} train")

    # Align P2/P3 lists to global case index. They were saved indexed by val position.
    p2_lists_by_case = [None] * n
    p2_scores_by_case = [None] * n
    p3_lists_by_case = [None] * n
    p3_scores_by_case = [None] * n
    for k, ci in enumerate(fold0_idx):
        # Each list element is [(tid, score), ...]
        p2 = p2_lists_raw[k]
        p3 = p3_lists_raw[k]
        p2_lists_by_case[ci] = [t for t, _ in p2]
        p2_scores_by_case[ci] = {t: float(s) for t, s in p2}
        p3_lists_by_case[ci] = [t for t, _ in p3]
        p3_scores_by_case[ci] = {t: float(s) for t, s in p3}

    print(f"\n{ts()} Building training features (folds 1-4, R39 baseline pool, 34 feats)...")
    X_tr, y_tr, g_tr = [], [], []
    tr_t = time.time()
    for ti, idx in enumerate(train_idx):
        src_lists = {
            "A": payload["src_a"][idx], "B": payload["src_b"][idx],
            "C": payload["src_c"][idx], "D": payload["src_d"][idx],
            "F": payload["src_f"][idx], "ALS": als_source[idx],
            "R21": r21_source[idx],
        }
        pool = weighted_rrf(src_lists, SW_BASE, topk=POOL_K, k=RRF_K)
        gt = cases[idx]["gt"]
        gi = pool.index(gt) if gt in pool else -1
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[idx][:300])}
        X = build_features(pool, cases[idx], src_lists, r21_rank_map,
                            als_factors, als_track_to_idx, als_vecs[idx],
                            track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album)
        s = len(pool)
        for k in range(s):
            X_tr.append(X[k])
            y_tr.append(1.0 if k == gi else 0.0)
        g_tr.append(s)
        if (ti + 1) % 1000 == 0:
            print(f"  {ti + 1}/{len(train_idx)} ({time.time() - tr_t:.0f}s)", flush=True)

    print(f"{ts()} Training LambdaRank (R39 baseline, 34 feats)...")
    ds_tr = lgb.Dataset(np.array(X_tr), label=np.array(y_tr),
                        group=g_tr, feature_name=list(FEAT_ALL))
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
              "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
              "verbose": -1, "seed": 0}
    model = lgb.train(params, ds_tr, num_boost_round=300, callbacks=[lgb.log_evaluation(0)])

    h7_fold0 = [i for i in fold0_idx if cases[i]["n_prior_music"] == 7]
    print(f"  h7 fold-0 cases: {len(h7_fold0)}")

    # Eval baseline
    print(f"\n{ts()} === Config: baseline R39 ===")
    base_rank, base_gt, base_pools = eval_config(
        "baseline", fold0_idx, cases, payload, r21_source, None, None,
        als_source, als_vecs, als_factors, als_track_to_idx,
        track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album, model, SW_BASE)
    h7_base_ndcg = float(np.mean([1.0 / np.log2(base_rank[i] + 2) if 0 <= base_rank[i] < 20 else 0.0
                                   for i in h7_fold0]))
    pool_hit_base = sum(1 for i in fold0_idx if base_gt[i] >= 0) / len(fold0_idx)
    print(f"  pool_hit@300: {pool_hit_base:.4f}")
    print(f"  h7 nDCG: {h7_base_ndcg:.5f}")

    # Eval +P2_w1.5
    print(f"\n{ts()} === Config: R39+P2_w1.5 ===")
    p2_rank, p2_gt, p2_pools = eval_config(
        "+P2", fold0_idx, cases, payload, r21_source, p2_lists_by_case, p2_scores_by_case,
        als_source, als_vecs, als_factors, als_track_to_idx,
        track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album, model,
        {**SW_BASE, "R54": 1.5})
    h7_p2_ndcg = float(np.mean([1.0 / np.log2(p2_rank[i] + 2) if 0 <= p2_rank[i] < 20 else 0.0
                                 for i in h7_fold0]))
    pool_hit_p2 = sum(1 for i in fold0_idx if p2_gt[i] >= 0) / len(fold0_idx)
    print(f"  pool_hit@300: {pool_hit_p2:.4f}")
    print(f"  h7 nDCG: {h7_p2_ndcg:.5f}  Δ vs baseline: {h7_p2_ndcg - h7_base_ndcg:+.5f}")

    # Eval +P3_w1.5
    print(f"\n{ts()} === Config: R39+P3_w1.5 ===")
    p3_rank, p3_gt, p3_pools = eval_config(
        "+P3", fold0_idx, cases, payload, r21_source, p3_lists_by_case, p3_scores_by_case,
        als_source, als_vecs, als_factors, als_track_to_idx,
        track_pop, max_pop, ta, tt, ttl, tat, tmt, track_album, model,
        {**SW_BASE, "R54": 1.5})
    h7_p3_ndcg = float(np.mean([1.0 / np.log2(p3_rank[i] + 2) if 0 <= p3_rank[i] < 20 else 0.0
                                 for i in h7_fold0]))
    pool_hit_p3 = sum(1 for i in fold0_idx if p3_gt[i] >= 0) / len(fold0_idx)
    print(f"  pool_hit@300: {pool_hit_p3:.4f}")
    print(f"  h7 nDCG: {h7_p3_ndcg:.5f}  Δ vs baseline: {h7_p3_ndcg - h7_base_ndcg:+.5f}")

    # Attribution
    p2_attr = attribution_table("Phase 2", h7_fold0, base_gt, p2_gt, base_rank, p2_rank,
                                  p2_lists_by_case, p2_scores_by_case, cases)
    p3_attr = attribution_table("Phase 3", h7_fold0, base_gt, p3_gt, base_rank, p3_rank,
                                  p3_lists_by_case, p3_scores_by_case, cases)

    # Side-by-side
    print(f"\n{ts()} === SIDE-BY-SIDE: P2 vs P3 fold-0 ===")
    print(f"  {'Metric':<35} {'P2':>10} {'P3':>10} {'Δ':>10}  {'Δ%':>8}")
    rows = [
        ("h7 newly admitted", p2_attr["newly_admitted"], p3_attr["newly_admitted"]),
        ("admitted -> top20", p2_attr["converted_top20"], p3_attr["converted_top20"]),
        ("admitted but buried", p2_attr["buried"], p3_attr["buried"]),
        ("conversion rate", round(p2_attr["conversion_rate"], 4), round(p3_attr["conversion_rate"], 4)),
    ]
    for name, p2v, p3v in rows:
        d = p3v - p2v
        pct = (d / max(p2v, 1)) * 100 if isinstance(p2v, int) else 0
        print(f"  {name:<35} {p2v:>10} {p3v:>10} {d:>+10}  {pct:>+7.1f}%")

    print(f"\n  {'fold-0 metric':<35} {'baseline':>10} {'+P2_w1.5':>10} {'+P3_w1.5':>10}")
    print(f"  {'pool_hit@300':<35} {pool_hit_base:>10.4f} {pool_hit_p2:>10.4f} {pool_hit_p3:>10.4f}")
    print(f"  {'h7 nDCG':<35} {h7_base_ndcg:>10.5f} {h7_p2_ndcg:>10.5f} {h7_p3_ndcg:>10.5f}")
    print(f"  {'Δh7 vs baseline':<35} {0:>10.5f} "
          f"{h7_p2_ndcg - h7_base_ndcg:>+10.5f} {h7_p3_ndcg - h7_base_ndcg:>+10.5f}")

    # Gate
    print(f"\n{ts()} === SUCCESS CONDITIONS (Phase 3 fold-0) ===")
    admissions_pct_gain = (p3_attr["newly_admitted"] - p2_attr["newly_admitted"]) / max(p2_attr["newly_admitted"], 1) * 100
    conversion_improved = p3_attr["conversion_rate"] > p2_attr["conversion_rate"] + 0.05
    h7_dgain = h7_p3_ndcg - h7_base_ndcg

    g1 = admissions_pct_gain >= 30
    g2 = conversion_improved
    g3 = h7_dgain >= 0.010
    print(f"  1. Admissions +30% vs P2: {admissions_pct_gain:+.1f}%  {'PASS' if g1 else 'FAIL'}")
    print(f"  2. Conversion rate improved materially: P2={p2_attr['conversion_rate']:.3f} "
          f"vs P3={p3_attr['conversion_rate']:.3f}  {'PASS' if g2 else 'FAIL'}")
    print(f"  3. fold-0 h7 Δ >= +0.010: {h7_dgain:+.5f}  {'PASS' if g3 else 'FAIL'}")

    any_pass = g1 or g2 or g3
    print(f"\n  Decision: {'PROCEED to full 5-fold Phase 3' if any_pass else 'STOP — R22b pattern repeating'}")

    out = {
        "fold0_metrics": {
            "baseline": {"pool_hit": pool_hit_base, "h7_ndcg": h7_base_ndcg},
            "P2_w1.5": {"pool_hit": pool_hit_p2, "h7_ndcg": h7_p2_ndcg,
                         "dh7_vs_baseline": h7_p2_ndcg - h7_base_ndcg},
            "P3_w1.5": {"pool_hit": pool_hit_p3, "h7_ndcg": h7_p3_ndcg,
                         "dh7_vs_baseline": h7_p3_ndcg - h7_base_ndcg},
        },
        "attribution": {"P2": p2_attr, "P3": p3_attr},
        "gates": {"admissions_pct_gain": admissions_pct_gain,
                  "conversion_improved": conversion_improved,
                  "h7_dgain": h7_dgain,
                  "g1_admissions_30pct": g1, "g2_conversion": g2, "g3_h7_010": g3,
                  "any_pass": any_pass},
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    out_path = REPO / "exp" / "eval" / "expR54_phase3_fold0_attribution.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{ts()} Attribution complete. Elapsed: {time.time() - t0:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
