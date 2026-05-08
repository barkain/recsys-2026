#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R37c: Two-model router — separate rankers per regime.

Variant B (safe):
- route_off (n_unique<5): use BASELINE LambdaRank (exact same model/pool)
- route_on (n_unique>=5): use SPECIALIST model with audio pool + features

This guarantees route_off cannot be damaged.
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
RRF_K = 20
POOL_K = 300
SW_BASE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
SW_AUDIO = {**SW_BASE, "AUDIO": 0.10}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_AUDIO = FEAT_BASE + ["audio_rank_inv", "audio_presence"]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def compute_n_unique(case, ta):
    return len({ta.get(t, "") for t in case["music_turns"]} - {""})


def build_row_features(case, payload_idx, payload, als_source, als_vecs,
                       als_factors, als_track_to_idx, track_pop, max_pop,
                       r21_source, audio_lists, pool, src_lists, use_audio, feat_names):
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    i = payload_idx
    n_feat = len(feat_names)
    n_base = len(FEAT_BASE)

    src_rank = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                for sn, sl in src_lists.items()}
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
    sv = als_vecs[i]
    pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
    artist_counts = Counter(a for a in pool_artists if a)
    r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}
    a_rm = {tid: r + 1 for r, tid in enumerate(audio_lists[i])} if audio_lists[i] else {}

    X = np.zeros((POOL_K, n_feat), dtype=np.float64)
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
        if use_audio:
            row[n_base + 0] = 1.0 / a_rm[tid] if tid in a_rm else 0.0
            row[n_base + 1] = 1.0 if tid in a_rm else 0.0
    return X


def main():
    t0 = time.time()
    print(f"{ts()} R37c: Two-Model Router (Variant B)")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    with open(R21_OOF) as f:
        r21_source = json.load(f)

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

    print(f"{ts()} Loading audio...")
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    audio_tids = []
    audio_raw = []
    for item in ds:
        audio_tids.append(str(item["track_id"]))
        v = item["audio-laion_clap"]
        audio_raw.append(v if v is not None and len(v) == 512 else [0.0] * 512)
    audio_embs = np.array(audio_raw, dtype=np.float32)
    norms = np.linalg.norm(audio_embs, axis=1, keepdims=True)
    audio_embs = audio_embs / np.where(norms > 0, norms, 1.0)
    audio_tid_to_idx = {tid: i for i, tid in enumerate(audio_tids)}
    del ds, audio_raw

    # Build audio lists
    audio_lists: list[list[str]] = []
    for c in cases:
        played = c["music_turns"]
        recent = played[-3:]
        ridx = [audio_tid_to_idx[t] for t in recent if t in audio_tid_to_idx]
        if not ridx:
            audio_lists.append([])
            continue
        avg = audio_embs[ridx].mean(axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        sims = audio_embs @ avg
        for t in played:
            if t in audio_tid_to_idx:
                sims[audio_tid_to_idx[t]] = -np.inf
        top = np.argpartition(-sims, 300)[:300]
        top = top[np.argsort(-sims[top])]
        audio_lists.append([audio_tids[j] for j in top])
    del audio_embs

    n = len(cases)
    folds = grouped_session_folds(sessions, seed=0)

    # Classify all cases
    is_route_on = [compute_n_unique(c, ta) >= 5 for c in cases]

    # Build pools + features for both regimes
    print(f"\n{ts()} Building features...")
    base_pools: list[list[str]] = []
    audio_pools: list[list[str]] = []
    base_X = np.zeros((n, POOL_K, len(FEAT_BASE)), dtype=np.float64)
    audio_X = np.zeros((n, POOL_K, len(FEAT_AUDIO)), dtype=np.float64)
    base_gt_idx = np.full(n, -1, dtype=np.int64)
    audio_gt_idx = np.full(n, -1, dtype=np.int64)
    base_sizes = np.zeros(n, dtype=np.int64)
    audio_sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        src_base = {"A": payload["src_a"][i], "B": payload["src_b"][i],
                    "C": payload["src_c"][i], "D": payload["src_d"][i],
                    "F": payload["src_f"][i], "ALS": als_source[i], "R21": r21_source[i]}
        src_audio = dict(src_base)
        if audio_lists[i]:
            src_audio["AUDIO"] = audio_lists[i]

        bp = weighted_rrf(src_base, SW_BASE, topk=POOL_K, k=RRF_K)
        ap = weighted_rrf(src_audio, SW_AUDIO, topk=POOL_K, k=RRF_K)
        base_pools.append(bp)
        audio_pools.append(ap)
        base_sizes[i] = len(bp)
        audio_sizes[i] = len(ap)
        if c["gt"] in bp:
            base_gt_idx[i] = bp.index(c["gt"])
        if c["gt"] in ap:
            audio_gt_idx[i] = ap.index(c["gt"])

        base_X[i] = build_row_features(c, i, payload, als_source, als_vecs,
                                         als_factors, als_track_to_idx, track_pop, max_pop,
                                         r21_source, audio_lists, bp, src_base, False, FEAT_BASE)
        audio_X[i] = build_row_features(c, i, payload, als_source, als_vecs,
                                          als_factors, als_track_to_idx, track_pop, max_pop,
                                          r21_source, audio_lists, ap, src_audio, True, FEAT_AUDIO)

    # CV5: train baseline model + route_on specialist
    print(f"\n{ts()} Training CV5 models...")
    case_ndcg_baseline = np.zeros(n)
    case_ndcg_router = np.zeros(n)

    for fi in range(5):
        val_set = set(folds[fi].tolist())
        tr_all = [j for j in range(n) if j not in val_set]
        va = sorted(val_set)

        # Baseline model (all cases, base features, base pool)
        X_tr, y_tr, g_tr = [], [], []
        X_va, y_va, g_va = [], [], []
        for idx in tr_all:
            s = int(base_sizes[idx])
            for k in range(s):
                X_tr.append(base_X[idx, k])
                y_tr.append(1.0 if k == base_gt_idx[idx] else 0.0)
            g_tr.append(s)
        for idx in va:
            s = int(base_sizes[idx])
            for k in range(s):
                X_va.append(base_X[idx, k])
                y_va.append(1.0 if k == base_gt_idx[idx] else 0.0)
            g_va.append(s)

        ds_tr = lgb.Dataset(np.array(X_tr), label=np.array(y_tr),
                            group=g_tr, feature_name=list(FEAT_BASE))
        ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                            group=g_va, reference=ds_tr)
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                  "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
                  "verbose": -1, "seed": 0}
        base_model = lgb.train(params, ds_tr, num_boost_round=300,
                               valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])

        # Score baseline
        preds_base = base_model.predict(np.array(X_va))
        offset = 0
        for idx in va:
            s = int(base_sizes[idx])
            sc = preds_base[offset:offset + s]
            offset += s
            if base_gt_idx[idx] < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == base_gt_idx[idx])[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg_baseline[idx] = 1.0 / np.log2(gt_pos[0] + 2)

        # Route_on specialist (audio features, audio pool, trained on route_on cases only)
        tr_on = [j for j in tr_all if is_route_on[j]]
        X_tr_on, y_tr_on, g_tr_on = [], [], []
        for idx in tr_on:
            s = int(audio_sizes[idx])
            for k in range(s):
                X_tr_on.append(audio_X[idx, k])
                y_tr_on.append(1.0 if k == audio_gt_idx[idx] else 0.0)
            g_tr_on.append(s)

        if not X_tr_on:
            continue

        ds_on = lgb.Dataset(np.array(X_tr_on), label=np.array(y_tr_on),
                            group=g_tr_on, feature_name=list(FEAT_AUDIO))
        on_model = lgb.train(params, ds_on, num_boost_round=300)

        # Route: use specialist for route_on val, baseline for route_off val
        for idx in va:
            if is_route_on[idx]:
                s = int(audio_sizes[idx])
                if s == 0:
                    continue
                sc = on_model.predict(audio_X[idx, :s].reshape(s, -1))
                gi = audio_gt_idx[idx]
            else:
                # Route_off: use baseline model + baseline pool (already scored)
                case_ndcg_router[idx] = case_ndcg_baseline[idx]
                continue

            if gi < 0:
                continue
            ranked = np.argsort(-sc)
            gt_pos = np.where(ranked == gi)[0]
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg_router[idx] = 1.0 / np.log2(gt_pos[0] + 2)

    # Metrics
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
               ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
    h7_diff = [i for i in h7 if i not in set(h7_same)]
    h7_on = [i for i in h7 if is_route_on[i]]
    h7_off = [i for i in h7 if not is_route_on[i]]

    def report(label, ndcg_arr):
        h7_v = float(np.mean([ndcg_arr[i] for i in h7]))
        same_v = float(np.mean([ndcg_arr[i] for i in h7_same])) if h7_same else 0
        diff_v = float(np.mean([ndcg_arr[i] for i in h7_diff])) if h7_diff else 0
        on_v = float(np.mean([ndcg_arr[i] for i in h7_on])) if h7_on else 0
        off_v = float(np.mean([ndcg_arr[i] for i in h7_off])) if h7_off else 0
        cv5_v = float(np.mean(ndcg_arr))
        return {"h7": h7_v, "same": same_v, "diff": diff_v,
                "on": on_v, "off": off_v, "cv5": cv5_v}

    base_r = report("baseline", case_ndcg_baseline)
    router_r = report("router", case_ndcg_router)

    # Recovered/lost
    rec_on = rec_off = lost_on = lost_off = 0
    for i in h7:
        b_hit = case_ndcg_baseline[i] > 0
        r_hit = case_ndcg_router[i] > 0
        if not b_hit and r_hit:
            if is_route_on[i]:
                rec_on += 1
            else:
                rec_off += 1
        if b_hit and not r_hit:
            if is_route_on[i]:
                lost_on += 1
            else:
                lost_off += 1

    sep = "=" * 70
    print(f"\n{sep}")
    print("R37c TWO-MODEL ROUTER (Variant B)")
    print(sep)
    print(f"  {'Config':<15} {'h7':>10} {'same':>10} {'diff':>10} "
          f"{'on(n≥5)':>10} {'off(n<5)':>10} {'Δh7':>10}")
    print(f"  {'-'*75}")
    dh7 = router_r["h7"] - base_r["h7"]
    print(f"  {'baseline':<15} {base_r['h7']:>10.5f} {base_r['same']:>10.5f} "
          f"{base_r['diff']:>10.5f} {base_r['on']:>10.5f} {base_r['off']:>10.5f}")
    print(f"  {'router_B':<15} {router_r['h7']:>10.5f} {router_r['same']:>10.5f} "
          f"{router_r['diff']:>10.5f} {router_r['on']:>10.5f} {router_r['off']:>10.5f} "
          f"{dh7:>+10.5f}")

    print("\n  Recovered/Lost:")
    print(f"    route_on:  rec={rec_on} lost={lost_on} net={rec_on-lost_on:+d}")
    print(f"    route_off: rec={rec_off} lost={lost_off} net={rec_off-lost_off:+d}")
    print(f"    total:     rec={rec_on+rec_off} lost={lost_on+lost_off} "
          f"net={rec_on+rec_off-lost_on-lost_off:+d}")

    # Gate
    print(f"\n{sep}")
    print("GATE CHECK")
    dsame = router_r["same"] - base_r["same"]
    g_h7 = dh7 >= 0.005
    g_same = dsame >= -0.003
    g_off = abs(router_r["off"] - base_r["off"]) < 0.001
    print(f"  Δh7 >= +0.005:          {dh7:+.5f} {'PASS' if g_h7 else 'FAIL'}")
    print(f"  same regress <= 0.003:  {dsame:+.5f} {'PASS' if g_same else 'FAIL'}")
    print(f"  route_off protected:    {router_r['off']-base_r['off']:+.5f} "
          f"{'PASS' if g_off else 'WARN'}")
    status = "PASS" if (g_h7 and g_same) else "PROMISING" if dh7 >= 0.002 and g_off else "FAIL"
    print(f"  Overall: {status}")

    out = {"baseline": base_r, "router": router_r, "delta_h7": dh7,
           "rec_on": rec_on, "lost_on": lost_on, "rec_off": rec_off, "lost_off": lost_off}
    out_path = REPO / "exp" / "eval" / "expR37c_two_model_router.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
