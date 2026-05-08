#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R37b: Conditional audio router — audio only when n_unique_artists >= 5.

Compares:
1. Baseline (no audio)
2. Global audio w=0.10 (from R36 Stage 3)
3. Conditional: audio w=0.10 only for n_unique>=5 cases
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
FEAT_WITH_AUDIO = FEAT_BASE + ["audio_rank_inv", "audio_presence"]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_audio_lists(cases, audio_embs, audio_track_ids, audio_tid_to_idx):
    audio_lists = []
    for c in cases:
        played = c["music_turns"]
        recent = played[-3:]
        recent_idx = [audio_tid_to_idx[t] for t in recent if t in audio_tid_to_idx]
        if not recent_idx:
            audio_lists.append([])
            continue
        avg = audio_embs[recent_idx].mean(axis=0)
        norm = np.linalg.norm(avg)
        if norm > 0:
            avg = avg / norm
        sims = audio_embs @ avg
        played_idx = {audio_tid_to_idx[t] for t in played if t in audio_tid_to_idx}
        for pi in played_idx:
            sims[pi] = -np.inf
        top = np.argpartition(-sims, 300)[:300]
        top = top[np.argsort(-sims[top])]
        audio_lists.append([audio_track_ids[j] for j in top])
    return audio_lists


def compute_n_unique(case, ta):
    artists = [ta.get(t, "") for t in case["music_turns"]]
    return len({a for a in artists if a})


def build_features_and_eval(cases, payload, als_source, als_vecs, als_factors,
                             als_track_to_idx, track_pop, r21_source,
                             audio_lists, sessions, ta, mode="baseline"):
    """Build features + CV5 LambdaRank for a given mode.

    mode: 'baseline', 'global_audio', 'conditional_audio'
    """
    n = len(cases)
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    use_audio = mode in ("global_audio", "conditional_audio")
    feat_names = FEAT_WITH_AUDIO if use_audio else FEAT_BASE
    n_feat = len(feat_names)
    n_feat_base = len(FEAT_BASE)

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        n_unique = compute_n_unique(c, ta)
        case_uses_audio = False
        if mode == "global_audio":
            case_uses_audio = True
        elif mode == "conditional_audio":
            case_uses_audio = n_unique >= 5

        src_lists: dict[str, list[str]] = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        sw = SW_BASE
        if case_uses_audio and audio_lists[i]:
            src_lists["AUDIO"] = audio_lists[i]
            sw = SW_AUDIO

        pool = weighted_rrf(src_lists, sw, topk=POOL_K, k=RRF_K)
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
        sv_als = als_vecs[i]
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}
        a_rm = {tid: r + 1 for r, tid in enumerate(audio_lists[i])} if audio_lists[i] else {}

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
            row[20] = sum(1 for sn in ["A", "B", "C", "D", "F", "ALS"]
                         if tid in src_rank.get(sn, {}))
            if sv_als is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv_als, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            row[28] = 1.0 if tid in r21_rank_map else 0.0

            if use_audio:
                row[n_feat_base + 0] = 1.0 / a_rm[tid] if tid in a_rm else 0.0
                row[n_feat_base + 1] = 1.0 if tid in a_rm else 0.0

    pool_hit = float(np.mean(gt_idx >= 0))

    # CV5 LambdaRank
    folds = grouped_session_folds(sessions, seed=0)
    case_ndcg = np.zeros(n)
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
            if len(gt_pos) > 0 and gt_pos[0] < 20:
                case_ndcg[idx] = 1.0 / np.log2(gt_pos[0] + 2)

    # Metrics by slice
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
               ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
    h7_diff = [i for i in h7 if i not in set(h7_same)]

    h7_route_on = [i for i in h7 if compute_n_unique(cases[i], ta) >= 5]
    h7_route_off = [i for i in h7 if compute_n_unique(cases[i], ta) < 5]

    return {
        "pool_hit": pool_hit,
        "h7_all": float(np.mean([case_ndcg[i] for i in h7])),
        "h7_same": float(np.mean([case_ndcg[i] for i in h7_same])) if h7_same else 0,
        "h7_diff": float(np.mean([case_ndcg[i] for i in h7_diff])) if h7_diff else 0,
        "h7_route_on": float(np.mean([case_ndcg[i] for i in h7_route_on])) if h7_route_on else 0,
        "h7_route_off": float(np.mean([case_ndcg[i] for i in h7_route_off])) if h7_route_off else 0,
        "n_route_on": len(h7_route_on),
        "n_route_off": len(h7_route_off),
        "cv5": float(np.mean(case_ndcg)),
    }


def main():
    t0 = time.time()
    print(f"{ts()} R37b: Conditional Audio Router")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    track_pop = build_popularity_stats()

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

    print(f"{ts()} Loading audio CLAP + building lists...")
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    audio_tids = []
    audio_vecs_raw = []
    for item in ds:
        audio_tids.append(str(item["track_id"]))
        v = item["audio-laion_clap"]
        if v is not None and len(v) == 512:
            audio_vecs_raw.append(v)
        else:
            audio_vecs_raw.append([0.0] * 512)
    audio_embs = np.array(audio_vecs_raw, dtype=np.float32)
    norms = np.linalg.norm(audio_embs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    audio_embs = audio_embs / norms
    audio_tid_to_idx = {tid: i for i, tid in enumerate(audio_tids)}
    del ds, audio_vecs_raw

    audio_lists = load_audio_lists(cases, audio_embs, audio_tids, audio_tid_to_idx)
    del audio_embs

    # Run 3 configs
    configs = ["baseline", "global_audio", "conditional_audio"]
    results = {}

    for mode in configs:
        print(f"\n{ts()} Config: {mode}")
        r = build_features_and_eval(
            cases, payload, als_source, als_vecs, als_factors,
            als_track_to_idx, track_pop, r21_source,
            audio_lists, sessions, ta, mode=mode)
        results[mode] = r
        print(f"  pool={r['pool_hit']:.4f}  h7={r['h7_all']:.5f}  "
              f"same={r['h7_same']:.5f}  diff={r['h7_diff']:.5f}  "
              f"on={r['h7_route_on']:.5f}({r['n_route_on']})  "
              f"off={r['h7_route_off']:.5f}({r['n_route_off']})")

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print("R37b CONDITIONAL AUDIO ROUTER")
    print(sep)
    base = results["baseline"]
    print(f"  {'Config':<22} {'pool':>7} {'h7':>10} {'same':>10} {'diff':>10} "
          f"{'on(n≥5)':>10} {'off(n<5)':>10} {'Δh7':>10}")
    print(f"  {'-'*89}")
    for mode, r in results.items():
        dh7 = r["h7_all"] - base["h7_all"]
        print(f"  {mode:<22} {r['pool_hit']:>7.4f} {r['h7_all']:>10.5f} "
              f"{r['h7_same']:>10.5f} {r['h7_diff']:>10.5f} "
              f"{r['h7_route_on']:>10.5f} {r['h7_route_off']:>10.5f} {dh7:>+10.5f}")

    # Gate
    print(f"\n{sep}")
    print("GATE CHECK")
    cond = results["conditional_audio"]
    dh7 = cond["h7_all"] - base["h7_all"]
    dsame = cond["h7_same"] - base["h7_same"]
    g_h7 = dh7 >= 0.005
    g_same = dsame >= -0.003
    print(f"  Δh7 >= +0.005:         {dh7:+.5f} {'PASS' if g_h7 else 'FAIL'}")
    print(f"  same regress <= 0.003: {dsame:+.5f} {'PASS' if g_same else 'FAIL'}")
    status = "PASS" if (g_h7 and g_same) else "PROMISING" if dh7 >= 0.002 and g_same else "FAIL"
    print(f"  Overall: {status}")

    out_path = REPO / "exp" / "eval" / "expR37b_conditional_audio.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
