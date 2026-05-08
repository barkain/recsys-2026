#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R36 Stage 3: Low-weight audio pool expansion.

Add audio CLAP avg_recent3 as a low-weight RRF source.
Features: rank/presence only (no raw cosine — too high-gain).
Sweep RRF weights: 0.05, 0.10, 0.20, 0.30.
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

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_AUDIO_LIGHT = ["audio_rank_inv", "audio_presence"]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_audio_clap():
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_ids = []
    vecs = []
    for item in ds:
        track_ids.append(str(item["track_id"]))
        v = item["audio-laion_clap"]
        if v is not None and len(v) == 512:
            vecs.append(v)
        else:
            vecs.append([0.0] * 512)
    arr = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    arr = arr / norms
    tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    return arr, track_ids, tid_to_idx


def build_audio_lists(cases, audio_embs, audio_track_ids, audio_tid_to_idx):
    """Build audio retrieval lists (avg_recent3) as track ID lists."""
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


def main():
    t0 = time.time()
    print(f"{ts()} R36 Stage 3: Audio Pool Expansion")
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

    print(f"{ts()} Loading audio CLAP + building retrieval lists...")
    audio_embs, audio_track_ids, audio_tid_to_idx = load_audio_clap()
    audio_lists = build_audio_lists(cases, audio_embs, audio_track_ids, audio_tid_to_idx)
    del audio_embs

    n = len(cases)
    folds = grouped_session_folds(sessions, seed=0)
    n_feat_base = len(FEAT_BASE)
    feat_names = FEAT_BASE + FEAT_AUDIO_LIGHT
    n_feat = len(feat_names)

    audio_weights = [0.0, 0.05, 0.10, 0.20, 0.30]
    results = {}

    for aw in audio_weights:
        label = f"audio_w={aw:.2f}" if aw > 0 else "baseline"
        sw = dict(SW_BASE)
        if aw > 0:
            sw["AUDIO"] = aw
        use_audio_feat = aw > 0

        print(f"\n{ts()} {label} ({n_feat if use_audio_feat else n_feat_base} features)")

        cur_n_feat = n_feat if use_audio_feat else n_feat_base
        cur_feat_names = feat_names if use_audio_feat else FEAT_BASE

        X = np.zeros((n, POOL_K, cur_n_feat), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)

        for i, c in enumerate(cases):
            src_lists: dict[str, list[str]] = {
                "A": payload["src_a"][i], "B": payload["src_b"][i],
                "C": payload["src_c"][i], "D": payload["src_d"][i],
                "F": payload["src_f"][i], "ALS": als_source[i],
                "R21": r21_source[i],
            }
            if aw > 0:
                src_lists["AUDIO"] = audio_lists[i]

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

                if use_audio_feat:
                    row[n_feat_base + 0] = 1.0 / a_rm[tid] if tid in a_rm else 0.0
                    row[n_feat_base + 1] = 1.0 if tid in a_rm else 0.0

        pool_hit = float(np.mean(gt_idx >= 0))
        print(f"  pool_hit@{POOL_K}: {pool_hit:.4f}")

        # CV5 LambdaRank
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
                                group=g_tr, feature_name=list(cur_feat_names))
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
                gt_pos_arr = np.where(ranked == gt_idx[idx])[0]
                if len(gt_pos_arr) > 0 and gt_pos_arr[0] < 20:
                    case_ndcg[idx] = 1.0 / np.log2(gt_pos_arr[0] + 2)

        # Metrics
        h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
        h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
                   ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
        h7_diff = [i for i in h7 if i not in set(h7_same)]

        h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))
        cv5 = float(np.mean(case_ndcg))
        same_ndcg = float(np.mean([case_ndcg[i] for i in h7_same])) if h7_same else 0
        diff_ndcg = float(np.mean([case_ndcg[i] for i in h7_diff])) if h7_diff else 0

        slices = {}
        for depth in range(8):
            idx_list = [i for i in range(n) if cases[i]["n_prior_music"] == depth]
            if idx_list:
                slices[f"hist_{depth}"] = float(np.mean([case_ndcg[i] for i in idx_list]))

        print(f"  h7={h7_ndcg:.5f}  cv5={cv5:.5f}  same={same_ndcg:.5f}  diff={diff_ndcg:.5f}")

        results[label] = {
            "pool_hit": pool_hit, "h7": h7_ndcg, "cv5": cv5,
            "same": same_ndcg, "diff": diff_ndcg, "slices": slices,
        }

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print("R36 STAGE 3 AUDIO EXPANSION SWEEP")
    print(sep)
    base = results["baseline"]
    print(f"  {'Config':<20} {'pool':>8} {'h7':>10} {'cv5':>10} {'same':>10} {'diff':>10} {'dh7':>10}")
    print(f"  {'-'*78}")
    for label, r in results.items():
        dh7 = r["h7"] - base["h7"]
        print(f"  {label:<20} {r['pool_hit']:>8.4f} {r['h7']:>10.5f} {r['cv5']:>10.5f} "
              f"{r['same']:>10.5f} {r['diff']:>10.5f} {dh7:>+10.5f}")

    out_path = REPO / "exp" / "eval" / "expR36_stage3_audio_expansion.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
