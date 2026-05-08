#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R36 Stage 2: Audio CLAP feature-only LambdaRank.

Add audio cosine/rank/presence features to existing pool@300 candidates.
No pool expansion. No global fusion. Single-source attribution.

No torch import — LambdaRank only, avoids loky segfault.
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
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

FEAT_BASE = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEAT_AUDIO = [
    "audio_cosine",
    "audio_rank_inv",
    "audio_presence",
    "audio_top20",
    "audio_top50",
]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_audio_clap():
    """Load audio CLAP embeddings, L2-normalize."""
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
    print(f"  Audio CLAP: {arr.shape}, {sum(1 for v in vecs if any(x != 0 for x in v))}/{len(vecs)} valid")
    return arr, track_ids, tid_to_idx


def build_audio_session_vec(played, audio_embs, tid_to_idx, recent_k=3):
    """Build session vector as normalized average of recent tracks' audio embeddings."""
    recent = played[-recent_k:]
    recent_idx = [tid_to_idx[t] for t in recent if t in tid_to_idx]
    if not recent_idx:
        return None
    avg = audio_embs[recent_idx].mean(axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg


def main():
    t0 = time.time()
    print(f"{ts()} R36 Stage 2: Audio CLAP Feature-Only LambdaRank")
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

    print(f"{ts()} Loading audio CLAP embeddings...")
    audio_embs, audio_track_ids, audio_tid_to_idx = load_audio_clap()

    # Precompute audio retrieval lists for rank features
    print(f"{ts()} Building audio retrieval lists (avg_recent3)...")
    audio_lists: list[list[int]] = []
    audio_session_vecs: list[np.ndarray | None] = []
    for i, c in enumerate(cases):
        played = c["music_turns"]
        sv = build_audio_session_vec(played, audio_embs, audio_tid_to_idx, recent_k=3)
        audio_session_vecs.append(sv)
        if sv is not None:
            sims = audio_embs @ sv
            played_idx = {audio_tid_to_idx[t] for t in played if t in audio_tid_to_idx}
            for pi in played_idx:
                sims[pi] = -np.inf
            top = np.argpartition(-sims, 300)[:300]
            top = top[np.argsort(-sims[top])]
            audio_lists.append(top.tolist())
        else:
            audio_lists.append([])

    audio_rank_maps = []
    for al in audio_lists:
        audio_rank_maps.append({audio_track_ids[j]: r + 1 for r, j in enumerate(al)})

    # Build features
    n = len(cases)
    n_feat_base = len(FEAT_BASE)

    configs = [
        ("baseline", FEAT_BASE, False),
        ("+audio", FEAT_BASE + FEAT_AUDIO, True),
    ]

    folds = grouped_session_folds(sessions, seed=0)

    for config_name, feat_names, use_audio in configs:
        n_feat = len(feat_names)
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
            sv_als = als_vecs[i]
            pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
            artist_counts = Counter(a for a in pool_artists if a)
            r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}

            a_sv = audio_session_vecs[i]
            a_rm = audio_rank_maps[i]

            for rank, tid in enumerate(pool[:POOL_K], start=1):
                ca = ta.get(tid, "")
                ct = tt.get(tid, set())
                row = X[i, rank - 1]

                # Base 29 features
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
                    # Audio cosine for ALL pool candidates
                    audio_idx = audio_tid_to_idx.get(tid)
                    if audio_idx is not None and a_sv is not None:
                        row[n_feat_base + 0] = float(np.dot(a_sv, audio_embs[audio_idx]))
                    row[n_feat_base + 1] = 1.0 / a_rm[tid] if tid in a_rm else 0.0
                    row[n_feat_base + 2] = 1.0 if tid in a_rm else 0.0
                    row[n_feat_base + 3] = 1.0 if a_rm.get(tid, 999) <= 20 else 0.0
                    row[n_feat_base + 4] = 1.0 if a_rm.get(tid, 999) <= 50 else 0.0

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

            if use_audio and fi == 4:
                importance = model.feature_importance(importance_type="gain")
                feat_imp = sorted(zip(feat_names, importance), key=lambda x: -x[1])
                print("  Top features by gain (last fold):")
                for fname, imp in feat_imp[:10]:
                    print(f"    {fname}: {imp:.1f}")

        # Report by slice
        slices = {}
        for depth in range(8):
            idx_list = [i for i in range(n) if cases[i]["n_prior_music"] == depth]
            if idx_list:
                slices[f"hist_{depth}"] = float(np.mean([case_ndcg[i] for i in idx_list]))

        h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
        h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
                   ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
        h7_diff = [i for i in h7 if i not in set(h7_same)]

        h7_ndcg = float(np.mean([case_ndcg[i] for i in h7]))
        cv5 = float(np.mean(case_ndcg))
        same_ndcg = float(np.mean([case_ndcg[i] for i in h7_same])) if h7_same else 0
        diff_ndcg = float(np.mean([case_ndcg[i] for i in h7_diff])) if h7_diff else 0

        print(f"  h7={h7_ndcg:.5f}  cv5={cv5:.5f}  same={same_ndcg:.5f}  diff={diff_ndcg:.5f}")
        for k, v in sorted(slices.items()):
            print(f"    {k}: {v:.5f}")


    print(f"\nElapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
