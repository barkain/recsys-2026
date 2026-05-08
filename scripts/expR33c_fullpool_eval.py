#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R33c full-pool evaluation: score all 300 candidates per hist_7 case.

Uses the trained MLP from R33c to rescore the full pool@300,
comparing against LambdaRank CV5 baseline on the same cases.
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightgbm as lgb
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank_grouped import grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector
from scripts.tune_postrank_v23 import tokens
from scripts.expR33c_neural_reranker import PoolRerankerMLP

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R21_EMBS = REPO / "cache" / "r21_production" / "track_embeddings.npy"
R21_IDS = REPO / "cache" / "r21_production" / "track_ids.json"
R33_DIR = REPO / "cache" / "r33c"
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
FEAT_NAMES = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
N_LR_FEAT = len(FEAT_NAMES)


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    t0 = time.time()
    print(f"{ts()} R33c Full-Pool Evaluation")
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

    r21_ids = json.loads(Path(R21_IDS).read_text())
    r21_id_to_idx = {tid: i for i, tid in enumerate(r21_ids)}
    r21_embs = np.load(R21_EMBS).astype(np.float32)
    r21_embs_norm = r21_embs / (np.linalg.norm(r21_embs, axis=1, keepdims=True) + 1e-8)

    query_embs = np.load(R33_DIR / "query_embs.npy")
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

    # MLP will be retrained on full pool below (original R33c didn't save checkpoint)
    print("  ERROR: MLP model not saved to disk. Need to retrain with checkpoint saving.")
    print("  Running LambdaRank baseline on full pool as reference...")

    # Fold-0 split
    folds = grouped_session_folds(sessions, seed=0)
    fold0_val = set(folds[0].tolist())
    n = len(cases)

    # Build full feature matrix + train LambdaRank CV5
    print(f"\n{ts()} Building features for all 8000 cases...")
    X = np.zeros((n, POOL_K, N_LR_FEAT), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)
    pools: list[list[str]] = []

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
        pools.append(pool)
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

    # Train CV5 LambdaRank baseline
    print(f"\n{ts()} Training CV5 LambdaRank baseline...")
    lr_scores: list[np.ndarray | None] = [None] * n
    for fi in range(5):
        val_idx = set(folds[fi].tolist())
        train_list = [j for j in range(n) if j not in val_idx]
        val_list = sorted(val_idx)

        X_tr, y_tr, g_tr = [], [], []
        X_va, y_va, g_va = [], [], []
        for idx in train_list:
            s = int(sizes[idx])
            for k in range(s):
                X_tr.append(X[idx, k])
                y_tr.append(1.0 if k == gt_idx[idx] else 0.0)
            g_tr.append(s)
        for idx in val_list:
            s = int(sizes[idx])
            for k in range(s):
                X_va.append(X[idx, k])
                y_va.append(1.0 if k == gt_idx[idx] else 0.0)
            g_va.append(s)

        ds_tr = lgb.Dataset(np.array(X_tr), label=np.array(y_tr),
                            group=g_tr, feature_name=list(FEAT_NAMES))
        ds_va = lgb.Dataset(np.array(X_va), label=np.array(y_va),
                            group=g_va, reference=ds_tr)
        params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                  "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
                  "verbose": -1, "seed": 0}
        lgb_model = lgb.train(params, ds_tr, num_boost_round=300,
                              valid_sets=[ds_va], callbacks=[lgb.log_evaluation(0)])

        preds = lgb_model.predict(np.array(X_va))
        offset = 0
        for idx in val_list:
            s = int(sizes[idx])
            lr_scores[idx] = preds[offset:offset + s].copy()
            offset += s

    # Now train MLP on full pool (retrain with all 300 candidates)
    print(f"\n{ts()} Training MLP on full pool@300...")

    # Build full-pool training data for hist_5+ non-fold-0
    h5_train = [i for i in range(n) if cases[i]["n_prior_music"] >= 5 and i not in fold0_val]
    h7_val = [i for i in range(n) if cases[i]["n_prior_music"] == 7 and i in fold0_val]

    model = PoolRerankerMLP(emb_dim=768, lr_feat_dim=N_LR_FEAT, hidden=256, dropout=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(20):
        np.random.shuffle(h5_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in h5_train:
            c = cases[i]
            pool = pools[i]
            if gt_idx[i] < 0 or sizes[i] == 0:
                continue

            q_emb = query_embs[i]
            gt_pool_pos = gt_idx[i]

            # Score all candidates in pool
            q_batch = []
            c_batch = []
            lr_batch = []
            inter_batch = []

            for j in range(int(sizes[i])):
                tid = pool[j]
                cand_idx = r21_id_to_idx.get(tid)
                c_emb = r21_embs_norm[cand_idx] if cand_idx is not None else np.zeros(768, dtype=np.float32)
                cosine = float(np.dot(q_emb, c_emb))
                dot_raw = float(np.dot(q_emb, r21_embs[cand_idx])) if cand_idx is not None else 0.0

                q_batch.append(q_emb)
                c_batch.append(c_emb)
                lr_batch.append(X[i, j].astype(np.float32))
                inter_batch.append(np.array([cosine, dot_raw, float(j + 1)], dtype=np.float32))

            q_t = torch.from_numpy(np.stack(q_batch))
            c_t = torch.from_numpy(np.stack(c_batch))
            lr_t = torch.from_numpy(np.stack(lr_batch))
            inter_t = torch.from_numpy(np.stack(inter_batch))

            scores = model(q_t, c_t, lr_t, inter_t)
            pos_score = scores[gt_pool_pos]
            neg_mask = torch.ones(len(scores), dtype=torch.bool)
            neg_mask[gt_pool_pos] = False
            neg_scores = scores[neg_mask]

            if len(neg_scores) == 0:
                continue

            loss = -torch.log(torch.sigmoid(pos_score - neg_scores) + 1e-8).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}", flush=True)

    # Full-pool eval on fold-0 hist_7
    print(f"\n{ts()} Full-pool evaluation on fold-0 hist_7...")
    model.training = False

    lr_ndcg = 0.0
    mlp_ndcg = 0.0
    same_lr = 0.0
    same_mlp = 0.0
    diff_lr = 0.0
    diff_mlp = 0.0
    n_same = 0
    n_diff = 0
    n_cases = 0
    recovered = 0
    lost = 0
    lr_gt_ranks = []
    mlp_gt_ranks = []

    with torch.no_grad():
        for i in h7_val:
            c = cases[i]
            pool = pools[i]
            s = int(sizes[i])
            if s == 0:
                n_cases += 1
                continue

            gt = c["gt"]
            gt_artist = ta.get(gt, "")
            is_same = gt_artist and gt_artist in {ta.get(t, "") for t in c["music_turns"]}
            gt_in_pool = gt in pool

            # LambdaRank baseline ranking
            sc_lr = lr_scores[i]
            if sc_lr is not None:
                lr_ranked = np.argsort(-sc_lr)
                lr_top20 = [pool[j] for j in lr_ranked[:20]]
            else:
                lr_top20 = pool[:20]
                lr_ranked = np.arange(s)

            # MLP scoring
            q_emb = query_embs[i]
            q_batch, c_batch, lr_batch, inter_batch = [], [], [], []
            for j in range(s):
                tid = pool[j]
                cand_idx = r21_id_to_idx.get(tid)
                c_emb = r21_embs_norm[cand_idx] if cand_idx is not None else np.zeros(768, dtype=np.float32)
                cosine = float(np.dot(q_emb, c_emb))
                dot_raw = float(np.dot(q_emb, r21_embs[cand_idx])) if cand_idx is not None else 0.0
                q_batch.append(q_emb)
                c_batch.append(c_emb)
                lr_batch.append(X[i, j].astype(np.float32))
                inter_batch.append(np.array([cosine, dot_raw, float(j + 1)], dtype=np.float32))

            mlp_scores = model(
                torch.from_numpy(np.stack(q_batch)),
                torch.from_numpy(np.stack(c_batch)),
                torch.from_numpy(np.stack(lr_batch)),
                torch.from_numpy(np.stack(inter_batch)),
            ).numpy()
            mlp_ranked = np.argsort(-mlp_scores)
            mlp_top20 = [pool[j] for j in mlp_ranked[:20]]

            # nDCG
            lr_v = 0.0
            if gt in lr_top20:
                pos = lr_top20.index(gt)
                lr_v = 1.0 / np.log2(pos + 2)
            mlp_v = 0.0
            if gt in mlp_top20:
                pos = mlp_top20.index(gt)
                mlp_v = 1.0 / np.log2(pos + 2)

            lr_ndcg += lr_v
            mlp_ndcg += mlp_v

            if is_same:
                same_lr += lr_v
                same_mlp += mlp_v
                n_same += 1
            else:
                diff_lr += lr_v
                diff_mlp += mlp_v
                n_diff += 1

            if gt_in_pool:
                gt_pool_pos = pool.index(gt)
                lr_gt_rank = np.where(lr_ranked == gt_pool_pos)[0]
                mlp_gt_rank = np.where(mlp_ranked == gt_pool_pos)[0]
                if len(lr_gt_rank) > 0:
                    lr_gt_ranks.append(int(lr_gt_rank[0]) + 1)
                if len(mlp_gt_rank) > 0:
                    mlp_gt_ranks.append(int(mlp_gt_rank[0]) + 1)

            if gt not in lr_top20 and gt in mlp_top20:
                recovered += 1
            if gt in lr_top20 and gt not in mlp_top20:
                lost += 1

            n_cases += 1

    lr_h7 = lr_ndcg / max(n_cases, 1)
    mlp_h7 = mlp_ndcg / max(n_cases, 1)
    dh7 = mlp_h7 - lr_h7

    sep = "=" * 70
    print(f"\n{sep}")
    print("R33c FULL-POOL EVALUATION (fold-0 hist_7)")
    print(sep)
    print(f"  LambdaRank baseline h7 nDCG@20:  {lr_h7:.5f}")
    print(f"  MLP reranker h7 nDCG@20:         {mlp_h7:.5f} ({dh7:+.5f})")
    if n_same > 0:
        print(f"  same_artist: LR={same_lr/n_same:.5f}  MLP={same_mlp/n_same:.5f}")
    if n_diff > 0:
        print(f"  diff_artist: LR={diff_lr/n_diff:.5f}  MLP={diff_mlp/n_diff:.5f}")
    print(f"  recovered (LR miss → MLP hit): {recovered}")
    print(f"  lost (LR hit → MLP miss):      {lost}")
    print(f"  net top20 change:              {recovered - lost:+d}")
    if lr_gt_ranks:
        print(f"  LR median GT rank:  {np.median(lr_gt_ranks):.0f}")
    if mlp_gt_ranks:
        print(f"  MLP median GT rank: {np.median(mlp_gt_ranks):.0f}")

    print(f"\n{sep}")
    print("GATE CHECK")
    g1 = dh7 >= 0.005
    print(f"  Δh7 nDCG@20 >= +0.005: {dh7:+.5f} {'PASS' if g1 else 'FAIL'}")
    print(f"  recovered={recovered}, lost={lost}, net={recovered-lost:+d}")

    out_path = REPO / "exp" / "eval" / "expR33c_fullpool.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "lr_h7": lr_h7, "mlp_h7": mlp_h7, "delta": dh7,
            "recovered": recovered, "lost": lost,
            "n_same": n_same, "n_diff": n_diff,
            "lr_median_gt_rank": float(np.median(lr_gt_ranks)) if lr_gt_ranks else -1,
            "mlp_median_gt_rank": float(np.median(mlp_gt_ranks)) if mlp_gt_ranks else -1,
        }, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
