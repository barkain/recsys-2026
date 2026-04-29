#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Epoch 2 vs Epoch 5 comprehensive comparison on devset.

Metrics per checkpoint:
- S standalone nDCG@20, hit@20, hit@200
- S unique GT hits vs ABCDF@200
- Median rank of S-unique GTs
- Overlap with A'/D/F individually
- Collapse rate
- History-depth breakdown
"""
import json, math, pickle, sys, time
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(".").resolve()))

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth
from datasets import Dataset
from mcrs.retrieval_modules.seq_model import SequenceRecommender
from scripts.expF1_cfbpr_retrieval import weighted_rrf

RUN_DIR = Path("cache/seq_model/runs/20260428_200502_fdddfe")
UTT_NPY = "cache/seq_model/utt_embeddings.npy"
UTT_INDEX = "cache/seq_model/utt_embedding_index.json"
TRACK_IDS_PATH = "cache/track_sim/metadata-qwen3_embedding_0.6b/track_ids.json"
TRACK_VECS_PATH = "cache/track_sim/metadata-qwen3_embedding_0.6b/vectors.npy"
R12_CACHE = "exp/eval/_R12_all_turns_payload.pkl"
RRF_K = 20


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("config", {})
    model = SequenceRecommender(
        track_emb_dim=config.get("track_emb_dim", 1024),
        utt_emb_dim=config.get("utt_emb_dim", 384),
        d_model=config.get("d_model", 256),
        nhead=config.get("nhead", 4),
        num_layers=config.get("num_layers", 4),
        output_dim=config.get("output_dim", 1024),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def eval_checkpoint(model, cases, payload, catalog_ids, catalog_matrix, catalog_t,
                    track_emb_dict, catalog_id_to_idx, utt_embs, utt_index, device):
    """Full evaluation of one checkpoint."""
    n = len(cases)

    def get_utt(sid, turn):
        key = f"{sid}:{turn}"
        idx = utt_index.get(key)
        if idx is not None and idx < len(utt_embs):
            return utt_embs[idx]
        return np.zeros(384, dtype=np.float32)

    # Per-case results
    results = []

    with torch.no_grad():
        for i, c in enumerate(cases):
            gt = c["gt"]
            sid = c["session_id"]
            turn = c["turn_number"]
            history = [(int(h["turn_number"]), str(h["content"]).strip())
                       for h in c["history"] if h["role"] == "music"]
            n_hist = len(history)

            if not history:
                results.append({
                    "gt": gt, "n_hist": 0, "s_hit_20": False, "s_hit_200": False,
                    "s_ndcg": 0.0, "s_top200": set(), "s_rank": None,
                    "collapse": False,
                })
                continue

            T_hist = len(history)
            T = T_hist + 1
            te = np.zeros((T_hist, 1024), dtype=np.float32)
            ue = np.zeros((T, 384), dtype=np.float32)
            for j, (t, tid) in enumerate(history):
                emb = track_emb_dict.get(tid)
                if emb is not None: te[j] = emb
                ue[j] = get_utt(sid, t)
            ue[T_hist] = get_utt(sid, turn)
            ac = np.ones(T_hist, dtype=np.int64)
            ti = np.zeros(T, dtype=np.int64)
            for j, (t, _) in enumerate(history): ti[j] = min(t-1, 7)
            ti[T_hist] = min(turn-1, 7)

            target = model(
                torch.from_numpy(te).unsqueeze(0).to(device),
                torch.from_numpy(ue).unsqueeze(0).to(device),
                torch.from_numpy(ac).unsqueeze(0).to(device),
                torch.from_numpy(ti).unsqueeze(0).to(device),
                torch.tensor([T], device=device),
            )
            scores = (target @ catalog_t.T).squeeze(0).cpu().numpy()

            played_set = {tid for _, tid in history}
            for tid in played_set:
                idx = catalog_id_to_idx.get(tid)
                if idx is not None: scores[idx] = -np.inf

            top200_idx = np.argpartition(-scores, 200)[:200]
            top200_idx = top200_idx[np.argsort(-scores[top200_idx])]
            s_top200 = [catalog_ids[j] for j in top200_idx]
            s_top200_set = set(s_top200)

            s_hit_200 = gt in s_top200_set
            s_hit_20 = gt in s_top200[:20]
            s_ndcg = 0.0
            s_rank = None
            for rank, tid in enumerate(s_top200):
                if tid == gt:
                    s_rank = rank + 1
                    if rank < 20:
                        s_ndcg = 1.0 / math.log2(rank + 2)
                    break

            # Collapse: top-1 == last played?
            top1 = s_top200[0] if s_top200 else None
            last_played = history[-1][1] if history else None
            collapse = (top1 == last_played)

            results.append({
                "gt": gt, "n_hist": n_hist,
                "s_hit_20": s_hit_20, "s_hit_200": s_hit_200,
                "s_ndcg": s_ndcg, "s_top200": s_top200_set,
                "s_rank": s_rank, "collapse": collapse,
            })

            if (i+1) % 2000 == 0:
                print(f"  [{datetime.now():%H:%M:%S}] {i+1}/{n}", flush=True)

    return results


def main():
    t0 = time.time()
    device = torch.device("cpu")

    # Load shared resources
    print(f"[{datetime.now():%H:%M:%S}] Loading resources...", flush=True)
    utt_embs = np.load(UTT_NPY)
    utt_index = json.load(open(UTT_INDEX))
    catalog_ids = json.load(open(TRACK_IDS_PATH))
    catalog_matrix = np.load(TRACK_VECS_PATH)
    catalog_t = torch.from_numpy(catalog_matrix).float()
    track_emb_dict = {tid: catalog_matrix[i] for i, tid in enumerate(catalog_ids)}
    catalog_id_to_idx = {tid: i for i, tid in enumerate(catalog_ids)}

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)

    # Build pipeline per-source top-200 for overlap analysis
    print(f"[{datetime.now():%H:%M:%S}] Building pipeline source pools...", flush=True)
    pipe_abcdf_pools = []
    pipe_a_pools = []
    pipe_d_pools = []
    pipe_f_pools = []
    for i in range(n):
        # ABCDF combined
        sources = {"A": payload["src_a"][i], "B": payload["src_b"][i],
                    "C": payload["src_c"][i], "D": payload["src_d"][i],
                    "F": payload["src_f"][i]}
        weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
        pool = weighted_rrf(sources, weights, topk=200, k=RRF_K)
        pipe_abcdf_pools.append(set(pool))
        pipe_a_pools.append(set(payload["src_a"][i][:200]))
        pipe_d_pools.append(set(payload["src_d"][i][:200]))
        pipe_f_pools.append(set(payload["src_f"][i][:200]))

    # Evaluate both checkpoints
    for epoch_num in [2, 5]:
        ckpt_path = RUN_DIR / f"epoch_{epoch_num}.pt"
        print(f"\n[{datetime.now():%H:%M:%S}] {'='*60}")
        print(f"[{datetime.now():%H:%M:%S}] EVALUATING EPOCH {epoch_num}")
        print(f"[{datetime.now():%H:%M:%S}] {'='*60}")

        model, ckpt = load_model(ckpt_path, device)
        results = eval_checkpoint(model, cases, payload, catalog_ids, catalog_matrix,
                                   catalog_t, track_emb_dict, catalog_id_to_idx,
                                   utt_embs, utt_index, device)

        # Aggregate metrics
        n_with_hist = sum(1 for r in results if r["n_hist"] > 0)
        s_ndcg = np.mean([r["s_ndcg"] for r in results])
        s_hit_20 = sum(r["s_hit_20"] for r in results)
        s_hit_200 = sum(r["s_hit_200"] for r in results)
        collapse_count = sum(r["collapse"] for r in results if r["n_hist"] > 0)
        collapse_rate = collapse_count / n_with_hist if n_with_hist else 0

        # Unique hits vs ABCDF
        s_unique = 0
        s_unique_ranks = []
        s_overlap_a = []
        s_overlap_d = []
        s_overlap_f = []
        for i, r in enumerate(results):
            gt = r["gt"]
            if r["s_hit_200"] and gt not in pipe_abcdf_pools[i]:
                s_unique += 1
                if r["s_rank"]: s_unique_ranks.append(r["s_rank"])
            if r["s_top200"]:
                s_overlap_a.append(len(r["s_top200"] & pipe_a_pools[i]) / len(r["s_top200"]))
                s_overlap_d.append(len(r["s_top200"] & pipe_d_pools[i]) / len(r["s_top200"]))
                s_overlap_f.append(len(r["s_top200"] & pipe_f_pools[i]) / len(r["s_top200"]))

        # History-depth breakdown
        hist_buckets = defaultdict(lambda: {"n": 0, "hit20": 0, "hit200": 0, "ndcg": [], "unique": 0})
        for i, r in enumerate(results):
            h = r["n_hist"]
            bk = f"hist_{min(h, 5)}" if h <= 5 else "hist_6plus"
            hist_buckets[bk]["n"] += 1
            hist_buckets[bk]["hit20"] += r["s_hit_20"]
            hist_buckets[bk]["hit200"] += r["s_hit_200"]
            hist_buckets[bk]["ndcg"].append(r["s_ndcg"])
            if r["s_hit_200"] and r["gt"] not in pipe_abcdf_pools[i]:
                hist_buckets[bk]["unique"] += 1

        print(f"\n  Standalone metrics:")
        print(f"    nDCG@20:    {s_ndcg:.4f}")
        print(f"    hit@20:     {s_hit_20}/{n} ({s_hit_20/n:.1%})")
        print(f"    hit@200:    {s_hit_200}/{n} ({s_hit_200/n:.1%})")
        print(f"    collapse:   {collapse_count}/{n_with_hist} ({collapse_rate:.1%})")

        print(f"\n  Unique hits vs ABCDF@200:")
        print(f"    S unique:   {s_unique}")
        if s_unique_ranks:
            print(f"    Median rank of unique GTs: {np.median(s_unique_ranks):.0f}")
            print(f"    Mean rank:  {np.mean(s_unique_ranks):.1f}")

        print(f"\n  Overlap with individual sources (mean % of S top-200):")
        print(f"    A' (qwen3): {np.mean(s_overlap_a):.3f}")
        print(f"    D (nbrs):   {np.mean(s_overlap_d):.3f}")
        print(f"    F (CF-BPR): {np.mean(s_overlap_f):.3f}")

        print(f"\n  History-depth breakdown:")
        print(f"    {'bucket':10s} {'n':>5s} {'hit@20':>7s} {'hit@200':>8s} {'nDCG':>7s} {'unique':>7s}")
        for bk in sorted(hist_buckets.keys()):
            d = hist_buckets[bk]
            bn = d["n"]
            print(f"    {bk:10s} {bn:5d} {d['hit20']/bn:7.1%} {d['hit200']/bn:8.1%} "
                  f"{np.mean(d['ndcg']):7.4f} {d['unique']:7d}")

        del model

    elapsed = time.time() - t0
    print(f"\n[{datetime.now():%H:%M:%S}] Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
