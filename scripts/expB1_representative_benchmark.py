#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""B1: Representative local benchmark across history slices.

Evaluates cfg0209, F1 CF-BPR, and ABCDFG on multiple eval slices
including synthetic low-history cases that better match Blind-A.

No API. No blind. No new models.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens
from scripts.expF1_cfbpr_retrieval import (
    build_cfbpr_index,
    cfbpr_max_recent,
    weighted_rrf,
)
from scripts.expR5_sequential_retrieval import SessionTransitionGraph
from offline_retrieval_sweep import CachedBM25, load_track_metadata, query_parts
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from datasets import Dataset, DownloadConfig, load_dataset
from scipy.optimize import minimize

POOL_K = 50
RRF_K = 20


def ndcg_at_k(predicted, gt_id, k=20):
    for i, tid in enumerate(predicted[:k]):
        if tid == gt_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


# ---- Load all data ---- #

def load_full_devset():
    """Load ALL test conversations with ALL turns (not just last turn)."""
    path = cached_test_arrow_path()
    if not path:
        raise FileNotFoundError("Cached devset test arrow not found")
    ds = Dataset.from_file(path)
    gt_map = build_ground_truth(ds)

    cases = []
    for item in ds:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))

        # Extract ALL user turns (not just last)
        user_turns = [c for c in convs if c["role"] == "user"]
        for ut in user_turns:
            turn_num = int(ut["turn_number"])
            user_query = str(ut["content"])
            history = [c for c in convs if int(c["turn_number"]) < turn_num]
            music_turns = [str(c["content"]).strip() for c in history if c["role"] == "music"]
            gt = lookup_ground_truth(gt_map, sid, uid, turn_num)
            if not gt:
                continue
            cases.append({
                "session_id": sid,
                "user_id": uid,
                "turn_number": turn_num,
                "user_query": user_query,
                "history": history,
                "music_turns": music_turns,
                "gt": str(gt),
                "n_prior_music": len(music_turns),
            })
    return cases


def build_slices(cases):
    """Build eval slices from all cases."""
    slices = {}

    # Full all-turn
    slices["all_turns"] = cases

    # Last-turn only (one per session, highest turn)
    by_session = defaultdict(list)
    for c in cases:
        by_session[c["session_id"]].append(c)
    last_turn = [max(v, key=lambda c: c["turn_number"]) for v in by_session.values()]
    slices["last_turn"] = last_turn

    # By history depth
    for max_hist in [0, 1, 2, 3]:
        slices[f"hist_{max_hist}"] = [c for c in cases if c["n_prior_music"] == max_hist]
    slices["hist_4plus"] = [c for c in cases if c["n_prior_music"] >= 4]
    slices["hist_5plus"] = [c for c in cases if c["n_prior_music"] >= 5]

    # Synthetic truncated: take all cases but truncate history
    for trunc_k in [0, 1, 2, 3]:
        truncated = []
        for c in cases:
            tc = dict(c)
            tc["music_turns"] = c["music_turns"][-trunc_k:] if trunc_k > 0 else []
            tc["_truncated_to"] = trunc_k
            truncated.append(tc)
        slices[f"trunc_{trunc_k}"] = truncated

    return slices


# ---- Per-case retrieval & scoring ---- #

class MultiSourceRetriever:
    """Retrieves from all sources for a single case."""

    def __init__(self, bm25, metadata, qwen_sim, cf_ids, cf_vecs, cf_idx, graph):
        self.bm25 = bm25
        self.metadata = metadata
        self.qwen_sim = qwen_sim
        self.cf_ids = cf_ids
        self.cf_vecs = cf_vecs
        self.cf_idx = cf_idx
        self.graph = graph

    def retrieve_case(self, case):
        """Returns per-source lists for a single case."""
        played = case["music_turns"]
        history = case["history"]
        user_query = case["user_query"]

        # A': qwen3 max_recent_5
        a_idxs = [self.qwen_sim._id_to_idx.get(str(t)) for t in played[-5:]]
        a_idxs = [i for i in a_idxs if i is not None]
        if a_idxs:
            anchor_vecs = self.qwen_sim.vectors[a_idxs]
            sims = self.qwen_sim.vectors @ anchor_vecs.T
            scores_a = sims.max(axis=1)
            exclude_a = {self.qwen_sim._id_to_idx[t] for t in played if t in self.qwen_sim._id_to_idx}
            cap = min(len(scores_a), 200 + len(exclude_a))
            cand = np.argpartition(-scores_a, cap - 1)[:cap]
            cand = cand[np.argsort(-scores_a[cand])]
            src_a = [self.qwen_sim.track_ids[int(ii)] for ii in cand if int(ii) not in exclude_a][:200]
        else:
            src_a = []

        # B: last_music_meta BM25
        q_b = " ".join(query_parts(history, user_query, self.metadata, "last_music_meta"))
        src_b = self.bm25.retrieve(q_b or user_query, topk=500)

        # C: full_history BM25
        q_c = " ".join(query_parts(history, user_query, self.metadata, "full"))
        src_c = self.bm25.retrieve(q_c or user_query, topk=500)

        # D: track neighbors (qwen3)
        anchor = played[-1] if played else None
        src_d = self.qwen_sim.track_id_to_neighbors(anchor, topk=200) if anchor else []

        # F: CF-BPR max_recent_5
        src_f = cfbpr_max_recent(played, self.cf_vecs, self.cf_idx, self.cf_ids, 5, 200) if played else []

        # G: session co-occurrence
        src_g = self.graph.g_session_cooccur(played, 200) if played else []

        return {
            "A": src_a, "B": src_b, "C": src_c,
            "D": src_d, "F": src_f, "G": src_g,
        }


def build_pool_and_score(sources, weights, gt, pool_k=50):
    """Build weighted-RRF pool and compute metrics."""
    pool = weighted_rrf(sources, weights, topk=pool_k, k=RRF_K)
    hit_20 = gt in pool[:20]
    hit_50 = gt in pool[:pool_k]
    ndcg = ndcg_at_k(pool, gt)
    gt_rank = pool.index(gt) + 1 if gt in pool else None
    return {
        "pool": pool,
        "hit_20": hit_20,
        "hit_50": hit_50,
        "ndcg": ndcg,
        "gt_rank": gt_rank,
    }


def eval_slice(slice_cases, retriever, configs, metadata, track_artist, track_tags):
    """Evaluate multiple configs on a slice."""
    results = {name: {"ndcg": [], "hit20": [], "hit50": [], "gt_ranks": [],
                       "cf_nonempty": 0, "g_nonempty": 0, "n": 0}
               for name in configs}

    for case in slice_cases:
        sources = retriever.retrieve_case(case)
        gt = case["gt"]

        for name, weights in configs.items():
            active_sources = {k: v for k, v in sources.items() if weights.get(k, 0) > 0}
            r = build_pool_and_score(active_sources, weights, gt)
            results[name]["ndcg"].append(r["ndcg"])
            results[name]["hit20"].append(r["hit_20"])
            results[name]["hit50"].append(r["hit_50"])
            if r["gt_rank"]:
                results[name]["gt_ranks"].append(r["gt_rank"])
            results[name]["n"] = len(slice_cases)

        # Source coverage
        if sources["F"]:
            for name in configs:
                results[name]["cf_nonempty"] += 1
        if sources["G"]:
            for name in configs:
                results[name]["g_nonempty"] += 1

    # Summarize
    summary = {}
    for name, r in results.items():
        n = r["n"]
        ndcg_arr = r["ndcg"]
        pool_hit_50 = sum(r["hit50"]) / n if n else 0
        ndcg_mean = sum(ndcg_arr) / n if n else 0
        conversion = ndcg_mean / pool_hit_50 if pool_hit_50 > 0 else 0
        summary[name] = {
            "n": n,
            "pool_hit_20": sum(r["hit20"]) / n if n else 0,
            "pool_hit_50": pool_hit_50,
            "ndcg_20": ndcg_mean,
            "hit_20_raw": sum(r["hit20"]),
            "hit_50_raw": sum(r["hit50"]),
            "conversion": conversion,
            "median_gt_rank": float(np.median(r["gt_ranks"])) if r["gt_ranks"] else None,
            "cf_coverage": r["cf_nonempty"] / n if n else 0,
            "g_coverage": r["g_nonempty"] / n if n else 0,
        }
    return summary


def main():
    t0 = time.time()

    # Load everything
    print("Loading full devset (all turns)...", flush=True)
    all_cases = load_full_devset()
    print(f"  Total cases: {len(all_cases)}")
    print(f"  Unique sessions: {len(set(c['session_id'] for c in all_cases))}")

    hist_dist = defaultdict(int)
    for c in all_cases:
        hist_dist[c["n_prior_music"]] += 1
    print(f"  History distribution: {dict(sorted(hist_dist.items()))}")

    slices = build_slices(all_cases)
    print(f"  Slices: {[(k, len(v)) for k, v in slices.items()]}")

    print("\nLoading retrievers...", flush=True)
    metadata = load_track_metadata()
    bm25 = CachedBM25()
    qwen_sim = TrackSimilarityRetriever(cache_dir="./cache")
    cf_ids, cf_vecs, cf_idx = build_cfbpr_index()

    print("Building session transition graph...", flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    session_sequences = {}
    for item in ds["train"]:
        sid = str(item["session_id"])
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        tracks = [str(c["content"]).strip() for c in convs if c["role"] == "music"]
        session_sequences[sid] = tracks
    graph = SessionTransitionGraph(session_sequences, metadata)

    retriever = MultiSourceRetriever(bm25, metadata, qwen_sim, cf_ids, cf_vecs, cf_idx, graph)

    # System configs
    configs = {
        "cfg0209_ABCD": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5},
        "F1_ABCDF": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0},
        "ABCDFG": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "G": 0.5},
    }

    # Evaluate key slices (not all — too expensive)
    eval_slices = ["last_turn", "all_turns", "hist_0", "hist_1", "hist_2",
                   "hist_3", "hist_4plus", "hist_5plus"]

    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS")
    print(f"{'='*70}")

    all_results = {}
    for sname in eval_slices:
        cases = slices[sname]
        if not cases:
            continue
        print(f"\n--- {sname} (n={len(cases)}) ---")
        summary = eval_slice(cases, retriever, configs, metadata,
                             {}, {})  # track_artist/tags not needed for pool eval
        all_results[sname] = summary

        # Print table
        print(f"  {'Config':20s} {'n':>4s} {'hit@20':>7s} {'hit@50':>7s} {'nDCG@20':>8s} "
              f"{'conv':>5s} {'med_rank':>9s} {'CF%':>5s} {'G%':>5s}")
        for cname in configs:
            s = summary[cname]
            mr = f"{s['median_gt_rank']:.0f}" if s['median_gt_rank'] else "-"
            print(f"  {cname:20s} {s['n']:4d} {s['pool_hit_20']:7.4f} {s['pool_hit_50']:7.4f} "
                  f"{s['ndcg_20']:8.4f} {s['conversion']:5.3f} {mr:>9s} "
                  f"{s['cf_coverage']:5.1%} {s['g_coverage']:5.1%}")

    # =====================================================================
    # DIAGNOSIS
    # =====================================================================
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")

    for sname in eval_slices:
        if sname not in all_results:
            continue
        s_cfg = all_results[sname]["cfg0209_ABCD"]
        s_f1 = all_results[sname]["F1_ABCDF"]

        hit50 = s_f1["pool_hit_50"]
        ndcg = s_f1["ndcg_20"]
        conv = s_f1["conversion"]
        cf_cov = s_f1["cf_coverage"]

        if hit50 < 0.25:
            limitation = "RETRIEVAL-LIMITED"
        elif conv < 0.30:
            limitation = "RANKING-LIMITED"
        elif cf_cov < 0.3:
            limitation = "CF-COVERAGE-LIMITED"
        else:
            limitation = "BALANCED"

        f1_lift = s_f1["ndcg_20"] - s_cfg["ndcg_20"]
        print(f"  {sname:15s}  hit@50={hit50:.3f}  nDCG={ndcg:.4f}  conv={conv:.3f}  "
              f"CF={cf_cov:.0%}  {limitation}  F1_lift={f1_lift:+.4f}")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expB1_representative_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
