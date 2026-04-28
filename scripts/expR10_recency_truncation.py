#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R10: Recency-truncated retrieval/fusion.

Hypothesis: using too much history dilutes retrieval. B1 showed hist_1 is best
(nDCG 0.178), hist_2/3 degrade. Test whether restricting sources to recent
context improves high-history cases.

Evaluates on last_turn (1000 cases) with per-history-depth breakdown.
No API. No blind.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth
from scripts.tune_postrank_v23 import tokens
from scripts.expF1_cfbpr_retrieval import (
    build_cfbpr_index,
    cfbpr_max_recent,
    weighted_rrf,
)
from scripts.expR5_sequential_retrieval import SessionTransitionGraph
from offline_retrieval_sweep import CachedBM25, load_track_metadata, query_parts
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from datasets import Dataset, DownloadConfig, load_dataset

POOL_K = 50
RRF_K = 20


def ndcg_at_k(predicted, gt_id, k=20):
    for i, tid in enumerate(predicted[:k]):
        if tid == gt_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def load_all_last_turn_cases():
    """Load last-turn cases for all 1000 dev sessions."""
    path = cached_test_arrow_path()
    ds = Dataset.from_file(path)
    gt_map = build_ground_truth(ds)

    cases = []
    for item in ds:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        user_turns = [c for c in convs if c["role"] == "user"]
        last_ut = max(user_turns, key=lambda c: int(c["turn_number"]))
        turn_num = int(last_ut["turn_number"])
        user_query = str(last_ut["content"])
        history = [c for c in convs if int(c["turn_number"]) < turn_num]
        music_turns = [str(c["content"]).strip() for c in history if c["role"] == "music"]
        gt = lookup_ground_truth(gt_map, sid, uid, turn_num)
        if not gt:
            continue
        cases.append({
            "session_id": sid, "user_id": uid,
            "turn_number": turn_num, "user_query": user_query,
            "history": history, "music_turns": music_turns,
            "gt": str(gt), "n_prior_music": len(music_turns),
        })
    return cases


def build_bm25_query_variants(history, user_query, metadata, music_turns):
    """Build BM25 query strings at different truncation levels."""
    # B variants: last_music_meta using different numbers of recent music turns
    variants = {}

    # Current B: last_music_meta (uses last 1 music turn's metadata)
    variants["B_full"] = " ".join(query_parts(history, user_query, metadata, "last_music_meta"))

    # Current C: full history
    variants["C_full"] = " ".join(query_parts(history, user_query, metadata, "full"))

    # C_recent: only last 2 user turns + user_query
    user_msgs = [str(h["content"]) for h in history if h["role"] == "user"]
    recent_user = user_msgs[-2:] if len(user_msgs) >= 2 else user_msgs
    variants["C_recent2"] = " ".join(recent_user + [user_query] * 3)

    # C_query_only: just the current user query (repeated for BM25 weight)
    variants["C_query_only"] = " ".join([user_query] * 5)

    # C_recent1: only last user turn + query
    recent1 = user_msgs[-1:] if user_msgs else []
    variants["C_recent1"] = " ".join(recent1 + [user_query] * 3)

    return variants


def main():
    t0 = time.time()

    print("Loading data...", flush=True)
    cases = load_all_last_turn_cases()
    metadata = load_track_metadata()
    bm25 = CachedBM25()
    print(f"  {len(cases)} last-turn cases")

    # History distribution
    hist_dist = defaultdict(int)
    for c in cases:
        hist_dist[c["n_prior_music"]] += 1
    print(f"  History distribution: {dict(sorted(hist_dist.items()))}")

    # Load retrievers
    print("Loading qwen3 track-sim...", flush=True)
    qwen_sim = TrackSimilarityRetriever(cache_dir="./cache")
    print("Loading CF-BPR...", flush=True)
    cf_ids, cf_vecs, cf_idx = build_cfbpr_index()
    print("Building session graph...", flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    seqs = {}
    for item in ds["train"]:
        sid = str(item["session_id"])
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        seqs[sid] = [str(c["content"]).strip() for c in convs if c["role"] == "music"]
    graph = SessionTransitionGraph(seqs, metadata)

    # =====================================================================
    # Pre-compute per-case source variants
    # =====================================================================
    print("\nPre-computing source variants...", flush=True)

    # BM25 queries at different truncation levels
    all_bm25_queries = {"B_full": [], "C_full": [], "C_recent2": [],
                         "C_query_only": [], "C_recent1": []}
    for c in cases:
        variants = build_bm25_query_variants(c["history"], c["user_query"],
                                              metadata, c["music_turns"])
        for k in all_bm25_queries:
            all_bm25_queries[k].append(variants.get(k, c["user_query"]) or c["user_query"])

    # Batch BM25 retrieval
    bm25_results = {}
    for qname, queries in all_bm25_queries.items():
        print(f"  BM25 {qname} ({len(queries)} queries)...", flush=True)
        bm25_results[qname] = bm25.retrieve_batch(queries, topk=500)

    # A' at different recent_k
    print("  A' variants...", flush=True)
    a_variants = {}
    for recent_k in [1, 2, 3, 5]:
        a_name = f"A_recent{recent_k}"
        a_results = []
        for c in cases:
            played = c["music_turns"]
            a_idxs = [qwen_sim._id_to_idx.get(str(t)) for t in played[-recent_k:]]
            a_idxs = [i for i in a_idxs if i is not None]
            if a_idxs:
                anchor_vecs = qwen_sim.vectors[a_idxs]
                sims = qwen_sim.vectors @ anchor_vecs.T
                scores = sims.max(axis=1)
                exclude = {qwen_sim._id_to_idx[t] for t in played if t in qwen_sim._id_to_idx}
                cap = min(len(scores), 200 + len(exclude))
                cand = np.argpartition(-scores, cap - 1)[:cap]
                cand = cand[np.argsort(-scores[cand])]
                out = [qwen_sim.track_ids[int(ii)] for ii in cand if int(ii) not in exclude][:200]
                a_results.append(out)
            else:
                a_results.append([])
        a_variants[a_name] = a_results
        print(f"    {a_name}: done", flush=True)

    # D: track neighbors (last track only — same for all variants)
    print("  D track neighbors...", flush=True)
    src_d = []
    for c in cases:
        anchor = c["music_turns"][-1] if c["music_turns"] else None
        src_d.append(qwen_sim.track_id_to_neighbors(anchor, topk=200) if anchor else [])

    # F at different recent_k
    print("  F CF-BPR variants...", flush=True)
    f_variants = {}
    for recent_k in [1, 2, 3, 5]:
        f_name = f"F_recent{recent_k}"
        f_results = []
        for c in cases:
            played = c["music_turns"]
            f_results.append(cfbpr_max_recent(played, cf_vecs, cf_idx, cf_ids, recent_k, 200) if played else [])
        f_variants[f_name] = f_results

    # G at different recent_k
    print("  G cooccur variants...", flush=True)
    g_variants = {}
    for recent_k in [1, 2, 3, 5]:
        g_name = f"G_recent{recent_k}"
        g_results = []
        for c in cases:
            played = c["music_turns"]
            # g_session_cooccur already uses last-5; modify for recent_k
            if played:
                truncated = played[-recent_k:]
                g_results.append(graph.g_session_cooccur(truncated, 200))
            else:
                g_results.append([])
        g_variants[g_name] = g_results

    # =====================================================================
    # Define fusion configs to test
    # =====================================================================

    configs = {
        # Baseline: current cfg0209 + F1
        "baseline_ABCDF_r5": {
            "A": ("A_recent5", 1.0), "B": ("B_full", 1.0),
            "C": ("C_full", 1.0), "D": ("D", 0.5), "F": ("F_recent5", 1.0),
        },
        # Baseline + G
        "baseline_ABCDFG": {
            "A": ("A_recent5", 1.0), "B": ("B_full", 1.0),
            "C": ("C_full", 1.0), "D": ("D", 0.5), "F": ("F_recent5", 1.0),
            "G": ("G_recent5", 0.5),
        },

        # === Truncated C variants ===
        "C_queryonly": {
            "A": ("A_recent5", 1.0), "B": ("B_full", 1.0),
            "C": ("C_query_only", 1.0), "D": ("D", 0.5), "F": ("F_recent5", 1.0),
        },
        "C_recent1": {
            "A": ("A_recent5", 1.0), "B": ("B_full", 1.0),
            "C": ("C_recent1", 1.0), "D": ("D", 0.5), "F": ("F_recent5", 1.0),
        },
        "C_recent2": {
            "A": ("A_recent5", 1.0), "B": ("B_full", 1.0),
            "C": ("C_recent2", 1.0), "D": ("D", 0.5), "F": ("F_recent5", 1.0),
        },
        "no_C": {
            "A": ("A_recent5", 1.0), "B": ("B_full", 1.0),
            "D": ("D", 0.5), "F": ("F_recent5", 1.0),
        },
        "C_half_weight": {
            "A": ("A_recent5", 1.0), "B": ("B_full", 1.0),
            "C": ("C_full", 0.25), "D": ("D", 0.5), "F": ("F_recent5", 1.0),
        },

        # === Truncated A/F/G ===
        "recent1_AFG": {
            "A": ("A_recent1", 1.0), "B": ("B_full", 1.0),
            "C": ("C_full", 1.0), "D": ("D", 0.5), "F": ("F_recent1", 1.0),
            "G": ("G_recent1", 0.5),
        },
        "recent2_AFG": {
            "A": ("A_recent2", 1.0), "B": ("B_full", 1.0),
            "C": ("C_full", 1.0), "D": ("D", 0.5), "F": ("F_recent2", 1.0),
            "G": ("G_recent2", 0.5),
        },
        "recent2_AFG_C_recent2": {
            "A": ("A_recent2", 1.0), "B": ("B_full", 1.0),
            "C": ("C_recent2", 1.0), "D": ("D", 0.5), "F": ("F_recent2", 1.0),
            "G": ("G_recent2", 0.5),
        },

        # === Maximum recency ===
        "max_recency": {
            "A": ("A_recent1", 1.0), "B": ("B_full", 1.0),
            "C": ("C_query_only", 1.0), "D": ("D", 0.5), "F": ("F_recent1", 1.0),
            "G": ("G_recent1", 0.5),
        },
    }

    # Source lookup
    def get_source(name, case_idx):
        if name == "D":
            return src_d[case_idx]
        if name.startswith("A_"):
            return a_variants[name][case_idx]
        if name.startswith("F_"):
            return f_variants[name][case_idx]
        if name.startswith("G_"):
            return g_variants[name][case_idx]
        if name.startswith("B_") or name.startswith("C_"):
            return bm25_results[name][case_idx]
        raise ValueError(f"Unknown source: {name}")

    # =====================================================================
    # Evaluate all configs
    # =====================================================================
    print(f"\n{'='*70}")
    print("EVALUATION")
    print(f"{'='*70}")

    n = len(cases)
    all_results = {}

    for cfg_name, cfg_sources in configs.items():
        # Per-case pool construction
        ndcgs = []
        hit20s = []
        hit50s = []
        per_hist = defaultdict(lambda: {"ndcg": [], "hit20": [], "hit50": []})

        for i, c in enumerate(cases):
            sources_dict = {}
            weights_dict = {}
            for src_label, (src_name, weight) in cfg_sources.items():
                sources_dict[src_label] = get_source(src_name, i)
                weights_dict[src_label] = weight

            pool = weighted_rrf(sources_dict, weights_dict, topk=POOL_K, k=RRF_K)
            gt = c["gt"]
            ndcg = ndcg_at_k(pool, gt)
            h20 = gt in pool[:20]
            h50 = gt in pool[:50]
            ndcgs.append(ndcg)
            hit20s.append(h20)
            hit50s.append(h50)

            h = min(c["n_prior_music"], 5)
            hk = f"hist_{h}" if h < 4 else "hist_4plus"
            per_hist[hk]["ndcg"].append(ndcg)
            per_hist[hk]["hit20"].append(h20)
            per_hist[hk]["hit50"].append(h50)

        overall = {
            "ndcg": float(np.mean(ndcgs)),
            "hit20": float(np.mean(hit20s)),
            "hit50": float(np.mean(hit50s)),
            "n": n,
        }
        hist_summary = {}
        for hk in sorted(per_hist.keys()):
            d = per_hist[hk]
            hist_summary[hk] = {
                "ndcg": float(np.mean(d["ndcg"])),
                "hit20": float(np.mean(d["hit20"])),
                "hit50": float(np.mean(d["hit50"])),
                "n": len(d["ndcg"]),
            }

        all_results[cfg_name] = {"overall": overall, "per_hist": hist_summary}

    # Print results
    bl_ndcg = all_results["baseline_ABCDF_r5"]["overall"]["ndcg"]

    print(f"\n  {'Config':30s} {'nDCG':>7s} {'Δ':>7s} {'hit@20':>7s} {'hit@50':>7s}  "
          f"{'h0':>6s} {'h1':>6s} {'h2':>6s} {'h3':>6s} {'h4+':>6s}")
    for cfg_name in configs:
        r = all_results[cfg_name]
        o = r["overall"]
        delta = o["ndcg"] - bl_ndcg
        hist_ndcgs = []
        for hk in ["hist_0", "hist_1", "hist_2", "hist_3", "hist_4plus"]:
            if hk in r["per_hist"]:
                hist_ndcgs.append(f"{r['per_hist'][hk]['ndcg']:.3f}")
            else:
                hist_ndcgs.append("  -  ")
        print(f"  {cfg_name:30s} {o['ndcg']:7.4f} {delta:+7.4f} {o['hit20']:7.4f} {o['hit50']:7.4f}  "
              f"{'  '.join(hist_ndcgs)}")

    # Best config
    best_name = max(all_results, key=lambda k: all_results[k]["overall"]["ndcg"])
    best = all_results[best_name]
    delta_best = best["overall"]["ndcg"] - bl_ndcg

    print(f"\n  Best: {best_name} → nDCG {best['overall']['ndcg']:.4f} (Δ {delta_best:+.4f})")

    if delta_best >= 0.015:
        verdict = "PASS"
    elif delta_best >= 0.005:
        # Check if hist_2plus improves
        bl_h2 = all_results["baseline_ABCDF_r5"]["per_hist"].get("hist_2", {}).get("ndcg", 0)
        bl_h3 = all_results["baseline_ABCDF_r5"]["per_hist"].get("hist_3", {}).get("ndcg", 0)
        best_h2 = best["per_hist"].get("hist_2", {}).get("ndcg", 0)
        best_h3 = best["per_hist"].get("hist_3", {}).get("ndcg", 0)
        if best_h2 > bl_h2 + 0.005 or best_h3 > bl_h3 + 0.005:
            verdict = "PROMISING"
        else:
            verdict = "WEAK"
    else:
        verdict = "FAIL"

    print(f"\n  GATE: {verdict}")

    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expR10_recency_truncation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Artifact: {out_path}")


if __name__ == "__main__":
    main()
