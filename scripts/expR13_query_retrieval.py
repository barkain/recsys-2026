#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R13: Query-to-track semantic retrieval using Qwen3-Embedding-0.6B.

Stage 1: Standalone diagnostic — embed user queries, search track embeddings.
Stage 2: Fusion eval with LambdaRank.

No API. No blind.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank_grouped import als_session_vector, build_als, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
TRACK_IDS_PATH = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "track_ids.json"
TRACK_VECS_PATH = REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "vectors.npy"
RRF_K = 20


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


_EMBED_MODEL = None

def embed_queries(queries, batch_size=32):
    """Embed queries with Qwen3-Embedding-0.6B on CPU (float16)."""
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from sentence_transformers import SentenceTransformer
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch.float16},
            device="cpu",
        )
    embeddings = _EMBED_MODEL.encode(queries, batch_size=batch_size,
                                     normalize_embeddings=True, show_progress_bar=True)
    return embeddings.astype(np.float32)


def retrieve_by_query(query_embs, track_vecs, track_ids, played_lists, topk=200):
    """For each query, find nearest tracks by cosine similarity."""
    results = []
    for i in range(len(query_embs)):
        scores = track_vecs @ query_embs[i]
        played_set = set(played_lists[i])
        # Exclude played tracks
        for tid_idx, tid in enumerate(track_ids):
            if tid in played_set:
                scores[tid_idx] = -np.inf
        top_idx = np.argpartition(-scores, topk)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results.append([track_ids[j] for j in top_idx])
    return results


EMB_CACHE_DIR = REPO_ROOT / "cache" / "r13_query_emb"


def embed_queries_cached():
    """Embed all queries in a subprocess to avoid memory conflicts with implicit."""
    import subprocess
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    current_path = EMB_CACHE_DIR / "emb_current.npy"
    context_path = EMB_CACHE_DIR / "emb_context.npy"
    if current_path.exists() and context_path.exists():
        print(f"  Using cached embeddings from {EMB_CACHE_DIR}", flush=True)
        return np.load(current_path), np.load(context_path)

    print(f"  Launching embedding subprocess...", flush=True)
    script = f'''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle, json, time, numpy as np, torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

REPO = Path("{REPO_ROOT}")
with open(REPO / "exp/eval/_R12_all_turns_payload.pkl", "rb") as f:
    cases = pickle.load(f)["cases"]

queries_current, queries_context = [], []
for c in cases:
    q = c["user_query"]
    queries_current.append(q)
    hist = [str(r["content"]) for r in c["history"] if r["role"] == "user"]
    queries_context.append((hist[-1] + " " + q) if hist else q)

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True,
                            model_kwargs={{"torch_dtype": torch.float16}}, device=device)
print(f"Device: {{device}}", flush=True)
print(f"Embedding {{len(queries_current)}} current queries...", flush=True)
t0 = time.time()
emb_c = model.encode(queries_current, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
print(f"  {{time.time()-t0:.1f}}s ({{len(queries_current)/(time.time()-t0):.1f}} q/s)", flush=True)

print(f"Embedding {{len(queries_context)}} context queries...", flush=True)
t0 = time.time()
emb_x = model.encode(queries_context, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
print(f"  {{time.time()-t0:.1f}}s ({{len(queries_context)/(time.time()-t0):.1f}} q/s)", flush=True)

out = Path("{EMB_CACHE_DIR}")
np.save(out / "emb_current.npy", emb_c.astype(np.float32))
np.save(out / "emb_context.npy", emb_x.astype(np.float32))
print("Saved.", flush=True)
'''
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True, text=True, timeout=3600,
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr}", flush=True)
        raise RuntimeError(f"Embedding subprocess failed with code {result.returncode}")
    return np.load(current_path), np.load(context_path)


def main():
    t0 = time.time()

    # Phase 1: Embed queries in isolated subprocess (no implicit loaded)
    print(f"{ts()} PHASE 1: Query embedding (isolated subprocess)", flush=True)
    t_emb = time.time()
    emb_current, emb_context = embed_queries_cached()
    emb_time = time.time() - t_emb
    n_q = len(emb_current)
    print(f"  {n_q} queries embedded in {emb_time:.1f}s", flush=True)

    # Phase 2: Load data and build pools (implicit can now load safely)
    print(f"\n{ts()} PHASE 2: Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    track_ids = json.load(open(TRACK_IDS_PATH))
    track_vecs = np.load(TRACK_VECS_PATH)
    track_id_set = set(track_ids)
    print(f"  {n} cases, {len(track_ids)} tracks in catalog")

    ta = payload["track_artist"]
    tt = payload["track_tags"]

    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_vecs.append(sv)
        if sv is not None:
            scores = als_factors @ sv
            played_idx = [als_track_to_idx[t] for t in played if t in als_track_to_idx]
            for idx in played_idx:
                scores[idx] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    print(f"{ts()} Building ABCDF+ALS@200 pools...", flush=True)
    base_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}
    abcdf_pools = []
    for i in range(n):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
        }
        pool = weighted_rrf(src_lists, base_weights, topk=200, k=RRF_K)
        abcdf_pools.append(set(pool))

    track_pop = build_popularity_stats()

    # Phase 3: Retrieve and evaluate
    print(f"\n{ts()} PHASE 3: Query-to-track retrieval & evaluation")

    played_lists = [c["music_turns"] for c in cases]

    # Retrieve
    print(f"{ts()} Retrieving top-200 (current query)...", flush=True)
    q_results_current = retrieve_by_query(emb_current, track_vecs, track_ids, played_lists, topk=200)
    print(f"{ts()} Retrieving top-200 (context query)...", flush=True)
    q_results_context = retrieve_by_query(emb_context, track_vecs, track_ids, played_lists, topk=200)

    # Compute baseline diff-artist pool_hit for gate comparison
    base_diff_artist_hit = 0
    diff_artist_total = 0
    base_hist0_hit = 0
    hist0_total = 0
    base_pop0_hit = 0
    pop0_total = 0
    for i, c in enumerate(cases):
        gt = c["gt"]
        played = c["music_turns"]
        n_hist = len(played)
        if played:
            last_artist = ta.get(played[-1], "")
            gt_artist = ta.get(gt, "")
            if gt_artist and last_artist and gt_artist != last_artist:
                diff_artist_total += 1
                if gt in abcdf_pools[i]:
                    base_diff_artist_hit += 1
        if n_hist == 0:
            hist0_total += 1
            if gt in abcdf_pools[i]:
                base_hist0_hit += 1
        if track_pop.get(gt, 0) == 0:
            pop0_total += 1
            if gt in abcdf_pools[i]:
                base_pop0_hit += 1

    base_diff_artist_rate = base_diff_artist_hit / diff_artist_total if diff_artist_total else 0
    base_hist0_rate = base_hist0_hit / hist0_total if hist0_total else 0
    base_pop0_rate = base_pop0_hit / pop0_total if pop0_total else 0
    print(f"\n{ts()} Baselines (ABCDF+ALS@200):")
    print(f"  diff-artist pool_hit: {base_diff_artist_hit}/{diff_artist_total} ({base_diff_artist_rate:.1%})")
    print(f"  hist_0 pool_hit:      {base_hist0_hit}/{hist0_total} ({base_hist0_rate:.1%})")
    print(f"  pop=0 pool_hit:       {base_pop0_hit}/{pop0_total} ({base_pop0_rate:.1%})")

    # Evaluate both variants
    stage1_results = {}
    for variant_name, q_results in [("Q_current", q_results_current),
                                     ("Q_context", q_results_context)]:
        print(f"\n{ts()} === {variant_name} ===")

        hit20 = hit50 = hit100 = hit200 = 0
        unique_vs_pool = 0
        unique_unreachable = 0
        pop0_recovery = 0
        diff_artist_recovery = 0
        hist0_recovery = 0

        # Fusion pool_hit (Q added to ABCDF+ALS)
        fused_diff_artist_hit = 0
        fused_hist0_hit = 0
        fused_pop0_hit = 0

        overlap_A = []
        overlap_D = []

        hist_buckets = defaultdict(lambda: {"n": 0, "hit200": 0, "unique": 0, "unreachable": 0})

        for i, c in enumerate(cases):
            gt = c["gt"]
            played = c["music_turns"]
            n_hist = len(played)
            bk = f"hist_{min(n_hist, 7)}"
            hist_buckets[bk]["n"] += 1

            q_set = set(q_results[i][:200])
            fused_pool = abcdf_pools[i] | q_set
            gt_hit = gt in q_set

            if gt in q_results[i][:20]: hit20 += 1
            if gt in q_results[i][:50]: hit50 += 1
            if gt in q_results[i][:100]: hit100 += 1
            if gt_hit:
                hit200 += 1
                hist_buckets[bk]["hit200"] += 1

                if gt not in abcdf_pools[i]:
                    unique_vs_pool += 1
                    hist_buckets[bk]["unique"] += 1

                    in_any_source = False
                    for sname in ["src_a", "src_b", "src_c", "src_d", "src_f"]:
                        if gt in payload[sname][i][:500]:
                            in_any_source = True
                            break
                    if not in_any_source and gt not in als_source[i][:500]:
                        unique_unreachable += 1
                        hist_buckets[bk]["unreachable"] += 1

                if track_pop.get(gt, 0) == 0:
                    pop0_recovery += 1

                if played:
                    last_artist = ta.get(played[-1], "")
                    gt_artist = ta.get(gt, "")
                    if gt_artist and last_artist and gt_artist != last_artist:
                        diff_artist_recovery += 1

                if n_hist == 0:
                    hist0_recovery += 1

            # Fused pool_hit by category
            if played:
                last_artist = ta.get(played[-1], "")
                gt_artist = ta.get(gt, "")
                if gt_artist and last_artist and gt_artist != last_artist:
                    if gt in fused_pool:
                        fused_diff_artist_hit += 1
            if n_hist == 0:
                if gt in fused_pool:
                    fused_hist0_hit += 1
            if track_pop.get(gt, 0) == 0:
                if gt in fused_pool:
                    fused_pop0_hit += 1

            a_set = set(payload["src_a"][i][:200])
            d_set = set(payload["src_d"][i][:200])
            if q_set:
                overlap_A.append(len(q_set & a_set) / len(q_set))
                overlap_D.append(len(q_set & d_set) / len(q_set))

        fused_diff_rate = fused_diff_artist_hit / diff_artist_total if diff_artist_total else 0
        fused_hist0_rate = fused_hist0_hit / hist0_total if hist0_total else 0
        fused_pop0_rate = fused_pop0_hit / pop0_total if pop0_total else 0
        diff_artist_lift = fused_diff_rate - base_diff_artist_rate

        print(f"  Standalone hit@20:  {hit20}/{n} ({hit20/n:.1%})")
        print(f"  Standalone hit@50:  {hit50}/{n} ({hit50/n:.1%})")
        print(f"  Standalone hit@100: {hit100}/{n} ({hit100/n:.1%})")
        print(f"  Standalone hit@200: {hit200}/{n} ({hit200/n:.1%})")
        print(f"  Unique vs ABCDF+ALS@200: {unique_vs_pool}")
        print(f"  Unique among unreachable: {unique_unreachable}")
        print(f"  Pop=0 GT recovery: {pop0_recovery}")
        print(f"  Different-artist recovery: {diff_artist_recovery}")
        print(f"  hist_0 recovery: {hist0_recovery}")
        print(f"  Overlap with A': {np.mean(overlap_A):.3f}")
        print(f"  Overlap with D:  {np.mean(overlap_D):.3f}")
        print(f"\n  Fused pool_hit (ABCDF+ALS+Q):")
        print(f"    diff-artist: {fused_diff_artist_hit}/{diff_artist_total} "
              f"({fused_diff_rate:.1%}, lift={diff_artist_lift:+.1%})")
        print(f"    hist_0:      {fused_hist0_hit}/{hist0_total} ({fused_hist0_rate:.1%}, "
              f"lift={fused_hist0_rate - base_hist0_rate:+.1%})")
        print(f"    pop=0:       {fused_pop0_hit}/{pop0_total} ({fused_pop0_rate:.1%}, "
              f"lift={fused_pop0_rate - base_pop0_rate:+.1%})")

        print(f"\n  By history depth:")
        print(f"    {'bucket':10s} {'n':>5s} {'hit@200':>8s} {'unique':>7s} {'unreach':>8s}")
        for bk in sorted(hist_buckets.keys()):
            d = hist_buckets[bk]
            print(f"    {bk:10s} {d['n']:5d} {d['hit200']/d['n']:8.1%} "
                  f"{d['unique']:7d} {d['unreachable']:8d}")

        # Gate checks
        gate_unreachable = unique_unreachable >= 150
        gate_diff_artist = diff_artist_lift >= 0.03
        mean_overlap_ad = (np.mean(overlap_A) + np.mean(overlap_D)) / 2
        gate_not_redundant = mean_overlap_ad < 0.5

        print(f"\n  GATES:")
        print(f"    unique unreachable >= 150:        {'PASS' if gate_unreachable else 'FAIL'} ({unique_unreachable})")
        print(f"    diff-artist pool_hit lift >= +3%: {'PASS' if gate_diff_artist else 'FAIL'} ({diff_artist_lift:+.1%})")
        print(f"    not redundant (A/D overlap < 50%):{'PASS' if gate_not_redundant else 'FAIL'} ({mean_overlap_ad:.1%})")

        stage1_results[variant_name] = {
            "hit20": hit20, "hit50": hit50, "hit100": hit100, "hit200": hit200,
            "unique_vs_pool": unique_vs_pool, "unique_unreachable": unique_unreachable,
            "pop0_recovery": pop0_recovery, "diff_artist_recovery": diff_artist_recovery,
            "hist0_recovery": hist0_recovery,
            "overlap_A": float(np.mean(overlap_A)), "overlap_D": float(np.mean(overlap_D)),
            "fused_diff_artist_rate": fused_diff_rate,
            "diff_artist_lift": diff_artist_lift,
            "gates": {"unreachable": gate_unreachable, "diff_artist": gate_diff_artist,
                      "not_redundant": gate_not_redundant},
        }

    elapsed = time.time() - t0
    print(f"\n{ts()} Stage 1 complete. Elapsed: {elapsed:.1f}s")

    any_pass = any(all(r["gates"].values()) for r in stage1_results.values())
    print(f"\nAny variant passes all gates: {'YES → proceed to Stage 2' if any_pass else 'NO → stop here'}")

    out_path = REPO_ROOT / "exp" / "eval" / "expR13_query_retrieval.json"
    with open(out_path, "w") as f:
        json.dump({"stage": "stage1", "results": stage1_results,
                   "baselines": {"diff_artist_pool_hit": base_diff_artist_rate,
                                 "hist0_pool_hit": base_hist0_rate,
                                 "pop0_pool_hit": base_pop0_rate},
                   "elapsed_s": elapsed}, f, indent=2, default=str)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
