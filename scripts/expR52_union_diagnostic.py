#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R52 Union Diagnostic: overlap/union analysis of Bucket E recovery across sources.

Computes which specific Bucket E cases are recovered by each of 4 source×strategy
combos, then measures pairwise overlap, union sizes, and incremental value of
attributes_qwen and image_siglip beyond audio_clap.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BUCKET_CACHE = REPO / "cache" / "r52_bucket_labels.json"
POOL_CACHE = REPO / "cache" / "r52_pools.pkl"
OUT_PATH = REPO / "exp" / "eval" / "expR52_union_diagnostic.json"

# Only the 3 sources we need
EMB_SOURCES = {
    "audio_clap": "audio-laion_clap",
    "attributes_qwen": "attributes-qwen3_embedding_0.6b",
    "image_siglip": "image-siglip2",
}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_single_embedding(col_name: str, hf_col: str):
    """Load a single embedding modality from HF, L2-normalize."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]

    track_ids = []
    vecs = []
    dim = None
    for item in ds:
        track_ids.append(str(item["track_id"]))
        v = item[hf_col]
        if v is not None and len(v) > 0:
            if dim is None:
                dim = len(v)
            vecs.append(v)
        else:
            vecs.append(None)

    if dim is None:
        print(f"  {col_name}: no valid embeddings found!")
        return None, track_ids, {}, 0

    arr = np.zeros((len(vecs), dim), dtype=np.float32)
    n_valid = 0
    for i, v in enumerate(vecs):
        if v is not None and len(v) == dim:
            arr[i] = v
            if np.any(arr[i] != 0):
                n_valid += 1

    # L2-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    arr = arr / norms

    tid_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    print(f"  {col_name}: {arr.shape}, valid={n_valid}/{len(vecs)} ({n_valid/len(vecs)*100:.1f}%)")

    del ds, vecs
    gc.collect()
    return arr, track_ids, tid_to_idx, n_valid


def retrieve_recent3_avg(played, emb_matrix, tid_to_idx, topk=300):
    """Average embedding of last 3 played tracks."""
    recent = played[-3:]
    recent_idx = [tid_to_idx[t] for t in recent if t in tid_to_idx]
    if not recent_idx:
        return []
    played_set = {tid_to_idx[t] for t in played if t in tid_to_idx}
    avg_emb = emb_matrix[recent_idx].mean(axis=0)
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb = avg_emb / norm
    sims = emb_matrix @ avg_emb
    for pi in played_set:
        sims[pi] = -np.inf
    top = np.argpartition(-sims, topk)[:topk]
    top = top[np.argsort(-sims[top])]
    return top.tolist()


def retrieve_history_recency(played, emb_matrix, tid_to_idx, topk=300):
    """All history tracks weighted by recency: weight = 0.9^(n-i)."""
    if not played:
        return []
    n_played = len(played)
    valid_idx = []
    weights = []
    for pos, t in enumerate(played):
        if t in tid_to_idx:
            valid_idx.append(tid_to_idx[t])
            weights.append(0.9 ** (n_played - 1 - pos))
    if not valid_idx:
        return []
    played_set = {tid_to_idx[t] for t in played if t in tid_to_idx}
    hist_embs = emb_matrix[valid_idx]
    w = np.array(weights, dtype=np.float32)
    w = w / w.sum()
    weighted_emb = (hist_embs * w[:, None]).sum(axis=0)
    norm = np.linalg.norm(weighted_emb)
    if norm > 0:
        weighted_emb = weighted_emb / norm
    sims = emb_matrix @ weighted_emb
    for pi in played_set:
        sims[pi] = -np.inf
    top = np.argpartition(-sims, topk)[:topk]
    top = top[np.argsort(-sims[top])]
    return top.tolist()


def main():
    t0 = time.time()
    print(f"{ts()} R52 Union Diagnostic: overlap/union of Bucket E recovery")
    print("=" * 70)

    # --- Load cached data ---
    with open(BUCKET_CACHE) as f:
        bucket_data = json.load(f)
    bucket_labels = bucket_data["bucket_labels"]
    h7_indices = bucket_data["h7_indices"]

    with open(POOL_CACHE, "rb") as f:
        pool_data = pickle.load(f)
    cases_h7_played = pool_data["cases_h7_played"]
    cases_h7_gt = pool_data["cases_h7_gt"]
    ta = pool_data["track_artist"]

    # Bucket E case indices
    bucket_e_indices = [i for i in h7_indices if bucket_labels[str(i)]["bucket"] == "E"]
    print(f"  Bucket E cases: {len(bucket_e_indices)}")

    # --- Run the 4 source×strategy combos ---
    # We need: audio_clap × {history_recency, recent3_avg}
    #          attributes_qwen × recent3_avg
    #          image_siglip × history_recency

    combos = [
        ("audio_hr",  "audio_clap",      "history_recency", retrieve_history_recency),
        ("audio_r3a", "audio_clap",      "recent3_avg",     retrieve_recent3_avg),
        ("attr_r3a",  "attributes_qwen", "recent3_avg",     retrieve_recent3_avg),
        ("image_hr",  "image_siglip",    "history_recency", retrieve_history_recency),
    ]

    # Group by embedding source to avoid loading the same source twice
    source_combos: dict[str, list] = {}
    for short, src, strat, fn in combos:
        source_combos.setdefault(src, []).append((short, strat, fn))

    # recovered_sets[short_name] = set of Bucket E case indices recovered
    recovered_sets: dict[str, set[int]] = {}

    for src_name, src_combos_list in source_combos.items():
        hf_col = EMB_SOURCES[src_name]
        print(f"\n{ts()} Loading embeddings: {src_name}")
        emb_matrix, track_ids, tid_to_idx, n_valid = load_single_embedding(src_name, hf_col)
        if emb_matrix is None:
            for short, _, _ in src_combos_list:
                recovered_sets[short] = set()
            continue

        for short, strat, fn in src_combos_list:
            print(f"  Running {short} ({src_name} × {strat})...")
            st = time.time()
            recovered = set()

            for case_i in bucket_e_indices:
                played = cases_h7_played[str(case_i)]
                gt = cases_h7_gt[str(case_i)]
                top_indices = fn(played, emb_matrix, tid_to_idx, topk=300)
                top_tids = set(track_ids[j] for j in top_indices)
                if gt in top_tids:
                    recovered.add(case_i)

            recovered_sets[short] = recovered
            print(f"    recovered {len(recovered)} E cases in {time.time()-st:.1f}s")

        del emb_matrix, track_ids, tid_to_idx
        gc.collect()

    # --- Analysis ---
    print(f"\n{ts()} ANALYSIS")
    print("=" * 70)

    # Per-source E sets
    print("\n--- Per-source Bucket E sets ---")
    for short in ["audio_hr", "audio_r3a", "attr_r3a", "image_hr"]:
        s = recovered_sets[short]
        print(f"  {short}: {len(s)} cases  indices={sorted(s)}")

    per_source_count = {short: len(recovered_sets[short])
                        for short in ["audio_hr", "audio_r3a", "attr_r3a", "image_hr"]}

    # Pairwise overlap
    print("\n--- Pairwise overlap ---")
    pairs = [
        ("audio_hr", "audio_r3a"),
        ("audio_hr", "attr_r3a"),
        ("audio_hr", "image_hr"),
        ("audio_r3a", "attr_r3a"),
        ("audio_r3a", "image_hr"),
        ("attr_r3a", "image_hr"),
    ]
    pairwise = {}
    for a, b in pairs:
        overlap = len(recovered_sets[a] & recovered_sets[b])
        key = f"{a}_x_{b}"
        pairwise[key] = overlap
        print(f"  |{a} ∩ {b}| = {overlap}")

    # Union recoveries
    print("\n--- Union recoveries ---")
    audio_only = recovered_sets["audio_hr"] | recovered_sets["audio_r3a"]
    audio_attr = audio_only | recovered_sets["attr_r3a"]
    audio_image = audio_only | recovered_sets["image_hr"]
    all_four = audio_only | recovered_sets["attr_r3a"] | recovered_sets["image_hr"]

    unique_r3a_beyond_hr = len(recovered_sets["audio_r3a"] - recovered_sets["audio_hr"])
    incr_attr = len(audio_attr) - len(audio_only)
    incr_image = len(audio_image) - len(audio_only)
    incr_all = len(all_four) - len(audio_only)

    union_sizes = {
        "audio_only": len(audio_only),
        "audio_attr": len(audio_attr),
        "audio_image": len(audio_image),
        "all_four": len(all_four),
    }

    print(f"  audio_only (audio_hr ∪ audio_r3a): {len(audio_only)}  "
          f"(unique r3a beyond hr: {unique_r3a_beyond_hr})")
    print(f"  audio+attr: {len(audio_attr)}  (incremental: +{incr_attr})")
    print(f"  audio+image: {len(audio_image)}  (incremental: +{incr_image})")
    print(f"  all 4: {len(all_four)}  (incremental beyond audio: +{incr_all})")

    # Incremental analysis with diff-artist check
    print("\n--- Incremental analysis ---")
    attr_incr_set = recovered_sets["attr_r3a"] - audio_only
    image_incr_set = recovered_sets["image_hr"] - audio_only

    def diff_artist_pct(case_set):
        if not case_set:
            return 0.0
        diff_count = 0
        for ci in case_set:
            gt = cases_h7_gt[str(ci)]
            played = cases_h7_played[str(ci)]
            gt_artist = ta.get(gt, "")
            last_artist = ta.get(played[-1], "") if played else ""
            if not gt_artist or not last_artist or gt_artist != last_artist:
                diff_count += 1
        return diff_count / len(case_set) * 100

    attr_diff_pct = diff_artist_pct(attr_incr_set)
    image_diff_pct = diff_artist_pct(image_incr_set)

    print(f"  attr incremental beyond audio: {len(attr_incr_set)} cases  "
          f"diff_artist={attr_diff_pct:.1f}%")
    print(f"    indices: {sorted(attr_incr_set)}")
    print(f"  image incremental beyond audio: {len(image_incr_set)} cases  "
          f"diff_artist={image_diff_pct:.1f}%")
    print(f"    indices: {sorted(image_incr_set)}")

    # Decision
    print("\n--- Decision ---")
    if len(attr_incr_set) >= 10 and attr_diff_pct >= 70 and len(image_incr_set) >= 10 and image_diff_pct >= 70:
        recommendation = "test multimodal with attr+image"
    elif len(attr_incr_set) >= 10 and attr_diff_pct >= 70:
        recommendation = "test multimodal with attr"
    elif len(image_incr_set) >= 10 and image_diff_pct >= 70:
        recommendation = "test multimodal with image"
    else:
        recommendation = "ignore attr/image"
    print(f"  Recommendation: {recommendation}")

    # Total Bucket E and recovery ceiling
    print(f"\n  Total Bucket E: {len(bucket_e_indices)}")
    print(f"  Best union recovery: {len(all_four)}/{len(bucket_e_indices)} "
          f"({len(all_four)/len(bucket_e_indices)*100:.1f}%)")

    # --- Save results ---
    output = {
        "experiment": "R52_union_diagnostic",
        "timestamp": datetime.now().isoformat(),
        "total_bucket_E": len(bucket_e_indices),
        "per_source_E_count": per_source_count,
        "per_source_E_indices": {k: sorted(v) for k, v in recovered_sets.items()},
        "pairwise_overlap": pairwise,
        "union_sizes": union_sizes,
        "union_details": {
            "audio_hr_unique_beyond_hr_alone": unique_r3a_beyond_hr,
            "audio_only_to_audio_attr_incr": incr_attr,
            "audio_only_to_audio_image_incr": incr_image,
            "audio_only_to_all_four_incr": incr_all,
        },
        "incremental_beyond_audio": {
            "attr": len(attr_incr_set),
            "image": len(image_incr_set),
            "attr_indices": sorted(attr_incr_set),
            "image_indices": sorted(image_incr_set),
            "attr_diff_artist_pct": round(attr_diff_pct, 1),
            "image_diff_artist_pct": round(image_diff_pct, 1),
        },
        "recommendation": recommendation,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n{ts()} Saved to {OUT_PATH}")
    print(f"  Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
