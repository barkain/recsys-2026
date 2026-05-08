#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R36: Multimodal track embedding source diagnostic.

Test audio-CLAP, lyrics-Qwen, attributes-Qwen as retrieval sources.
For each: build candidate lists from recent played tracks, evaluate hit@k.

No LambdaRank, no lightgbm import (avoid segfault).
"""
from __future__ import annotations

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

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_embeddings():
    """Load all track embedding modalities from HF cache."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]

    track_ids = []
    modalities: dict[str, list] = {
        "audio_clap": [], "lyrics_qwen": [], "attributes_qwen": [],
        "image_siglip": [],
    }
    col_map = {
        "audio-laion_clap": "audio_clap",
        "lyrics-qwen3_embedding_0.6b": "lyrics_qwen",
        "attributes-qwen3_embedding_0.6b": "attributes_qwen",
        "image-siglip2": "image_siglip",
    }

    for item in ds:
        track_ids.append(str(item["track_id"]))
        for col, key in col_map.items():
            modalities[key].append(item[col])

    result: dict[str, object] = {"track_ids": track_ids}
    for key, vecs in modalities.items():
        dim = None
        for v in vecs:
            if v is not None and len(v) > 0:
                dim = len(v)
                break
        if dim is None:
            print(f"  {key}: no valid embeddings found!")
            continue
        arr = np.zeros((len(vecs), dim), dtype=np.float32)
        n_valid = 0
        for i, v in enumerate(vecs):
            if v is not None and len(v) == dim:
                arr[i] = v
                n_valid += 1
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        result[key] = arr / norms
        print(f"  {key}: {arr.shape}, valid={n_valid}/{len(vecs)}")

    result["tid_to_idx"] = {tid: i for i, tid in enumerate(track_ids)}
    return result


def retrieve_max_recent(played, emb_matrix, tid_to_idx, topk=300, recent_k=3):
    """Retrieve by max similarity to recent played tracks."""
    recent = played[-recent_k:]
    recent_idx = [tid_to_idx[t] for t in recent if t in tid_to_idx]
    if not recent_idx:
        return []

    played_set = {tid_to_idx[t] for t in played if t in tid_to_idx}
    recent_embs = emb_matrix[recent_idx]
    sims = emb_matrix @ recent_embs.T
    max_sims = sims.max(axis=1)

    for pi in played_set:
        max_sims[pi] = -np.inf

    top_idx = np.argpartition(-max_sims, topk)[:topk]
    top_idx = top_idx[np.argsort(-max_sims[top_idx])]
    return top_idx.tolist()


def retrieve_avg_recent(played, emb_matrix, tid_to_idx, topk=300, recent_k=3):
    """Retrieve by average embedding of recent tracks."""
    recent = played[-recent_k:]
    recent_idx = [tid_to_idx[t] for t in recent if t in tid_to_idx]
    if not recent_idx:
        return []

    played_set = {tid_to_idx[t] for t in played if t in tid_to_idx}
    avg_emb = emb_matrix[recent_idx].mean(axis=0)
    avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
    sims = emb_matrix @ avg_emb

    for pi in played_set:
        sims[pi] = -np.inf

    top_idx = np.argpartition(-sims, topk)[:topk]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    return top_idx.tolist()


def evaluate_source(cases, source_lists, track_ids, r21_lists, pool_sets, ta, label):
    """Evaluate a source on hist_7 and all cases."""
    n = len(cases)
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]

    print(f"\n  {label}:")
    for subset_name, subset in [("hist_7", h7), ("all", list(range(n)))]:
        row = ""
        for k in [20, 100, 300]:
            hit = sum(1 for i in subset
                     if cases[i]["gt"] in set(track_ids[j] for j in source_lists[i][:k]))
            if k == 20:
                row = f"    {subset_name:<8} @20={hit:>4}"
            else:
                row += f"  @{k}={hit:>4}"
        print(row)

    # Unique GTs vs pool@300
    unique_h7 = sum(1 for i in h7
                    if cases[i]["gt"] in set(track_ids[j] for j in source_lists[i][:300])
                    and cases[i]["gt"] not in pool_sets[i])
    unique_all = sum(1 for i in range(n)
                     if cases[i]["gt"] in set(track_ids[j] for j in source_lists[i][:300])
                     and cases[i]["gt"] not in pool_sets[i])

    # Same/diff artist for hist_7
    h7_same = [i for i in h7 if ta.get(cases[i]["gt"], "") and
               ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
    h7_diff = [i for i in h7 if i not in set(h7_same)]

    same_hit = sum(1 for i in h7_same
                   if cases[i]["gt"] in set(track_ids[j] for j in source_lists[i][:300]))
    diff_hit = sum(1 for i in h7_diff
                   if cases[i]["gt"] in set(track_ids[j] for j in source_lists[i][:300]))

    print(f"    unique_h7_vs_pool: {unique_h7}  unique_all: {unique_all}")
    print(f"    h7 same_artist@300: {same_hit}/{len(h7_same)}  "
          f"diff_artist@300: {diff_hit}/{len(h7_diff)}")

    return {
        "unique_h7": unique_h7, "unique_all": unique_all,
        "same_hit": same_hit, "diff_hit": diff_hit,
        "n_same": len(h7_same), "n_diff": len(h7_diff),
    }


def main():
    t0 = time.time()
    print(f"{ts()} R36: Multimodal Track Embedding Audit")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload["track_artist"]

    with open(R21_OOF) as f:
        r21_lists = json.load(f)

    # Build current pool sets for unique-GT comparison
    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector

    print(f"{ts()} Building ALS + pools...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
    pool_sets = []
    for i, c in enumerate(cases):
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_list = []
        if sv is not None:
            sc = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc[als_track_to_idx[t]] = -np.inf
            top = np.argpartition(-sc, 200)[:200]
            top = top[np.argsort(-sc[top])]
            als_list = [als_track_ids[j] for j in top]
        sl = {"A": payload["src_a"][i], "B": payload["src_b"][i],
              "C": payload["src_c"][i], "D": payload["src_d"][i],
              "F": payload["src_f"][i], "ALS": als_list, "R21": r21_lists[i]}
        pool_sets.append(set(weighted_rrf(sl, sw, topk=300, k=20)))
    del als_factors, als_track_ids, als_track_to_idx

    print(f"\n{ts()} Loading track embeddings...")
    embs = load_track_embeddings()
    track_ids = embs["track_ids"]
    tid_to_idx = embs["tid_to_idx"]
    n = len(cases)

    # Test each modality with different retrieval strategies
    modalities = ["audio_clap", "lyrics_qwen", "attributes_qwen"]
    strategies = [
        ("max_recent3", lambda played, emb: retrieve_max_recent(played, emb, tid_to_idx, 300, 3)),
        ("avg_recent3", lambda played, emb: retrieve_avg_recent(played, emb, tid_to_idx, 300, 3)),
        ("max_recent1", lambda played, emb: retrieve_max_recent(played, emb, tid_to_idx, 300, 1)),
    ]

    results = {}
    for mod_name in modalities:
        emb_matrix: np.ndarray = embs[mod_name]  # type: ignore[assignment]
        print(f"\n{'='*70}")
        print(f"MODALITY: {mod_name} ({emb_matrix.shape[1]}d)")
        print("=" * 70)

        for strat_name, strat_fn in strategies:
            source_lists = []
            for i, c in enumerate(cases):
                played = c["music_turns"]
                if played:
                    source_lists.append(strat_fn(played, emb_matrix))
                else:
                    source_lists.append([])
                if (i + 1) % 2000 == 0:
                    print(f"    {i+1}/{n} retrieved", flush=True)

            key = f"{mod_name}_{strat_name}"
            r = evaluate_source(cases, source_lists, track_ids, r21_lists,
                               pool_sets, ta, key)
            results[key] = r

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print("R36 SUMMARY")
    print(sep)
    print(f"  {'Source':<35} {'uniq_h7':>8} {'uniq_all':>9} "
          f"{'same@300':>10} {'diff@300':>10}")
    print(f"  {'-'*72}")
    for key, r in results.items():
        print(f"  {key:<35} {r['unique_h7']:>8} {r['unique_all']:>9} "
              f"{r['same_hit']}/{r['n_same']:>10} {r['diff_hit']}/{r['n_diff']:>10}")

    out_path = REPO / "exp" / "eval" / "expR36_multimodal_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
