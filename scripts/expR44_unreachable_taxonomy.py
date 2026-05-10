#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R44: Taxonomy of Bucket E (unreachable) cases.

GT is not sourced by any current retriever (A, B, C, D, F, ALS, R21).
This is the largest miss bucket (25.7% of hist_7 misses in R42).
Goal: understand what these GTs look like and whether any existing or
feasible retriever could reach them.
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

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector
from scripts.expS2_lr_v2 import build_popularity_stats

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_albums():
    """Load track_id -> album_id mapping."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_album = {}
    for item in ds:
        tid = str(item["track_id"])
        alb_id = item.get("album_id", [])
        if isinstance(alb_id, list) and alb_id:
            track_album[tid] = str(alb_id[0])
        else:
            alb_name = item.get("album_name", [])
            if isinstance(alb_name, list) and alb_name:
                track_album[tid] = str(alb_name[0])
            else:
                track_album[tid] = ""
    return track_album


def load_track_artists_from_meta():
    """Load track_id -> artist_name mapping from metadata."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_artist_meta = {}
    for item in ds:
        tid = str(item["track_id"])
        artist = item.get("artist_name", [])
        if isinstance(artist, list) and artist:
            track_artist_meta[tid] = str(artist[0]).lower().strip()
        else:
            track_artist_meta[tid] = str(artist).lower().strip() if artist else ""
    return track_artist_meta


def load_training_track_set():
    """Get all track_ids that appear in training sessions."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    train = ds["train"]
    seen = set()
    for item in train:
        for c in item["conversations"]:
            if c["role"] == "music":
                seen.add(str(c["content"]).strip())
    return seen


def main():
    t0 = time.time()
    print(f"{ts()} R44: Unreachable (Bucket E) Taxonomy")
    print("=" * 70)

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    print(f"{ts()} Loading payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload["track_artist"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    print(f"{ts()} Loading popularity stats...")
    track_pop = build_popularity_stats()

    print(f"{ts()} Loading album mapping...")
    track_album = load_track_albums()

    print(f"{ts()} Loading artist metadata...")
    track_artist_meta = load_track_artists_from_meta()

    print(f"{ts()} Loading training track set...")
    training_tracks = load_training_track_set()
    print(f"  {len(training_tracks)} unique tracks seen in training")

    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
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
        else:
            als_source.append([])

    # -------------------------------------------------------------------
    # Step 1: Identify Bucket E cases (hist_7 only)
    # -------------------------------------------------------------------
    n = len(cases)
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    print(f"\n{ts()} Identifying Bucket E among {len(h7)} hist_7 cases...")

    bucket_a_cases = []  # GT in pool and ranked top-20 (approximate: GT in pool)
    bucket_e_cases = []
    bucket_other_cases = []  # D or in-pool misses

    for i in h7:
        c = cases[i]
        gt = c["gt"]

        # Build source union (full source lists, not just pool@300)
        src_union = set()
        src_union.update(payload["src_a"][i])
        src_union.update(payload["src_b"][i])
        src_union.update(payload["src_c"][i])
        src_union.update(payload["src_d"][i])
        src_union.update(payload["src_f"][i])
        src_union.update(als_source[i])
        src_union.update(r21_source[i])

        if gt not in src_union:
            bucket_e_cases.append(i)
        else:
            # Check if GT is in pool@300 for hit vs D classification
            src_lists = {
                "A": payload["src_a"][i], "B": payload["src_b"][i],
                "C": payload["src_c"][i], "D": payload["src_d"][i],
                "F": payload["src_f"][i], "ALS": als_source[i],
                "R21": r21_source[i],
            }
            pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
            if gt in pool:
                bucket_a_cases.append(i)  # in-pool (A/B/C)
            else:
                bucket_other_cases.append(i)  # D cases

    print(f"  Bucket E (unreachable): {len(bucket_e_cases)} ({len(bucket_e_cases)/len(h7)*100:.1f}%)")
    print(f"  In-pool (A/B/C):        {len(bucket_a_cases)} ({len(bucket_a_cases)/len(h7)*100:.1f}%)")
    print(f"  Bucket D (source miss): {len(bucket_other_cases)} ({len(bucket_other_cases)/len(h7)*100:.1f}%)")

    # -------------------------------------------------------------------
    # Step 2: GT properties analysis for Bucket E
    # -------------------------------------------------------------------
    print(f"\n{ts()} Analyzing {len(bucket_e_cases)} Bucket E GT properties...")

    e_props = {
        "same_artist": 0, "diff_artist": 0,
        "same_album": 0,
        "seen_in_training": 0, "unseen": 0,
        "artist_in_query": 0,
        "n_history_tracks": [],
        "gt_popularity": [],
    }

    # Per-case details for actionability
    e_case_details = []

    for i in bucket_e_cases:
        c = cases[i]
        gt = c["gt"]
        played = c["music_turns"]
        query = c["user_query"]

        # same_artist / diff_artist
        played_artists = {ta.get(t, "") for t in played} - {""}
        gt_artist = ta.get(gt, "")
        is_same_artist = bool(gt_artist and gt_artist in played_artists)
        if is_same_artist:
            e_props["same_artist"] += 1
        else:
            e_props["diff_artist"] += 1

        # same_album
        played_album_ids = {track_album.get(t, "") for t in played} - {""}
        gt_album = track_album.get(gt, "")
        is_same_album = bool(gt_album and gt_album in played_album_ids)
        if is_same_album:
            e_props["same_album"] += 1

        # seen_in_training
        is_seen = gt in training_tracks
        if is_seen:
            e_props["seen_in_training"] += 1
        else:
            e_props["unseen"] += 1

        # popularity
        pop = track_pop.get(gt, 0)
        e_props["gt_popularity"].append(pop)

        # n_history_tracks
        e_props["n_history_tracks"].append(len(played))

        # artist_in_query
        gt_artist_name = track_artist_meta.get(gt, "")
        is_artist_in_query = bool(
            gt_artist_name and len(gt_artist_name) > 2 and gt_artist_name in query.lower()
        )
        if is_artist_in_query:
            e_props["artist_in_query"] += 1

        e_case_details.append({
            "case_idx": i,
            "gt": gt,
            "same_artist": is_same_artist,
            "same_album": is_same_album,
            "seen_in_training": is_seen,
            "popularity": pop,
            "artist_in_query": is_artist_in_query,
        })

    # -------------------------------------------------------------------
    # Step 3: Extended source check
    # -------------------------------------------------------------------
    print(f"\n{ts()} Extended source check for Bucket E...")

    # Source sizes:
    #   src_a: 200, src_b: 500, src_c: 500, src_d: 200, src_f: 200
    #   ALS: 200, R21: 300
    # These are the FULL lists in the payload — no way to extend without
    # re-running the retriever. Report what we have.

    # Check if GT appears in individual full source lists
    e_in_source = Counter()
    for i in bucket_e_cases:
        gt = cases[i]["gt"]
        if gt in payload["src_a"][i]:
            e_in_source["A_full"] += 1
        if gt in payload["src_b"][i]:
            e_in_source["B_full"] += 1
        if gt in payload["src_c"][i]:
            e_in_source["C_full"] += 1
        if gt in payload["src_d"][i]:
            e_in_source["D_full"] += 1
        if gt in payload["src_f"][i]:
            e_in_source["F_full"] += 1
        if gt in als_source[i]:
            e_in_source["ALS_full"] += 1
        if gt in r21_source[i]:
            e_in_source["R21_full"] += 1

    # By definition, all should be 0 (we already checked union)
    print("  Sanity check — GT in individual sources (should all be 0):")
    for s in ["A_full", "B_full", "C_full", "D_full", "F_full", "ALS_full", "R21_full"]:
        print(f"    {s}: {e_in_source[s]}")

    # Check if GT is in ALS catalog at all (has embedding)
    e_in_als_catalog = sum(1 for i in bucket_e_cases
                           if cases[i]["gt"] in als_track_to_idx)
    print(f"\n  GT in ALS catalog (has embedding): {e_in_als_catalog}/{len(bucket_e_cases)} "
          f"({e_in_als_catalog/len(bucket_e_cases)*100:.1f}%)")

    # Check if GT is in the full track metadata catalog
    e_in_meta = sum(1 for i in bucket_e_cases
                    if cases[i]["gt"] in track_artist_meta)
    print(f"  GT in track metadata:              {e_in_meta}/{len(bucket_e_cases)} "
          f"({e_in_meta/len(bucket_e_cases)*100:.1f}%)")

    print("\n  Source list sizes in payload:")
    print("    A (cf-bpr):  max 200 items")
    print("    B (BM25):    max 500 items")
    print("    C (BM25):    max 500 items")
    print("    D (cf-bpr):  max 200 items")
    print("    F (cf-bpr):  max 200 items")
    print("    ALS:         max 200 items")
    print("    R21 (dense): max 300 items")
    print("  Note: GT not in any of these full lists. Extending to top-1000")
    print("  would require re-running each retriever.")

    # -------------------------------------------------------------------
    # Step 4: Distribution analysis
    # -------------------------------------------------------------------
    print(f"\n{ts()} Distribution analysis")
    sep = "=" * 70
    print(sep)

    ne = len(bucket_e_cases)
    print(f"\n{'Property':<25s} | {'Count':>6s} | {'% of E':>7s}")
    print("-" * 45)
    print(f"{'seen_in_training':<25s} | {e_props['seen_in_training']:>6d} | {e_props['seen_in_training']/ne*100:>6.1f}%")
    print(f"{'unseen':<25s} | {e_props['unseen']:>6d} | {e_props['unseen']/ne*100:>6.1f}%")
    print(f"{'same_artist':<25s} | {e_props['same_artist']:>6d} | {e_props['same_artist']/ne*100:>6.1f}%")
    print(f"{'diff_artist':<25s} | {e_props['diff_artist']:>6d} | {e_props['diff_artist']/ne*100:>6.1f}%")
    print(f"{'same_album':<25s} | {e_props['same_album']:>6d} | {e_props['same_album']/ne*100:>6.1f}%")
    print(f"{'artist_in_query':<25s} | {e_props['artist_in_query']:>6d} | {e_props['artist_in_query']/ne*100:>6.1f}%")
    print(f"{'in_als_catalog':<25s} | {e_in_als_catalog:>6d} | {e_in_als_catalog/ne*100:>6.1f}%")

    # Popularity distribution: E-case GTs vs hit-case GTs (bucket A)
    e_pops = np.array(e_props["gt_popularity"])
    a_pops = np.array([track_pop.get(cases[i]["gt"], 0) for i in bucket_a_cases])

    print("\nPopularity distribution (raw count in training):")
    print(f"{'Stat':<12s} | {'E (unreachable)':>16s} | {'A (hit)':>16s}")
    print("-" * 50)
    for stat_name, fn in [("median", np.median), ("mean", np.mean),
                           ("p25", lambda x: np.percentile(x, 25)),
                           ("p75", lambda x: np.percentile(x, 75)),
                           ("p90", lambda x: np.percentile(x, 90)),
                           ("max", np.max), ("zero_pop", lambda x: np.sum(x == 0))]:
        if len(e_pops) > 0:
            e_val = fn(e_pops)
        else:
            e_val = 0
        if len(a_pops) > 0:
            a_val = fn(a_pops)
        else:
            a_val = 0
        print(f"{stat_name:<12s} | {e_val:>16.1f} | {a_val:>16.1f}")

    # n_history_tracks distribution (should all be 7 for hist_7)
    hist_arr = np.array(e_props["n_history_tracks"])
    print(f"\nn_history_tracks: min={hist_arr.min()}, max={hist_arr.max()}, "
          f"mean={hist_arr.mean():.1f} (should be ~7 for hist_7)")

    # -------------------------------------------------------------------
    # Step 5: Actionability assessment
    # -------------------------------------------------------------------
    print(f"\n{ts()} Actionability assessment")
    print(sep)

    cf_recoverable = 0      # seen + same_artist
    dense_recoverable = 0   # seen + diff_artist
    cold_start = 0          # unseen + diff_artist
    catalog_gap = 0         # unseen + not popular (pop=0)

    # Extended categories
    seen_same_album = 0     # seen + same_album
    artist_query_unseen = 0 # artist_in_query but unseen

    for d in e_case_details:
        seen = d["seen_in_training"]
        same_art = d["same_artist"]
        pop = d["popularity"]
        same_alb = d["same_album"]
        art_q = d["artist_in_query"]

        if seen and same_art:
            cf_recoverable += 1
        elif seen and not same_art:
            dense_recoverable += 1
        elif not seen and not same_art:
            if pop == 0:
                catalog_gap += 1
            else:
                cold_start += 1
        elif not seen and same_art:
            # Unseen but same artist — content-based could help
            cold_start += 1

        if seen and same_alb:
            seen_same_album += 1
        if art_q and not seen:
            artist_query_unseen += 1

    print(f"\n{'Category':<35s} | {'Count':>6s} | {'% of E':>7s} | Approach")
    print("-" * 100)
    print(f"{'CF-recoverable (seen+same_art)':<35s} | {cf_recoverable:>6d} | {cf_recoverable/ne*100:>6.1f}% | Better CF model (ALS, BPR) should surface these")
    print(f"{'Dense-recoverable (seen+diff_art)':<35s} | {dense_recoverable:>6d} | {dense_recoverable/ne*100:>6.1f}% | Broader supervised dense retriever (beyond R21)")
    print(f"{'Cold-start (unseen+pop>0)':<35s} | {cold_start:>6d} | {cold_start/ne*100:>6.1f}% | Content-based retrieval (audio, tags, text)")
    print(f"{'Catalog-gap (unseen+pop=0)':<35s} | {catalog_gap:>6d} | {catalog_gap/ne*100:>6.1f}% | Fundamental coverage issue, hard to fix")
    print(f"{'---':<35s} | {'---':>6s} | {'---':>7s} |")
    print(f"{'TOTAL':<35s} | {ne:>6d} | {'100.0':>6s}% |")

    print("\n  Additional cross-cuts:")
    print(f"    seen + same_album:       {seen_same_album:>4d} — album-based retriever could help")
    print(f"    artist_in_query + unseen: {artist_query_unseen:>4d} — query mentions GT artist but track unseen")

    # -------------------------------------------------------------------
    # Step 6: Detailed sub-population analysis
    # -------------------------------------------------------------------
    print(f"\n{ts()} Detailed sub-population analysis")
    print(sep)

    # CF-recoverable details: why didn't ALS/cf-bpr find them?
    cf_in_als = sum(1 for d in e_case_details
                    if d["seen_in_training"] and d["same_artist"]
                    and d["gt"] in als_track_to_idx)
    print(f"\n  CF-recoverable ({cf_recoverable} cases):")
    print(f"    GT has ALS embedding: {cf_in_als}/{cf_recoverable} — "
          f"these are in the CF space but scored too low for top-200")
    cf_not_als = cf_recoverable - cf_in_als
    if cf_not_als > 0:
        print(f"    GT missing ALS embedding: {cf_not_als} — "
              f"not in co-occurrence matrix")

    # Dense-recoverable: what are the popularity levels?
    dense_pops = [d["popularity"] for d in e_case_details
                  if d["seen_in_training"] and not d["same_artist"]]
    if dense_pops:
        dense_arr = np.array(dense_pops)
        print(f"\n  Dense-recoverable ({dense_recoverable} cases):")
        print(f"    Popularity: median={np.median(dense_arr):.0f}, "
              f"mean={np.mean(dense_arr):.1f}, p25={np.percentile(dense_arr, 25):.0f}, "
              f"p75={np.percentile(dense_arr, 75):.0f}")
        dense_has_als = sum(1 for d in e_case_details
                            if d["seen_in_training"] and not d["same_artist"]
                            and d["gt"] in als_track_to_idx)
        print(f"    GT has ALS embedding: {dense_has_als}/{dense_recoverable}")

    # Cold-start: popularity distribution
    cold_pops = [d["popularity"] for d in e_case_details
                 if not d["seen_in_training"] and d["popularity"] > 0]
    if cold_pops:
        cold_arr = np.array(cold_pops)
        print(f"\n  Cold-start ({cold_start} cases, pop>0):")
        print(f"    Popularity: median={np.median(cold_arr):.0f}, "
              f"mean={np.mean(cold_arr):.1f}, p25={np.percentile(cold_arr, 25):.0f}, "
              f"p75={np.percentile(cold_arr, 75):.0f}")

    # Artist overlap analysis: for same_artist E cases, how many tracks
    # does the GT artist have in training?
    print(f"\n  Artist analysis for same_artist E cases ({e_props['same_artist']} cases):")
    same_art_artist_track_counts = []
    for d in e_case_details:
        if d["same_artist"]:
            gt_art = ta.get(d["gt"], "")
            if gt_art:
                # Count how many training tracks this artist has
                art_count = sum(1 for tid in training_tracks if ta.get(tid, "") == gt_art)
                same_art_artist_track_counts.append(art_count)
    if same_art_artist_track_counts:
        sac = np.array(same_art_artist_track_counts)
        print(f"    Artist training tracks: median={np.median(sac):.0f}, "
              f"mean={np.mean(sac):.1f}, min={np.min(sac)}, max={np.max(sac)}")

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print(f"\n{ts()} SUMMARY")
    print(sep)
    print(f"Total hist_7 cases:  {len(h7)}")
    print(f"Bucket E (unreachable): {ne} ({ne/len(h7)*100:.1f}% of hist_7)")
    print("\nRecoverability breakdown:")
    print(f"  CF-recoverable:    {cf_recoverable:>4d} ({cf_recoverable/ne*100:.1f}%) — plausible with better CF")
    print(f"  Dense-recoverable: {dense_recoverable:>4d} ({dense_recoverable/ne*100:.1f}%) — needs broader dense retriever")
    print(f"  Cold-start:        {cold_start:>4d} ({cold_start/ne*100:.1f}%) — needs content-based")
    print(f"  Catalog-gap:       {catalog_gap:>4d} ({catalog_gap/ne*100:.1f}%) — hard/impossible")

    theoretically_recoverable = cf_recoverable + dense_recoverable
    print(f"\n  Theoretically recoverable (CF + Dense): {theoretically_recoverable} "
          f"({theoretically_recoverable/ne*100:.1f}% of E, "
          f"{theoretically_recoverable/len(h7)*100:.1f}% of h7)")
    print(f"  Fundamentally hard (cold + gap):         {cold_start + catalog_gap} "
          f"({(cold_start+catalog_gap)/ne*100:.1f}% of E)")

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    out_data = {
        "h7_count": len(h7),
        "bucket_e_count": ne,
        "bucket_e_pct_of_h7": round(ne / len(h7) * 100, 1),
        "bucket_a_count": len(bucket_a_cases),
        "bucket_d_count": len(bucket_other_cases),
        "properties": {
            "seen_in_training": e_props["seen_in_training"],
            "unseen": e_props["unseen"],
            "same_artist": e_props["same_artist"],
            "diff_artist": e_props["diff_artist"],
            "same_album": e_props["same_album"],
            "artist_in_query": e_props["artist_in_query"],
            "in_als_catalog": e_in_als_catalog,
        },
        "popularity": {
            "e_median": float(np.median(e_pops)) if len(e_pops) > 0 else 0,
            "e_mean": float(np.mean(e_pops)) if len(e_pops) > 0 else 0,
            "e_p25": float(np.percentile(e_pops, 25)) if len(e_pops) > 0 else 0,
            "e_p75": float(np.percentile(e_pops, 75)) if len(e_pops) > 0 else 0,
            "a_median": float(np.median(a_pops)) if len(a_pops) > 0 else 0,
            "a_mean": float(np.mean(a_pops)) if len(a_pops) > 0 else 0,
            "a_p25": float(np.percentile(a_pops, 25)) if len(a_pops) > 0 else 0,
            "a_p75": float(np.percentile(a_pops, 75)) if len(a_pops) > 0 else 0,
        },
        "actionability": {
            "cf_recoverable": cf_recoverable,
            "dense_recoverable": dense_recoverable,
            "cold_start": cold_start,
            "catalog_gap": catalog_gap,
            "theoretically_recoverable": theoretically_recoverable,
        },
        "source_sizes": {
            "A": 200, "B": 500, "C": 500, "D": 200, "F": 200,
            "ALS": 200, "R21": 300,
        },
    }

    out_path = REPO / "exp" / "eval" / "expR44_unreachable_taxonomy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
