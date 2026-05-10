#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R45: Diagnostic analysis of 104 dense-recoverable (seen+diff_artist) cases from Bucket E.

GT is seen in training, GT artist differs from session artists, but no
current retriever surfaces the GT.  Goal: understand *why* these are missed
and classify the miss reason to guide next retrieval improvements.

No lightgbm — analysis only.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R21_MODEL_DIR = REPO / "cache" / "r21_production" / "model"
R21_TRACK_EMBS = REPO / "cache" / "r21_production" / "track_embeddings.npy"
R21_TRACK_IDS = REPO / "cache" / "r21_production" / "track_ids.json"
R21_DEV_QUERY_EMBS = REPO / "cache" / "r21_production" / "dev_query_embeddings.npy"

RRF_K = 20
POOL_K = 300
SW = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


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


def load_training_sessions():
    """Load training sessions as list of lists of track_ids."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    train = ds["train"]
    sessions = []
    for item in train:
        tracks = []
        for c in item["conversations"]:
            if c["role"] == "music":
                tracks.append(str(c["content"]).strip())
        if tracks:
            sessions.append(tracks)
    return sessions


def load_track_metadata():
    """Load full track metadata for tag overlap analysis."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    meta = {}
    for item in ds:
        tid = str(item["track_id"])
        tags = item.get("tag_list", [])
        if isinstance(tags, list):
            tag_set = set(str(t).lower().strip() for t in tags if t)
        else:
            tag_set = set()
        artist = item.get("artist_name", [])
        if isinstance(artist, list) and artist:
            artist_name = str(artist[0]).lower().strip()
        else:
            artist_name = str(artist).lower().strip() if artist else ""
        album_id = item.get("album_id", [])
        if isinstance(album_id, list) and album_id:
            alb = str(album_id[0])
        else:
            alb_name = item.get("album_name", [])
            if isinstance(alb_name, list) and alb_name:
                alb = str(alb_name[0])
            else:
                alb = ""
        # release_date
        rd = item.get("release_date", [])
        if isinstance(rd, list) and rd:
            release_date = str(rd[0])
        else:
            release_date = str(rd) if rd else ""
        meta[tid] = {
            "tags": tag_set,
            "artist": artist_name,
            "album": alb,
            "release_date": release_date,
        }
    return meta


def main():
    t0 = time.time()
    print(f"{ts()} R45: Dense-Recoverable Diagnostic Analysis", flush=True)
    print("=" * 70, flush=True)

    # -------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------
    print(f"{ts()} Loading payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload["track_artist"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    print(f"{ts()} Loading training track set...", flush=True)
    training_tracks = load_training_track_set()
    print(f"  {len(training_tracks)} unique tracks seen in training", flush=True)

    print(f"{ts()} Building ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()

    als_source = []
    als_vecs = []
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

    # -------------------------------------------------------------------
    # Step 1: Identify dense-recoverable cases
    # -------------------------------------------------------------------
    n = len(cases)
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    print(f"\n{ts()} Identifying dense-recoverable among {len(h7)} hist_7 cases...", flush=True)

    dense_recoverable_cases = []

    for i in h7:
        c = cases[i]
        gt = c["gt"]
        played = c["music_turns"]

        # Build source union
        src_union = set()
        src_union.update(payload["src_a"][i])
        src_union.update(payload["src_b"][i])
        src_union.update(payload["src_c"][i])
        src_union.update(payload["src_d"][i])
        src_union.update(payload["src_f"][i])
        src_union.update(als_source[i])
        src_union.update(r21_source[i])

        if gt in src_union:
            continue  # Not Bucket E

        # Check seen in training
        if gt not in training_tracks:
            continue  # Not seen

        # Check diff_artist
        played_artists = set()
        for t in played:
            a = ta.get(t, "")
            if isinstance(a, list):
                for aa in a:
                    if aa:
                        played_artists.add(aa.lower().strip() if isinstance(aa, str) else str(aa).lower().strip())
            elif a:
                played_artists.add(a.lower().strip() if isinstance(a, str) else str(a).lower().strip())

        gt_artist_raw = ta.get(gt, "")
        if isinstance(gt_artist_raw, list):
            gt_artist = gt_artist_raw[0].lower().strip() if gt_artist_raw else ""
        else:
            gt_artist = str(gt_artist_raw).lower().strip() if gt_artist_raw else ""

        if gt_artist and gt_artist in played_artists:
            continue  # same_artist, not dense-recoverable

        dense_recoverable_cases.append(i)

    print(f"  Found {len(dense_recoverable_cases)} dense-recoverable cases", flush=True)

    # -------------------------------------------------------------------
    # Step 2: Extended source rank analysis
    # -------------------------------------------------------------------
    print(f"\n{ts()} Step 2: Extended source rank analysis", flush=True)
    print("=" * 70, flush=True)

    # Load R21 embeddings for full ranking
    print(f"{ts()} Loading R21 track embeddings...", flush=True)
    r21_track_embs = np.load(R21_TRACK_EMBS)  # (N_tracks, D)
    with open(R21_TRACK_IDS) as f:
        r21_track_ids = json.load(f)
    r21_tid_to_idx = {tid: i for i, tid in enumerate(r21_track_ids)}

    # Load cached dev query embeddings
    print(f"{ts()} Loading dev query embeddings...", flush=True)
    dev_query_embs = np.load(R21_DEV_QUERY_EMBS)  # (N_cases, D)
    print(f"  Track embs: {r21_track_embs.shape}, Query embs: {dev_query_embs.shape}", flush=True)

    # Results storage
    r21_ranks = []  # R21 full rank of GT (1-indexed)
    r21_oof_present = []  # GT in R21 OOF top-300?
    als_ranks = []  # ALS full rank of GT (1-indexed)

    for progress_idx, case_idx in enumerate(dense_recoverable_cases):
        if progress_idx % 10 == 0:
            print(f"{ts()}   Processing case {progress_idx+1}/{len(dense_recoverable_cases)} "
                  f"(case_idx={case_idx})...", flush=True)

        c = cases[case_idx]
        gt = c["gt"]
        played = c["music_turns"]

        # --- R21 OOF check ---
        r21_oof_list = r21_source[case_idx]
        r21_oof_present.append(gt in r21_oof_list)

        # --- R21 full rank ---
        gt_r21_idx = r21_tid_to_idx.get(gt)
        if gt_r21_idx is not None and case_idx < len(dev_query_embs):
            q_emb = dev_query_embs[case_idx]  # (D,)
            scores_r21 = r21_track_embs @ q_emb  # (N_tracks,)
            # Rank (1-indexed)
            gt_score = scores_r21[gt_r21_idx]
            rank_r21 = int(np.sum(scores_r21 > gt_score)) + 1
            r21_ranks.append(rank_r21)
        else:
            r21_ranks.append(-1)  # GT not in R21 catalog

        # --- ALS full rank ---
        sv = als_vecs[case_idx]
        gt_als_idx = als_track_to_idx.get(gt)
        if sv is not None and gt_als_idx is not None:
            scores_als = als_factors @ sv
            # Mask played tracks
            for t in played:
                if t in als_track_to_idx:
                    scores_als[als_track_to_idx[t]] = -np.inf
            gt_score_als = scores_als[gt_als_idx]
            rank_als = int(np.sum(scores_als > gt_score_als)) + 1
            als_ranks.append(rank_als)
        else:
            als_ranks.append(-1)

    r21_ranks_arr = np.array(r21_ranks)
    als_ranks_arr = np.array(als_ranks)
    r21_oof_hit_count = int(sum(r21_oof_present))

    # Print R21 rank distribution
    print(f"\n{'='*70}", flush=True)
    print("R21 full-catalog rank distribution for dense-recoverable cases:", flush=True)
    valid_r21 = r21_ranks_arr[r21_ranks_arr > 0]
    bins = [
        ("rank 1-300", 1, 300),
        ("rank 301-500", 301, 500),
        ("rank 501-1000", 501, 1000),
        ("rank 1001-5000", 1001, 5000),
        ("rank 5001-10000", 5001, 10000),
        ("rank 10000+", 10001, 999999),
    ]
    for label, lo, hi in bins:
        cnt = int(np.sum((valid_r21 >= lo) & (valid_r21 <= hi)))
        print(f"  {label:<20s}: {cnt:>4d}", flush=True)
    not_in_catalog = int(np.sum(r21_ranks_arr < 0))
    print(f"  {'not in R21 catalog':<20s}: {not_in_catalog:>4d}", flush=True)
    if len(valid_r21) > 0:
        print(f"  Stats: median={np.median(valid_r21):.0f}, "
              f"mean={np.mean(valid_r21):.0f}, "
              f"min={np.min(valid_r21)}, max={np.max(valid_r21)}", flush=True)

    # Print ALS rank distribution
    print("\nALS full-catalog rank distribution:", flush=True)
    valid_als = als_ranks_arr[als_ranks_arr > 0]
    for label, lo, hi in bins:
        cnt = int(np.sum((valid_als >= lo) & (valid_als <= hi)))
        print(f"  {label:<20s}: {cnt:>4d}", flush=True)
    not_in_als = int(np.sum(als_ranks_arr < 0))
    print(f"  {'not in ALS catalog':<20s}: {not_in_als:>4d}", flush=True)
    if len(valid_als) > 0:
        print(f"  Stats: median={np.median(valid_als):.0f}, "
              f"mean={np.mean(valid_als):.0f}, "
              f"min={np.min(valid_als)}, max={np.max(valid_als)}", flush=True)

    # R21 OOF sanity
    print(f"\nR21 OOF (top-300) contains GT: {r21_oof_hit_count}/{len(r21_oof_present)} "
          f"(should be 0 by definition)", flush=True)

    # -------------------------------------------------------------------
    # Step 3: Training co-occurrence analysis
    # -------------------------------------------------------------------
    print(f"\n{ts()} Step 3: Training co-occurrence analysis", flush=True)
    print("=" * 70, flush=True)

    print(f"{ts()} Loading training sessions...", flush=True)
    train_sessions = load_training_sessions()
    print(f"  {len(train_sessions)} training sessions loaded", flush=True)

    # Build co-occurrence index: track -> set of session indices
    print(f"{ts()} Building co-occurrence index...", flush=True)
    track_to_sessions = defaultdict(set)
    for si, session in enumerate(train_sessions):
        for tid in session:
            track_to_sessions[tid].add(si)

    # For each dense-recoverable case, check co-occurrence
    gt_session_counts = []
    covisit_any = []  # Does GT co-occur with any played track?
    covisit_fractions = []  # Fraction of played tracks that co-occur with GT
    covisit_counts = []  # Number of played tracks that co-occur with GT

    for case_idx in dense_recoverable_cases:
        c = cases[case_idx]
        gt = c["gt"]
        played = c["music_turns"]

        # How many training sessions contain GT?
        gt_sessions = track_to_sessions.get(gt, set())
        gt_session_counts.append(len(gt_sessions))

        # Do any played tracks co-occur with GT?
        n_covisit = 0
        for t in played:
            t_sessions = track_to_sessions.get(t, set())
            if gt_sessions & t_sessions:
                n_covisit += 1

        covisit_any.append(n_covisit > 0)
        covisit_fractions.append(n_covisit / len(played) if played else 0)
        covisit_counts.append(n_covisit)

    gt_sc = np.array(gt_session_counts)
    covisit_any_arr = np.array(covisit_any)
    covisit_frac_arr = np.array(covisit_fractions)

    print("\nGT training session count distribution:", flush=True)
    print(f"  min={gt_sc.min()}, max={gt_sc.max()}, median={np.median(gt_sc):.0f}, "
          f"mean={np.mean(gt_sc):.1f}", flush=True)
    session_bins = [(0, 0), (1, 1), (2, 5), (6, 20), (21, 100), (101, 99999)]
    for lo, hi in session_bins:
        cnt = int(np.sum((gt_sc >= lo) & (gt_sc <= hi)))
        if hi == 99999:
            label = f"  sessions {lo}+"
        elif lo == hi:
            label = f"  sessions = {lo}"
        else:
            label = f"  sessions {lo}-{hi}"
        print(f"{label:<25s}: {cnt:>4d}", flush=True)

    print("\nCo-visitation:", flush=True)
    print(f"  GT co-occurs with >=1 played track: {int(covisit_any_arr.sum())} cases", flush=True)
    print(f"  GT co-occurs with  0 played tracks: {int((~covisit_any_arr).sum())} cases", flush=True)
    if covisit_any_arr.sum() > 0:
        cv_fracs = covisit_frac_arr[covisit_any_arr]
        print(f"  When co-visiting: mean fraction of played tracks = {cv_fracs.mean():.3f}", flush=True)
        cv_cnts = np.array(covisit_counts)[covisit_any_arr]
        print(f"  When co-visiting: mean count = {cv_cnts.mean():.1f}, "
              f"median = {np.median(cv_cnts):.0f}", flush=True)

    # -------------------------------------------------------------------
    # Step 4: Nearest-track relation
    # -------------------------------------------------------------------
    print(f"\n{ts()} Step 4: Nearest-track relation analysis", flush=True)
    print("=" * 70, flush=True)

    print(f"{ts()} Loading full track metadata...", flush=True)
    track_meta = load_track_metadata()

    # Also build artist co-occurrence from training: artist -> set of co-occurring artists
    print(f"{ts()} Building artist co-occurrence from training...", flush=True)
    session_artist_sets = []
    for session in train_sessions:
        artists_in_session = set()
        for tid in session:
            a = track_meta.get(tid, {}).get("artist", "")
            if a:
                artists_in_session.add(a)
        session_artist_sets.append(artists_in_session)

    artist_cooccur = defaultdict(set)
    for artists_in_session in session_artist_sets:
        for a in artists_in_session:
            artist_cooccur[a].update(artists_in_session)
    # Remove self
    for a in artist_cooccur:
        artist_cooccur[a].discard(a)

    tag_jaccard_maxs = []
    tag_jaccard_means = []
    same_album_cases = []
    same_decade_cases = []
    artist_cooccur_cases = []

    for case_idx in dense_recoverable_cases:
        c = cases[case_idx]
        gt = c["gt"]
        played = c["music_turns"]

        gt_m = track_meta.get(gt, {})
        gt_tags = gt_m.get("tags", set())
        gt_album = gt_m.get("album", "")
        gt_artist = gt_m.get("artist", "")
        gt_rd = gt_m.get("release_date", "")

        # Tag overlap
        jaccards = []
        for t in played:
            t_m = track_meta.get(t, {})
            t_tags = t_m.get("tags", set())
            if gt_tags or t_tags:
                j = len(gt_tags & t_tags) / len(gt_tags | t_tags) if (gt_tags | t_tags) else 0
            else:
                j = 0
            jaccards.append(j)

        if jaccards:
            tag_jaccard_maxs.append(max(jaccards))
            tag_jaccard_means.append(np.mean(jaccards))
        else:
            tag_jaccard_maxs.append(0)
            tag_jaccard_means.append(0)

        # Same album check
        played_albums = {track_meta.get(t, {}).get("album", "") for t in played} - {""}
        is_same_album = bool(gt_album and gt_album in played_albums)
        same_album_cases.append(is_same_album)

        # Same decade check
        gt_decade = gt_rd[:3] if len(gt_rd) >= 3 else ""
        played_decades = set()
        for t in played:
            t_rd = track_meta.get(t, {}).get("release_date", "")
            if len(t_rd) >= 3:
                played_decades.add(t_rd[:3])
        same_decade_cases.append(bool(gt_decade and gt_decade in played_decades))

        # Artist co-occurrence check
        played_artists = set()
        for t in played:
            a = track_meta.get(t, {}).get("artist", "")
            if a:
                played_artists.add(a)

        gt_cooccur = artist_cooccur.get(gt_artist, set())
        has_artist_link = bool(gt_cooccur & played_artists)
        artist_cooccur_cases.append(has_artist_link)

    tj_max = np.array(tag_jaccard_maxs)
    tj_mean = np.array(tag_jaccard_means)

    print("\nTag Jaccard (GT vs played tracks):", flush=True)
    print(f"  Max:  mean={tj_max.mean():.4f}, median={np.median(tj_max):.4f}, "
          f"p75={np.percentile(tj_max, 75):.4f}, p90={np.percentile(tj_max, 90):.4f}", flush=True)
    print(f"  Mean: mean={tj_mean.mean():.4f}, median={np.median(tj_mean):.4f}", flush=True)
    tj_zero = int(np.sum(tj_max == 0))
    print(f"  Cases with max Jaccard = 0: {tj_zero}/{len(tj_max)}", flush=True)

    print(f"\nSame album (GT album matches any played): {sum(same_album_cases)}/{len(same_album_cases)}", flush=True)
    print(f"Same decade: {sum(same_decade_cases)}/{len(same_decade_cases)}", flush=True)
    print("Artist co-occurrence (GT artist appears in training with a played artist): "
          f"{sum(artist_cooccur_cases)}/{len(artist_cooccur_cases)}", flush=True)

    # -------------------------------------------------------------------
    # Step 5: Classification of miss reason
    # -------------------------------------------------------------------
    print(f"\n{ts()} Step 5: Miss reason classification", flush=True)
    print("=" * 70, flush=True)

    classifications = []
    for idx_in_list, case_idx in enumerate(dense_recoverable_cases):
        r21_rank = r21_ranks[idx_in_list]
        als_rank = als_ranks[idx_in_list]
        has_covisit = covisit_any[idx_in_list]

        # Primary classification based on R21 rank
        if 0 < r21_rank <= 500:
            primary = "r21_shallow"
        elif 0 < r21_rank <= 5000:
            primary = "r21_deep"
        elif r21_rank > 5000 or r21_rank < 0:
            primary = "r21_miss"
        else:
            primary = "r21_miss"

        # ALS classification
        if 0 < als_rank <= 500:
            als_class = "als_shallow"
        else:
            als_class = "als_deep"

        # Co-visitation
        has_cv = has_covisit

        # Artist link
        has_art_link = artist_cooccur_cases[idx_in_list]

        # Composite classification (pick the most actionable)
        if primary == "r21_shallow":
            classification = "r21_shallow"
        elif als_class == "als_shallow":
            classification = "als_shallow"
        elif primary == "r21_deep":
            classification = "r21_deep"
        elif has_cv:
            classification = "co_visit"
        elif has_art_link:
            classification = "artist_bridge"
        else:
            classification = "no_bridge"

        classifications.append(classification)

    class_counts = Counter(classifications)
    print("\nMiss reason classification:", flush=True)
    for cls in ["r21_shallow", "als_shallow", "r21_deep", "co_visit", "artist_bridge", "no_bridge"]:
        cnt = class_counts.get(cls, 0)
        pct = cnt / len(classifications) * 100 if classifications else 0
        print(f"  {cls:<20s}: {cnt:>4d} ({pct:>5.1f}%)", flush=True)

    # -------------------------------------------------------------------
    # Step 6: Detailed output
    # -------------------------------------------------------------------
    print(f"\n{ts()} Step 6: Summary", flush=True)
    print("=" * 70, flush=True)

    print(f"\nTotal dense-recoverable cases: {len(dense_recoverable_cases)}", flush=True)

    print("\nR21 rank distribution:", flush=True)
    for label, lo, hi in bins:
        cnt = int(np.sum((valid_r21 >= lo) & (valid_r21 <= hi)))
        print(f"  {label:<20s}: {cnt:>4d}", flush=True)
    print(f"  {'not in catalog':<20s}: {not_in_catalog:>4d}", flush=True)

    print("\nALS rank distribution:", flush=True)
    for label, lo, hi in bins:
        cnt = int(np.sum((valid_als >= lo) & (valid_als <= hi)))
        print(f"  {label:<20s}: {cnt:>4d}", flush=True)
    print(f"  {'not in catalog':<20s}: {not_in_als:>4d}", flush=True)

    print("\nCo-visitation:", flush=True)
    print(f"  GT co-occurs with >=1 played track: {int(covisit_any_arr.sum())} cases", flush=True)
    print(f"  GT co-occurs with  0 played tracks: {int((~covisit_any_arr).sum())} cases", flush=True)

    print("\nMiss reason classification:", flush=True)
    for cls in ["r21_shallow", "als_shallow", "r21_deep", "co_visit", "artist_bridge", "no_bridge"]:
        cnt = class_counts.get(cls, 0)
        pct = cnt / len(classifications) * 100 if classifications else 0
        print(f"  {cls:<20s}: {cnt:>4d} ({pct:>5.1f}%)", flush=True)

    # Actionability summary
    shallow = class_counts.get("r21_shallow", 0) + class_counts.get("als_shallow", 0)
    deep = class_counts.get("r21_deep", 0) + class_counts.get("co_visit", 0) + class_counts.get("artist_bridge", 0)
    hard = class_counts.get("no_bridge", 0)
    print("\nActionability:", flush=True)
    print(f"  Shallow retrieval fix (r21/als rank 301-500): {shallow}", flush=True)
    print(f"  Needs new signal (r21_deep + co_visit + artist_bridge): {deep}", flush=True)
    print(f"  No bridge (fundamentally hard): {hard}", flush=True)

    # -------------------------------------------------------------------
    # Save per-case details
    # -------------------------------------------------------------------
    per_case = []
    for idx_in_list, case_idx in enumerate(dense_recoverable_cases):
        c = cases[case_idx]
        per_case.append({
            "case_idx": case_idx,
            "gt": c["gt"],
            "r21_rank": r21_ranks[idx_in_list],
            "als_rank": als_ranks[idx_in_list],
            "gt_train_sessions": gt_session_counts[idx_in_list],
            "covisit_any": bool(covisit_any[idx_in_list]),
            "covisit_count": covisit_counts[idx_in_list],
            "tag_jaccard_max": round(tag_jaccard_maxs[idx_in_list], 4),
            "tag_jaccard_mean": round(tag_jaccard_means[idx_in_list], 4),
            "same_album": same_album_cases[idx_in_list],
            "same_decade": same_decade_cases[idx_in_list],
            "artist_cooccur": artist_cooccur_cases[idx_in_list],
            "classification": classifications[idx_in_list],
        })

    out_data = {
        "n_dense_recoverable": len(dense_recoverable_cases),
        "r21_rank_distribution": {
            "rank_1_300": int(np.sum((valid_r21 >= 1) & (valid_r21 <= 300))),
            "rank_301_500": int(np.sum((valid_r21 >= 301) & (valid_r21 <= 500))),
            "rank_501_1000": int(np.sum((valid_r21 >= 501) & (valid_r21 <= 1000))),
            "rank_1001_5000": int(np.sum((valid_r21 >= 1001) & (valid_r21 <= 5000))),
            "rank_5001_10000": int(np.sum((valid_r21 >= 5001) & (valid_r21 <= 10000))),
            "rank_10000_plus": int(np.sum(valid_r21 > 10000)),
            "not_in_catalog": not_in_catalog,
        },
        "r21_rank_stats": {
            "median": float(np.median(valid_r21)) if len(valid_r21) > 0 else None,
            "mean": float(np.mean(valid_r21)) if len(valid_r21) > 0 else None,
            "min": int(np.min(valid_r21)) if len(valid_r21) > 0 else None,
            "max": int(np.max(valid_r21)) if len(valid_r21) > 0 else None,
        },
        "als_rank_distribution": {
            "rank_1_300": int(np.sum((valid_als >= 1) & (valid_als <= 300))),
            "rank_301_500": int(np.sum((valid_als >= 301) & (valid_als <= 500))),
            "rank_501_1000": int(np.sum((valid_als >= 501) & (valid_als <= 1000))),
            "rank_1001_5000": int(np.sum((valid_als >= 1001) & (valid_als <= 5000))),
            "rank_5001_10000": int(np.sum((valid_als >= 5001) & (valid_als <= 10000))),
            "rank_10000_plus": int(np.sum(valid_als > 10000)),
            "not_in_catalog": not_in_als,
        },
        "als_rank_stats": {
            "median": float(np.median(valid_als)) if len(valid_als) > 0 else None,
            "mean": float(np.mean(valid_als)) if len(valid_als) > 0 else None,
            "min": int(np.min(valid_als)) if len(valid_als) > 0 else None,
            "max": int(np.max(valid_als)) if len(valid_als) > 0 else None,
        },
        "covisitation": {
            "gt_cooccurs_with_played": int(covisit_any_arr.sum()),
            "gt_no_cooccurrence": int((~covisit_any_arr).sum()),
            "gt_train_session_count_median": float(np.median(gt_sc)),
            "gt_train_session_count_mean": float(np.mean(gt_sc)),
        },
        "tag_jaccard": {
            "max_mean": float(tj_max.mean()),
            "max_median": float(np.median(tj_max)),
            "mean_mean": float(tj_mean.mean()),
        },
        "relations": {
            "same_album": sum(same_album_cases),
            "same_decade": sum(same_decade_cases),
            "artist_cooccur": sum(artist_cooccur_cases),
        },
        "classification": dict(class_counts),
        "actionability": {
            "shallow_fix": shallow,
            "needs_new_signal": deep,
            "no_bridge": hard,
        },
        "per_case": per_case,
    }

    out_path = REPO / "exp" / "eval" / "expR45_dense_recoverable.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}", flush=True)
    print(f"Elapsed: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
