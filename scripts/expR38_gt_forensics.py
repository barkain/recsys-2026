#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R38: GT forensics — what distinguishes GT from LambdaRank's wrong top candidates.

For hist_7 LR miss-in-pool cases, compare GT vs wrong top-1/top-5 candidates
to find the discriminative feature we're missing.
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

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R34_LR = REPO / "cache" / "r34_residual" / "lr_scores.npy"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    t0 = time.time()
    print(f"{ts()} R38: GT Forensics")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload["track_artist"]
    tt = payload["track_tags"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    lr_scores = np.load(R34_LR)

    # Build pools
    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector

    als_factors, als_track_ids, als_track_to_idx = build_als()
    sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

    n = len(cases)
    h7 = [i for i in range(n) if cases[i]["n_prior_music"] == 7]

    # Load track metadata for deeper analysis
    from datasets import DownloadConfig, load_dataset
    meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                           download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    track_meta: dict[str, dict] = {}
    for item in meta_ds:
        tid = str(item["track_id"])
        track_meta[tid] = {
            "name": item.get("track_name", [""])[0] if isinstance(item.get("track_name"), list) else str(item.get("track_name", "")),
            "artist": ", ".join(item.get("artist_name", [])) if isinstance(item.get("artist_name"), list) else str(item.get("artist_name", "")),
            "album": item.get("album_name", [""])[0] if isinstance(item.get("album_name"), list) else str(item.get("album_name", "")),
            "tags": set(str(t) for t in item.get("tag_list", [])[:20]) if isinstance(item.get("tag_list"), list) else set(),
        }
    del meta_ds

    print(f"{ts()} Analyzing {len(h7)} hist_7 cases...")

    # Categories for GT vs wrong top candidates
    gt_properties = Counter()
    distinguishing_features = Counter()
    examples = []

    for i in h7:
        c = cases[i]
        gt = c["gt"]
        played = c["music_turns"]
        query = c["user_query"]

        # Build pool
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
              "F": payload["src_f"][i], "ALS": als_list, "R21": r21_source[i]}
        pool = weighted_rrf(sl, sw, topk=300, k=20)

        if gt not in pool:
            continue

        gt_pool_idx = pool.index(gt)
        ps = len(pool)
        sc = lr_scores[i, :ps]
        lr_ranked = np.argsort(-sc)
        gt_lr_rank = np.where(lr_ranked == gt_pool_idx)[0]
        if len(gt_lr_rank) == 0:
            continue
        gt_rank = int(gt_lr_rank[0]) + 1

        if gt_rank <= 20:
            continue

        # This is a LR miss case — GT in pool but ranked >20
        wrong_top5_idx = [lr_ranked[j] for j in range(min(5, ps)) if lr_ranked[j] != gt_pool_idx]
        wrong_top1_tid = pool[wrong_top5_idx[0]] if wrong_top5_idx else None

        # GT properties
        gt_meta = track_meta.get(gt, {})
        gt_artist = gt_meta.get("artist", "")
        gt_tags = gt_meta.get("tags", set())
        gt_album = gt_meta.get("album", "")
        gt_name = gt_meta.get("name", "")

        played_artists = {ta.get(t, "") for t in played} - {""}
        last_artist = ta.get(played[-1], "")
        played_tags = set()
        for t in played[-3:]:
            played_tags |= tt.get(t, set())
        played_albums = {track_meta.get(t, {}).get("album", "") for t in played[-3:]} - {""}

        # GT relation to history
        gt_same_last = gt_artist == last_artist and gt_artist
        gt_same_any = gt_artist in played_artists and gt_artist
        gt_same_album = gt_album in played_albums and gt_album
        gt_tag_overlap = len(gt_tags & played_tags) if gt_tags and played_tags else 0
        gt_tag_jaccard = gt_tag_overlap / len(gt_tags | played_tags) if (gt_tags or played_tags) else 0

        # Query mentions
        query_lower = query.lower()
        gt_artist_mentioned = gt_artist.lower() in query_lower if gt_artist else False
        gt_name_mentioned = gt_name.lower() in query_lower if gt_name and len(gt_name) > 3 else False

        # Source ranks
        gt_r21_rank = r21_source[i].index(gt) + 1 if gt in r21_source[i][:300] else -1
        gt_bm25c_rank = payload["src_c"][i].index(gt) + 1 if gt in payload["src_c"][i][:300] else -1

        # Wrong top-1 properties (for comparison)
        w1_meta = track_meta.get(wrong_top1_tid, {}) if wrong_top1_tid else {}
        w1_artist = w1_meta.get("artist", "")
        w1_same_any = w1_artist in played_artists and w1_artist
        w1_tags = w1_meta.get("tags", set())
        w1_tag_jaccard = len(w1_tags & played_tags) / len(w1_tags | played_tags) if (w1_tags or played_tags) else 0

        # Classify the distinguishing feature
        reasons = []
        if gt_same_any and not w1_same_any:
            reasons.append("gt_same_artist_wrong_diff")
        if not gt_same_any and w1_same_any:
            reasons.append("gt_diff_artist_wrong_same")
        if gt_artist_mentioned:
            reasons.append("gt_artist_in_query")
        if gt_name_mentioned:
            reasons.append("gt_name_in_query")
        if gt_tag_jaccard > w1_tag_jaccard + 0.1:
            reasons.append("gt_better_tags")
        if gt_same_album:
            reasons.append("gt_same_album")
        if gt_r21_rank > 0 and gt_r21_rank < 20:
            reasons.append("gt_r21_top20_but_lr_missed")
        if not reasons:
            reasons.append("no_clear_distinction")

        for r in reasons:
            distinguishing_features[r] += 1

        # Track GT properties
        if gt_same_any:
            gt_properties["same_artist"] += 1
        else:
            gt_properties["diff_artist"] += 1
        if gt_same_album:
            gt_properties["same_album"] += 1
        if gt_tag_jaccard > 0.3:
            gt_properties["strong_tag_overlap"] += 1
        if gt_artist_mentioned:
            gt_properties["artist_in_query"] += 1

        if len(examples) < 30:
            examples.append({
                "case_idx": i,
                "gt": gt,
                "gt_artist": gt_artist,
                "gt_name": gt_name,
                "gt_rank": gt_rank,
                "gt_same_any": gt_same_any,
                "gt_same_last": gt_same_last,
                "gt_same_album": gt_same_album,
                "gt_tag_jaccard": round(gt_tag_jaccard, 3),
                "gt_artist_mentioned": gt_artist_mentioned,
                "gt_r21_rank": gt_r21_rank,
                "gt_bm25c_rank": gt_bm25c_rank,
                "wrong_top1": wrong_top1_tid,
                "w1_artist": w1_artist,
                "w1_same_any": w1_same_any,
                "w1_tag_jaccard": round(w1_tag_jaccard, 3),
                "reasons": reasons,
                "query": query[:120],
            })

    # ---------------------------------------------------------------
    # Reports
    # ---------------------------------------------------------------
    sep = "=" * 70
    n_miss = sum(distinguishing_features.values())
    print(f"\n{sep}")
    print(f"R38 GT FORENSICS — {n_miss} LR miss-in-pool cases")
    print(sep)

    print("\nGT Properties (in miss cases):")
    for prop, count in gt_properties.most_common():
        print(f"  {prop:<30} {count:>5} ({count/n_miss*100:.1f}%)")

    print("\nDistinguishing Features (GT vs wrong top-1):")
    for feat, count in distinguishing_features.most_common():
        print(f"  {feat:<40} {count:>5} ({count/n_miss*100:.1f}%)")

    print("\nConcrete Examples (first 20):")
    for ex in examples[:20]:
        print(f"\n  Case {ex['case_idx']} — GT rank {ex['gt_rank']}")
        print(f"    GT: {ex['gt_name']} by {ex['gt_artist']}")
        print(f"    same_artist={ex['gt_same_any']} same_album={ex['gt_same_album']} "
              f"tag_j={ex['gt_tag_jaccard']} mentioned={ex['gt_artist_mentioned']}")
        print(f"    R21={ex['gt_r21_rank']} BM25c={ex['gt_bm25c_rank']}")
        print(f"    Wrong#1: {ex['w1_artist']} same={ex['w1_same_any']} tag_j={ex['w1_tag_jaccard']}")
        print(f"    Reasons: {ex['reasons']}")
        print(f"    Query: {ex['query']}")

    out_path = REPO / "exp" / "eval" / "expR38_gt_forensics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "n_miss": n_miss,
            "gt_properties": dict(gt_properties),
            "distinguishing_features": dict(distinguishing_features),
            "examples": examples,
        }, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
