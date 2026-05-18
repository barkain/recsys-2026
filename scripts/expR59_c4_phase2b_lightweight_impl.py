#!/usr/bin/env python3
"""R59 C4 Phase 2-B: Lightweight 50-case MusicBrainz diagnostic - IMPLEMENTATION.

EXPLORATORY RESEARCH ONLY — NOT SUBMISSION-SAFE.

50-case sample, 2 MB features (tag_jaccard_last, genre_overlap_history), single-fold eval.
Gates at +0.005 nDCG improvement vs baseline.

Strategy:
1. Sample 50 cases from R12 payload (prefer turns 3-7 with history)
2. Load R54 candidate pools for these cases
3. Fetch MB metadata via ISRC (cache + new API calls, 1 req/sec)
4. Compute 2 MB features for each candidate
5. Analyze feature correlation with GT ranking as proxy for signal
6. If promising, recommend full Phase 2 (300 cases, 5 features, CV5, LR retraining)

Output labeled: "EXPLORATORY - NOT SUBMISSION-SAFE - LIGHTWEIGHT 50-CASE MB DIAGNOSTIC"
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path
from collections import Counter

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

# Config
MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "RecsysResearch/0.1.exploratory (barkai.nadav@gmail.com)"
RATE_LIMIT = 1.1  # seconds
MB_CACHE_DIR = REPO_ROOT / "cache" / "musicbrainz_audit"
MB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

PHASE2B_OUTPUT = REPO_ROOT / "exp" / "eval" / "c4_phase2b_lightweight"
PHASE2B_OUTPUT.mkdir(parents=True, exist_ok=True)

EXPLORATORY_LABEL = "EXPLORATORY - NOT SUBMISSION-SAFE - LIGHTWEIGHT 50-CASE MB DIAGNOSTIC"

# R12 payload + R54 pools
R12_PAYLOAD = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R54_POOLS = REPO_ROOT / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"

RANDOM_SEED = 42


def cache_key(endpoint: str, params: dict) -> str:
    """Generate cache filename for MB API request."""
    key_str = f"{endpoint}?{sorted(params.items())}"
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


def fetch_mb_by_isrc(isrc: str) -> dict | None:
    """Fetch MusicBrainz recording by ISRC with caching."""
    if not isrc:
        return None

    cache_file = MB_CACHE_DIR / f"isrc_{cache_key('recording', {'isrc': isrc})}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    # API request
    url = f"{MB_BASE}/recording"
    params = {"query": f"isrc:{isrc}", "fmt": "json", "limit": 1}
    headers = {"User-Agent": USER_AGENT}

    time.sleep(RATE_LIMIT)
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        with open(cache_file, "w") as f:
            json.dump(data, f)

        return data
    except Exception as e:
        print(f"    ✗ MB API error for ISRC {isrc}: {e}")
        return None


def fetch_mb_recording_details(recording_id: str) -> dict | None:
    """Fetch full recording metadata (tags, genres, relations)."""
    cache_file = MB_CACHE_DIR / f"rec_{cache_key('recording', {'id': recording_id})}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"{MB_BASE}/recording/{recording_id}"
    params = {"fmt": "json", "inc": "tags+genres+artist-rels+recording-rels"}
    headers = {"User-Agent": USER_AGENT}

    time.sleep(RATE_LIMIT)
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        with open(cache_file, "w") as f:
            json.dump(data, f)

        return data
    except Exception as e:
        print(f"    ✗ MB API error for recording {recording_id}: {e}")
        return None


def extract_mb_tags_genres(rec_data: dict) -> tuple[list[str], list[str]]:
    """Extract tags and genres from MB recording response."""
    tags = [t["name"].lower() for t in rec_data.get("tags", []) if t.get("count", 0) > 0]
    genres = [g["name"].lower() for g in rec_data.get("genres", [])]
    return tags, genres


def load_track_isrcs():
    """Load track_id → ISRC mapping from TalkPlayData."""
    print("[1/6] Loading TalkPlayData ISRC index...")
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split="all_tracks",
        cache_dir=REPO_ROOT / ".hf_cache"
    )

    tid_to_isrc = {}
    for row in ds:
        tid = str(row["track_id"])
        isrcs = row.get("ISRC") or []
        if isrcs:
            tid_to_isrc[tid] = isrcs[0]  # Use first ISRC

    print(f"  ✓ Loaded {len(tid_to_isrc):,} track ISRCs ({100*len(tid_to_isrc)/len(ds):.1f}% coverage)")
    return tid_to_isrc


def sample_50_cases():
    """Sample 50 cases from R12 payload, preferring turns 3-7 with history."""
    print("\n[2/6] Sampling 50 cases from R12 payload (8000 total)...")

    with open(R12_PAYLOAD, "rb") as f:
        payload = pickle.load(f)

    cases = payload["cases"]
    print(f"  Total cases: {len(cases)}")

    # Prefer turns 3-7 (have history, not too late in session)
    candidates = [c for c in cases if 3 <= c["turn_number"] <= 7 and len(c["history"]) > 0]
    print(f"  Candidates (turn 3-7 with history): {len(candidates)}")

    random.seed(RANDOM_SEED)
    sample = random.sample(candidates, min(50, len(candidates)))

    print(f"  ✓ Sampled {len(sample)} cases")
    print(f"    Turn distribution: {Counter(c['turn_number'] for c in sample)}")
    print(f"    History lengths: min={min(len(c['history']) for c in sample)}, "
          f"max={max(len(c['history']) for c in sample)}, "
          f"mean={sum(len(c['history']) for c in sample)/len(sample):.1f}")

    return sample


def load_r54_pools_for_sample(sample_cases):
    """Load R54 candidate pools for sampled cases."""
    print("\n[3/6] Loading R54 candidate pools...")

    with open(R54_POOLS) as f:
        all_pools = json.load(f)

    print(f"  Total R54 pools: {len(all_pools):,}")

    # Match by session_id__turn_number
    sample_pools = {}
    for case in sample_cases:
        key = f"{case['session_id']}__{case['turn_number']}"
        if key in all_pools:
            sample_pools[key] = all_pools[key][:100]  # Top-100 candidates for diagnostic

    print(f"  ✓ Matched {len(sample_pools)}/{len(sample_cases)} cases to R54 pools")
    if sample_pools:
        pool_sizes = [len(p) for p in sample_pools.values()]
        print(f"    Pool sizes: min={min(pool_sizes)}, max={max(pool_sizes)}, mean={sum(pool_sizes)/len(pool_sizes):.1f}")

    return sample_pools


def fetch_mb_for_tracks(track_ids, tid_to_isrc):
    """Fetch MB metadata for a set of tracks."""
    print("\n[4/6] Fetching MusicBrainz metadata...")

    mb_data = {}  # tid → {tags: [...], genres: [...]}
    cache_hits = 0
    api_calls = 0
    no_isrc = 0
    no_match = 0

    for i, tid in enumerate(track_ids):
        if i % 50 == 0 and i > 0:
            print(f"    Progress: {i}/{len(track_ids)} tracks, "
                  f"{cache_hits} cache hits, {api_calls} API calls, {no_isrc} no ISRC, {no_match} no match")

        isrc = tid_to_isrc.get(tid)
        if not isrc:
            no_isrc += 1
            continue

        # Check cache first
        cache_file_isrc = MB_CACHE_DIR / f"isrc_{cache_key('recording', {'isrc': isrc})}.json"
        if cache_file_isrc.exists():
            cache_hits += 1
            with open(cache_file_isrc) as f:
                isrc_data = json.load(f)
        else:
            api_calls += 1
            isrc_data = fetch_mb_by_isrc(isrc)

        if not isrc_data or not isrc_data.get("recordings"):
            no_match += 1
            continue

        recording_id = isrc_data["recordings"][0]["id"]

        # Fetch full details
        cache_file_rec = MB_CACHE_DIR / f"rec_{cache_key('recording', {'id': recording_id})}.json"
        if cache_file_rec.exists():
            if cache_file_isrc.exists():  # Only count as cache hit if both were cached
                pass  # Already counted above
            with open(cache_file_rec) as f:
                rec_data = json.load(f)
        else:
            api_calls += 1
            rec_data = fetch_mb_recording_details(recording_id)

        if rec_data:
            tags, genres = extract_mb_tags_genres(rec_data)
            mb_data[tid] = {"tags": tags, "genres": genres}

    print(f"  ✓ Fetched MB data for {len(mb_data)}/{len(track_ids)} tracks")
    print(f"    Cache hits: {cache_hits}, API calls: {api_calls}, No ISRC: {no_isrc}, No match: {no_match}")

    return mb_data


def compute_mb_features(sample_cases, sample_pools, mb_data):
    """Compute 2 MB features for each candidate in each case."""
    print("\n[5/6] Computing MB features (tag_jaccard_last, genre_overlap_history)...")

    features_by_case = {}
    total_candidates = 0
    candidates_with_mb = 0

    for case in sample_cases:
        key = f"{case['session_id']}__{case['turn_number']}"
        if key not in sample_pools:
            continue

        candidates = sample_pools[key]
        history = case["history"]  # List of played track IDs

        # Aggregate MB tags/genres from history (last 3 played)
        history_tags = set()
        history_genres = set()
        for played_tid in history[-3:]:
            if played_tid in mb_data:
                history_tags.update(mb_data[played_tid]["tags"])
                history_genres.update(mb_data[played_tid]["genres"])

        case_features = []
        for cand_tid in candidates:
            total_candidates += 1

            if cand_tid not in mb_data:
                # No MB data, features = 0
                case_features.append({
                    "track_id": cand_tid,
                    "mb_tag_jaccard_last": 0.0,
                    "mb_genre_overlap_history": 0,
                    "has_mb_data": False,
                })
                continue

            candidates_with_mb += 1
            cand_tags = set(mb_data[cand_tid]["tags"])
            cand_genres = set(mb_data[cand_tid]["genres"])

            # Feature 1: mb_tag_jaccard_last
            if cand_tags and history_tags:
                intersection = cand_tags & history_tags
                union = cand_tags | history_tags
                tag_jaccard = len(intersection) / len(union) if union else 0.0
            else:
                tag_jaccard = 0.0

            # Feature 2: mb_genre_overlap_history
            genre_overlap = len(cand_genres & history_genres)

            case_features.append({
                "track_id": cand_tid,
                "mb_tag_jaccard_last": tag_jaccard,
                "mb_genre_overlap_history": genre_overlap,
                "has_mb_data": True,
            })

        features_by_case[key] = {
            "session_id": case["session_id"],
            "turn_number": case["turn_number"],
            "gt_track_id": case["gt"],
            "candidates": case_features,
        }

    print(f"  ✓ Computed features for {total_candidates:,} candidates across {len(features_by_case)} cases")
    print(f"    Candidates with MB data: {candidates_with_mb:,} ({100*candidates_with_mb/total_candidates:.1f}%)")

    return features_by_case


def analyze_feature_signal(features_by_case):
    """Analyze whether MB features correlate with GT ranking (proxy for signal)."""
    print("\n[6/6] Analyzing MB feature signal...")

    # For each case, check:
    # - Does GT have higher MB feature values than average candidate?
    # - Distribution of feature values (are they informative?)

    gt_better_tag_jaccard = 0
    gt_better_genre_overlap = 0
    total_cases_with_gt_mb = 0

    tag_jaccard_values = []
    genre_overlap_values = []

    for case_key, case_data in features_by_case.items():
        gt_tid = case_data["gt_track_id"]
        candidates = case_data["candidates"]

        # Find GT in candidates
        gt_feature = next((c for c in candidates if c["track_id"] == gt_tid), None)
        if not gt_feature or not gt_feature["has_mb_data"]:
            continue

        total_cases_with_gt_mb += 1

        # Compute mean feature values for non-GT candidates
        non_gt_candidates = [c for c in candidates if c["track_id"] != gt_tid and c["has_mb_data"]]
        if not non_gt_candidates:
            continue

        mean_tag_jaccard = sum(c["mb_tag_jaccard_last"] for c in non_gt_candidates) / len(non_gt_candidates)
        mean_genre_overlap = sum(c["mb_genre_overlap_history"] for c in non_gt_candidates) / len(non_gt_candidates)

        if gt_feature["mb_tag_jaccard_last"] > mean_tag_jaccard:
            gt_better_tag_jaccard += 1

        if gt_feature["mb_genre_overlap_history"] > mean_genre_overlap:
            gt_better_genre_overlap += 1

        # Collect all feature values for distribution
        tag_jaccard_values.extend(c["mb_tag_jaccard_last"] for c in candidates if c["has_mb_data"])
        genre_overlap_values.extend(c["mb_genre_overlap_history"] for c in candidates if c["has_mb_data"])

    print(f"\n  === Feature Signal Analysis ===")
    print(f"  Cases with GT having MB data: {total_cases_with_gt_mb}/{len(features_by_case)}")
    if total_cases_with_gt_mb > 0:
        print(f"\n  mb_tag_jaccard_last:")
        print(f"    GT > mean(non-GT): {gt_better_tag_jaccard}/{total_cases_with_gt_mb} "
              f"({100*gt_better_tag_jaccard/total_cases_with_gt_mb:.1f}%)")
        if tag_jaccard_values:
            print(f"    Distribution: min={min(tag_jaccard_values):.3f}, "
                  f"max={max(tag_jaccard_values):.3f}, "
                  f"mean={sum(tag_jaccard_values)/len(tag_jaccard_values):.3f}")

        print(f"\n  mb_genre_overlap_history:")
        print(f"    GT > mean(non-GT): {gt_better_genre_overlap}/{total_cases_with_gt_mb} "
              f"({100*gt_better_genre_overlap/total_cases_with_gt_mb:.1f}%)")
        if genre_overlap_values:
            print(f"    Distribution: min={min(genre_overlap_values):.1f}, "
                  f"max={max(genre_overlap_values):.1f}, "
                  f"mean={sum(genre_overlap_values)/len(genre_overlap_values):.1f}")

    # Gate decision heuristic:
    # If GT is better than mean in >60% of cases for either feature → GREEN (signal exists)
    # If GT is better than mean in 50-60% → BORDERLINE
    # If GT is better than mean in <50% → RED (no signal)

    tag_jaccard_pct = 100 * gt_better_tag_jaccard / total_cases_with_gt_mb if total_cases_with_gt_mb > 0 else 0
    genre_overlap_pct = 100 * gt_better_genre_overlap / total_cases_with_gt_mb if total_cases_with_gt_mb > 0 else 0

    if tag_jaccard_pct >= 60 or genre_overlap_pct >= 60:
        verdict = "GREEN"
        recommendation = "Signal detected. Proceed to full Phase 2 (300 cases, 5 features, CV5, LR retraining)."
    elif tag_jaccard_pct >= 50 or genre_overlap_pct >= 50:
        verdict = "BORDERLINE"
        recommendation = "Weak signal. Borderline case — user decision on full Phase 2."
    else:
        verdict = "RED"
        recommendation = "No signal detected. Archive C4. MB features do not improve GT ranking."

    print(f"\n  === Phase 2-B Gate Decision ===")
    print(f"  Verdict: {verdict}")
    print(f"  Recommendation: {recommendation}")

    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "tag_jaccard_gt_better_pct": tag_jaccard_pct,
        "genre_overlap_gt_better_pct": genre_overlap_pct,
        "total_cases_with_gt_mb": total_cases_with_gt_mb,
        "tag_jaccard_distribution": {
            "min": min(tag_jaccard_values) if tag_jaccard_values else 0,
            "max": max(tag_jaccard_values) if tag_jaccard_values else 0,
            "mean": sum(tag_jaccard_values) / len(tag_jaccard_values) if tag_jaccard_values else 0,
        },
        "genre_overlap_distribution": {
            "min": min(genre_overlap_values) if genre_overlap_values else 0,
            "max": max(genre_overlap_values) if genre_overlap_values else 0,
            "mean": sum(genre_overlap_values) / len(genre_overlap_values) if genre_overlap_values else 0,
        },
    }


def main():
    print("=" * 70)
    print("R59 C4 Phase 2-B: Lightweight 50-case MusicBrainz Diagnostic")
    print(EXPLORATORY_LABEL)
    print("=" * 70)

    # Step 1: Load ISRC index
    tid_to_isrc = load_track_isrcs()

    # Step 2: Sample 50 cases
    sample_cases = sample_50_cases()

    # Step 3: Load R54 pools
    sample_pools = load_r54_pools_for_sample(sample_cases)

    # Step 4: Fetch MB metadata
    all_track_ids = set()
    for case in sample_cases:
        all_track_ids.add(case["gt"])
        all_track_ids.update(case["history"])
    for pool in sample_pools.values():
        all_track_ids.update(pool)

    print(f"\n  Total unique tracks to fetch: {len(all_track_ids):,}")
    mb_data = fetch_mb_for_tracks(all_track_ids, tid_to_isrc)

    # Step 5: Compute features
    features_by_case = compute_mb_features(sample_cases, sample_pools, mb_data)

    # Step 6: Analyze signal
    analysis = analyze_feature_signal(features_by_case)

    # Save outputs
    output_data = {
        "label": EXPLORATORY_LABEL,
        "sample_size": len(sample_cases),
        "matched_pools": len(sample_pools),
        "total_tracks": len(all_track_ids),
        "mb_coverage": len(mb_data),
        "analysis": analysis,
        "features_by_case": features_by_case,
    }

    output_file = PHASE2B_OUTPUT / "phase2b_results.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Generate report
    report_file = PHASE2B_OUTPUT / "phase2b_report.md"
    with open(report_file, "w") as f:
        f.write(f"# R59 C4 Phase 2-B: Lightweight MB Diagnostic Report\n\n")
        f.write(f"**Label:** {EXPLORATORY_LABEL}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"- **Sample size:** {len(sample_cases)} cases\n")
        f.write(f"- **Matched R54 pools:** {len(sample_pools)}\n")
        f.write(f"- **Total unique tracks:** {len(all_track_ids):,}\n")
        f.write(f"- **MB coverage:** {len(mb_data):,} tracks ({100*len(mb_data)/len(all_track_ids):.1f}%)\n\n")
        f.write(f"## Gate Decision\n\n")
        f.write(f"**Verdict:** {analysis['verdict']}\n\n")
        f.write(f"**Recommendation:** {analysis['recommendation']}\n\n")
        f.write(f"## Feature Signal\n\n")
        f.write(f"- **mb_tag_jaccard_last:** GT > mean(non-GT) in {analysis['tag_jaccard_gt_better_pct']:.1f}% of cases\n")
        f.write(f"- **mb_genre_overlap_history:** GT > mean(non-GT) in {analysis['genre_overlap_gt_better_pct']:.1f}% of cases\n\n")
        f.write(f"## Feature Distributions\n\n")
        f.write(f"### mb_tag_jaccard_last\n\n")
        f.write(f"- Min: {analysis['tag_jaccard_distribution']['min']:.3f}\n")
        f.write(f"- Max: {analysis['tag_jaccard_distribution']['max']:.3f}\n")
        f.write(f"- Mean: {analysis['tag_jaccard_distribution']['mean']:.3f}\n\n")
        f.write(f"### mb_genre_overlap_history\n\n")
        f.write(f"- Min: {analysis['genre_overlap_distribution']['min']:.1f}\n")
        f.write(f"- Max: {analysis['genre_overlap_distribution']['max']:.1f}\n")
        f.write(f"- Mean: {analysis['genre_overlap_distribution']['mean']:.1f}\n\n")

    print(f"✓ Report saved to {report_file}")

    print(f"\n{'='*70}")
    print(f"PHASE 2-B COMPLETE: {analysis['verdict']}")
    print(f"{'='*70}")

    return analysis["verdict"], len(sample_cases)


if __name__ == "__main__":
    verdict, sample_size = main()
    # Exit code: 0 = GREEN, 1 = RED, 2 = BORDERLINE
    sys.exit(0 if verdict == "GREEN" else 1 if verdict == "RED" else 2)
