#!/usr/bin/env python3
"""R59 C4 Phase 2: 300-case MusicBrainz diagnostic with API calls.

EXPLORATORY RESEARCH ONLY — NOT SUBMISSION-SAFE.

Tests 5 MB-derived features on 300-case dev sample (100 DEMOTED, 100 UNREACHABLE, 100 HIT).
Gates at +0.005 nDCG improvement vs R39+R54 baseline.

MusicBrainz API: 1 req/sec rate limit, aggressive caching.
NO blind evaluation. NO production integration.

Outputs labeled: "EXPLORATORY - NOT SUBMISSION-SAFE"

Phase 2 Steps:
1. Load dev data, sample 300 cases (stratified by bucket)
2. Load R39+R54 baseline features (37 features)
3. Fetch MB metadata via ISRC (use cache/musicbrainz_audit/, make new API calls as needed)
4. Compute 5 MB features per candidate:
   - mb_tag_jaccard_last: Jaccard(MB tags for candidate, union MB tags for last 3 played)
   - mb_genre_overlap_history: count MB genres for candidate in ANY played track's genres
   - mb_recording_relation_played: binary flag for MB recording relation to played tracks
   - mb_artist_relation_played: binary flag for MB artist relation to played artists
   - mb_same_label_last: binary flag for same release label as last played
5. Train LambdaRank CV5 on 300-case sample with 37+5=42 features
6. Compare metrics: baseline (37 features) vs MB-enriched (42 features)

Gates:
- GREEN: nDCG@20 improvement ≥ +0.005, same-artist no regress → scale to full 8K (Phase 3)
- RED: regress OR flat → archive C4, report failure mode
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from collections import Counter

import requests
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# MusicBrainz API config
MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "RecsysResearch/0.1.exploratory (barkai.nadav@gmail.com)"
RATE_LIMIT = 1.1  # seconds between requests
MB_CACHE_DIR = REPO_ROOT / "cache" / "musicbrainz_audit"
MB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Phase 2 output
PHASE2_OUTPUT_DIR = REPO_ROOT / "exp" / "eval" / "c4_phase2_exploratory"
PHASE2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# EXPLORATORY label
EXPLORATORY_LABEL = "EXPLORATORY - NOT SUBMISSION-SAFE - EXTERNAL DATA (MusicBrainz)"


def cache_key(url: str, params: dict) -> str:
    """Generate cache key for MB API request."""
    key_str = f"{url}?{sorted(params.items())}"
    return hashlib.md5(key_str.encode()).hexdigest()[:8]


def fetch_mb_recording_by_isrc(isrc: str) -> dict | None:
    """Fetch MusicBrainz recording by ISRC with caching."""
    cache_file = MB_CACHE_DIR / f"{cache_key('recording', {'isrc': isrc})}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    # Make API request
    url = f"{MB_BASE}/recording"
    params = {
        "query": f"isrc:{isrc}",
        "fmt": "json",
        "limit": 1,
    }
    headers = {"User-Agent": USER_AGENT}

    time.sleep(RATE_LIMIT)
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        # Cache response
        with open(cache_file, "w") as f:
            json.dump(data, f)

        return data
    except Exception as e:
        print(f"  ✗ MB API error for ISRC {isrc}: {e}")
        return None


def fetch_mb_recording_details(recording_id: str) -> dict | None:
    """Fetch full recording details including tags, genres, relations."""
    cache_file = MB_CACHE_DIR / f"{cache_key('recording_detail', {'id': recording_id})}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"{MB_BASE}/recording/{recording_id}"
    params = {
        "fmt": "json",
        "inc": "tags+genres+artist-rels+recording-rels+work-rels+releases",
    }
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
        print(f"  ✗ MB API error for recording {recording_id}: {e}")
        return None


def extract_mb_metadata(recording_data: dict) -> dict:
    """Extract tags, genres, relations from MB recording response."""
    tags = [t["name"] for t in recording_data.get("tags", []) if t.get("count", 0) > 0]
    genres = [g["name"] for g in recording_data.get("genres", [])]

    # Relations
    artist_rels = []
    recording_rels = []

    for rel in recording_data.get("relations", []):
        if rel.get("type") in ["member of band", "collaboration"]:
            artist_rels.append({
                "name": rel.get("artist", {}).get("name"),
                "type": rel["type"],
            })
        elif rel.get("type") in ["cover", "remix", "samples material", "live performance"]:
            recording_rels.append({
                "type": rel["type"],
                "target_id": rel.get("recording", {}).get("id"),
            })

    # Release label (from first release if available)
    label = None
    releases = recording_data.get("releases", [])
    if releases:
        # MB API with inc=releases doesn't give label; would need separate release API call
        # For Phase 2 diagnostic, skip label feature (too expensive: 1 recording + 1 release = 2 API calls per track)
        pass

    return {
        "tags": tags,
        "genres": genres,
        "artist_relations": artist_rels,
        "recording_relations": recording_rels,
        "label": label,
    }


def compute_mb_features_for_candidate(
    candidate_tid: str,
    candidate_mb: dict | None,
    played_track_mbs: list[dict],
) -> dict:
    """Compute 5 MB features for a candidate track."""
    features = {
        "mb_tag_jaccard_last": 0.0,
        "mb_genre_overlap_history": 0,
        "mb_recording_relation_played": 0,
        "mb_artist_relation_played": 0,
        # Skip mb_same_label_last (requires additional API calls per track)
    }

    if not candidate_mb or not candidate_mb.get("tags"):
        return features  # No MB data, features = 0

    candidate_tags = set(candidate_mb["tags"])
    candidate_genres = set(candidate_mb["genres"])
    candidate_artist_rels = {r["name"] for r in candidate_mb.get("artist_relations", []) if r.get("name")}
    candidate_rec_rels = {r["target_id"] for r in candidate_mb.get("recording_relations", []) if r.get("target_id")}

    # Aggregate played track MB data
    played_tags_union = set()
    played_genres_all = set()
    played_artist_rels_all = set()
    played_recording_ids = set()

    for played_mb in played_track_mbs[-3:]:  # last 3 played
        if played_mb:
            played_tags_union.update(played_mb.get("tags", []))
            played_genres_all.update(played_mb.get("genres", []))
            played_artist_rels_all.update(
                r["name"] for r in played_mb.get("artist_relations", []) if r.get("name")
            )
            # Would need recording IDs from played tracks; for now assume unavailable

    # Feature 1: mb_tag_jaccard_last
    if candidate_tags and played_tags_union:
        intersection = candidate_tags & played_tags_union
        union = candidate_tags | played_tags_union
        features["mb_tag_jaccard_last"] = len(intersection) / len(union) if union else 0.0

    # Feature 2: mb_genre_overlap_history
    features["mb_genre_overlap_history"] = len(candidate_genres & played_genres_all)

    # Feature 3: mb_recording_relation_played (needs played recording IDs; skip for Phase 2)
    # features["mb_recording_relation_played"] = 1 if (candidate_rec_rels & played_recording_ids) else 0

    # Feature 4: mb_artist_relation_played
    features["mb_artist_relation_played"] = 1 if (candidate_artist_rels & played_artist_rels_all) else 0

    return features


def load_300_case_sample():
    """Load 300-case stratified sample (100 DEMOTED, 100 UNREACHABLE, 100 HIT).

    For Phase 2 diagnostic, we need:
    - Dev session data (conversations, played tracks, GT)
    - Candidate pools (top-50 or top-100 from retriever)
    - R39+R54 baseline features

    TODO: Implement proper stratified sampling from dev set.
    For now, return placeholder structure.
    """
    print("[Phase 2.1] Loading 300-case sample...")
    print("  ⚠ Placeholder: proper sampling from dev set not implemented yet")
    print("  ⚠ Phase 2 will need:")
    print("    - Dev session data (h7 split, 8000 cases)")
    print("    - Bucket assignments (DEMOTED/UNREACHABLE/HIT from prior forensics)")
    print("    - Candidate pools (R54 retriever output)")
    print("    - R39+R54 baseline features (37 features per candidate)")

    # Return empty for now
    return {
        "sessions": [],
        "sample_size": 0,
        "buckets": {"DEMOTED": 0, "UNREACHABLE": 0, "HIT": 0},
    }


def main():
    print("=" * 70)
    print("R59 C4 Phase 2: 300-case MusicBrainz Diagnostic")
    print(EXPLORATORY_LABEL)
    print("=" * 70)

    # Phase 2.1: Load sample
    sample_data = load_300_case_sample()

    if sample_data["sample_size"] == 0:
        print("\n✗ PHASE 2 BLOCKED: Need proper dev sample implementation")
        print("\nRequired for Phase 2:")
        print("  1. Dev session data with played tracks + GT")
        print("  2. Bucket stratification (DEMOTED/UNREACHABLE/HIT)")
        print("  3. Candidate pools per session")
        print("  4. R39+R54 baseline features (37 features)")
        print("  5. Track ISRC index for MB API join")
        print("\nRecommendation:")
        print("  - Reuse existing dev evaluation infrastructure from R54/R56/R57")
        print("  - Sample 300 cases from h7 dev set (8000 total)")
        print("  - Load R54 retriever output (candidate pools)")
        print("  - Load R54 baseline features from cache/r54_phase3_*.pkl")
        print("\nPhase 2 implementation continues in next iteration.")

        # Write placeholder report
        report_path = PHASE2_OUTPUT_DIR / "phase2_diagnostic_report.md"
        with open(report_path, "w") as f:
            f.write(f"# R59 C4 Phase 2: MusicBrainz Diagnostic Report\n\n")
            f.write(f"**Status:** BLOCKED - Need dev sample implementation\n\n")
            f.write(f"**Label:** {EXPLORATORY_LABEL}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"Phase 2 requires:\n")
            f.write(f"1. 300-case stratified sample from h7 dev set\n")
            f.write(f"2. Candidate pools from R54 retriever\n")
            f.write(f"3. R39+R54 baseline features (37 features)\n")
            f.write(f"4. Track ISRC index for MB API join\n\n")
            f.write(f"Recommendation: Reuse R54/R56/R57 evaluation infrastructure.\n")

        print(f"\n✓ Placeholder report: {report_path}")
        return "BLOCKED"

    # Phase 2.2: Fetch MB metadata for sample
    # Phase 2.3: Compute MB features
    # Phase 2.4: Train LambdaRank CV5 with 42 features
    # Phase 2.5: Compare with baseline
    # Phase 2.6: Gate decision

    print("\n✓ Phase 2 script structure complete (implementation continues)")
    return "IN_PROGRESS"


if __name__ == "__main__":
    status = main()
    sys.exit(0 if status == "GREEN" else 1 if status == "RED" else 2)
