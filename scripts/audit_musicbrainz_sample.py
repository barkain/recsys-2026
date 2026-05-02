#!/usr/bin/env python3
"""MusicBrainz 100-track sample audit: ISRC lookup + relation quality."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_PATH = REPO_ROOT / "cache" / "mb_audit_sample.json"
CACHE_DIR = REPO_ROOT / "cache" / "musicbrainz_audit"
RESULTS_PATH = REPO_ROOT / "cache" / "mb_audit_results.json"

MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "RecsysResearch/0.1 (barkai.nadav@gmail.com)"
RATE_LIMIT = 1.1  # seconds between requests

# Load full catalog for cross-referencing
from datasets import Dataset, concatenate_datasets

def load_catalog_isrc_index():
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    splits = []
    for split in ["all_tracks", "test_tracks"]:
        matches = sorted(hf_cache.glob(
            f"talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            f"talk_play_data-challenge-track-metadata-{split}.arrow"))
        if matches:
            splits.append(Dataset.from_file(str(matches[-1])))
    ds = concatenate_datasets(splits)
    cols = ds.to_dict()

    isrc_to_tid = {}
    tid_to_meta = {}
    artist_to_tids = {}

    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        isrcs = cols["ISRC"][i] or []
        artists = cols["artist_name"][i] or []
        artist_ids = cols["artist_id"][i] or []

        tid_to_meta[tid] = {
            "track_name": cols["track_name"][i],
            "artist_name": artists,
            "artist_id": artist_ids,
        }

        for isrc in isrcs:
            if isrc:
                isrc_to_tid[isrc] = tid

        for aid in artist_ids:
            if aid not in artist_to_tids:
                artist_to_tids[aid] = []
            artist_to_tids[aid].append(tid)

    return isrc_to_tid, tid_to_meta, artist_to_tids


def mb_request(endpoint, params=None):
    """Make a MusicBrainz API request with caching and rate limiting."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache_key = endpoint.replace("/", "_") + "_" + str(sorted((params or {}).items()))
    cache_key = cache_key.replace(" ", "").replace("'", "")[:200]
    cache_file = CACHE_DIR / f"{hash(cache_key) & 0xFFFFFFFF:08x}.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    url = f"{MB_BASE}/{endpoint}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    for attempt in range(3):
        time.sleep(RATE_LIMIT)
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=60)
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
            print(f"  MB request attempt {attempt+1} failed: {e}", flush=True)
            if attempt < 2:
                time.sleep(5)
                continue
            return {"error": "timeout", "status": 0}

        if resp.status_code == 200:
            data = resp.json()
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return data
        elif resp.status_code == 404:
            data = {"error": "not_found", "status": 404}
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return data
        elif resp.status_code == 503:
            print(f"  MB 503 — rate limited, waiting 5s...", flush=True)
            time.sleep(5)
            continue
        else:
            print(f"  MB API error: {resp.status_code} for {endpoint}", flush=True)
            return {"error": f"http_{resp.status_code}", "status": resp.status_code}
    return {"error": "max_retries", "status": 0}


def lookup_by_isrc(isrc):
    """Look up a recording by ISRC (no inc params — not supported on ISRC endpoint)."""
    return mb_request("isrc/" + isrc)


def get_recording_relations(recording_id):
    """Get relations for a recording."""
    return mb_request(f"recording/{recording_id}",
                      {"inc": "artist-credits+tags+genres+releases+url-rels"})


def get_artist_recordings(artist_id, limit=100):
    """Get recordings by an artist."""
    return mb_request("recording",
                      {"artist": artist_id, "limit": str(limit), "offset": "0"})


def get_artist_relations(artist_id):
    """Get artist relations (members, collaborators, etc.)."""
    return mb_request(f"artist/{artist_id}",
                      {"inc": "artist-rels+tags+genres"})


def main():
    print("Loading sample and catalog...", flush=True)
    with open(SAMPLE_PATH) as f:
        sample = json.load(f)

    isrc_to_tid, tid_to_meta, artist_to_tids = load_catalog_isrc_index()
    catalog_tids = set(tid_to_meta.keys())
    print(f"Catalog: {len(catalog_tids)} tracks, {len(isrc_to_tid)} ISRCs indexed")

    results = []
    request_count = 0

    for idx, track in enumerate(sample):
        tid = track["tid"]
        isrc_list = track["isrc"]
        isrc = isrc_list[0] if isrc_list and len(isrc_list) > 0 else None
        category = track["category"]

        print(f"\n[{idx+1}/100] {category}: {track['track_name']} by {track['artist_name']}")

        result = {
            "tid": tid,
            "category": category,
            "track_name": track["track_name"],
            "artist_name": track["artist_name"],
            "popularity": track["popularity"],
            "has_isrc": bool(isrc),
            "mb_match": False,
            "mb_recording_id": None,
            "title_match": False,
            "artist_match": False,
            "mb_tags": [],
            "mb_genres": [],
            "artist_relations": [],
            "same_artist_catalog_count": 0,
            "related_artist_catalog_count": 0,
            "related_recordings_in_catalog": 0,
        }

        if not isrc:
            print("  No ISRC — skipping MB lookup")
            results.append(result)
            continue

        # Step 1: ISRC lookup
        isrc_data = lookup_by_isrc(isrc)
        request_count += 1

        if "error" in isrc_data:
            print(f"  ISRC lookup failed: {isrc_data['error']}")
            results.append(result)
            continue

        recordings = isrc_data.get("recordings", [])
        if not recordings:
            print("  No recordings found for ISRC")
            results.append(result)
            continue

        rec = recordings[0]
        mb_id = rec.get("id", "")
        mb_title = rec.get("title", "")

        result["mb_match"] = True
        result["mb_recording_id"] = mb_id
        result["mb_title"] = mb_title

        # Step 1b: Get full recording details (artist-credit, tags, genres)
        rec_detail = get_recording_relations(mb_id)
        request_count += 1

        mb_artists = []
        mb_artist_ids_mb = []
        mb_tags = []
        mb_genres = []
        if "error" not in rec_detail:
            mb_artists = [a.get("artist", {}).get("name", "") for a in rec_detail.get("artist-credit", [])]
            mb_artist_ids_mb = [a.get("artist", {}).get("id", "") for a in rec_detail.get("artist-credit", [])]
            mb_tags = [t.get("name", "") for t in rec_detail.get("tags", [])]
            mb_genres = [g.get("name", "") for g in rec_detail.get("genres", [])]

        result["mb_artists"] = mb_artists
        result["mb_tags"] = mb_tags
        result["mb_genres"] = mb_genres

        local_title = " ".join(track["track_name"]) if isinstance(track["track_name"], list) else str(track["track_name"])
        local_artist = " ".join(track["artist_name"]) if isinstance(track["artist_name"], list) else str(track["artist_name"])
        result["title_match"] = mb_title.lower().strip() in local_title.lower() or local_title.lower() in mb_title.lower().strip()
        result["artist_match"] = any(a.lower() in local_artist.lower() or local_artist.lower() in a.lower() for a in mb_artists) if mb_artists else False

        print(f"  MB: '{mb_title}' by {mb_artists}")
        print(f"  Title match: {result['title_match']}, Artist match: {result['artist_match']}")
        print(f"  Tags: {mb_tags[:5]}, Genres: {mb_genres[:5]}")

        # Step 2: Get artist recordings (same-artist catalog expansion)
        mb_artist_ids = mb_artist_ids_mb
        same_artist_in_catalog = set()

        for mb_aid in mb_artist_ids[:2]:  # limit to first 2 artists
            if not mb_aid:
                continue
            artist_recs = get_artist_recordings(mb_aid, limit=100)
            request_count += 1

            if "error" not in artist_recs:
                for ar in artist_recs.get("recordings", []):
                    ar_isrcs = []
                    # Check if any of these recordings are in our catalog
                    # We'd need ISRC for each, but MB doesn't return ISRCs in browse results
                    # Instead, check by title match against catalog
                    pass
                result["mb_artist_recording_count"] = artist_recs.get("recording-count", 0)
                print(f"  Artist {mb_aid[:8]}... has {artist_recs.get('recording-count', 0)} recordings on MB")

        # Count same-artist tracks already in our catalog (using local artist_id)
        local_artist_ids = tid_to_meta.get(tid, {}).get("artist_id", [])
        for aid in local_artist_ids:
            same_artist_in_catalog.update(artist_to_tids.get(aid, []))
        same_artist_in_catalog.discard(tid)
        result["same_artist_catalog_count"] = len(same_artist_in_catalog)

        # Step 3: Get artist relations (collaborators, members)
        for mb_aid in mb_artist_ids[:1]:  # only first artist
            if not mb_aid:
                continue
            artist_rels = get_artist_relations(mb_aid)
            request_count += 1

            if "error" not in artist_rels:
                relations = artist_rels.get("relations", [])
                related_artists = []
                for rel in relations:
                    if rel.get("type") in ("member of band", "collaboration", "vocal", "instrument",
                                           "producer", "mix", "remix", "tribute"):
                        target = rel.get("artist", {})
                        if target:
                            related_artists.append({
                                "name": target.get("name", ""),
                                "id": target.get("id", ""),
                                "type": rel.get("type", ""),
                            })
                result["artist_relations"] = related_artists[:10]
                print(f"  Artist relations: {len(related_artists)} related artists")
                for ra in related_artists[:3]:
                    print(f"    - {ra['name']} ({ra['type']})")

        results.append(result)

        if request_count % 10 == 0:
            print(f"\n  --- {request_count} requests made so far ---\n", flush=True)

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump({"results": results, "request_count": request_count}, f, indent=2)

    # Summary
    print(f"\n{'='*60}")
    print(f"MUSICBRAINZ AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Total requests: {request_count}")

    for cat in ["unreachable_pop0", "pop0_random", "popular", "diff_artist_miss"]:
        cat_results = [r for r in results if r["category"] == cat]
        n = len(cat_results)
        has_isrc = sum(1 for r in cat_results if r["has_isrc"])
        matched = sum(1 for r in cat_results if r["mb_match"])
        title_ok = sum(1 for r in cat_results if r["title_match"])
        artist_ok = sum(1 for r in cat_results if r["artist_match"])
        has_tags = sum(1 for r in cat_results if r["mb_tags"])
        has_genres = sum(1 for r in cat_results if r["mb_genres"])
        has_rels = sum(1 for r in cat_results if r["artist_relations"])

        print(f"\n  {cat} ({n} tracks):")
        print(f"    ISRC present:     {has_isrc}/{n}")
        print(f"    MB match:         {matched}/{n} ({matched/n:.0%})")
        print(f"    Title match:      {title_ok}/{n}")
        print(f"    Artist match:     {artist_ok}/{n}")
        print(f"    Has tags:         {has_tags}/{n}")
        print(f"    Has genres:       {has_genres}/{n}")
        print(f"    Has artist rels:  {has_rels}/{n}")

    # Gate checks
    total_matched = sum(1 for r in results if r["mb_match"])
    total_with_data = sum(1 for r in results if r["mb_tags"] or r["mb_genres"] or r["artist_relations"])
    print(f"\nGATES:")
    print(f"  ISRC→recording match >=70%: {'PASS' if total_matched/100 >= 0.70 else 'FAIL'} ({total_matched}/100 = {total_matched}%)")
    print(f"  Meaningful data >=40%:       {'PASS' if total_with_data/100 >= 0.40 else 'FAIL'} ({total_with_data}/100 = {total_with_data}%)")


if __name__ == "__main__":
    main()
