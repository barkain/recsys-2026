#!/usr/bin/env python3
"""MusicBrainz Stage 1B: Can MB relations recover held-out dev GTs?

Bounded diagnostic — max 500 MB requests, cached.
Tests whether MB artist-relations from played tracks lead to GT artist/track.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CACHE_DIR = REPO_ROOT / "cache" / "musicbrainz_audit"
UNREACHABLE_PATH = REPO_ROOT / "cache" / "mb_unreachable_cases.json"

MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "RecsysResearch/0.1 (barkai.nadav@gmail.com)"
RATE_LIMIT = 1.1
MAX_REQUESTS = 500

request_count = 0


def mb_request(endpoint, params=None):
    global request_count
    if request_count >= MAX_REQUESTS:
        return {"error": "budget_exhausted"}

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_key = endpoint.replace("/", "_") + "_" + str(sorted((params or {}).items()))
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
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
            if attempt < 2:
                time.sleep(5)
                continue
            return {"error": "timeout"}

        request_count += 1

        if resp.status_code == 200:
            data = resp.json()
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return data
        elif resp.status_code == 404:
            data = {"error": "not_found"}
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return data
        elif resp.status_code == 503:
            time.sleep(5)
            continue
        else:
            return {"error": f"http_{resp.status_code}"}
    return {"error": "max_retries"}


def get_artist_relations(mb_artist_id):
    return mb_request(f"artist/{mb_artist_id}", {"inc": "artist-rels+tags+genres"})


def lookup_isrc(isrc):
    return mb_request(f"isrc/{isrc}")


def get_recording_detail(mb_recording_id):
    return mb_request(f"recording/{mb_recording_id}",
                      {"inc": "artist-credits+tags+genres"})


def main():
    global request_count

    # Load catalog data
    from datasets import Dataset, concatenate_datasets
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

    # Build lookup indices
    tid_to_isrc = {}
    tid_to_artist_ids = {}
    artist_id_to_tids = defaultdict(set)
    tid_to_meta = {}

    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        isrcs = cols["ISRC"][i] or []
        artist_ids = cols["artist_id"][i] or []
        tid_to_isrc[tid] = isrcs[0] if isrcs else None
        tid_to_artist_ids[tid] = artist_ids
        tid_to_meta[tid] = {
            "track_name": cols["track_name"][i],
            "artist_name": cols["artist_name"][i],
        }
        for aid in artist_ids:
            artist_id_to_tids[aid].add(tid)

    catalog_tids = set(tid_to_meta.keys())
    print(f"Catalog: {len(catalog_tids)} tracks, {len(artist_id_to_tids)} artists")

    # Load unreachable cases
    with open(UNREACHABLE_PATH) as f:
        unreachable = json.load(f)
    print(f"Unreachable cases: {len(unreachable)}")

    # Strategy: for each unreachable case, check if MB artist-relations from
    # played artists can reach the GT artist. We need to:
    # 1. Look up played artists' MB IDs (via their tracks' ISRCs)
    # 2. Get MB artist relations
    # 3. Check if any related MB artist matches the GT artist

    # But we have local artist_ids (internal UUIDs), not MB artist IDs.
    # Bridge: played track ISRC → MB recording → MB artist ID
    # Then: MB artist ID → MB artist relations → related MB artist names
    # Then: match related artist names against GT artist name (fuzzy)

    # This is expensive. Let's be strategic:
    # - Focus on the most common played artists (they cover many cases)
    # - Cache aggressively

    # Count which played artist_ids appear most often in unreachable cases
    played_aid_freq = Counter()
    for u in unreachable:
        for aid in u["played_artist_ids"]:
            played_aid_freq[aid] += 1

    # Pick a representative track for each frequent artist to get their ISRC→MB ID
    aid_to_sample_tid = {}
    for aid in played_aid_freq:
        tids = artist_id_to_tids.get(aid, set())
        for tid in tids:
            if tid_to_isrc.get(tid):
                aid_to_sample_tid[aid] = tid
                break

    # Focus on top-100 most frequent played artists (covers most cases)
    top_aids = [aid for aid, _ in played_aid_freq.most_common(100)]
    print(f"\nTop 100 played artists cover cases:")
    covered = sum(1 for u in unreachable
                  if any(aid in top_aids for aid in u["played_artist_ids"]))
    print(f"  {covered}/{len(unreachable)} ({covered/len(unreachable):.0%})")

    # Step 1: Map top played artists to MB artist IDs via ISRC
    print(f"\nStep 1: Mapping played artists to MB IDs...")
    aid_to_mb_id = {}  # local artist_id → MB artist ID
    aid_to_mb_name = {}

    for aid in top_aids:
        if request_count >= MAX_REQUESTS:
            print(f"  Budget exhausted at {request_count} requests")
            break

        sample_tid = aid_to_sample_tid.get(aid)
        if not sample_tid:
            continue
        isrc = tid_to_isrc[sample_tid]
        if not isrc:
            continue

        isrc_data = lookup_isrc(isrc)
        if "error" in isrc_data:
            continue

        recordings = isrc_data.get("recordings", [])
        if not recordings:
            continue

        mb_rec_id = recordings[0]["id"]
        rec_detail = get_recording_detail(mb_rec_id)
        if "error" in rec_detail:
            continue

        for ac in rec_detail.get("artist-credit", []):
            mb_artist = ac.get("artist", {})
            mb_aid = mb_artist.get("id", "")
            mb_name = mb_artist.get("name", "")
            if mb_aid and mb_name:
                # Match to local artist by name
                local_names = [n for n in (tid_to_meta.get(sample_tid, {}).get("artist_name") or [])]
                if any(mb_name.lower() in n.lower() or n.lower() in mb_name.lower()
                       for n in local_names):
                    aid_to_mb_id[aid] = mb_aid
                    aid_to_mb_name[aid] = mb_name
                    break

    print(f"  Mapped {len(aid_to_mb_id)} / {len(top_aids)} played artists to MB IDs")
    print(f"  Requests so far: {request_count}")

    # Step 2: Get MB artist relations for mapped artists
    print(f"\nStep 2: Getting MB artist relations...")
    mb_id_to_relations = {}  # MB artist ID → list of related MB artist names

    for aid, mb_id in aid_to_mb_id.items():
        if request_count >= MAX_REQUESTS:
            break
        if mb_id in mb_id_to_relations:
            continue

        rel_data = get_artist_relations(mb_id)
        if "error" in rel_data:
            mb_id_to_relations[mb_id] = []
            continue

        relations = []
        for rel in rel_data.get("relations", []):
            target = rel.get("artist", {})
            if target:
                relations.append({
                    "name": target.get("name", ""),
                    "id": target.get("id", ""),
                    "type": rel.get("type", ""),
                })
        mb_id_to_relations[mb_id] = relations

    print(f"  Got relations for {len(mb_id_to_relations)} MB artists")
    print(f"  Requests so far: {request_count}")

    # Step 3: For each unreachable case, check if MB relations bridge to GT
    print(f"\nStep 3: Checking GT recovery...")

    # Build reverse index: MB artist name (lowered) → set of related MB artist names
    played_aid_to_related_names = {}
    for aid, mb_id in aid_to_mb_id.items():
        rels = mb_id_to_relations.get(mb_id, [])
        related_names = set()
        for r in rels:
            related_names.add(r["name"].lower())
        played_aid_to_related_names[aid] = related_names

    # Also build: local artist_id → all catalog track_ids by that artist
    # (already have artist_id_to_tids)

    recovered_same_artist = 0
    recovered_relation = 0
    recovered_union = 0
    pop0_recovered = 0
    diff_artist_recovered = 0
    hist0_recovered = 0
    examples = []

    for u in unreachable:
        gt = u["gt"]
        gt_artist_ids = u["gt_artist_ids"]
        gt_artist_names = [n for n in (tid_to_meta.get(gt, {}).get("artist_name") or [])]
        gt_names_lower = set(n.lower() for n in gt_artist_names)
        played = u["played"]
        played_aids = u["played_artist_ids"]
        pop = u["pop"]
        n_hist = u["n_hist"]

        recovered = False

        # Check 1: Same-artist discography in catalog
        # If GT artist has other tracks played by user, same-artist retrieval should find it
        # But we know these are *unreachable* — so same-artist retrieval already failed
        # The question is: can MB discography surface THIS specific track?
        # For same-artist: the track is already in catalog with the same artist_id
        # So same-artist retrieval should have found it... unless it's a cold track
        # that ALS/CF-BPR never saw. MB wouldn't help here beyond what we already have.

        # Check 2: MB relation bridge (different-artist)
        for aid in played_aids:
            if aid not in played_aid_to_related_names:
                continue
            related_names = played_aid_to_related_names[aid]
            if related_names & gt_names_lower:
                recovered = True
                recovered_relation += 1
                if pop == 0:
                    pop0_recovered += 1
                if not u["same_artist"] and played:
                    diff_artist_recovered += 1
                if n_hist == 0:
                    hist0_recovered += 1
                if len(examples) < 20:
                    played_name = aid_to_mb_name.get(aid, aid[:8])
                    matched_name = related_names & gt_names_lower
                    examples.append({
                        "gt": gt,
                        "gt_name": gt_artist_names,
                        "gt_track": tid_to_meta.get(gt, {}).get("track_name"),
                        "played_artist": played_name,
                        "matched_via": list(matched_name),
                        "pop": pop,
                        "relation_type": "artist-relation",
                    })
                break

        if recovered:
            recovered_union += 1

    # Project to full dataset
    total_cases = 8000
    unreachable_count = len(unreachable)

    print(f"\n{'='*60}")
    print(f"MUSICBRAINZ RECOVERY DIAGNOSTIC")
    print(f"{'='*60}")
    print(f"Total requests used: {request_count}/{MAX_REQUESTS}")
    print(f"Artists mapped to MB: {len(aid_to_mb_id)}")
    print(f"Artist relations fetched: {len(mb_id_to_relations)}")
    print(f"\nRecovery (among {unreachable_count} unreachable cases):")
    print(f"  Via MB artist-relation bridge: {recovered_relation}")
    print(f"  Pop=0 recovered: {pop0_recovered}")
    print(f"  Different-artist recovered: {diff_artist_recovered}")
    print(f"  hist_0 recovered: {hist0_recovered}")
    print(f"  Union recovered: {recovered_union}")

    print(f"\nProjection:")
    # We only checked top-100 played artists. Full coverage would check all ~2355
    coverage_ratio = covered / unreachable_count if unreachable_count else 0
    print(f"  Top-100 artists cover {covered}/{unreachable_count} ({coverage_ratio:.0%}) of unreachable cases")
    print(f"  Recovery rate among covered: {recovered_union}/{covered} ({recovered_union/covered:.1%})" if covered else "")
    projected = int(recovered_union / coverage_ratio) if coverage_ratio > 0 else recovered_union
    print(f"  Projected full-artist recovery: ~{projected}")
    print(f"  Gate (>=150 unreachable): {'PASS' if projected >= 150 else 'FAIL'}")

    print(f"\nExamples of GT recovery via MB relations:")
    for ex in examples[:10]:
        print(f"  GT: {ex['gt_track']} by {ex['gt_name']} (pop={ex['pop']})")
        print(f"    Played artist: {ex['played_artist']}")
        print(f"    MB relation match: {ex['matched_via']}")
        print()

    # Save results
    out = {
        "request_count": request_count,
        "artists_mapped": len(aid_to_mb_id),
        "recovered_relation": recovered_relation,
        "recovered_union": recovered_union,
        "pop0_recovered": pop0_recovered,
        "diff_artist_recovered": diff_artist_recovered,
        "hist0_recovered": hist0_recovered,
        "projected_full": projected,
        "examples": examples,
    }
    out_path = REPO_ROOT / "cache" / "mb_recovery_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
