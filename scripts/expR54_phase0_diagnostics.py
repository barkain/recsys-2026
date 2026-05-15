#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R54 Phase 0: Data inventory and diagnostics. No model training.

1. Count train-split sessions and extractable (query, GT) pairs.
2. Verify GT track coverage in catalog metadata.
3. Count same-session positive candidates and assess coverage.
4. Inventory structural metadata fields (year, duration) coverage.
5. Output: exp/eval/expR54_phase0_diagnostics.json
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def load_catalog():
    from datasets import Dataset  # type: ignore[reportMissingImports]
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    if not matches:
        raise FileNotFoundError("all_tracks arrow not found")
    ds = Dataset.from_file(str(matches[-1]))
    cols = ds.to_dict()
    meta = {}
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        meta[tid] = {k: cols[k][i] for k in cols}
    return meta


def extract_year(meta_entry):
    rd = meta_entry.get("release_date")
    if not rd:
        return None
    rd_str = str(rd)
    m = re.match(r"(\d{4})", rd_str)
    if m:
        y = int(m.group(1))
        if 1900 <= y <= 2030:
            return y
    return None


def extract_duration(meta_entry):
    d = meta_entry.get("duration")
    if d is None:
        return None
    try:
        val = float(d)
        if val > 0:
            return val
    except (ValueError, TypeError):
        pass
    d_ms = meta_entry.get("duration_ms")
    if d_ms is not None:
        try:
            val = float(d_ms) / 1000.0
            if val > 0:
                return val
        except (ValueError, TypeError):
            pass
    return None


def main():
    t0 = time.time()
    results = {}

    # --- 1. Load dev payload ---
    print(f"{ts()} Loading dev payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    dev_sessions = set(c["session_id"] for c in cases)
    dev_gts = set(c["gt"] for c in cases)
    print(f"  Dev: {len(cases)} cases, {len(dev_sessions)} sessions, {len(dev_gts)} unique GTs")
    results["dev"] = {
        "n_cases": len(cases),
        "n_sessions": len(dev_sessions),
        "n_unique_gts": len(dev_gts),
    }

    # --- 2. Load catalog metadata ---
    print(f"{ts()} Loading catalog metadata...", flush=True)
    meta = load_catalog()
    print(f"  Catalog: {len(meta)} tracks")
    results["catalog_size"] = len(meta)

    # --- 3. Train-split inventory ---
    print(f"{ts()} Loading train split...", flush=True)
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]

    n_train_sessions = 0
    n_train_dev_overlap = 0
    n_pairs = 0
    n_pairs_gt_in_catalog = 0
    n_pairs_gt_missing = 0
    gt_track_ids = set()
    music_turns_per_session = []

    for item in train_ds:
        sid = item["session_id"]
        if sid in dev_sessions:
            n_train_dev_overlap += 1
            continue
        n_train_sessions += 1

        convs = item["conversations"]
        session_music_count = 0
        has_prior_user = False
        for conv in convs:
            role = conv["role"]
            if role == "user":
                has_prior_user = True
            elif role == "music" and has_prior_user:
                track_id = str(conv["content"]).strip()
                session_music_count += 1
                gt_track_ids.add(track_id)
                if track_id in meta:
                    n_pairs_gt_in_catalog += 1
                else:
                    n_pairs_gt_missing += 1
                n_pairs += 1

        music_turns_per_session.append(session_music_count)

    mt_arr = np.array(music_turns_per_session)
    print(f"  Train sessions: {n_train_sessions}")
    print(f"  Train-dev overlap: {n_train_dev_overlap}")
    print(f"  Total pairs: {n_pairs}")
    print(f"  Pairs with GT in catalog: {n_pairs_gt_in_catalog} ({n_pairs_gt_in_catalog/max(n_pairs,1):.1%})")
    print(f"  Pairs with GT missing: {n_pairs_gt_missing}")
    print(f"  Unique GT tracks: {len(gt_track_ids)}")
    print(f"  Music turns/session: mean={mt_arr.mean():.1f}, median={np.median(mt_arr):.0f}, "
          f"min={mt_arr.min()}, max={mt_arr.max()}")

    results["train_split"] = {
        "n_sessions": n_train_sessions,
        "n_dev_overlap": n_train_dev_overlap,
        "n_pairs": n_pairs,
        "n_pairs_gt_in_catalog": n_pairs_gt_in_catalog,
        "n_pairs_gt_missing": n_pairs_gt_missing,
        "n_unique_gts": len(gt_track_ids),
        "music_turns_per_session_mean": float(mt_arr.mean()),
        "music_turns_per_session_median": float(np.median(mt_arr)),
        "music_turns_per_session_min": int(mt_arr.min()),
        "music_turns_per_session_max": int(mt_arr.max()),
    }

    # --- 4. Dev GT coverage in catalog ---
    dev_gt_in_catalog = sum(1 for gt in dev_gts if gt in meta)
    print(f"\n{ts()} Dev GT coverage: {dev_gt_in_catalog}/{len(dev_gts)} "
          f"({dev_gt_in_catalog/len(dev_gts):.1%}) in catalog")
    results["dev_gt_coverage"] = {
        "in_catalog": dev_gt_in_catalog,
        "total": len(dev_gts),
        "rate": dev_gt_in_catalog / len(dev_gts),
    }

    # --- 5. Same-session positive candidates ---
    print(f"\n{ts()} Counting same-session positive candidates...", flush=True)
    n_same_session_candidates = 0
    n_sessions_with_candidates = 0

    for item in train_ds:
        sid = item["session_id"]
        convs = item["conversations"]

        music_tracks = []
        user_positive_after_music = []

        for i, conv in enumerate(convs):
            role = conv["role"]
            content = str(conv["content"])

            if role == "music":
                track_id = content.strip()
                music_tracks.append((i, track_id))
            elif role == "user" and music_tracks:
                low = content.lower()
                positive_signals = ["love", "great", "perfect", "yes", "more like",
                                    "exactly", "amazing", "awesome", "fantastic",
                                    "that's what i", "keep going", "nice"]
                if any(s in low for s in positive_signals):
                    last_music_idx, last_track = music_tracks[-1]
                    if last_track in meta:
                        user_positive_after_music.append(last_track)

        if user_positive_after_music:
            n_sessions_with_candidates += 1
            n_same_session_candidates += len(user_positive_after_music)

    # Also check dev sessions
    n_dev_same_session = 0
    for c in cases:
        history = c.get("history", [])
        music_tracks_dev = []
        for h in history:
            role = h.get("role", "")
            content = str(h.get("content", ""))
            if role == "music":
                music_tracks_dev.append(content.strip())
            elif role == "user" and music_tracks_dev:
                low = content.lower()
                positive_signals = ["love", "great", "perfect", "yes", "more like",
                                    "exactly", "amazing", "awesome", "fantastic",
                                    "that's what i", "keep going", "nice"]
                if any(s in low for s in positive_signals):
                    last_track = music_tracks_dev[-1]
                    if last_track in meta:
                        n_dev_same_session += 1

    print(f"  Train: {n_same_session_candidates} candidates across {n_sessions_with_candidates} sessions")
    print(f"  Dev: {n_dev_same_session} candidates")
    total_same_session = n_same_session_candidates + n_dev_same_session
    print(f"  Total: {total_same_session} (threshold: 50)")
    print(f"  Decision: {'USE' if total_same_session >= 50 else 'SKIP'}")

    results["same_session_positives"] = {
        "train_candidates": n_same_session_candidates,
        "train_sessions_with_candidates": n_sessions_with_candidates,
        "dev_candidates": n_dev_same_session,
        "total": total_same_session,
        "decision": "use" if total_same_session >= 50 else "skip",
    }

    # --- 6. Structural metadata coverage ---
    print(f"\n{ts()} Inventorying structural metadata...", flush=True)
    n_tracks = len(meta)
    year_count = 0
    duration_count = 0
    album_count = 0
    tag_count = 0
    artist_count = 0

    year_dist = Counter()
    duration_dist = {"short": 0, "medium": 0, "long": 0}
    tag_lengths = []

    for tid, m in meta.items():
        y = extract_year(m)
        if y is not None:
            year_count += 1
            decade = (y // 10) * 10
            year_dist[decade] += 1

        d = extract_duration(m)
        if d is not None:
            duration_count += 1
            d_sec = d / 1000.0 if d > 1000 else d
            if d_sec < 180:
                duration_dist["short"] += 1
            elif d_sec <= 300:
                duration_dist["medium"] += 1
            else:
                duration_dist["long"] += 1

        album = m.get("album_name")
        if album:
            if isinstance(album, list):
                if album and album[0]:
                    album_count += 1
            elif str(album).strip():
                album_count += 1

        artists = m.get("artist_name")
        if artists:
            if isinstance(artists, list):
                if artists and artists[0]:
                    artist_count += 1
            elif str(artists).strip():
                artist_count += 1

        tags = m.get("tag_list")
        if tags:
            if isinstance(tags, list) and tags:
                tag_count += 1
                tag_lengths.append(len(tags))
            elif str(tags).strip():
                tag_count += 1
                tag_lengths.append(1)

    tag_arr = np.array(tag_lengths) if tag_lengths else np.array([0])

    print(f"  Year:     {year_count}/{n_tracks} ({year_count/n_tracks:.1%})")
    print(f"  Duration: {duration_count}/{n_tracks} ({duration_count/n_tracks:.1%})")
    print(f"  Album:    {album_count}/{n_tracks} ({album_count/n_tracks:.1%})")
    print(f"  Artist:   {artist_count}/{n_tracks} ({artist_count/n_tracks:.1%})")
    print(f"  Tags:     {tag_count}/{n_tracks} ({tag_count/n_tracks:.1%})")
    print(f"  Tags/track: mean={tag_arr.mean():.1f}, median={np.median(tag_arr):.0f}, max={tag_arr.max()}")
    print(f"  Year decades: {dict(sorted(year_dist.items()))}")
    print(f"  Duration buckets: {duration_dist}")

    results["metadata_coverage"] = {
        "year": {"count": year_count, "total": n_tracks, "rate": year_count / n_tracks},
        "duration": {"count": duration_count, "total": n_tracks, "rate": duration_count / n_tracks},
        "album": {"count": album_count, "total": n_tracks, "rate": album_count / n_tracks},
        "artist": {"count": artist_count, "total": n_tracks, "rate": artist_count / n_tracks},
        "tags": {
            "count": tag_count, "total": n_tracks, "rate": tag_count / n_tracks,
            "tags_per_track_mean": float(tag_arr.mean()),
            "tags_per_track_median": float(np.median(tag_arr)),
        },
        "year_decades": {str(k): v for k, v in sorted(year_dist.items())},
        "duration_buckets": duration_dist,
    }

    # --- 7. Coverage of enriched track text on GT tracks ---
    print(f"\n{ts()} Checking enriched-text coverage on dev GTs...", flush=True)
    gt_year = sum(1 for gt in dev_gts if gt in meta and extract_year(meta[gt]) is not None)
    gt_dur = sum(1 for gt in dev_gts if gt in meta and extract_duration(meta[gt]) is not None)
    gt_album = sum(1 for gt in dev_gts if gt in meta and (
        (isinstance(meta[gt].get("album_name"), list) and meta[gt]["album_name"] and meta[gt]["album_name"][0])
        or (not isinstance(meta[gt].get("album_name"), list) and str(meta[gt].get("album_name", "")).strip())
    ))
    n_dev_gt_in_cat = sum(1 for gt in dev_gts if gt in meta)
    print(f"  Dev GTs in catalog: {n_dev_gt_in_cat}")
    print(f"  Year on dev GTs: {gt_year}/{n_dev_gt_in_cat} ({gt_year/max(n_dev_gt_in_cat,1):.1%})")
    print(f"  Duration on dev GTs: {gt_dur}/{n_dev_gt_in_cat} ({gt_dur/max(n_dev_gt_in_cat,1):.1%})")
    print(f"  Album on dev GTs: {gt_album}/{n_dev_gt_in_cat} ({gt_album/max(n_dev_gt_in_cat,1):.1%})")

    results["dev_gt_enrichment"] = {
        "n_in_catalog": n_dev_gt_in_cat,
        "year": {"count": gt_year, "rate": gt_year / max(n_dev_gt_in_cat, 1)},
        "duration": {"count": gt_dur, "rate": gt_dur / max(n_dev_gt_in_cat, 1)},
        "album": {"count": gt_album, "rate": gt_album / max(n_dev_gt_in_cat, 1)},
    }

    # --- Summary ---
    elapsed = time.time() - t0
    results["elapsed_s"] = elapsed
    results["created_at"] = datetime.now().isoformat()

    out_path = REPO / "exp" / "eval" / "expR54_phase0_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{ts()} Phase 0 complete. Elapsed: {elapsed:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
