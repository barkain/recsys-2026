#!/usr/bin/env python3
"""R497: resolve played tracks to MusicBrainz recording MBIDs by title/artist.

This complements the ISRC bridge. It is lower precision than an ISRC lookup, but
it can cover MLHD-era rows whose challenge catalog metadata has no usable ISRC.

Output is intentionally scanner-compatible: each played track receives an
`isrc_matches` list containing `best_recording`, so
`expR497_stream_mlhdplus_scan.py` can load this file unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import requests

TARGETS = Path("exp/eval/expR497_source_fingerprint_targets.json")
OUT = Path("exp/eval/expR497_musicbrainz_text_bridge.json")
CACHE = Path("cache/r497_musicbrainz_text")

MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "Recsys2026-R497-Research/0.1 (https://github.com/nadavbarkai/recsys-2026)"
RATE_LIMIT_SEC = 1.1


def scalar(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return "" if value is None else str(value)


def norm_text(value: object) -> str:
    value = scalar(value).lower()
    value = value.replace("&", " and ")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\[[^)]*\]", " ", value)
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def loose_title(value: object) -> str:
    s = norm_text(value)
    # Strip common version/remaster decorations while preserving meaningful title.
    s = re.sub(r"\b(20\\d\\d|19\\d\\d|remaster(ed)?|version|explicit|mono|stereo|live|edit|single|album|digital|rarities?)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cache_path(title: str, artist: str) -> Path:
    key = f"{title}__{artist}".lower()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", key)[:180]
    return CACHE / f"{safe}.json"


def search_recording(title: str, artist: str, *, refresh: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(title, artist)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    query = f'recording:"{title}" AND artist:"{artist}"'
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    params = {"fmt": "json", "query": query, "limit": 10}
    for attempt in range(4):
        time.sleep(RATE_LIMIT_SEC if attempt == 0 else RATE_LIMIT_SEC * (attempt + 1))
        try:
            response = requests.get(f"{MB_BASE}/recording", params=params, headers=headers, timeout=60)
        except requests.RequestException as exc:
            if attempt == 3:
                data = {"error": "request_exception", "detail": str(exc)}
                path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
                return data
            continue
        if response.status_code == 200:
            data = response.json()
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            return data
        if response.status_code in {429, 503}:
            time.sleep(5 + attempt * 5)
            continue
        data = {"error": f"http_{response.status_code}", "detail": response.text[:500]}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return data
    data = {"error": "max_retries"}
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return data


def recording_artists(recording: dict) -> list[str]:
    return [
        (credit.get("artist") or {}).get("name")
        for credit in recording.get("artist-credit", [])
        if isinstance(credit, dict) and (credit.get("artist") or {}).get("name")
    ]


def compact_recording(recording: dict, track: dict) -> dict:
    local_title = norm_text(track.get("track_name"))
    local_title_loose = loose_title(track.get("track_name"))
    mb_title = norm_text(recording.get("title"))
    mb_title_loose = loose_title(recording.get("title"))
    local_artist = norm_text(track.get("artist_name"))
    artists = recording_artists(recording)
    artist_ok = any(norm_text(a) == local_artist or local_artist in norm_text(a) or norm_text(a) in local_artist for a in artists)
    title_exact = bool(local_title and local_title == mb_title)
    title_loose = bool(local_title_loose and (local_title_loose == mb_title_loose or local_title_loose in mb_title_loose or mb_title_loose in local_title_loose))
    score = float(recording.get("score") or 0)
    same_track = bool(artist_ok and (title_exact or title_loose) and score >= 80)
    return {
        "recording_mbid": recording.get("id"),
        "title": recording.get("title"),
        "artists": artists,
        "score": score,
        "disambiguation": recording.get("disambiguation"),
        "length": recording.get("length"),
        "title_exact": title_exact,
        "title_loose": title_loose,
        "artist_match": artist_ok,
        "looks_like_same_track": same_track,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--bucket", action="append", default=["MLHD_STRONG"])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.targets.read_text())
    buckets = set(args.bucket)
    targets = [t for t in data["targets"] if t["target_bucket"] in buckets]

    rows = []
    total = 0
    matched = 0
    for ti, target in enumerate(targets, start=1):
        out_tracks = []
        print(f"session [{ti}/{len(targets)}] {target['session_prefix']}", flush=True)
        for track in target["played_tracks"]:
            total += 1
            title = scalar(track.get("track_name"))
            artist = scalar(track.get("artist_name"))
            payload = search_recording(title, artist, refresh=args.refresh)
            compact = [compact_recording(r, track) for r in payload.get("recordings") or []]
            best = next((r for r in compact if r["looks_like_same_track"]), compact[0] if compact else None)
            if best and best.get("recording_mbid") and best.get("looks_like_same_track"):
                matched += 1
            out_tracks.append(
                {
                    "track_id": track.get("track_id"),
                    "track_name": track.get("track_name"),
                    "artist_name": track.get("artist_name"),
                    "isrc_matches": [
                        {
                            "isrc": None,
                            "bridge_method": "musicbrainz_text_search",
                            "error": payload.get("error"),
                            "recording_count": len(payload.get("recordings") or []),
                            "best_recording": best if best and best.get("looks_like_same_track") else None,
                            "top_recordings": compact[:5],
                        }
                    ],
                }
            )
        rows.append(
            {
                "session_id": target["session_id"],
                "session_prefix": target["session_prefix"],
                "session_date": target["session_date"],
                "target_bucket": target["target_bucket"],
                "profile_key": target.get("profile_key"),
                "query": target.get("query"),
                "played_tracks": out_tracks,
            }
        )

    summary = {
        "target_buckets": sorted(buckets),
        "sessions": len(rows),
        "track_text_lookups": total,
        "lookups_with_confident_recording": matched,
        "coverage": matched / total if total else 0.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
