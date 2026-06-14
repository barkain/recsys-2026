#!/usr/bin/env python3
"""R497: resolve played-track ISRCs to MusicBrainz recording MBIDs.

Research-only bridge for the source-session feasibility path. The challenge track
metadata has ISRCs, while MLHD-style listening logs are MusicBrainz-recording-MBID
based. This script maps only the provided played tracks in selected R497 target
buckets; it does not use hidden labels or ground-truth tracks.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

TARGETS = Path("exp/eval/expR497_source_fingerprint_targets.json")
OUT = Path("exp/eval/expR497_musicbrainz_isrc_bridge.json")
CACHE = Path("cache/r497_musicbrainz_isrc")

MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "Recsys2026-R497-Research/0.1 (https://github.com/nadavbarkai/recsys-2026)"
RATE_LIMIT_SEC = 1.1


def norm_text(value: object) -> str:
    if isinstance(value, list):
        value = " ".join(str(x) for x in value if x)
    value = "" if value is None else str(value)
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def text_match(local: object, mb: object) -> bool:
    a = norm_text(local)
    b = norm_text(mb)
    return bool(a and b and (a in b or b in a))


def cache_path(isrc: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", isrc.upper())
    return CACHE / f"{safe}.json"


def lookup_isrc(isrc: str, *, refresh: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(isrc)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    url = f"{MB_BASE}/isrc/{quote(isrc)}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    params = {"fmt": "json"}
    for attempt in range(4):
        if attempt:
            time.sleep(RATE_LIMIT_SEC * (attempt + 1))
        else:
            time.sleep(RATE_LIMIT_SEC)
        try:
            response = requests.get(url, params=params, headers=headers, timeout=60)
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
        if response.status_code in {404, 400}:
            data = {"error": f"http_{response.status_code}", "detail": response.text[:500]}
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


def compact_recording(recording: dict, track: dict) -> dict:
    artists = [
        (credit.get("artist") or {}).get("name")
        for credit in recording.get("artist-credit", [])
        if isinstance(credit, dict)
    ]
    local_title = track.get("track_name")
    local_artist = track.get("artist_name")
    title_ok = text_match(local_title, recording.get("title"))
    artist_ok = any(text_match(local_artist, artist) for artist in artists)
    same_track = bool(title_ok and (artist_ok or not artists))
    return {
        "recording_mbid": recording.get("id"),
        "title": recording.get("title"),
        "artists": artists,
        "disambiguation": recording.get("disambiguation"),
        "length": recording.get("length"),
        "title_match": title_ok,
        "artist_match": artist_ok,
        "artist_credit_present": bool(artists),
        "looks_like_same_track": same_track,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=TARGETS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--bucket",
        action="append",
        default=["MLHD_STRONG"],
        help="target bucket to include; repeatable. default: MLHD_STRONG",
    )
    parser.add_argument("--limit-sessions", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.targets.read_text())
    buckets = set(args.bucket)
    targets = [t for t in data["targets"] if t["target_bucket"] in buckets]
    if args.limit_sessions:
        targets = targets[: args.limit_sessions]

    unique_isrcs = []
    seen = set()
    for target in targets:
        for track in target["played_tracks"]:
            for isrc in track.get("isrc") or []:
                isrc = str(isrc).strip().upper()
                if isrc and isrc not in seen:
                    seen.add(isrc)
                    unique_isrcs.append(isrc)

    isrc_payloads = {}
    for i, isrc in enumerate(unique_isrcs, start=1):
        print(f"[{i}/{len(unique_isrcs)}] {isrc}", flush=True)
        isrc_payloads[isrc] = lookup_isrc(isrc, refresh=args.refresh)

    rows = []
    matched = 0
    total = 0
    for target in targets:
        out_tracks = []
        for track in target["played_tracks"]:
            track_matches = []
            for isrc in track.get("isrc") or []:
                total += 1
                payload = isrc_payloads.get(str(isrc).strip().upper(), {})
                recordings = payload.get("recordings") or []
                compact = [compact_recording(r, track) for r in recordings[:5]]
                best = next((r for r in compact if r["looks_like_same_track"]), compact[0] if compact else None)
                if best and best.get("recording_mbid"):
                    matched += 1
                track_matches.append(
                    {
                        "isrc": isrc,
                        "error": payload.get("error"),
                        "recording_count": len(recordings),
                        "best_recording": best,
                        "top_recordings": compact,
                    }
                )
            out_tracks.append(
                {
                    "track_id": track.get("track_id"),
                    "track_name": track.get("track_name"),
                    "artist_name": track.get("artist_name"),
                    "isrc_matches": track_matches,
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
        "unique_isrcs": len(unique_isrcs),
        "track_isrc_lookups": total,
        "lookups_with_recording": matched,
        "coverage": matched / total if total else 0.0,
        "interpretation": (
            "If coverage is high, these rows are mechanically bridgeable from challenge ISRCs "
            "to MLHD recording MBIDs. This is necessary but not sufficient for lawful source-session matching."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
