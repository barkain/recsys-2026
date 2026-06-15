#!/usr/bin/env python3
"""R497: map recovered MLHD+ source-day recording MBIDs to challenge catalog IDs.

Input is the stream match report from `expR497_stream_mlhdplus_scan.py`.
For each source-day MBID, query MusicBrainz recording details for ISRCs, then join
those ISRCs to the TalkPlayData challenge catalog.

Research-only. Does not use hidden labels.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import quote

import requests

MATCHES = Path("exp/eval/expR497_mlhdplus_complete_stream_matches.json")
OUT = Path("exp/eval/expR497_source_day_catalog_pool.json")
CACHE = Path("cache/r497_musicbrainz_recording")

MB_BASE = "https://musicbrainz.org/ws/2"
USER_AGENT = "Recsys2026-R497-Research/0.1 (https://github.com/barkain/recsys-2026)"
RATE_LIMIT_SEC = 1.1


def scalar_list(value):
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def cache_path(mbid: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", mbid)
    return CACHE / f"{safe}.json"


def mb_recording(mbid: str, *, refresh: bool = False) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = cache_path(mbid)
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    url = f"{MB_BASE}/recording/{quote(mbid)}"
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    params = {"fmt": "json", "inc": "isrcs+artist-credits+releases"}
    for attempt in range(4):
        time.sleep(RATE_LIMIT_SEC if attempt == 0 else RATE_LIMIT_SEC * (attempt + 1))
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


def load_catalog_by_isrc() -> tuple[dict[str, list[dict]], dict[str, dict]]:
    import glob
    import pyarrow as pa

    pattern = (
        "/Users/nadavbarkai/.cache/huggingface/datasets/"
        "talkpl-ai___talk_play_data-challenge-track-metadata/"
        "default/0.0.0/*/talk_play_data-challenge-track-metadata-all_tracks.arrow"
    )
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit("challenge track metadata Arrow not found")

    by_isrc: dict[str, list[dict]] = {}
    by_tid: dict[str, dict] = {}
    with pa.memory_map(paths[-1], "r") as source:
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            cols = batch.to_pydict()
            for i, tid in enumerate(cols["track_id"]):
                row = {
                    "track_id": tid,
                    "track_name": scalar_list(cols.get("track_name", [None])[i]),
                    "artist_name": scalar_list(cols.get("artist_name", [None])[i]),
                    "album_name": scalar_list(cols.get("album_name", [None])[i]),
                    "isrc": scalar_list(cols.get("ISRC", [None])[i]),
                    "popularity": cols.get("popularity", [None])[i],
                }
                by_tid[tid] = row
                for isrc in row["isrc"]:
                    if isrc:
                        by_isrc.setdefault(str(isrc).upper(), []).append(row)
    return by_isrc, by_tid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=Path, default=MATCHES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--session", default="2bfd631e")
    parser.add_argument("--candidate-index", type=int, default=0)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    match_report = json.loads(args.matches.read_text())
    if "matches" in match_report:
        rows = match_report["matches"].get(args.session) or []
        if not rows:
            raise SystemExit(f"no matches for session {args.session}")
        match = rows[args.candidate_index]
    elif match_report.get("session") == args.session and "day_sample_mbids" in match_report:
        match = match_report
    else:
        raise SystemExit(f"no matches for session {args.session}")
    mbids = list(dict.fromkeys(match["day_sample_mbids"]))
    by_isrc, _ = load_catalog_by_isrc()

    pool = []
    for i, mbid in enumerate(mbids, start=1):
        print(f"[{i}/{len(mbids)}] {mbid}", flush=True)
        rec = mb_recording(mbid, refresh=args.refresh)
        isrcs = [str(x).upper() for x in rec.get("isrcs") or [] if x]
        catalog_hits = []
        for isrc in isrcs:
            catalog_hits.extend(by_isrc.get(isrc, []))
        artists = [
            (credit.get("artist") or {}).get("name")
            for credit in rec.get("artist-credit", [])
            if isinstance(credit, dict)
        ]
        pool.append(
            {
                "source_mbid": mbid,
                "mb_title": rec.get("title"),
                "mb_artists": artists,
                "mb_isrcs": isrcs,
                "catalog_hits": catalog_hits,
            }
        )

    summary = {
        "session": args.session,
        "source_user": match["user_id"],
        "source_member": match["member"],
        "source_day_unique_recordings": match["day_unique_recording_count"],
        "source_day_recordings": match["day_recording_count"],
        "mapped_mbids": sum(1 for p in pool if p["catalog_hits"]),
        "catalog_track_ids": sorted({h["track_id"] for p in pool for h in p["catalog_hits"]}),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "pool": pool}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
