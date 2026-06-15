#!/usr/bin/env python3
"""R497: stream MLHD+ tar shards and look for source-session fingerprints.

This is research-only and does not use hidden labels. It scans official MLHD+
tar shards over HTTP or from local disk without extracting the dataset. The
target signal is whether a per-user listening log contains the challenge's
played-track MusicBrainz recording MBIDs on the challenge session date.

Default target rows are the three bridgeable MLHD-era rows from R497:
  ca8cbe02, a1df8767, 2bfd631e.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import BinaryIO, Iterable

import requests

BRIDGE = Path("exp/eval/expR497_musicbrainz_isrc_bridge.json")
OUT = Path("exp/eval/expR497_mlhdplus_stream_matches.json")
BASE_URL = "https://data.musicbrainz.org/pub/musicbrainz/listenbrainz/mlhd/"
DEFAULT_SHARDS = [f"mlhdplus-complete-{x}.tar" for x in "0123456789abcdef"]
DEFAULT_SESSIONS = {"ca8cbe02", "a1df8767", "2bfd631e"}


@dataclass
class Target:
    session_id: str
    session_prefix: str
    session_date: str
    start_ts: int
    end_ts: int
    mbids: set[str]
    track_names_by_mbid: dict[str, str]


@dataclass
class FileHit:
    source: str
    member: str
    user_id: str
    day_recording_count: int = 0
    day_unique: set[str] = field(default_factory=set)
    matched_mbids: set[str] = field(default_factory=set)
    sample_mbids: list[str] = field(default_factory=list)


def parse_date_window(date: str, window_days: int) -> tuple[int, int]:
    center = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start = center - timedelta(days=window_days)
    end = center + timedelta(days=window_days + 1)
    return int(start.timestamp()), int(end.timestamp())


def parse_ts(value: bytes) -> int | None:
    s = value.strip()
    if re.fullmatch(rb"\d{9,10}", s):
        return int(s)
    if re.fullmatch(rb"\d{12,13}", s):
        return int(s) // 1000
    return None


def user_id_from_member(name: str) -> str:
    basename = name.rsplit("/", 1)[-1]
    for suffix in (".txt", ".tsv", ".csv", ".gz", ".bz2", ".zst"):
        if basename.endswith(suffix):
            basename = basename[: -len(suffix)]
    return basename


def load_targets(path: Path, sessions: set[str], window_days: int) -> list[Target]:
    bridge = json.loads(path.read_text())
    targets = []
    for row in bridge["rows"]:
        if sessions and row["session_prefix"] not in sessions and row["session_id"] not in sessions:
            continue
        mbids: set[str] = set()
        names = {}
        for track in row["played_tracks"]:
            label = f"{track.get('track_name')} — {track.get('artist_name')}"
            for match in track.get("isrc_matches") or []:
                best = match.get("best_recording") or {}
                mbid = best.get("recording_mbid")
                if mbid:
                    mbids.add(mbid)
                    names[mbid] = label
        if not mbids:
            continue
        start, end = parse_date_window(row["session_date"], window_days)
        targets.append(
            Target(
                session_id=row["session_id"],
                session_prefix=row["session_prefix"],
                session_date=row["session_date"],
                start_ts=start,
                end_ts=end,
                mbids=mbids,
                track_names_by_mbid=names,
            )
        )
    return targets


def iter_sources(args: argparse.Namespace) -> Iterable[str]:
    if args.source:
        yield from args.source
        return
    for shard in DEFAULT_SHARDS:
        yield BASE_URL + shard


def open_source(source: str) -> tuple[BinaryIO, requests.Response | None]:
    if source.startswith(("http://", "https://")):
        response = requests.get(source, stream=True, timeout=120)
        response.raise_for_status()
        response.raw.decode_content = True
        return response.raw, response
    return Path(source).open("rb"), None


def scan_member(
    source: str,
    member_name: str,
    fh: BinaryIO,
    targets: list[Target],
    sample_limit: int,
) -> dict[str, FileHit]:
    hits = {
        t.session_prefix: FileHit(source=source, member=member_name, user_id=user_id_from_member(member_name))
        for t in targets
    }
    any_relevant = False

    zreader = None
    if member_name.endswith(".zst"):
        import zstandard

        zreader = zstandard.ZstdDecompressor().stream_reader(fh)
        reader: BinaryIO = io.BufferedReader(zreader)
    else:
        reader = fh

    try:
        data = reader.read()
        target_needles = [mbid.encode("utf-8") for target in targets for mbid in target.mbids]
        if not any(needle in data for needle in target_needles):
            return {}

        for raw_line in data.splitlines():
            parts = raw_line.rstrip(b"\n").split(b"\t")
            if len(parts) < 4:
                parts = raw_line.rstrip(b"\n").split(b",")
            if len(parts) < 4:
                continue
            ts = parse_ts(parts[0])
            if ts is None:
                continue
            rec = parts[3].decode("utf-8", errors="ignore").strip()
            if not rec:
                continue
            for target in targets:
                if not (target.start_ts <= ts < target.end_ts):
                    continue
                hit = hits[target.session_prefix]
                hit.day_recording_count += 1
                hit.day_unique.add(rec)
                if len(hit.sample_mbids) < sample_limit:
                    hit.sample_mbids.append(rec)
                if rec in target.mbids:
                    any_relevant = True
                    hit.matched_mbids.add(rec)
    finally:
        if zreader is not None:
            zreader.close()

    if not any_relevant:
        return {}
    return {k: v for k, v in hits.items() if v.matched_mbids}


def scan_tar_source(source: str, targets: list[Target], args: argparse.Namespace) -> dict[str, list[FileHit]]:
    print(f"scan source: {source}", flush=True)
    raw, response = open_source(source)
    out: dict[str, list[FileHit]] = {t.session_prefix: [] for t in targets}
    members_seen = 0
    start = time.time()
    try:
        with tarfile.open(fileobj=raw, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                members_seen += 1
                if args.max_members and members_seen > args.max_members:
                    break
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                member_hits = scan_member(source, member.name, extracted, targets, args.sample_limit)
                for prefix, hit in member_hits.items():
                    if len(hit.matched_mbids) >= args.min_matches:
                        out[prefix].append(hit)
                        print(
                            f"  HIT {prefix} user={hit.user_id} "
                            f"matched={len(hit.matched_mbids)} day_unique={len(hit.day_unique)}",
                            flush=True,
                        )
                if members_seen % args.progress_every == 0:
                    elapsed = max(time.time() - start, 1.0)
                    print(f"  members={members_seen} elapsed={elapsed/60:.1f}m", flush=True)
    finally:
        raw.close()
        if response is not None:
            response.close()
    return out


def serialize(matches: dict[str, list[FileHit]], targets: list[Target], sources: list[str]) -> dict:
    target_by_prefix = {t.session_prefix: t for t in targets}
    serial_matches = {}
    for prefix, rows in matches.items():
        target = target_by_prefix[prefix]
        rows = sorted(rows, key=lambda h: (-len(h.matched_mbids), len(h.day_unique), h.user_id))
        serial_matches[prefix] = [
            {
                "source": h.source,
                "member": h.member,
                "user_id": h.user_id,
                "matched_count": len(h.matched_mbids),
                "matched_mbids": sorted(h.matched_mbids),
                "matched_tracks": [target.track_names_by_mbid.get(m, m) for m in sorted(h.matched_mbids)],
                "day_recording_count": h.day_recording_count,
                "day_unique_recording_count": len(h.day_unique),
                "day_sample_mbids": h.sample_mbids,
            }
            for h in rows[:100]
        ]
    return {
        "summary": {
            "sources": sources,
            "targets": [t.session_prefix for t in targets],
            "candidate_counts": {k: len(v) for k, v in serial_matches.items()},
            "feasible_under_100": {k: 0 < len(v) <= 100 for k, v in serial_matches.items()},
        },
        "matches": serial_matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", help="tar path or URL; repeatable. default: all 16 partial shards")
    parser.add_argument("--bridge", type=Path, default=BRIDGE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--session", action="append", help="session prefix/id; default: three bridgeable rows")
    parser.add_argument("--window-days", type=int, default=0)
    parser.add_argument("--min-matches", type=int, default=2)
    parser.add_argument("--max-members", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    sessions = set(args.session or DEFAULT_SESSIONS)
    targets = load_targets(args.bridge, sessions, args.window_days)
    if not targets:
        raise SystemExit("no selected targets with MBIDs")
    print("targets:")
    for t in targets:
        print(f"  {t.session_prefix} {t.session_date} mbids={len(t.mbids)}", flush=True)

    sources = list(iter_sources(args))
    all_matches: dict[str, list[FileHit]] = {t.session_prefix: [] for t in targets}
    for source in sources:
        source_matches = scan_tar_source(source, targets, args)
        for prefix, rows in source_matches.items():
            all_matches[prefix].extend(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(serialize(all_matches, targets, sources), indent=2, ensure_ascii=False) + "\n")

    result = serialize(all_matches, targets, sources)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result["summary"], indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
