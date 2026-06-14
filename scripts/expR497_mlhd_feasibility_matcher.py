#!/usr/bin/env python3
"""R497: CPU-only MLHD feasibility matcher.

This is a no-GT source-session feasibility tool. It scans MLHD-style per-user
listening logs and asks whether the provided challenge played tracks can identify
a source user/day using MusicBrainz recording MBIDs.

Expected MLHD row shape is headerless TSV:
    timestamp, artist_mbid, release_mbid, recording_mbid

The script streams files line-by-line. It does not load MLHD into memory.
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, TextIO

BRIDGE = Path("exp/eval/expR497_musicbrainz_isrc_bridge.json")
OUT = Path("exp/eval/expR497_mlhd_feasibility_matches.json")


@dataclass
class Target:
    session_id: str
    session_prefix: str
    session_date: str
    mbids: set[str]
    track_names_by_mbid: dict[str, str]


@dataclass
class Hit:
    user_id: str
    file: str
    matched_mbids: set[str] = field(default_factory=set)
    day_recording_count: int = 0
    day_unique_recording_count: int = 0
    day_sample_mbids: list[str] = field(default_factory=list)


def parse_date(date: str) -> datetime:
    return datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def parse_ts(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    if re.fullmatch(r"\d{9,10}", value):
        return datetime.fromtimestamp(int(value), timezone.utc)
    if re.fullmatch(r"\d{12,13}", value):
        return datetime.fromtimestamp(int(value) / 1000.0, timezone.utc)
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value[:19] if fmt != "%Y-%m-%d" else value[:10], fmt).replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            pass
    return None


def open_text(path: Path) -> TextIO:
    raw = path.open("rb")
    if path.suffix == ".gz":
        return gzip.open(raw, mode="rt", errors="replace")
    if path.suffix == ".bz2":
        return bz2.open(raw, mode="rt", errors="replace")
    if path.suffix in {".zst", ".zstd"}:
        try:
            import zstandard
        except ImportError:
            sys.exit("zstd input requires `uv pip install zstandard`")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), errors="replace")
    return path.open("r", errors="replace")


def iter_files(root: Path, pattern: str) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    yield from root.rglob(pattern)


def load_targets(path: Path, prefixes: set[str] | None) -> list[Target]:
    bridge = json.loads(path.read_text())
    targets = []
    for row in bridge["rows"]:
        if prefixes and row["session_prefix"] not in prefixes and row["session_id"] not in prefixes:
            continue
        mbids: set[str] = set()
        names: dict[str, str] = {}
        for track in row["played_tracks"]:
            label = f"{track.get('track_name')} — {track.get('artist_name')}"
            for match in track.get("isrc_matches") or []:
                best = match.get("best_recording") or {}
                mbid = best.get("recording_mbid")
                if mbid:
                    mbids.add(mbid)
                    names[mbid] = label
        if mbids:
            targets.append(
                Target(
                    session_id=row["session_id"],
                    session_prefix=row["session_prefix"],
                    session_date=row["session_date"],
                    mbids=mbids,
                    track_names_by_mbid=names,
                )
            )
    return targets


def user_id_from_path(path: Path) -> str:
    stem = path.name
    for suffix in (".txt.gz", ".tsv.gz", ".csv.gz", ".txt", ".tsv", ".csv"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem


def scan_file(path: Path, targets: list[Target], window_days: int, sample_limit: int) -> dict[str, Hit]:
    windows = {}
    for t in targets:
        center = parse_date(t.session_date)
        windows[t.session_prefix] = (
            center - timedelta(days=window_days),
            center + timedelta(days=window_days + 1),
        )

    hits = {
        t.session_prefix: Hit(user_id=user_id_from_path(path), file=str(path), day_sample_mbids=[])
        for t in targets
    }
    day_unique = {t.session_prefix: set() for t in targets}

    try:
        fh = open_text(path)
    except OSError:
        return {}

    with fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                parts = line.rstrip("\n").split(",")
            if len(parts) < 4:
                continue
            ts = parse_ts(parts[0])
            if ts is None:
                continue
            rec_mbid = parts[3].strip()
            if not rec_mbid:
                continue
            for target in targets:
                start, end = windows[target.session_prefix]
                if not (start <= ts < end):
                    continue
                hit = hits[target.session_prefix]
                hit.day_recording_count += 1
                day_unique[target.session_prefix].add(rec_mbid)
                if len(hit.day_sample_mbids) < sample_limit:
                    hit.day_sample_mbids.append(rec_mbid)
                if rec_mbid in target.mbids:
                    hit.matched_mbids.add(rec_mbid)

    out = {}
    for prefix, hit in hits.items():
        hit.day_unique_recording_count = len(day_unique[prefix])
        if hit.matched_mbids:
            out[prefix] = hit
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mlhd_root", type=Path)
    parser.add_argument("--bridge", type=Path, default=BRIDGE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--session", action="append", help="session prefix/id to include; repeatable")
    parser.add_argument("--pattern", default="*", help="file glob under mlhd_root; default scans all")
    parser.add_argument("--window-days", type=int, default=0, help="date window around session_date")
    parser.add_argument("--min-matches", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=80)
    args = parser.parse_args()

    targets = load_targets(args.bridge, set(args.session or []) or None)
    if not targets:
        sys.exit("no targets with bridged MBIDs selected")
    print("targets:")
    for t in targets:
        print(f"  {t.session_prefix} {t.session_date} mbids={len(t.mbids)}")

    matches: dict[str, list[Hit]] = defaultdict(list)
    files_seen = 0
    for path in iter_files(args.mlhd_root, args.pattern):
        if not path.is_file():
            continue
        files_seen += 1
        if files_seen % 1000 == 0:
            print(f"scanned {files_seen} files", flush=True)
        for prefix, hit in scan_file(path, targets, args.window_days, args.sample_limit).items():
            if len(hit.matched_mbids) >= args.min_matches:
                matches[prefix].append(hit)
        if args.max_files and files_seen >= args.max_files:
            break

    serial = {}
    for target in targets:
        rows = sorted(
            matches.get(target.session_prefix, []),
            key=lambda h: (-len(h.matched_mbids), h.day_unique_recording_count, h.user_id),
        )
        serial[target.session_prefix] = [
            {
                "user_id": h.user_id,
                "file": h.file,
                "matched_count": len(h.matched_mbids),
                "matched_mbids": sorted(h.matched_mbids),
                "matched_tracks": [target.track_names_by_mbid.get(m, m) for m in sorted(h.matched_mbids)],
                "day_recording_count": h.day_recording_count,
                "day_unique_recording_count": h.day_unique_recording_count,
                "day_sample_mbids": h.day_sample_mbids,
            }
            for h in rows[:50]
        ]

    summary = {
        "files_seen": files_seen,
        "targets": len(targets),
        "window_days": args.window_days,
        "min_matches": args.min_matches,
        "candidate_counts": {k: len(v) for k, v in serial.items()},
        "feasible_under_100": {k: 0 < len(v) <= 100 for k, v in serial.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "matches": serial}, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
