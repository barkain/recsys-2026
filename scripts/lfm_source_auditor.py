#!/usr/bin/env python3
# ruff: noqa: T201
"""LFM source-session auditor — plug in any discovered listening-data file and report
whether it carries the fields needed for session-window reconstruction. Reads only a
bounded number of LEADING rows (a prefix), so it never fully loads multi-GB files on this
Mac. (The prefix is row-bounded, not byte-bounded; a pathological single huge line could
still be read in full.)

Per the reconstruction research track (2026-06-12 pivot): the nDCG gap is a
source-session reconstruction problem (TalkPlayData2 GT is sampled from a hidden 16-32
track pool drawn from a real LFM-2b session). Blind-A was measured locally as
2009-06-16..2018-12-28, so a candidate external dataset is useful ONLY if it has
per-event session structure: user + timestamp + track identity, ideally with
demographics and Spotify/MB/Last.fm mappings, covering that 2009-2018 window.

Usage:
    python scripts/lfm_source_auditor.py PATH [--rows N] [--no-header]

Supports .csv/.tsv/.json/.jsonl, optionally .gz/.bz2/.zst (streams a prefix only).
KNOWN LIMITATIONS (flagged, not auto-handled):
  - Some logs are HEADERLESS per-user files (e.g. MLHD: <timestamp, artist-MBID,
    release-MBID, recording-MBID>) with the user_id encoded in the FILENAME, not in rows.
    Use --no-header for those; user_id presence must be judged from the path, not columns.
  - Track identity via an internal/opaque item_id (no title, no MBID, no Spotify) is NOT
    resolvable without the dataset's own catalog — the verdict flags this.
Reports: detected columns, required-field coverage (fuzzy-matched), an inferred timestamp
range from sampled rows, Blind-A-era overlap, and a usefulness verdict.
"""
from __future__ import annotations
import argparse
import bz2
import csv
import gzip
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# field family -> candidate column-name substrings (lowercased)
FIELDS = {
    "user_id":    ["user_id", "userid", "user_name", "username", "user-sha", "uid"],
    "timestamp":  ["timestamp", "listened_at", "listened", "created_at", "play_time", "scrobble"],
    "track_id":   ["track_id", "trackid", "track-id", "recording_mbid", "recording_id", "spotify", "isrc", "song_id", "item_id"],
    "track_name": ["track_name", "track-name", "track_title", "title", "song_name", "recording_name"],
    "artist":     ["artist_name", "artist-name", "artist_id", "artist_mbid", "artistname"],
    "album":      ["album_name", "album_id", "release_name", "release_mbid"],
    "demographics": ["country", "age", "gender", "sex"],
    "mb_id":      ["mbid", "musicbrainz", "recording_mbid", "artist_mbid", "release_mbid"],
    "spotify_id": ["spotify", "spotifyid", "spotify_uri", "spotify_id"],
}
RESOLVABLE_TRACK = ["track_name", "mb_id", "spotify_id"]   # an internal-only track_id is NOT resolvable
REQUIRED_CORE = ["user_id", "timestamp"]
BLIND_A_START = datetime(2009, 1, 1, tzinfo=timezone.utc).timestamp()
BLIND_A_END = datetime(2018, 12, 31, 23, 59, 59, tzinfo=timezone.utc).timestamp()


def open_prefix(path: Path):
    """Text stream over a (optionally compressed) file; callers read only a row prefix."""
    raw = path.open("rb")
    if path.suffix == ".gz":
        return io.TextIOWrapper(gzip.GzipFile(fileobj=raw), errors="replace")
    if path.suffix == ".bz2":
        return io.TextIOWrapper(bz2.BZ2File(raw), errors="replace")
    if path.suffix in (".zst", ".zstd"):
        try:
            import zstandard
        except ImportError:
            sys.exit("this is a zstd file — `uv pip install zstandard` first (common for ListenBrainz/MLHD+ dumps)")
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), errors="replace")
    return io.TextIOWrapper(raw, errors="replace")


def parse_ts(v):
    if v is None:
        return None
    s = str(v).strip()
    if re.fullmatch(r"\d{9,10}", s):
        return float(s)
    if re.fullmatch(r"\d{12,13}", s):
        return float(s) / 1000.0
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19] if ("T" in s or " " in s) else s, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def flatten(d, prefix="", out=None):
    """Flatten one+ levels of nested dicts so ListenBrainz track_metadata/additional_info fields show up."""
    out = {} if out is None else out
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                flatten(v, key + ".", out)
            else:
                out[key] = v
    return out


def sniff_columns(stream, max_rows, no_header):
    """Detect columns + sample rows for csv/tsv/json/jsonl. Returns (cols, rows, kind)."""
    head = stream.readline()
    if head.lstrip()[:1] in ("{", "["):  # json / jsonl
        rows = []
        for ln in [head] + [stream.readline() for _ in range(max_rows)]:
            if not ln:
                break
            try:
                obj = json.loads(ln)
                if isinstance(obj, dict):
                    rows.append(flatten(obj))
            except json.JSONDecodeError:
                pass
        cols = sorted({k for r in rows for k in r}) if rows else []
        return cols, rows, "json"
    # delimited — use csv module for proper quoting
    delim = "\t" if head.count("\t") >= head.count(",") else ","
    first = next(csv.reader([head.rstrip("\n")], delimiter=delim), [])
    if no_header:
        cols = [f"col{i}" for i in range(len(first))]
        data_lines = [head] + [stream.readline() for _ in range(max_rows)]
    else:
        cols = [c.strip().strip('"') for c in first]
        data_lines = [stream.readline() for _ in range(max_rows)]
    rows = []
    for ln in data_lines:
        if not ln:
            break
        vals = next(csv.reader([ln.rstrip("\n")], delimiter=delim), [])
        rows.append(dict(zip(cols, vals)))
    return cols, rows, ("delim/no-header" if no_header else "delim")


def match_fields(cols):
    low = [c.lower() for c in cols]
    found = {}
    for fam, subs in FIELDS.items():
        found[fam] = next((cols[i] for i, c in enumerate(low) if any(s in c for s in subs)), None)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--no-header", action="store_true",
                    help="file has no header row (e.g. MLHD per-user logs); columns named col0..colN")
    a = ap.parse_args()
    path = Path(a.path)
    if not path.exists():
        sys.exit(f"no such file: {path}")
    print(f"== auditing {path}  ({path.stat().st_size/1e6:.1f} MB on disk) ==")
    cols, rows, kind = sniff_columns(open_prefix(path), a.rows, a.no_header)
    print(f"format: {kind}  columns ({len(cols)}): {cols[:25]}")
    found = match_fields(cols)
    print("\nrequired-field coverage:")
    for fam, hit in found.items():
        print(f"  [{'OK ' if hit else '-- '}] {fam:12s} -> {hit}")
    if not found["user_id"]:
        print("  NOTE: user_id absent from columns — if this is a per-user file, the user is the FILENAME "
              "(re-run with --no-header and treat user from the path).")

    ts_col = found["timestamp"]
    tmin = tmax = None
    if ts_col:
        for r in rows:
            t = parse_ts(r.get(ts_col))
            if t:
                tmin = t if tmin is None else min(tmin, t)
                tmax = t if tmax is None else max(tmax, t)
    overlaps = None
    if tmin:
        fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
        overlaps = tmax >= BLIND_A_START and tmin <= BLIND_A_END
        print(f"\nsampled timestamp range: {fmt(tmin)} .. {fmt(tmax)}  (PREFIX sample only — not full file)")
        print(f"Blind-A era overlap (2009-2018) in sample: {'YES' if overlaps else 'no'}")

    has_core = all(found[f] for f in REQUIRED_CORE)
    resolvable = any(found[f] for f in RESOLVABLE_TRACK)
    internal_only = (found["track_id"] and not resolvable)
    if not has_core:
        verdict = "REJECT (missing user_id and/or timestamp — no session structure)"
    elif internal_only:
        verdict = "PARTIAL (session structure but track identity is an INTERNAL id — needs the dataset's catalog to resolve titles/Spotify/MBID)"
    elif not resolvable:
        verdict = "REJECT (no resolvable track identity: title/MBID/Spotify)"
    elif overlaps is False:
        verdict = "PARTIAL (session-level + resolvable but sampled dates miss Blind-A 2009-2018 — check full min/max)"
    else:
        verdict = "USEFUL (session-level, resolvable track identity)"
    print(f"\nVERDICT: {verdict}")
    print("  caveats: timestamp range is from the PREFIX only; fuzzy column-name matching may mis-tag — "
          "confirm full date range and that track ids resolve to titles before relying on this.")


if __name__ == "__main__":
    main()
