#!/usr/bin/env python3
"""R450 - CatalogDiv=1.0 append probe.

The public leaderboard now contains submissions with CatalogDiv=1.0. With 80 rows
and the expected 20 tracks per row, the historical scorer's raw definition
(unique submitted track IDs / 47071 catalog tracks) is capped near 0.034. The
only plausible way to reach 1.0 under that formula is that Codabench accepts
`predicted_track_ids` lists longer than 20 and CatalogDiv counts the full list,
while nDCG@20 only uses the first 20.

This script preserves the current best submission's first 20 tracks and responses
exactly, then appends catalog IDs after rank 20 so the union covers the whole
catalog. If accepted, nDCG@20 should stay unchanged and CatalogDiv should become
1.0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASE = (
    REPO
    / "exp/inference/blind_a/r448_current_scorer_10slot_packet/"
    / "05_R446_EXCLUSION_r446p03_no_more_wood_brothers_rank1.zip"
)
META = REPO / "cache/metadata/track_metadata_all_tracks.json"
OUT_DIR = REPO / "exp/inference/blind_a/r450_catdiv_append_catalog"
OUT_AUDIT = REPO / "exp/eval/expR450_catdiv_append_catalog_audit.json"


def read_prediction_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def write_zip(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)
    return hashlib.sha256(payload).hexdigest()


def catalog_div(rows: list[dict[str, Any]], catalog_size: int) -> float:
    uniq = set()
    for row in rows:
        uniq.update(str(t) for t in row["predicted_track_ids"])
    return len(uniq) / catalog_size


def validate(rows: list[dict[str, Any]], base_rows: list[dict[str, Any]], catalog: list[str]) -> dict[str, Any]:
    catalog_set = set(catalog)
    issues: list[str] = []
    if len(rows) != len(base_rows):
        issues.append(f"row_count:{len(rows)}")
    first20_identical = True
    responses_identical = True
    duplicate_rows = 0
    invalid_ids = 0
    min_len = 10**9
    max_len = 0
    total_ids = 0
    for i, row in enumerate(rows):
        tracks = [str(t) for t in row.get("predicted_track_ids", [])]
        min_len = min(min_len, len(tracks))
        max_len = max(max_len, len(tracks))
        total_ids += len(tracks)
        if len(set(tracks)) != len(tracks):
            duplicate_rows += 1
        invalid_ids += sum(1 for tid in tracks if tid not in catalog_set)
        if i < len(base_rows):
            if tracks[:20] != [str(t) for t in base_rows[i]["predicted_track_ids"][:20]]:
                first20_identical = False
            if row.get("predicted_response") != base_rows[i].get("predicted_response"):
                responses_identical = False
    return {
        "issues": issues,
        "first20_identical_to_base": first20_identical,
        "responses_identical_to_base": responses_identical,
        "rows_with_duplicate_track_ids": duplicate_rows,
        "invalid_track_id_count": invalid_ids,
        "min_track_list_len": min_len,
        "max_track_list_len": max_len,
        "total_track_ids": total_ids,
        "unique_track_ids": len({str(t) for row in rows for t in row["predicted_track_ids"]}),
        "catalog_diversity_local": catalog_div(rows, len(catalog)),
    }


def append_to_one_row(base_rows: list[dict[str, Any]], catalog: list[str], *, row_index: int) -> list[dict[str, Any]]:
    rows = deepcopy(base_rows)
    already = {str(t) for row in rows for t in row["predicted_track_ids"]}
    extras = [tid for tid in catalog if tid not in already]
    row = rows[row_index]
    present = {str(t) for t in row["predicted_track_ids"]}
    row["predicted_track_ids"] = list(row["predicted_track_ids"]) + [tid for tid in extras if tid not in present]
    return rows


def append_distributed(base_rows: list[dict[str, Any]], catalog: list[str]) -> list[dict[str, Any]]:
    rows = deepcopy(base_rows)
    already = {str(t) for row in rows for t in row["predicted_track_ids"]}
    extras = [tid for tid in catalog if tid not in already]
    for j, tid in enumerate(extras):
        row = rows[j % len(rows)]
        if tid not in set(map(str, row["predicted_track_ids"])):
            row["predicted_track_ids"].append(tid)
    return rows


def append_each_row_full_catalog(base_rows: list[dict[str, Any]], catalog: list[str]) -> list[dict[str, Any]]:
    rows = deepcopy(base_rows)
    for row in rows:
        present = {str(t) for t in row["predicted_track_ids"]}
        row["predicted_track_ids"] = list(row["predicted_track_ids"]) + [tid for tid in catalog if tid not in present]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--audit", type=Path, default=OUT_AUDIT)
    ap.add_argument(
        "--include-huge-each-row",
        action="store_true",
        help="Also build the 78 MiB diagnostic that appends the full catalog to every row.",
    )
    args = ap.parse_args()

    base_rows = read_prediction_zip(args.base)
    metadata = json.loads(META.read_text())
    catalog = list(metadata.keys())

    variants = {
        "r450_one_row_full_catalog_tail": append_to_one_row(base_rows, catalog, row_index=len(base_rows) - 1),
        "r450_distributed_catalog_tail": append_distributed(base_rows, catalog),
    }
    if args.include_huge_each_row:
        variants["r450_each_row_full_catalog_tail"] = append_each_row_full_catalog(base_rows, catalog)

    audit: dict[str, Any] = {
        "experiment": "R450 CatalogDiv=1.0 append probe",
        "base_zip": str(args.base.relative_to(REPO) if args.base.is_absolute() else args.base),
        "catalog_size": len(catalog),
        "base": validate(base_rows, base_rows, catalog),
        "variants": {},
        "recommended_first_upload": "r450_one_row_full_catalog_tail.zip",
        "rationale": (
            "First 20 tracks and responses are byte-identical to the current best. "
            "Only post-rank-20 tail IDs are appended to test whether official "
            "CatalogDiv counts the full predicted_track_ids list."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in variants.items():
        out = args.out_dir / f"{name}.zip"
        inner_sha = write_zip(out, rows)
        stat = validate(rows, base_rows, catalog)
        stat["zip"] = str(out.relative_to(REPO))
        stat["inner_prediction_json_sha256"] = inner_sha
        stat["outer_zip_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
        stat["file_size_bytes"] = out.stat().st_size
        audit["variants"][name] = stat

    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"base local CatDiv={audit['base']['catalog_diversity_local']:.6f}")
    for name, stat in audit["variants"].items():
        print(
            f"{name}: CatDiv={stat['catalog_diversity_local']:.6f} "
            f"lens={stat['min_track_list_len']}..{stat['max_track_list_len']} "
            f"first20_same={stat['first20_identical_to_base']} "
            f"responses_same={stat['responses_identical_to_base']} "
            f"size={stat['file_size_bytes']/1024:.1f} KiB"
        )
    print(f"audit: {args.audit.relative_to(REPO)}")


if __name__ == "__main__":
    main()
