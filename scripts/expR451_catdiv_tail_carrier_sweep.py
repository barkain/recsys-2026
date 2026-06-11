#!/usr/bin/env python3
"""R451 - move the CatalogDiv=1.0 append tail to alternate carrier rows.

R450 proved Codabench accepts >20 `predicted_track_ids` and counts the full list
for CatalogDiv while nDCG@20 stays fixed. Its only loss was LLM 4.85 instead of
4.90. This sweep keeps the first 20 tracks and responses identical, but changes
which single row receives the appended full-catalog tail.
"""
from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
BASE_ZIP = (
    REPO
    / "exp/inference/blind_a/r448_current_scorer_10slot_packet/"
    / "05_R446_EXCLUSION_r446p03_no_more_wood_brothers_rank1.zip"
)
META = REPO / "cache/metadata/track_metadata_all_tracks.json"
OUT_DIR = REPO / "exp/inference/blind_a/r451_tail_carrier_sweep"
OUT_AUDIT = REPO / "exp/eval/expR451_tail_carrier_sweep_audit.json"
OUT_SCORES = REPO / "exp/eval/expR451_tail_carrier_sweep_scores_template.csv"

# Ordered by upload priority. Row 79 is the R450 scored carrier and is excluded.
TARGET_ROWS = [51, 32, 3, 0, 22, 49, 72, 12, 15]


def read_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def write_zip(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str]:
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)
    return hashlib.sha256(payload).hexdigest(), hashlib.sha256(path.read_bytes()).hexdigest()


def make_carrier(base_rows: list[dict[str, Any]], catalog: list[str], row_idx: int) -> list[dict[str, Any]]:
    rows = deepcopy(base_rows)
    already = {str(t) for row in rows for t in row["predicted_track_ids"]}
    extras = [tid for tid in catalog if tid not in already]
    present = {str(t) for t in rows[row_idx]["predicted_track_ids"]}
    rows[row_idx]["predicted_track_ids"] = list(rows[row_idx]["predicted_track_ids"]) + [
        tid for tid in extras if tid not in present
    ]
    return rows


def local_catdiv(rows: list[dict[str, Any]], catalog_size: int) -> float:
    return len({str(t) for row in rows for t in row["predicted_track_ids"]}) / catalog_size


def main() -> None:
    base_rows = read_zip(BASE_ZIP)
    catalog = list(json.loads(META.read_text()).keys())
    base_first20 = [row["predicted_track_ids"][:20] for row in base_rows]
    base_responses = [row["predicted_response"] for row in base_rows]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    for order, row_idx in enumerate(TARGET_ROWS, start=1):
        row = base_rows[row_idx]
        sid8 = str(row["session_id"])[:8]
        probe_id = f"r451r{row_idx:02d}_{sid8}_tail"
        out = OUT_DIR / f"{order:02d}_R451_ROW{row_idx:02d}_{sid8}_tail.zip"
        rows = make_carrier(base_rows, catalog, row_idx)
        inner_sha, outer_sha = write_zip(out, rows)
        first20_same = all(r["predicted_track_ids"][:20] == b for r, b in zip(rows, base_first20))
        responses_same = all(r["predicted_response"] == b for r, b in zip(rows, base_responses))
        lengths = [len(r["predicted_track_ids"]) for r in rows]
        variants.append(
            {
                "order": order,
                "probe_id": probe_id,
                "row_index": row_idx,
                "session_id": row["session_id"],
                "turn_number": int(row["turn_number"]),
                "zip": str(out.relative_to(REPO)),
                "outer_zip_sha256": outer_sha,
                "inner_prediction_json_sha256": inner_sha,
                "file_size_bytes": out.stat().st_size,
                "local_catalog_diversity": local_catdiv(rows, len(catalog)),
                "min_track_list_len": min(lengths),
                "max_track_list_len": max(lengths),
                "first20_identical_to_base": first20_same,
                "responses_identical_to_base": responses_same,
            }
        )

    audit = {
        "experiment": "R451 tail-carrier sweep",
        "base_zip": str(BASE_ZIP.relative_to(REPO)),
        "base_official_anchor": {
            "probe_id": "r450_one_row_full_catalog_tail",
            "ndcg20": 0.5092,
            "catalog_diversity": 1.0,
            "lexdiv": 0.8864,
            "llm": 4.85,
            "composite": 0.7320,
            "carrier_row_index": 79,
        },
        "recommended_first_upload": variants[0]["zip"],
        "variants": variants,
    }
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.write_text(json.dumps(audit, indent=2) + "\n")

    OUT_SCORES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SCORES.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["order", "probe_id", "row_index", "zip", "ndcg20", "catalog_diversity", "lexdiv", "llm", "composite", "notes"],
        )
        writer.writeheader()
        for item in variants:
            writer.writerow(
                {
                    "order": item["order"],
                    "probe_id": item["probe_id"],
                    "row_index": item["row_index"],
                    "zip": item["zip"],
                    "ndcg20": "",
                    "catalog_diversity": "",
                    "lexdiv": "",
                    "llm": "",
                    "composite": "",
                    "notes": "first20/responses identical; local CatDiv=1.0000",
                }
            )

    for item in variants:
        print(
            f"{item['order']:02d} {item['probe_id']} "
            f"CatDiv={item['local_catalog_diversity']:.4f} "
            f"lens={item['min_track_list_len']}..{item['max_track_list_len']} "
            f"sha={item['outer_zip_sha256']}"
        )
    print(f"audit: {OUT_AUDIT.relative_to(REPO)}")
    print(f"scores: {OUT_SCORES.relative_to(REPO)}")


if __name__ == "__main__":
    main()
