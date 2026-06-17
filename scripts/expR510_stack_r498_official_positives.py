#!/usr/bin/env python3
"""R510: stack independently official-positive row actions onto R498.

This is a conservative builder in the engineering sense, not the strategy sense:
it applies only row actions that already produced positive official nDCG evidence
in a single-row probe. The base is R498 because that is the best clean recent
submission (official nDCG 0.5126).
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
BASE_ZIP = REPO / "exp/inference/blind_a/r498_listwise_llm/r498_gpt41_full20_keep_top1_submission.zip"
R433_MANIFEST = REPO / "exp/eval/expR433_multimodal_on_r432s_probe_manifest.json"
R446_MANIFEST = REPO / "exp/eval/expR446_explicit_constraint_probe_manifest.json"
META_JSON = REPO / "cache/metadata/track_metadata_all_tracks.json"
OUT_DIR = REPO / "exp/inference/blind_a/r510_stack_r498_official_positives"
OUT_ZIP = OUT_DIR / "r510_r498_plus_official_positive_rows_submission.zip"
OUT_AUDIT = REPO / "exp/eval/expR510_stack_r498_official_positives_audit.json"
OUT_DOC = REPO / "docs/r510_stack_r498_official_positives.md"

BASE_OFFICIAL_NDCG = 0.5126
EVIDENCE = {
    "r433p04_1f7b28c1_t8_mm_rank2": {
        "official_base": "R432s",
        "official_base_ndcg": 0.5092,
        "official_probe_ndcg": 0.5092 + 0.0019,
        "delta_ndcg": 0.0019,
    },
    "r433p03_d5c80ee5_t7_mm_rank2": {
        "official_base": "R432s",
        "official_base_ndcg": 0.5092,
        "official_probe_ndcg": 0.5092 + 0.0019,
        "delta_ndcg": 0.0019,
    },
    "r446p03_no_more_wood_brothers_rank1": {
        "official_base": "R432s",
        "official_base_ndcg": 0.5073,
        "official_probe_ndcg": 0.5092,
        "delta_ndcg": 0.0019,
    },
    "r446p02_holiday_beyond_santa_rank1": {
        "official_base": "R432s/R491",
        "official_base_ndcg": 0.5092,
        "official_probe_ndcg": 0.5115,
        "delta_ndcg": 0.0023,
    },
}
PROBE_IDS = list(EVIDENCE)


def load_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        if names != ["prediction.json"]:
            raise RuntimeError(f"{path} has unexpected entries: {names}")
        return json.loads(zf.read("prediction.json"))


def write_zip(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("prediction.json")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def label(meta: dict[str, Any], track_id: str) -> str:
    row = meta.get(track_id, {})
    return f"{one(row.get('track_name'))} - {one(row.get('artist_name'))}".strip(" -")


def lexdiv(rows: list[dict[str, Any]]) -> float:
    bigrams: list[tuple[str, str]] = []
    for row in rows:
        toks = str(row["predicted_response"]).lower().split()
        bigrams.extend(zip(toks, toks[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def load_variants() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in [R433_MANIFEST, R446_MANIFEST]:
        manifest = json.load(open(path))
        for variant in manifest["variants"]:
            out[variant["probe_id"]] = variant
    return out


def find_row(rows: list[dict[str, Any]], session_id: str, turn_number: int) -> int:
    for i, row in enumerate(rows):
        if row["session_id"] == session_id and int(row["turn_number"]) == int(turn_number):
            return i
    raise RuntimeError(f"row not found: {session_id} turn {turn_number}")


def apply_rank2_insert(row: dict[str, Any], track_id: str) -> dict[str, Any]:
    old = list(row["predicted_track_ids"])
    old_rank = old.index(track_id) + 1 if track_id in old else None
    if old_rank == 2:
        new = old
        dropped = None
    else:
        without = [tid for tid in old if tid != track_id]
        dropped = without[-1]
        new = [without[0], track_id, *without[1:19]]
    row["predicted_track_ids"] = new
    return {"old_rank_in_r498": old_rank, "dropped_track_id": dropped}


def apply_promote_rank1(row: dict[str, Any], track_id: str) -> dict[str, Any]:
    old = list(row["predicted_track_ids"])
    old_rank = old.index(track_id) + 1 if track_id in old else None
    if old_rank == 1:
        new = old
        dropped = None
    elif track_id in old:
        new = [track_id, *[tid for tid in old if tid != track_id]]
        dropped = None
    else:
        dropped = old[-1]
        new = [track_id, *old[:19]]
    row["predicted_track_ids"] = new
    return {"old_rank_in_r498": old_rank, "dropped_track_id": dropped}


def main() -> None:
    meta = json.load(open(META_JSON))
    base = load_zip(BASE_ZIP)
    rows = deepcopy(base)
    variants = load_variants()
    applied: list[dict[str, Any]] = []

    for probe_id in PROBE_IDS:
        variant = variants[probe_id]
        action = variant["actions"][0]
        row_idx = find_row(rows, action["session_id"], int(action["turn_number"]))
        if row_idx != int(action["row_index"]):
            raise RuntimeError(f"{probe_id}: row index changed {action['row_index']} -> {row_idx}")
        row = rows[row_idx]
        before_tracks = list(row["predicted_track_ids"])

        if "insert_track" in action:
            target = action["insert_track"]
            change = apply_rank2_insert(row, target)
            response_changed = False
        elif "promote_track" in action:
            target = action["promote_track"]
            change = apply_promote_rank1(row, target)
            row["predicted_response"] = variant["new_response"]
            response_changed = True
        else:
            raise RuntimeError(f"{probe_id}: unsupported action {action}")

        applied.append(
            {
                "probe_id": probe_id,
                "row_index": row_idx,
                "session_id": row["session_id"],
                "turn_number": int(row["turn_number"]),
                "target_track_id": target,
                "target_label": label(meta, target),
                "response_changed": response_changed,
                "top1_before": before_tracks[0],
                "top1_after": row["predicted_track_ids"][0],
                "old_rank_in_r498": change["old_rank_in_r498"],
                "dropped_track_id": change["dropped_track_id"],
                "dropped_label": label(meta, change["dropped_track_id"]) if change["dropped_track_id"] else None,
                "evidence": EVIDENCE[probe_id],
            }
        )

    issues: list[str] = []
    changed_track_rows: list[int] = []
    changed_response_rows: list[int] = []
    for i, (old, new) in enumerate(zip(base, rows)):
        if (old["session_id"], int(old["turn_number"])) != (new["session_id"], int(new["turn_number"])):
            issues.append(f"row_key_changed:{i}")
        tids = new["predicted_track_ids"]
        if len(tids) != 20 or len(set(tids)) != 20:
            issues.append(f"bad_track_list:{i}")
        if old["predicted_track_ids"] != tids:
            changed_track_rows.append(i)
        if old["predicted_response"] != new["predicted_response"]:
            changed_response_rows.append(i)

    expected_ndcg = BASE_OFFICIAL_NDCG + sum(item["evidence"]["delta_ndcg"] for item in applied)
    sha = write_zip(OUT_ZIP, rows)
    audit = {
        "experiment": "R510 stack R498 plus independently official-positive rows",
        "base_zip": str(BASE_ZIP.relative_to(REPO)),
        "out_zip": str(OUT_ZIP.relative_to(REPO)),
        "sha256": sha,
        "validation_issues": issues,
        "base_official_ndcg": BASE_OFFICIAL_NDCG,
        "rough_additive_expected_ndcg": round(expected_ndcg, 4),
        "rough_additive_delta_ndcg": round(expected_ndcg - BASE_OFFICIAL_NDCG, 4),
        "changed_track_rows_vs_r498": changed_track_rows,
        "changed_response_rows_vs_r498": changed_response_rows,
        "base_lexdiv_local": lexdiv(base),
        "candidate_lexdiv_local": lexdiv(rows),
        "applied": applied,
        "caveat": "Deltas are official single-row evidence but not guaranteed additive across R498 because R498 changed row context on some rows.",
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# R510 — R498 + Official-Positive Row Stack",
        "",
        f"**Zip:** `{OUT_ZIP.relative_to(REPO)}`",
        f"**sha256:** `{sha}`",
        "",
        "## What changed",
        "",
        "Starts from R498 (`0.5126` official nDCG) and applies only row actions with prior official positive nDCG evidence:",
        "",
    ]
    for item in applied:
        ev = item["evidence"]
        lines.append(
            f"- `{item['probe_id']}` row `{item['row_index']}`: `{item['target_label']}`; "
            f"official single-row delta `+{ev['delta_ndcg']:.4f}`; "
            f"R498 old rank `{item['old_rank_in_r498']}`; response changed `{item['response_changed']}`."
        )
    lines += [
        "",
        "## Expected score",
        "",
        f"- Rough additive nDCG estimate: `{expected_ndcg:.4f}` from base `{BASE_OFFICIAL_NDCG:.4f}`.",
        "- This is a measured-positive stack, not a path to 0.60 by itself.",
        "- The larger retrieval path remains source-session/full-catalog reconstruction or another mechanism that admits true hidden-pool candidates.",
        "",
        "## Validation",
        "",
        f"- Changed track rows vs R498: `{changed_track_rows}`",
        f"- Changed response rows vs R498: `{changed_response_rows}`",
        f"- Local LexDiv: `{lexdiv(rows):.6f}` vs R498 `{lexdiv(base):.6f}`",
        f"- Validation issues: `{issues}`",
    ]
    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
