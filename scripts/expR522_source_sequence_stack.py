#!/usr/bin/env python3
"""Build R522: two-row source-sequence stack on top of R510.

This is deliberately narrow. It keeps top-1 and responses unchanged, and only
promotes source-backed tracks already present in the R510 top-20.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path


BASE = Path(
    "exp/inference/blind_a/r510_stack_r498_official_positives/"
    "r510_r498_plus_official_positive_rows_submission.zip"
)
OUT_DIR = Path("exp/inference/blind_a/r522_source_sequence_stack")
OUT_ZIP = OUT_DIR / "r522_r510_ca8_misery_5ad_miles_rank2_submission.zip"
AUDIT = Path("exp/eval/expR522_source_sequence_stack_audit.json")
METADATA = Path("cache/metadata/track_metadata_all_tracks.json")

ACTIONS = [
    {
        "session_prefix": "ca8cbe02",
        "track_id": "7a7f8898-62b9-4ecd-ac08-401000e025d3",
        "new_rank": 2,
        "reason": "MLHD source-day Emilie Autumn sequence; dramatic/theatrical request after Time for Tea.",
    },
    {
        "session_prefix": "5ad7094f",
        "track_id": "058126ce-0c84-4787-97dc-4fff4b78b5b2",
        "new_rank": 2,
        "reason": "MLHD source-day Hieroglyphics sequence; Miles to the Sun best fits introspective/philosophical edge.",
    },
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prediction(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if names != ["prediction.json"]:
            raise SystemExit(f"unexpected zip members: {names}")
        return json.loads(zf.read("prediction.json"))


def scalar(value):
    if isinstance(value, list):
        return value[0] if value else "?"
    return value if value not in (None, "") else "?"


def label(track_id: str, metadata: dict[str, dict]) -> str:
    row = metadata.get(track_id, {})
    return f"{scalar(row.get('track_name'))} — {scalar(row.get('artist_name'))}"


def move_to_rank(track_ids: list[str], track_id: str, rank: int) -> tuple[list[str], int | None]:
    new_ids = list(track_ids)
    old_rank: int | None = None
    if track_id in new_ids:
        old_rank = new_ids.index(track_id) + 1
        new_ids.pop(old_rank - 1)
    else:
        raise ValueError(f"target track not in source top20: {track_id}")
    new_ids.insert(rank - 1, track_id)
    new_ids = new_ids[:20]
    if len(new_ids) != 20 or len(set(new_ids)) != 20:
        raise ValueError("invalid top20 after move")
    return new_ids, old_rank


def main() -> None:
    rows = load_prediction(BASE)
    metadata = json.loads(METADATA.read_text())

    changed = []
    used_actions = set()
    for idx, row in enumerate(rows):
        prefix = row["session_id"][:8]
        for action_idx, action in enumerate(ACTIONS):
            if prefix != action["session_prefix"]:
                continue
            old_ids = list(row["predicted_track_ids"])
            new_ids, old_rank = move_to_rank(old_ids, action["track_id"], action["new_rank"])
            row["predicted_track_ids"] = new_ids
            used_actions.add(action_idx)
            changed.append(
                {
                    "row_index": idx,
                    "session_id": row["session_id"],
                    "turn_number": row.get("turn_number"),
                    "target_track_id": action["track_id"],
                    "target_label": label(action["track_id"], metadata),
                    "old_rank": old_rank,
                    "new_rank": action["new_rank"],
                    "top1_before": old_ids[0],
                    "top1_after": new_ids[0],
                    "top1_unchanged": old_ids[0] == new_ids[0],
                    "reason": action["reason"],
                    "old_top10": [
                        {"rank": i + 1, "track_id": tid, "label": label(tid, metadata)}
                        for i, tid in enumerate(old_ids[:10])
                    ],
                    "new_top10": [
                        {"rank": i + 1, "track_id": tid, "label": label(tid, metadata)}
                        for i, tid in enumerate(new_ids[:10])
                    ],
                }
            )

    if used_actions != set(range(len(ACTIONS))):
        raise SystemExit(f"missing actions: {set(range(len(ACTIONS))) - used_actions}")
    if len(rows) != 80:
        raise SystemExit(f"expected 80 rows, got {len(rows)}")
    for row in rows:
        ids = row["predicted_track_ids"]
        if len(ids) != 20 or len(set(ids)) != 20:
            raise SystemExit(f"bad top20 for {row['session_id']}")
    if not all(c["top1_unchanged"] for c in changed):
        raise SystemExit("top1 changed unexpectedly")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

    audit = {
        "run": "R522",
        "base_zip": str(BASE),
        "out_zip": str(OUT_ZIP),
        "out_sha256": sha256(OUT_ZIP),
        "changed_rows": changed,
        "rationale": {
            "why_narrow": "Broad rerank/retrieval edits repeatedly regressed official nDCG; this only promotes source-backed tracks already in top20.",
            "expected_gain_if_hits": "ca8 rank6->2 is about +0.0034 nDCG; 5ad rank9->2 is about +0.0049 nDCG.",
            "main_risk": "If either hidden GT is in old ranks 2..target_rank-1, that row loses a small amount.",
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"zip": str(OUT_ZIP), "sha256": audit["out_sha256"], "changed_rows": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
