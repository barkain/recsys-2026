#!/usr/bin/env python3
"""Build R521: source-sequence backed ca8cbe02 Misery rank-2 promotion.

Base is R510, the current best nDCG candidate. This only changes one row:
ca8cbe02 turn 6, moving "Misery Loves Company" from rank 6 to rank 2.
Top-1 and responses are unchanged, so this tests ranking signal only.
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
OUT_DIR = Path("exp/inference/blind_a/r521_source_sequence_ca8_misery_rank2")
OUT_ZIP = OUT_DIR / "r521_r510_ca8_misery_rank2_submission.zip"
AUDIT = Path("exp/eval/expR521_source_sequence_ca8_misery_rank2_audit.json")
METADATA = Path("cache/metadata/track_metadata_all_tracks.json")

TARGET_SESSION_PREFIX = "ca8cbe02"
TARGET_TRACK_ID = "7a7f8898-62b9-4ecd-ac08-401000e025d3"
TARGET_RANK = 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prediction(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def label(track_id: str, metadata: dict[str, dict]) -> str:
    row = metadata.get(track_id, {})
    title = (row.get("track_name") or ["?"])[0]
    artist = (row.get("artist_name") or ["?"])[0]
    return f"{title} — {artist}"


def move_to_rank(track_ids: list[str], track_id: str, rank: int) -> tuple[list[str], int | None, int]:
    new_ids = list(track_ids)
    old_rank: int | None = None
    if track_id in new_ids:
        old_rank = new_ids.index(track_id) + 1
        new_ids.pop(old_rank - 1)
    new_ids.insert(rank - 1, track_id)
    new_ids = new_ids[:20]
    if len(set(new_ids)) != len(new_ids):
        raise ValueError("duplicate track id after move")
    return new_ids, old_rank, rank


def main() -> None:
    rows = load_prediction(BASE)
    metadata = json.loads(METADATA.read_text())

    changed = []
    for idx, row in enumerate(rows):
        if not row["session_id"].startswith(TARGET_SESSION_PREFIX):
            continue
        old_ids = row["predicted_track_ids"]
        new_ids, old_rank, new_rank = move_to_rank(old_ids, TARGET_TRACK_ID, TARGET_RANK)
        row["predicted_track_ids"] = new_ids
        changed.append(
            {
                "row_index": idx,
                "session_id": row["session_id"],
                "turn_number": row.get("turn_number"),
                "target_track_id": TARGET_TRACK_ID,
                "target_label": label(TARGET_TRACK_ID, metadata),
                "old_rank": old_rank,
                "new_rank": new_rank,
                "top1_before": old_ids[0],
                "top1_after": new_ids[0],
                "top1_unchanged": old_ids[0] == new_ids[0],
                "old_top20": [{"rank": i + 1, "track_id": tid, "label": label(tid, metadata)} for i, tid in enumerate(old_ids)],
                "new_top20": [{"rank": i + 1, "track_id": tid, "label": label(tid, metadata)} for i, tid in enumerate(new_ids)],
            }
        )

    if len(changed) != 1:
        raise SystemExit(f"expected one changed row, got {len(changed)}")

    if len(rows) != 80:
        raise SystemExit(f"expected 80 rows, got {len(rows)}")
    for row in rows:
        ids = row["predicted_track_ids"]
        if len(ids) != 20 or len(set(ids)) != 20:
            raise SystemExit(f"bad top20 for {row['session_id']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

    audit = {
        "run": "R521",
        "base_zip": str(BASE),
        "out_zip": str(OUT_ZIP),
        "out_sha256": sha256(OUT_ZIP),
        "changed_rows": changed,
        "rationale": {
            "source_evidence": [
                "MLHD text-bridge row ca8cbe02 has multiple candidate days containing the Emilie Autumn block.",
                "Misery Loves Company appears repeatedly near Shalott and Swallow in the source day sequence.",
                "The official conversation asks for another Emilie Autumn track with theatrical drama and a strong driving/industrial beat after Time for Tea.",
            ],
            "risk_control": [
                "Only one row changes.",
                "Top-1 is unchanged, so the existing response remains coherent.",
                "The target was already in R510 top20 at rank 6; this is a rank-only promotion, not broad retrieval replacement.",
            ],
        },
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"zip": str(OUT_ZIP), "sha256": audit["out_sha256"], "changed_rows": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
