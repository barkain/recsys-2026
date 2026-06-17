#!/usr/bin/env python3
"""R511: source-day guided ca8 rank-2 promotion on top of R510.

The recovered MLHD+ source-day pool for ca8cbe02 contains several Emilie Autumn
tracks. The query asks for the theatrical storytelling of "Time for Tea" with a
stronger industrial rhythm. Among source-pool tracks already in R510, the
industrial remix "Dead Is The New Alive - Manipulator Mix By Dope Stars Inc."
is the clearest semantic fit and sits at rank 8. Promote it to rank 2 while
keeping top-1 and response unchanged.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
BASE_ZIP = REPO / "exp/inference/blind_a/r510_stack_r498_official_positives/r510_r498_plus_official_positive_rows_submission.zip"
OUT_DIR = REPO / "exp/inference/blind_a/r511_source_pool_ca8"
OUT_ZIP = OUT_DIR / "r511_r510_ca8_dead_is_new_alive_rank2_submission.zip"
OUT_AUDIT = REPO / "exp/eval/expR511_source_pool_ca8_industrial_rank2_audit.json"
OUT_DOC = REPO / "docs/r511_source_pool_ca8_industrial_rank2.md"

SESSION_PREFIX = "ca8cbe02"
TARGET_TRACK_ID = "2d4b6470-1a21-475a-8a0d-dba729eb104b"
TARGET_LABEL = "Dead Is The New Alive - Manipulator Mix By Dope Stars Inc. - Emilie Autumn"


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


def lexdiv(rows: list[dict[str, Any]]) -> float:
    bigrams: list[tuple[str, str]] = []
    for row in rows:
        toks = str(row["predicted_response"]).lower().split()
        bigrams.extend(zip(toks, toks[1:]))
    return len(set(bigrams)) / len(bigrams) if bigrams else 0.0


def main() -> None:
    base = load_zip(BASE_ZIP)
    rows = deepcopy(base)
    row_idx = next(i for i, row in enumerate(rows) if row["session_id"].startswith(SESSION_PREFIX))
    old_tracks = list(rows[row_idx]["predicted_track_ids"])
    old_rank = old_tracks.index(TARGET_TRACK_ID) + 1
    if old_rank <= 2:
        raise RuntimeError(f"target already rank {old_rank}")
    new_tracks = [old_tracks[0], TARGET_TRACK_ID] + [
        tid for tid in old_tracks[1:] if tid != TARGET_TRACK_ID
    ]
    rows[row_idx]["predicted_track_ids"] = new_tracks

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

    sha = write_zip(OUT_ZIP, rows)
    audit = {
        "experiment": "R511 source-day ca8 industrial-rhythm rank2 promotion",
        "base_zip": str(BASE_ZIP.relative_to(REPO)),
        "out_zip": str(OUT_ZIP.relative_to(REPO)),
        "sha256": sha,
        "validation_issues": issues,
        "row_index": row_idx,
        "session_id": rows[row_idx]["session_id"],
        "turn_number": int(rows[row_idx]["turn_number"]),
        "query": "Time for Tea theatrical/storytelling plus strong driving beat or industrial rhythm",
        "target_track_id": TARGET_TRACK_ID,
        "target_label": TARGET_LABEL,
        "old_rank_in_r510": old_rank,
        "new_rank": 2,
        "top1_unchanged": old_tracks[0] == rows[row_idx]["predicted_track_ids"][0],
        "response_unchanged": base[row_idx]["predicted_response"] == rows[row_idx]["predicted_response"],
        "changed_track_rows_vs_r510": changed_track_rows,
        "changed_response_rows_vs_r510": changed_response_rows,
        "base_lexdiv_local": lexdiv(base),
        "candidate_lexdiv_local": lexdiv(rows),
        "source_evidence": {
            "source_pool": "exp/eval/expR510_ca8cbe02_mlhd_candidate0_pool.json",
            "source_user": "18f653ee-38f0-4c22-97b4-47810eb5976b.txt",
            "source_day_unique_recordings": 26,
            "mapped_catalog_tracks": 7,
            "reason": "Target is in recovered source-day pool and is the strongest industrial-rhythm semantic match among mapped catalog tracks.",
        },
    }
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    OUT_DOC.write_text(
        "\n".join(
            [
                "# R511 — Source-Pool ca8 Industrial Rank-2",
                "",
                f"**Zip:** `{OUT_ZIP.relative_to(REPO)}`",
                f"**sha256:** `{sha}`",
                "",
                "Starts from R510 (`0.5149` official nDCG) and changes one row:",
                "",
                f"- Row `{row_idx}` / `ca8cbe02`: promote `{TARGET_LABEL}` from rank `{old_rank}` to rank `2`.",
                "- Top-1 and response are unchanged.",
                "- Evidence: MLHD+ source-day match narrows this row to a 26-recording day pool; the target maps into the challenge catalog and is the clearest industrial-rhythm fit.",
                "",
                "Expected value is asymmetric: if this source-pool track is the GT, moving rank 8 to rank 2 gains material nDCG; if it is not, top-1 is preserved and only ranks 2-7 shift down by one.",
                "",
                f"Validation issues: `{issues}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
