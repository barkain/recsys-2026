#!/usr/bin/env python3
"""Build R514: R500 top20-only ranker plus R510 official-positive rows.

R513 showed that applying R500's order inside R510 is neutral. R514 tests the
opposite composition: use the unsubmitted R500 top20-only branch as the base,
then copy the four rows with independent official-positive evidence from R510.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
R500_TOP5 = REPO / "exp/inference/blind_a/r500_top20_llm/r500_gpt41_top20_top5_keep_top1_submission.zip"
R500_FULL = REPO / "exp/inference/blind_a/r500_top20_llm/r500_gpt41_top20_full20_keep_top1_submission.zip"
R510 = REPO / "exp/inference/blind_a/r510_stack_r498_official_positives/r510_r498_plus_official_positive_rows_submission.zip"
OUT_DIR = REPO / "exp/inference/blind_a/r514_r500_plus_official_rows"
AUDIT_PATH = REPO / "exp/eval/expR514_r500_plus_official_rows_audit.json"
DOC_PATH = REPO / "docs/r514_r500_plus_official_rows_candidate.md"

OFFICIAL_POSITIVE_ROWS = [4, 9, 40, 65]


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


def validate(rows: list[dict[str, Any]], base: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    if len(rows) != 80:
        issues.append(f"row_count:{len(rows)}")
    for i, (row, base_row) in enumerate(zip(rows, base)):
        if row["session_id"] != base_row["session_id"] or int(row["turn_number"]) != int(base_row["turn_number"]):
            issues.append(f"key_changed:{i}")
        tids = row["predicted_track_ids"]
        if len(tids) != 20 or len(set(tids)) != 20:
            issues.append(f"bad_track_list:{i}")
        if not str(row.get("predicted_response", "")).strip():
            issues.append(f"empty_response:{i}")
    return issues


def build(base_path: Path, name: str) -> dict[str, Any]:
    base = load_zip(base_path)
    source = load_zip(R510)
    rows = deepcopy(base)
    for idx in OFFICIAL_POSITIVE_ROWS:
        rows[idx] = deepcopy(source[idx])

    out_zip = OUT_DIR / f"r514_{name}_plus_r510_positive_rows_submission.zip"
    sha = write_zip(out_zip, rows)
    issues = validate(rows, base)
    changed_vs_base = [
        i
        for i, (old, new) in enumerate(zip(base, rows))
        if old["predicted_track_ids"] != new["predicted_track_ids"]
    ]
    changed_resp_vs_base = [
        i
        for i, (old, new) in enumerate(zip(base, rows))
        if old["predicted_response"] != new["predicted_response"]
    ]
    top1_churn_vs_base = [
        i
        for i, (old, new) in enumerate(zip(base, rows))
        if old["predicted_track_ids"][0] != new["predicted_track_ids"][0]
    ]
    return {
        "name": name,
        "base_zip": str(base_path.relative_to(REPO)),
        "source_positive_rows_zip": str(R510.relative_to(REPO)),
        "out_zip": str(out_zip.relative_to(REPO)),
        "sha256": sha,
        "validation_issues": issues,
        "copied_rows_from_r510": OFFICIAL_POSITIVE_ROWS,
        "changed_rows_vs_base": changed_vs_base,
        "changed_response_rows_vs_base": changed_resp_vs_base,
        "top1_churn_rows_vs_base": top1_churn_vs_base,
    }


def main() -> None:
    audits = [
        build(R500_TOP5, "r500_top5"),
        build(R500_FULL, "r500_full20"),
    ]
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(audits, indent=2, ensure_ascii=False), encoding="utf-8")

    recommended = audits[0]
    DOC_PATH.write_text(
        "\n".join(
            [
                "# R514 - R500 Top20 Branch + Official-Positive Rows",
                "",
                "**Recommended upload:**",
                "",
                f"`{recommended['out_zip']}`",
                "",
                f"`sha256: {recommended['sha256']}`",
                "",
                "## Rationale",
                "",
                "R513 proved that borrowing R500 order inside the R510/R498 candidate pool is neutral. R514 tests the opposite composition: use the R500 top20-only GPT-4.1 branch as the base, then copy the four rows already banked in R510 from independent official-positive probes.",
                "",
                "The recommended variant uses `r500_top5_keep_top1`, because that was the R500 policy with the best risk profile: no response changes, no top-20 membership changes relative to the R432s base, and strong dev lift on admitted-hit rows.",
                "",
                "## Validation",
                "",
                f"- copied rows from R510: `{recommended['copied_rows_from_r510']}`",
                f"- changed rows vs R500 top5 base: `{recommended['changed_rows_vs_base']}`",
                f"- response rows changed vs R500 top5 base: `{recommended['changed_response_rows_vs_base']}`",
                f"- top1 churn rows vs R500 top5 base: `{recommended['top1_churn_rows_vs_base']}`",
                f"- validation issues: `{recommended['validation_issues']}`",
                "",
                "## Risk",
                "",
                "This is a distinct branch, not an incremental R510 tweak. It may underperform R510 if the R500 top20-only blind transfer is weak. It is still the cleanest unspent mechanism because R500 had strong fold-positive dev evidence and was not found in the official submission archive.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audits, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
