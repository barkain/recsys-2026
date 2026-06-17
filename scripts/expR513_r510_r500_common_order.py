#!/usr/bin/env python3
"""Build R513: R510 candidate set + R500 in-set listwise order.

R512 failed because it appended new tail candidates selected by an LLM. R513
does not admit any new candidates. It starts from R510 and, for high-agreement
rows only, reorders tracks already present in R510 according to R500's top-20
listwise GPT-4.1 order.
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
ORDER_ZIP = REPO / "exp/inference/blind_a/r500_top20_llm/r500_gpt41_top20_full20_keep_top1_submission.zip"
R500_OUTPUTS = REPO / "exp/eval/r500_top20_llm/r500_blind_top20_gpt41_outputs.jsonl"
OUT_DIR = REPO / "exp/inference/blind_a/r513_r510_r500_common_order"
AUDIT_PATH = REPO / "exp/eval/expR513_r510_r500_common_order_audit.json"


VARIANTS = [
    # Top-5-only matches the best R500 deployment shape: preserve top-1 and
    # only alter high-leverage visible head positions.
    {"name": "top5_ov14_conf70", "mode": "top5", "min_overlap": 14, "min_conf": 0.70},
    {"name": "top5_ov16_conf65", "mode": "top5", "min_overlap": 16, "min_conf": 0.65},
    {"name": "top5_ov17_conf80", "mode": "top5", "min_overlap": 17, "min_conf": 0.80},
    # Full-common variants are staged for analysis but are riskier because they
    # reorder many tail positions that the judge may read.
    {"name": "full_ov17_conf80", "mode": "full", "min_overlap": 17, "min_conf": 0.80},
    {"name": "full_ov16_conf65", "mode": "full", "min_overlap": 16, "min_conf": 0.65},
]


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


def load_r500_confidence() -> dict[str, float]:
    out: dict[str, float] = {}
    with R500_OUTPUTS.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            raw = json.loads(rec["raw"])
            sid = rec["session_id"].split("-")[0]
            out[sid] = float(raw.get("confidence", 0.0))
    return out


def reorder(base: list[str], order: list[str], mode: str) -> list[str]:
    top1 = base[0]
    pref = [tid for tid in order if tid in base and tid != top1]
    if mode == "top5":
        selected = pref[:4]
        return [top1] + selected + [tid for tid in base[1:] if tid not in selected]
    if mode == "full":
        return [top1] + pref + [tid for tid in base[1:] if tid not in pref]
    raise ValueError(f"unknown mode: {mode}")


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
        if set(tids) != set(base_row["predicted_track_ids"]):
            issues.append(f"membership_changed:{i}")
        if tids[0] != base_row["predicted_track_ids"][0]:
            issues.append(f"top1_changed:{i}")
        if row["predicted_response"] != base_row["predicted_response"]:
            issues.append(f"response_changed:{i}")
    return issues


def main() -> None:
    base = load_zip(BASE_ZIP)
    order_rows = load_zip(ORDER_ZIP)
    confidence = load_r500_confidence()
    all_audits: list[dict[str, Any]] = []

    for variant in VARIANTS:
        rows = deepcopy(base)
        applied: list[dict[str, Any]] = []
        for idx, (base_row, order_row) in enumerate(zip(base, order_rows)):
            sid = base_row["session_id"].split("-")[0]
            base_tids = list(base_row["predicted_track_ids"])
            order_tids = list(order_row["predicted_track_ids"])
            overlap = len(set(base_tids) & set(order_tids))
            conf = confidence.get(sid, 0.0)
            if overlap < variant["min_overlap"] or conf < variant["min_conf"]:
                continue
            new_tids = reorder(base_tids, order_tids, variant["mode"])
            if new_tids == base_tids:
                continue
            rows[idx]["predicted_track_ids"] = new_tids
            applied.append(
                {
                    "row_index": idx,
                    "session_short": sid,
                    "turn_number": int(base_row["turn_number"]),
                    "confidence": conf,
                    "overlap": overlap,
                    "changed_positions": [j + 1 for j, (a, b) in enumerate(zip(base_tids, new_tids)) if a != b],
                    "base_top5": base_tids[:5],
                    "new_top5": new_tids[:5],
                }
            )

        name = variant["name"]
        out_zip = OUT_DIR / f"r513_r510_r500_common_{name}_submission.zip"
        sha = write_zip(out_zip, rows)
        issues = validate(rows, base)
        changed_rows = [
            i
            for i, (old, new) in enumerate(zip(base, rows))
            if old["predicted_track_ids"] != new["predicted_track_ids"]
        ]
        audit = {
            "name": name,
            "mode": variant["mode"],
            "min_overlap": variant["min_overlap"],
            "min_conf": variant["min_conf"],
            "base_zip": str(BASE_ZIP.relative_to(REPO)),
            "order_zip": str(ORDER_ZIP.relative_to(REPO)),
            "out_zip": str(out_zip.relative_to(REPO)),
            "sha256": sha,
            "validation_issues": issues,
            "changed_rows": changed_rows,
            "changed_row_count": len(changed_rows),
            "top1_changed_count": 0,
            "response_changed_count": 0,
            "membership_changed_count": 0,
            "applied": applied,
        }
        all_audits.append(audit)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(all_audits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(all_audits, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
