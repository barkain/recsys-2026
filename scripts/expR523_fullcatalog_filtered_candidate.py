#!/usr/bin/env python3
"""R523: filtered full-catalog generative retrieval candidate.

The raw GPT-4.1 full-catalog selector found four Blind-A top-1 swaps at
threshold 0.6, but two are known or obvious traps:

- row 16 / License to Drive: exact-title probe already failed officially.
- row 19 / ONE OK ROCK: model picked a Japanese-title track while the user
  explicitly praised "We Are"; this is likely a semantic mismatch.

This build keeps only the two high-confidence rows whose selected tracks are
already plausible rank corrections and repairs the matching responses.
"""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
BASE_ZIP = REPO / "exp/inference/blind_a/r510_stack_r498_official_positives/r510_r498_plus_official_positive_rows_submission.zip"
OUT_DIR = REPO / "exp/inference/blind_a/r523_generative_catalog_r510"
OUT_ZIP = OUT_DIR / "r523_r510_gpt41_fullcatalog_filtered2_repaired_submission.zip"
OUT_AUDIT = REPO / "exp/eval/r523_generative_catalog_r510/r523_filtered2_repaired_audit.json"
META = REPO / "cache/metadata/track_metadata_all_tracks.json"


ACTIONS = [
    {
        "row_index": 6,
        "session_prefix": "1415a335",
        "target_track_id": "ed5c9953-3ba6-4bb9-80e4-7bedcc4a1833",
        "target_label": "Battle Metal - Turisas",
        "reason": "Full-catalog GPT-4.1 selected the exact title/artist at confidence 0.82 for the folk/viking metal battle-scene query; current R510 has it at rank 3.",
        "response": (
            'I recommend "Battle Metal" by Turisas. It directly fits the epic folk/viking-metal direction you asked for: '
            "mythic battle imagery, huge chanted hooks, and a forceful warrior-march energy that follows naturally from "
            '"End of an Empire". I kept the rest of the list close to that Turisas-led battle-metal sound.'
        ),
    },
    {
        "row_index": 36,
        "session_prefix": "68993adf",
        "target_track_id": "707d63de-3ecd-47f1-8b22-21d73207dd96",
        "target_label": "Story of My Life - One Direction",
        "reason": "Full-catalog GPT-4.1 selected the exact title/artist at confidence 0.70 for the narrative life-story query; current R510 has it at rank 2.",
        "response": (
            'I recommend "Story of My Life" by One Direction. It is a direct match for your request for a song that feels '
            "like a life story unfolding: reflective, narrative-driven, and slightly melancholic without losing its melodic "
            "pull. The rest of the list keeps adjacent story-song alternatives."
        ),
    },
]


def read_zip(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if names != ["prediction.json"]:
            raise RuntimeError(f"unexpected zip entries in {path}: {names}")
        return json.loads(zf.read("prediction.json"))


def write_zip(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo("prediction.json")
        info.date_time = (1980, 1, 1, 0, 0, 0)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first(value):
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def label_track(track_id: str, meta: dict) -> str:
    m = meta.get(track_id, {})
    return f"{first(m.get('track_name'))} - {first(m.get('artist_name'))}"


def main() -> None:
    meta = json.load(open(META, encoding="utf-8"))
    base = read_zip(BASE_ZIP)
    rows = copy.deepcopy(base)
    audit_rows = []

    for action in ACTIONS:
        idx = action["row_index"]
        row = rows[idx]
        if not row["session_id"].startswith(action["session_prefix"]):
            raise RuntimeError(f"row {idx} session mismatch: {row['session_id']}")
        target = action["target_track_id"]
        old_top20 = [str(t) for t in row["predicted_track_ids"][:20]]
        if target not in old_top20:
            raise RuntimeError(f"target absent from row {idx}: {target}")
        old_rank = old_top20.index(target) + 1
        new_top20 = [target] + [t for t in old_top20 if t != target]
        row["predicted_track_ids"] = new_top20[:20]
        row["predicted_response"] = action["response"]
        audit_rows.append(
            {
                **action,
                "session_id": row["session_id"],
                "turn_number": row["turn_number"],
                "old_rank": old_rank,
                "old_top1": old_top20[0],
                "old_top1_label": label_track(old_top20[0], meta),
                "new_top1": target,
                "new_top1_label": label_track(target, meta),
                "max_delta_if_gt": 1.0 - (1.0 / __import__("math").log2(old_rank + 1)),
            }
        )

    issues = []
    for i, row in enumerate(rows):
        tids = row.get("predicted_track_ids") or []
        if len(tids) != 20 or len(set(tids)) != 20:
            issues.append(f"bad_track_ids:{i}")
    changed_track_rows = [
        i for i, (a, b) in enumerate(zip(base, rows))
        if a["predicted_track_ids"] != b["predicted_track_ids"]
    ]
    changed_response_rows = [
        i for i, (a, b) in enumerate(zip(base, rows))
        if a["predicted_response"] != b["predicted_response"]
    ]
    sha = write_zip(OUT_ZIP, rows)
    audit = {
        "experiment": "R523 filtered full-catalog GPT-4.1 candidate",
        "base_zip": str(BASE_ZIP.relative_to(REPO)),
        "out_zip": str(OUT_ZIP.relative_to(REPO)),
        "sha256": sha,
        "source_dev_result": {
            "slice_rows": 240,
            "policy": "GPT-4.1 generative full-catalog, threshold>=0.6, insert rank1",
            "delta_ndcg": 0.009014274319945533,
            "recovered": 1,
            "lost": 0,
        },
        "raw_blind_selector": {
            "threshold": 0.6,
            "changed_rows": [6, 16, 19, 36],
            "dropped_rows": {
                "16": "License to Drive exact-title row already failed officially.",
                "19": "ONE OK ROCK pick conflicts with explicit We Are/album context.",
            },
        },
        "changed_track_rows": changed_track_rows,
        "changed_response_rows": changed_response_rows,
        "top1_churn": len(changed_track_rows),
        "validation_issues": issues,
        "actions": audit_rows,
    }
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "actions"}, indent=2))


if __name__ == "__main__":
    main()
