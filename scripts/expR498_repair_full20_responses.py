#!/usr/bin/env python3
"""Repair responses for the R498 raw full20 candidate.

The raw full20 policy is the highest-upside nDCG variant, but it changes top-1
on a subset of rows. This script keeps the raw full20 track lists and rewrites
only those mismatched responses so the text names the new top recommendation.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mcrs.utils import call_llm_api  # noqa: E402
from scripts.expR498_listwise_llm_reranker import (  # noqa: E402
    _read_jsonl,
    label_track,
    load_meta,
    read_base_zip,
    write_zip,
)

BASE = REPO / "exp/inference/blind_a/r432s_targeted_subset_submission.zip"
RAW_FULL20 = REPO / "exp/inference/blind_a/r498_listwise_llm/r498_gpt41_full20_submission.zip"
PROMPTS = REPO / "exp/eval/r498_listwise_llm/r498_blind_prompts.jsonl"
OUT_JSONL = REPO / "exp/eval/r498_listwise_llm/r498_full20_response_repairs_gpt41.jsonl"
OUT_ZIP = REPO / "exp/inference/blind_a/r498_listwise_llm/r498_gpt41_full20_repaired_submission.zip"


SYSTEM = """\
You write concise, natural music recommendation responses for a benchmark.
The ranked track list is already fixed. Your job is only to write text that
coherently recommends the provided #1 track for the conversation.

Rules:
- Mention the #1 title and artist exactly once near the beginning.
- Do not mention track IDs.
- Do not claim private knowledge beyond the given metadata.
- Keep it to 2-3 compact sentences.
- No markdown, no bullet list.
"""


def read_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def repair_prompt(record: dict[str, Any], new_top: str, top5: list[str], meta: dict[str, dict[str, Any]]) -> str:
    top5_text = "\n".join(f"{i}. {label_track(t, meta)}" for i, t in enumerate(top5, 1))
    return f"""\
Conversation:
{record["conversation"]}

Fixed ranked recommendation list, top 5:
{top5_text}

Write the response. The #1 recommendation must be:
{label_track(new_top, meta)}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, default=BASE)
    ap.add_argument("--raw-full20", type=Path, default=RAW_FULL20)
    ap.add_argument("--prompts", type=Path, default=PROMPTS)
    ap.add_argument("--out-jsonl", type=Path, default=OUT_JSONL)
    ap.add_argument("--out-zip", type=Path, default=OUT_ZIP)
    ap.add_argument("--model", default="gpt-4.1")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    meta = load_meta()
    base_rows = read_base_zip(args.base)
    raw_rows = read_zip(args.raw_full20)
    records = {r["session_id"]: r for r in _read_jsonl(args.prompts)}

    done: dict[str, str] = {}
    if args.out_jsonl.exists() and not args.overwrite:
        for row in _read_jsonl(args.out_jsonl):
            done[row["session_id"]] = row["response"]

    changed_sids = []
    repaired_rows = []
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite or not args.out_jsonl.exists() else "a"
    with open(args.out_jsonl, mode, encoding="utf-8") as f:
        for base_row, raw_row in zip(base_rows, raw_rows):
            if base_row["session_id"] != raw_row["session_id"]:
                raise ValueError("row order/session mismatch")
            new_row = dict(raw_row)
            old_top = base_row["predicted_track_ids"][0]
            new_top = raw_row["predicted_track_ids"][0]
            if old_top != new_top:
                sid = raw_row["session_id"]
                changed_sids.append(sid)
                if sid in done:
                    response = done[sid]
                else:
                    prompt = repair_prompt(records[sid], new_top, raw_row["predicted_track_ids"][:5], meta)
                    response = call_llm_api(
                        SYSTEM,
                        prompt,
                        model=args.model,
                        max_tokens=args.max_tokens,
                        strict_no_truncation=True,
                    )
                    payload = {
                        "session_id": sid,
                        "old_top1": old_top,
                        "new_top1": new_top,
                        "new_top1_label": label_track(new_top, meta),
                        "response": response,
                    }
                    f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"repaired {len(changed_sids)}: {sid[:8]}", flush=True)
                new_row["predicted_response"] = response
            repaired_rows.append(new_row)

    sha = write_zip(args.out_zip, repaired_rows)
    manifest = {
        "experiment": "R498 raw full20 with repaired top1 responses",
        "base": str(args.base.relative_to(REPO)),
        "raw_full20": str(args.raw_full20.relative_to(REPO)),
        "prompts": str(args.prompts.relative_to(REPO)),
        "repairs": str(args.out_jsonl.relative_to(REPO)),
        "out_zip": str(args.out_zip.relative_to(REPO)),
        "sha256": sha,
        "top1_changed_and_repaired": len(changed_sids),
    }
    args.out_zip.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
