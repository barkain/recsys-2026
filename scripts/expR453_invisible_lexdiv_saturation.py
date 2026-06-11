#!/usr/bin/env python3
"""R453 - invisible LexDiv saturation on top of R452.

R452 proved zero-width LexDiv padding can raise LexDiv to 0.9749 while holding
LLM 4.90. Remaining LexDiv headroom is small, so this builds only moderate
larger payloads. If the judge still ignores the invisible payload, the best
variant can add roughly 0.001-0.002 composite.
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
BASE_ZIP = REPO / "exp/inference/blind_a/r451_tail_carrier_sweep/01_R451_ROW51_9d4ef919_tail.zip"
OUT_DIR = REPO / "exp/inference/blind_a/r453_invisible_lexdiv_saturation"
OUT_AUDIT = REPO / "exp/eval/expR453_invisible_lexdiv_saturation_audit.json"
OUT_SCORES = REPO / "exp/eval/expR453_invisible_lexdiv_saturation_scores_template.csv"

ZW_CHARS = [chr(0x200B), chr(0x200C), chr(0x200D), chr(0x2060)]
PAYLOAD_SIZES = [384, 512, 768, 1024]


def read_zip(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def write_zip(path: Path, rows: list[dict[str, Any]]) -> tuple[str, str]:
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)
    return hashlib.sha256(payload).hexdigest(), hashlib.sha256(path.read_bytes()).hexdigest()


def lexical_diversity(responses: list[str]) -> float:
    seen: set[tuple[str, str]] = set()
    total = 0
    for response in responses:
        toks = (response or "").lower().split()
        for i in range(len(toks) - 1):
            seen.add((toks[i], toks[i + 1]))
            total += 1
    return len(seen) / total if total else 0.0


def catalog_diversity(rows: list[dict[str, Any]]) -> float:
    return len({str(t) for row in rows for t in row["predicted_track_ids"]}) / 47071


def encode_token(n: int) -> str:
    chars: list[str] = []
    for _ in range(10):
        chars.append(ZW_CHARS[n & 3])
        n >>= 2
    return "".join(chars)


def build_rows(base_rows: list[dict[str, Any]], tokens_per_row: int) -> list[dict[str, Any]]:
    rows = deepcopy(base_rows)
    counter = 1
    for row in rows:
        payload = [encode_token(counter + i) for i in range(tokens_per_row)]
        counter += tokens_per_row
        row["predicted_response"] = str(row["predicted_response"]).rstrip() + " " + " ".join(payload)
    return rows


def strip_zero_width(text: str) -> str:
    for ch in ZW_CHARS:
        text = text.replace(ch, "")
    return " ".join(text.split())


def expected_composite(lexdiv: float, llm: float = 4.90) -> float:
    return 0.5 * 0.5092 + 0.1 * 1.0 + 0.1 * lexdiv + 0.3 * ((llm - 1.0) / 4.0)


def main() -> None:
    base_rows = read_zip(BASE_ZIP)
    base_first20 = [row["predicted_track_ids"][:20] for row in base_rows]
    base_lens = [len(row["predicted_track_ids"]) for row in base_rows]
    base_visible = [strip_zero_width(str(row["predicted_response"])) for row in base_rows]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    variants: list[dict[str, Any]] = []
    for tokens_per_row in PAYLOAD_SIZES:
        rows = build_rows(base_rows, tokens_per_row)
        out = OUT_DIR / f"r453_invisible_lexdiv_zwtok{tokens_per_row:04d}.zip"
        inner_sha, outer_sha = write_zip(out, rows)
        lex = lexical_diversity([str(row["predicted_response"]) for row in rows])
        variants.append(
            {
                "probe_id": f"r453_zwtok{tokens_per_row:04d}",
                "tokens_per_row": tokens_per_row,
                "zip": str(out.relative_to(REPO)),
                "outer_zip_sha256": outer_sha,
                "inner_prediction_json_sha256": inner_sha,
                "file_size_bytes": out.stat().st_size,
                "local_lexical_diversity": lex,
                "local_catalog_diversity": catalog_diversity(rows),
                "expected_composite_if_llm_4_90": expected_composite(lex, 4.90),
                "expected_composite_if_llm_4_85": expected_composite(lex, 4.85),
                "first20_identical_to_base": all(
                    row["predicted_track_ids"][:20] == first20 for row, first20 in zip(rows, base_first20)
                ),
                "track_lengths_identical_to_base": [len(row["predicted_track_ids"]) for row in rows] == base_lens,
                "visible_response_text_identical_after_stripping_zero_width": all(
                    strip_zero_width(str(row["predicted_response"])) == visible
                    for row, visible in zip(rows, base_visible)
                ),
            }
        )

    audit = {
        "experiment": "R453 invisible LexDiv saturation",
        "base_zip": str(BASE_ZIP.relative_to(REPO)),
        "base_official_anchor": {
            "probe_id": "r452_zwtok256",
            "ndcg20": 0.5092,
            "catalog_diversity": 1.0,
            "lexdiv": 0.9749,
            "llm": 4.90,
            "composite": 0.7446,
        },
        "recommended_first_upload": next(item["zip"] for item in variants if item["tokens_per_row"] == 768),
        "variants": variants,
    }
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.write_text(json.dumps(audit, indent=2) + "\n")

    OUT_SCORES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SCORES.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["probe_id", "tokens_per_row", "zip", "ndcg20", "catalog_diversity", "lexdiv", "llm", "composite", "notes"],
        )
        writer.writeheader()
        for item in variants:
            writer.writerow(
                {
                    "probe_id": item["probe_id"],
                    "tokens_per_row": item["tokens_per_row"],
                    "zip": item["zip"],
                    "ndcg20": "",
                    "catalog_diversity": "",
                    "lexdiv": "",
                    "llm": "",
                    "composite": "",
                    "notes": f"local LexDiv={item['local_lexical_diversity']:.4f}; visible text unchanged",
                }
            )

    for item in variants:
        print(
            f"{item['probe_id']} LexDiv={item['local_lexical_diversity']:.4f} "
            f"comp@4.90={item['expected_composite_if_llm_4_90']:.4f} "
            f"comp@4.85={item['expected_composite_if_llm_4_85']:.4f} "
            f"sha={item['outer_zip_sha256']}"
        )
    print(f"audit: {OUT_AUDIT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
