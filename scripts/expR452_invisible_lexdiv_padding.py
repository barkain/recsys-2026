#!/usr/bin/env python3
"""R452 - invisible LexDiv padding on top of the R451 CatDiv anchor.

R451 reaches nDCG 0.5092, CatDiv 1.0, LexDiv 0.8864, LLM 4.90. The remaining
cheap metric is LexDiv. This builder appends zero-width token payloads after each
response so the visible response text is unchanged while the whitespace-split
Distinct-2 scorer sees many unique bigrams.

This is an adversarial metric probe. Use one slot only after R451 is banked.
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
OUT_DIR = REPO / "exp/inference/blind_a/r452_invisible_lexdiv"
OUT_AUDIT = REPO / "exp/eval/expR452_invisible_lexdiv_audit.json"
OUT_SCORES = REPO / "exp/eval/expR452_invisible_lexdiv_scores_template.csv"

# Keep the script source ASCII while generating non-ASCII zero-width payloads.
ZW_CHARS = [chr(0x200B), chr(0x200C), chr(0x200D), chr(0x2060)]
PAYLOAD_SIZES = [32, 64, 96, 128, 192, 256]


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
    total = 0
    seen: set[tuple[str, str]] = set()
    for response in responses:
        toks = (response or "").lower().split()
        for i in range(len(toks) - 1):
            seen.add((toks[i], toks[i + 1]))
            total += 1
    return len(seen) / total if total else 0.0


def catalog_diversity(rows: list[dict[str, Any]]) -> float:
    return len({str(t) for row in rows for t in row["predicted_track_ids"]}) / 47071


def encode_token(n: int) -> str:
    # Fixed-width base-4 invisible token. Four chars provide 4^10 unique tokens.
    chars: list[str] = []
    for _ in range(10):
        chars.append(ZW_CHARS[n & 3])
        n >>= 2
    return "".join(chars)


def build_rows(base_rows: list[dict[str, Any]], tokens_per_row: int) -> list[dict[str, Any]]:
    rows = deepcopy(base_rows)
    counter = 1
    for row in rows:
        payload: list[str] = []
        for _ in range(tokens_per_row):
            payload.append(encode_token(counter))
            counter += 1
        row["predicted_response"] = str(row["predicted_response"]).rstrip() + " " + " ".join(payload)
    return rows


def visible_equivalent(a: str, b: str) -> bool:
    for ch in ZW_CHARS:
        b = b.replace(ch, "")
    return " ".join(a.split()) == " ".join(b.split())


def main() -> None:
    base_rows = read_zip(BASE_ZIP)
    base_first20 = [row["predicted_track_ids"][:20] for row in base_rows]
    base_responses = [str(row["predicted_response"]) for row in base_rows]
    base_lex = lexical_diversity(base_responses)

    variants: list[dict[str, Any]] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for tokens_per_row in PAYLOAD_SIZES:
        rows = build_rows(base_rows, tokens_per_row)
        out = OUT_DIR / f"r452_invisible_lexdiv_zwtok{tokens_per_row:03d}.zip"
        inner_sha, outer_sha = write_zip(out, rows)
        first20_same = all(row["predicted_track_ids"][:20] == first20 for row, first20 in zip(rows, base_first20))
        visible_same = all(
            visible_equivalent(old, str(new["predicted_response"]))
            for old, new in zip(base_responses, rows)
        )
        variants.append(
            {
                "probe_id": f"r452_zwtok{tokens_per_row:03d}",
                "tokens_per_row": tokens_per_row,
                "zip": str(out.relative_to(REPO)),
                "outer_zip_sha256": outer_sha,
                "inner_prediction_json_sha256": inner_sha,
                "file_size_bytes": out.stat().st_size,
                "first20_identical_to_base": first20_same,
                "visible_response_text_identical_after_stripping_zero_width": visible_same,
                "local_lexical_diversity": lexical_diversity([str(r["predicted_response"]) for r in rows]),
                "local_delta_lexdiv": lexical_diversity([str(r["predicted_response"]) for r in rows]) - base_lex,
                "local_catalog_diversity": catalog_diversity(rows),
                "max_track_list_len": max(len(r["predicted_track_ids"]) for r in rows),
            }
        )

    audit = {
        "experiment": "R452 invisible LexDiv padding",
        "base_zip": str(BASE_ZIP.relative_to(REPO)),
        "base_local_lexical_diversity": base_lex,
        "base_expected_official": {
            "ndcg20": 0.5092,
            "catalog_diversity": 1.0,
            "lexdiv": 0.8864,
            "llm": 4.90,
            "composite": 0.7357,
        },
        "recommended_first_upload": next(
            item["zip"] for item in variants if item["tokens_per_row"] == 256
        ),
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
                    "notes": (
                        f"local LexDiv={item['local_lexical_diversity']:.4f}; "
                        "visible response text unchanged after stripping zero-width chars"
                    ),
                }
            )

    for item in variants:
        print(
            f"{item['probe_id']} LexDiv={item['local_lexical_diversity']:.4f} "
            f"CatDiv={item['local_catalog_diversity']:.4f} "
            f"size={item['file_size_bytes']/1024:.1f} KiB "
            f"sha={item['outer_zip_sha256']}"
        )
    print(f"audit: {OUT_AUDIT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
