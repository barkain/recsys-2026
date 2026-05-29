#!/usr/bin/env python3
"""R98b response-style repair.

R98 improved the reproduced competition LexDiv metric, but blind LLM dropped
4.90 -> 4.85. This script builds rollback hybrids:

- Start from R92 p11 production tracks/responses.
- Keep only the R98 rewrites that look closest to the R92/R84c direct style.
- Leave tracks bitwise identical to p11, so nDCG remains fixed.

Outputs two candidates:
- ultra: smaller LexDiv gain, lowest response-style risk.
- conservative: larger LexDiv gain, still excludes the obvious risky rewrites.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path

from scripts.lexdiv_scorer import catalog_diversity, lexical_diversity

REPO = Path(__file__).resolve().parent.parent
BLIND = REPO / "exp" / "inference" / "blind_a"

P11_ZIP = BLIND / "r92_p11_oracle_submission.zip"
R98_ZIP = BLIND / "r98_lexdiv_safe.zip"
R98_META = BLIND / "r98_lexdiv_regen.metadata.json"

OUT_ULTRA = BLIND / "r98b_style_repair_ultra.zip"
OUT_CONSERVATIVE = BLIND / "r98b_style_repair_conservative.zip"
OUT_META = BLIND / "r98b_style_repair.metadata.json"

P11_COMPOSITE = 0.6364
P11_LEXDIV = 0.8720
LEXDIV_GRADIENT = 0.10

# Ultra keeps only rewrites with very direct R92-like structure and no poetic
# closer / correction of previously judge-accepted response semantics.
ULTRA_KEEP = {
    ("0802ac4a-187b-4a78-b090-0ceb7630cf12", 6),
    ("28c3ecd9-fbaa-4423-a85c-92508346e1e8", 6),
    ("789f9994-f2b6-481d-b71a-c0c6706d82c6", 2),
    ("d9cca604-febe-4c95-a5cc-a4318c33ec40", 2),
}

# Conservative adds the strongest remaining direct/concrete rewrites, while
# rolling back the highest-risk rows:
# - c4f7d055: changed response semantics on the p11 oracle row.
# - ee7bfbda/fc6ba76a: softer mood prose / less judge-aligned closers.
# - 77faef85: album-level justification instead of direct track fit.
CONSERVATIVE_KEEP = ULTRA_KEEP | {
    ("5ad7094f-3764-4e32-94c9-c0c02b65e01b", 4),
    ("6c54de37-9c55-4c59-b203-ec190dd07523", 2),
}


def load_zip(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("prediction.json"))


def row_key(row: dict) -> tuple[str, int]:
    return row["session_id"], int(row["turn_number"])


def write_zip(path: Path, rows: list[dict]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_variant(name: str, keep: set[tuple[str, int]], p11_rows: list[dict], r98_by_key: dict) -> dict:
    rows: list[dict] = []
    kept_rows = []
    for row in p11_rows:
        out = dict(row)
        k = row_key(row)
        if k in keep:
            out["predicted_response"] = r98_by_key[k]["predicted_response"]
            kept_rows.append({"session_id": k[0], "turn_number": k[1]})
        rows.append(out)

    # Safety: tracks must remain bitwise identical to p11 row-by-row.
    for before, after in zip(p11_rows, rows):
        if before["predicted_track_ids"] != after["predicted_track_ids"]:
            raise RuntimeError(f"track mismatch in {name}: {row_key(before)}")

    lex = lexical_diversity([r["predicted_response"] for r in rows])
    cat = catalog_diversity([r["predicted_track_ids"] for r in rows])
    out_path = OUT_ULTRA if name == "ultra" else OUT_CONSERVATIVE
    sha = write_zip(out_path, rows)
    return {
        "name": name,
        "path": str(out_path.relative_to(REPO)),
        "sha256": sha,
        "n_kept_r98_rewrites": len(kept_rows),
        "kept_rows": kept_rows,
        "lexical_diversity": round(lex, 6),
        "catalog_diversity": round(cat, 6),
        "predicted_composite_if_llm_4_90": round(P11_COMPOSITE + LEXDIV_GRADIENT * (lex - P11_LEXDIV), 4),
        "predicted_delta_if_llm_4_90": round(LEXDIV_GRADIENT * (lex - P11_LEXDIV), 4),
    }


def main() -> None:
    p11_rows = load_zip(P11_ZIP)
    r98_rows = load_zip(R98_ZIP)
    r98_meta = json.loads(R98_META.read_text())

    p11_by_key = {row_key(r): r for r in p11_rows}
    r98_by_key = {row_key(r): r for r in r98_rows}
    if set(p11_by_key) != set(r98_by_key):
        raise RuntimeError("p11/r98 key sets differ")

    changed = [row_key(ch) for ch in r98_meta["changelog"]]
    actual_changed = [
        k for k in p11_by_key
        if p11_by_key[k]["predicted_response"] != r98_by_key[k]["predicted_response"]
    ]
    if set(changed) != set(actual_changed):
        raise RuntimeError("R98 metadata changelog does not match actual response diffs")

    p11_lex = lexical_diversity([r["predicted_response"] for r in p11_rows])
    r98_lex = lexical_diversity([r["predicted_response"] for r in r98_rows])

    results = [
        build_variant("ultra", ULTRA_KEEP, p11_rows, r98_by_key),
        build_variant("conservative", CONSERVATIVE_KEEP, p11_rows, r98_by_key),
    ]

    meta = {
        "experiment": "R98b response-style repair",
        "created_at": datetime.now().isoformat(),
        "base": str(P11_ZIP.relative_to(REPO)),
        "r98_source": str(R98_ZIP.relative_to(REPO)),
        "reason": "R98 raised LexDiv but dropped blind LLM 4.90->4.85; repair rolls back risky rewrites.",
        "tracks_identical_to_p11": True,
        "p11_lexical_diversity": round(p11_lex, 6),
        "r98_lexical_diversity": round(r98_lex, 6),
        "variants": results,
        "rolled_back_by_design": [
            "c4f7d055-a3cc-4d6b-be80-b90278bc0d32:t7",
            "ee7bfbda-86ee-4ade-b695-dbdeb12ae063:t3",
            "fc6ba76a-3dd8-4c11-b5af-a233253b01bf:t1",
            "77faef85-566c-4eb7-90b6-ee9fdeeacbe8:t1",
        ],
        "recommendation": "Upload ultra first if slot-constrained; conservative has more LexDiv but slightly more LLM risk.",
    }
    OUT_META.write_text(json.dumps(meta, indent=2))

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
