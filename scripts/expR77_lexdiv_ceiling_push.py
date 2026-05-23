#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R77 LexDiv ceiling push — iterate R74's bigram-repeat-density approach on R74 corpus.

R74 lifted LexDiv 0.8536→0.8719 (+0.0183) via 15-row regen on R73 base.
R77 takes R74 as input and runs the SAME audit + regen prompt. The
bigram audit will pick the new highest-density rows (different from
R74's set because R74 already reduced those bigrams).

Tracks bitwise identical to R74 (zero nDCG risk). LLM ceiling 4.85
preserved by keeping R74's prompt and archetype rotation unchanged.

Goal: LexDiv 0.8719 → 0.88+. Composite +0.003-0.005 → ~0.628-0.630.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Reuse R74's logic with patched input/output paths
import scripts.expR74_lexdiv_polish as r74

# R74 final rows become R77 input
r74.R73_ROWS = REPO / "exp" / "inference" / "blind_a" / "r74_lexdiv_rows_final.jsonl"

# R77 output paths
r74.PERSISTED_ROWS = REPO / "exp" / "inference" / "blind_a" / "r77_ceiling_rows_persisted.jsonl"
r74.FINAL_ROWS = REPO / "exp" / "inference" / "blind_a" / "r77_ceiling_rows_final.jsonl"
r74.OUT_ZIP = REPO / "exp" / "inference" / "blind_a" / "r77_ceiling_submission.zip"
r74.OUT_METADATA = REPO / "exp" / "inference" / "blind_a" / "r77_ceiling_submission.metadata.json"
r74.OUT_DOC = REPO / "docs" / "r77_ceiling_result.md"
r74.AUDIT_JSON = REPO / "exp" / "eval" / "expR77_bigram_audit.json"
r74.PROMPT_VERSION = (
    "R77 v1; LexDiv ceiling push on R74 base; same prompt and archetypes as "
    "R74; re-audited bigram density on R74 corpus to find residual high-"
    "density rows"
)

if __name__ == "__main__":
    r74.main()
