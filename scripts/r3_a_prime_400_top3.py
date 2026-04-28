#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Powell CV5 only for the top-3 A' configs by pool_hit@50.

Reuses cached A' lists + 400-row payload. No API, no inference, no model dl.
Top configs (by cheap pool_hit@50 prune from r3_confirm_400_a_prime.py results):
  1. A_max_recent_5 w=2.0  → pool_hit 0.3250
  2. A_max_recent_3 w=2.0  → pool_hit 0.3100
  3. A_max_recent_5 w=1.0  → pool_hit 0.3050
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3_confirm_400_a_prime import (  # noqa: E402
    A_PRIME_CACHE,
    augment_metadata_for_a,
    eval_one,
)
from scripts.r3_confirm_400_deterministic import CACHE_FILE  # noqa: E402

TOP_CONFIGS = [
    ("A_max_recent_5", 2.0),
    ("A_max_recent_3", 2.0),
    ("A_max_recent_5", 1.0),
]
RESULTS_FILE = "exp/eval/r3_a_prime_400_top3_results.json"


def main() -> None:
    t0 = time.time()
    with open(CACHE_FILE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    with open(A_PRIME_CACHE, "rb") as f:
        a_variants = pickle.load(f)  # noqa: S301

    # Augment metadata for any A' tids missing from baseline payload (idempotent).
    augment_metadata_for_a(payload, a_variants)

    seeds = [0, 1, 2]
    out: dict[str, Any] = {
        "meta": {
            "n": len(payload["cases"]),
            "seeds": seeds,
            "fixed_BCD_weights": [2.0, 0.5, 1.0],
            "selection_protocol": (
                "top-3 by cheap pool_hit@50 from r3_confirm_400_a_prime "
                "(per Codex efficiency directive)."
            ),
            "no_llm_calls": True,
        },
        "results": {},
    }

    for v_name, w_a in TOP_CONFIGS:
        key = f"{v_name}__wA{w_a}"
        print(f"\n[{key}] Powell CV5 start", flush=True)
        res = eval_one(payload, a_variants[v_name], w_a, seeds)
        out["results"][key] = {"variant": v_name, **res}
        agg = res["aggregate"]
        print(
            f"[{key}] mean_cv5={agg['mean_cv5']:.4f}  "
            f"std={agg['std_of_cv5_means']:.4f}  "
            f"pool_hit={res['pool_hit_rate']:.4f}",
            flush=True,
        )
        Path(RESULTS_FILE).parent.mkdir(parents=True, exist_ok=True)
        out["meta"]["elapsed_sec"] = time.time() - t0
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)

    out["meta"]["elapsed_sec"] = time.time() - t0
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {RESULTS_FILE}  elapsed={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
