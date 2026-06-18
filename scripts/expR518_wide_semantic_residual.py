#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201
"""R518: wide semantic residual reranker.

R516 was positive but capped because its candidate pool was only production top20
+ R480 insertion candidates.  R515 showed R54/R84 add reachability but rank-only
fusion cannot select safely.  This experiment combines both:

  production top20 + R480 insertion candidates + R54/R84 retrieval candidates

and reuses the R516 semantic LGBM feature set, evaluated all-dev OOF.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_goal65_eval import evaluate, load_dev  # noqa: E402
from scripts.expR516_semantic_residual_reranker import (  # noqa: E402
    OUT_JSON as _R516_OUT_JSON,
    build_matrix,
    blended_rankings,
    load_payload_bits,
    load_r480_rows,
    model_rankings,
    train_oof,
    unique_keep_order,
)

OUT_JSON = REPO / "exp/eval/expR518_wide_semantic_residual.json"
OUT_MD = REPO / "docs/r518_wide_semantic_residual.md"

CONFIGS = [
    {"name": "r48080_r54r84_50", "r480_depth": 80, "r54_depth": 50, "r84_depth": 50},
    {"name": "r48080_r54r84_100", "r480_depth": 80, "r54_depth": 100, "r84_depth": 100},
]
BASE_WEIGHTS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.5, 5.0, 8.0]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def build_wide_pools(dev: dict[str, Any], r480_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[list[str]], dict[str, float]]:
    pools = []
    stats = Counter()
    for i in range(dev["n"]):
        pool = unique_keep_order(
            dev["lr_top20"][i]
            + list(r480_rows[i]["cands"])[: int(cfg["r480_depth"])]
            + dev["r54pool"][i][: int(cfg["r54_depth"])]
            + dev["r84pool"][i][: int(cfg["r84_depth"])]
        )
        pools.append(pool)
        gt = dev["gt"][i]
        if gt in pool:
            stats["pool_hit"] += 1
        if gt in dev["lr_top20"][i]:
            stats["base_hit"] += 1
        if gt not in dev["lr_top20"][i] and gt in pool:
            stats["new_reachable"] += 1
    return pools, {
        "mean_pool_size": float(np.mean([len(p) for p in pools])),
        "pool_hit": stats["pool_hit"] / dev["n"],
        "base_hit": stats["base_hit"] / dev["n"],
        "new_reachable": stats["new_reachable"] / dev["n"],
    }


def run_config(
    dev: dict[str, Any],
    r480_rows: list[dict[str, Any]],
    played: list[list[str]],
    track_artist: dict[str, str],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    print(f"{ts()} {cfg['name']}: building pools/features", flush=True)
    pools, pool_stats = build_wide_pools(dev, r480_rows, cfg)
    X, y, starts, counts = build_matrix(dev, pools, r480_rows, played, track_artist)
    print(f"{ts()} {cfg['name']}: matrix {X.shape}, positives={int(y.sum())}, pool_hit={pool_stats['pool_hit']:.4f}", flush=True)
    scores, train_report = train_oof(dev, X, y, starts, counts)
    policies = []
    for name, rankings in [
        ("pure_model", model_rankings(pools, scores, dev["lr_top20"], keep_top1=False)),
        ("keep_top1_model", model_rankings(pools, scores, dev["lr_top20"], keep_top1=True)),
    ]:
        policies.append({"policy": name, **evaluate(dev, rankings)})
    for keep in [False, True]:
        for bw in BASE_WEIGHTS:
            rankings = blended_rankings(pools, scores, dev["lr_top20"], base_weight=bw, keep_top1=keep)
            policies.append({"policy": f"blend_bw{bw:g}_keep{int(keep)}", "base_weight": bw, "keep_top1": keep, **evaluate(dev, rankings)})
    policies.sort(key=lambda r: r["dNDCG_all"], reverse=True)
    best = policies[0]
    print(
        f"{ts()} {cfg['name']}: best {best['policy']} dNDCG={best['dNDCG_all']:.5f} "
        f"nDCG={best['nDCG@20']:.5f} churn80={best['churn_top1_per80']} overlap={best['overlap@20']}",
        flush=True,
    )
    del X, y, starts, counts, scores
    gc.collect()
    return {"config": cfg, "pool_stats": pool_stats, "train_report": train_report, "policies": policies[:16]}


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R518 wide semantic residual", flush=True)
    dev = load_dev()
    r480_rows = load_r480_rows()
    played, track_artist = load_payload_bits()
    base_metrics = evaluate(dev, dev["lr_top20"])
    results = [run_config(dev, r480_rows, played, track_artist, cfg) for cfg in CONFIGS]
    best_rows = []
    for result in results:
        for policy in result["policies"]:
            best_rows.append({"config_name": result["config"]["name"], **policy})
    best_rows.sort(key=lambda r: r["dNDCG_all"], reverse=True)
    best = best_rows[0]
    verdict = "GO" if best["dNDCG_all"] >= 0.010 and best["all_gates_pass"] else "NO_GO"
    out = {
        "experiment": "R518 wide semantic residual",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "r516_reference_json": str(_R516_OUT_JSON),
        "base_metrics": base_metrics,
        "best_overall": best_rows[:24],
        "config_results": results,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    lines = [
        "# R518 - Wide Semantic Residual",
        "",
        f"**Date:** {datetime.now():%Y-%m-%d}",
        f"**Verdict:** **{verdict}**",
        "",
        "## Best Offline Result",
        "",
        f"- config: `{best['config_name']}`",
        f"- policy: `{best['policy']}`",
        f"- nDCG@20: `{best['nDCG@20']:.6f}` vs base `{best['base_nDCG@20']:.6f}`",
        f"- dNDCG: `{best['dNDCG_all']:.6f}`",
        f"- same/diff/h7 delta: `{best['dNDCG_same']:.6f}` / `{best['dNDCG_diff']:.6f}` / `{best['dNDCG_h7']:.6f}`",
        f"- churn top1 per 80: `{best['churn_top1_per80']}`",
        f"- overlap@20: `{best['overlap@20']}`",
        "",
        "## Interpretation",
        "",
        "This tests whether the positive R516 semantic residual can exploit the extra reachability "
        "from R54/R84 retrieval candidates. A blind build is justified only if this clears about "
        "+0.010 all-dev dNDCG with sane churn.",
        "",
        f"Full JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"{ts()} wrote {OUT_JSON} and {OUT_MD}", flush=True)
    print(f"{ts()} verdict={verdict} best_config={best['config_name']} best_policy={best['policy']} dNDCG={best['dNDCG_all']:.6f}", flush=True)


if __name__ == "__main__":
    main()
