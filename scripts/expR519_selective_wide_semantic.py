#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201
"""R519: leakage-aware selector for the R518 wide semantic residual.

R518 nearly hits the primary dNDCG gate but has low overlap because it changes
too many top20 items.  This script reruns the best R518 config, then searches
GT-independent deployment rules.  The selector is chosen fold-held-out:
for each validation fold, pick the best rule on the other four folds, apply it
to the held-out fold, and evaluate the final all-dev rankings.
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_goal65_eval import evaluate, load_dev  # noqa: E402
from scripts.expR516_semantic_residual_reranker import (  # noqa: E402
    build_matrix,
    load_payload_bits,
    load_r480_rows,
    train_oof,
    unique_keep_order,
)

OUT_JSON = REPO / "exp/eval/expR519_selective_wide_semantic.json"
OUT_MD = REPO / "docs/r519_selective_wide_semantic.md"

CFG = {"name": "r48080_r54r84_100", "r480_depth": 80, "r54_depth": 100, "r84_depth": 100}
BASE_WEIGHTS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.5, 5.0]
TOP_K = 20


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_at_rank(rank: int) -> float:
    return 1.0 / math.log2(rank + 1) if 0 < rank <= TOP_K else 0.0


def rank_of(items: list[str], target: str) -> int:
    try:
        return items.index(target) + 1
    except ValueError:
        return -1


def build_wide_pools(dev: dict[str, Any], r480_rows: list[dict[str, Any]]) -> tuple[list[list[str]], dict[str, float]]:
    pools = []
    n_hit = n_base = n_new = 0
    for i in range(dev["n"]):
        pool = unique_keep_order(
            dev["lr_top20"][i]
            + list(r480_rows[i]["cands"])[: CFG["r480_depth"]]
            + dev["r54pool"][i][: CFG["r54_depth"]]
            + dev["r84pool"][i][: CFG["r84_depth"]]
        )
        pools.append(pool)
        gt = dev["gt"][i]
        n_hit += gt in pool
        n_base += gt in dev["lr_top20"][i]
        n_new += gt not in dev["lr_top20"][i] and gt in pool
    return pools, {
        "mean_pool_size": float(np.mean([len(p) for p in pools])),
        "pool_hit": n_hit / dev["n"],
        "base_hit": n_base / dev["n"],
        "new_reachable": n_new / dev["n"],
    }


def blended_case(pool: list[str], sc: np.ndarray, base: list[str], bw: float, keep_top1: bool) -> tuple[list[str], dict[str, float]]:
    br = {t: r for r, t in enumerate(base, 1)}
    vals = np.asarray([float(sc[j]) + (bw / br[tid] if tid in br else 0.0) for j, tid in enumerate(pool)], dtype=np.float32)
    order = np.argsort(-vals, kind="mergesort")
    ranked = [pool[int(j)] for j in order]
    if keep_top1 and base:
        ranked = [base[0]] + [t for t in ranked if t != base[0]]
    top_vals = vals[order[:25]]
    margin12 = float(top_vals[0] - top_vals[1]) if len(top_vals) > 1 else 0.0
    margin20 = float(top_vals[19] - top_vals[20]) if len(top_vals) > 20 else 0.0
    return ranked[:TOP_K], {"margin12": margin12, "margin20": margin20}


@dataclass(frozen=True)
class Rule:
    name: str
    fn: Callable[[dict[str, float]], bool]


def make_rules(feature_rows: list[dict[str, float]]) -> list[Rule]:
    rules: list[Rule] = [Rule("all", lambda f: True)]
    for ov in range(14, 21):
        rules.append(Rule(f"overlap_ge_{ov}", lambda f, ov=ov: f["overlap"] >= ov))
    for changed in range(0, 8):
        rules.append(Rule(f"changed_le_{changed}", lambda f, changed=changed: f["changed"] <= changed))
    rules.append(Rule("top1_same", lambda f: f["top1_changed"] == 0))
    rules.append(Rule("top1_same_overlap_ge16", lambda f: f["top1_changed"] == 0 and f["overlap"] >= 16))
    margins = np.asarray([f["margin12"] for f in feature_rows], dtype=np.float32)
    for q in [0.25, 0.4, 0.5, 0.6, 0.75]:
        thr = float(np.quantile(margins, q))
        rules.append(Rule(f"margin12_ge_q{q:g}", lambda f, thr=thr: f["margin12"] >= thr))
        rules.append(Rule(f"overlap_ge16_margin12_ge_q{q:g}", lambda f, thr=thr: f["overlap"] >= 16 and f["margin12"] >= thr))
    for npv in [0, 1, 3, 5, 7]:
        rules.append(Rule(f"nprior_ge_{npv}", lambda f, npv=npv: f["n_prior"] >= npv))
    for npv in [0, 1, 3, 5, 7]:
        rules.append(Rule(f"overlap_ge16_nprior_ge_{npv}", lambda f, npv=npv: f["overlap"] >= 16 and f["n_prior"] >= npv))
    return rules


def subset_metrics(dev: dict[str, Any], base: list[list[str]], cand: list[list[str]], select: list[bool], idx: list[int]) -> dict[str, float]:
    deltas = []
    churn = 0
    overlap = 0.0
    selected = 0
    for i in idx:
        use = select[i]
        ranking = cand[i] if use else base[i]
        gt = dev["gt"][i]
        deltas.append(ndcg_at_rank(rank_of(ranking, gt)) - ndcg_at_rank(rank_of(base[i], gt)))
        if use:
            selected += 1
        if ranking and base[i] and ranking[0] != base[i][0]:
            churn += 1
        overlap += len(set(ranking[:20]) & set(base[i][:20]))
    n = max(len(idx), 1)
    return {
        "dNDCG": float(np.mean(deltas)),
        "selected_rate": selected / n,
        "churn80": churn / n * 80.0,
        "overlap": overlap / n,
    }


def nested_select(dev: dict[str, Any], base: list[list[str]], cand: list[list[str]], feats: list[dict[str, float]], policy_name: str) -> dict[str, Any]:
    rules = make_rules(feats)
    selected = [False] * dev["n"]
    chosen = []
    folds = sorted(set(dev["fold"]))
    for fold in folds:
        train_idx = [i for i, f in enumerate(dev["fold"]) if f != fold]
        val_idx = [i for i, f in enumerate(dev["fold"]) if f == fold]
        scored = []
        for rule in rules:
            sel = [rule.fn(feats[i]) for i in range(dev["n"])]
            m = subset_metrics(dev, base, cand, sel, train_idx)
            # Prefer material gain, then overlap.  Require positive train gain.
            if m["dNDCG"] > 0:
                scored.append((m["dNDCG"], m["overlap"], -m["churn80"], rule, m))
        if not scored:
            rule = Rule("none", lambda f: False)
            train_m = subset_metrics(dev, base, cand, [False] * dev["n"], train_idx)
        else:
            scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            rule = scored[0][3]
            train_m = scored[0][4]
        for i in val_idx:
            selected[i] = rule.fn(feats[i])
        val_m = subset_metrics(dev, base, cand, selected, val_idx)
        chosen.append({"fold": int(fold), "rule": rule.name, "train": train_m, "val_partial": val_m})
    final_rankings = [cand[i] if selected[i] else base[i] for i in range(dev["n"])]
    final = evaluate(dev, final_rankings)
    return {
        "policy": policy_name,
        "selected_count": int(sum(selected)),
        "selected_rate": float(np.mean(selected)),
        "chosen_by_fold": chosen,
        "metrics": final,
    }


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R519 selective wide semantic", flush=True)
    dev = load_dev()
    r480_rows = load_r480_rows()
    played, track_artist = load_payload_bits()
    pools, pool_stats = build_wide_pools(dev, r480_rows)
    X, y, starts, counts = build_matrix(dev, pools, r480_rows, played, track_artist)
    print(f"{ts()} matrix {X.shape}, positives={int(y.sum())}, pool_hit={pool_stats['pool_hit']:.4f}", flush=True)
    scores, train_report = train_oof(dev, X, y, starts, counts)
    del X, y, starts, counts
    gc.collect()

    base = dev["lr_top20"]
    results = []
    for keep in [False, True]:
        for bw in BASE_WEIGHTS:
            cand = []
            feats = []
            for i, (pool, sc, b) in enumerate(zip(pools, scores, base, strict=True)):
                ranking, extra = blended_case(pool, sc, b, bw, keep)
                overlap = len(set(ranking[:20]) & set(b[:20]))
                feats.append({
                    "overlap": float(overlap),
                    "changed": float(20 - overlap),
                    "top1_changed": float(bool(ranking and b and ranking[0] != b[0])),
                    "n_prior": float(dev["n_prior"][i]),
                    **extra,
                })
                cand.append(ranking)
            raw = evaluate(dev, cand)
            sel = nested_select(dev, base, cand, feats, f"blend_bw{bw:g}_keep{int(keep)}")
            results.append({"raw_policy": {"policy": f"blend_bw{bw:g}_keep{int(keep)}", **raw}, "selective": sel})
            print(
                f"{ts()} bw={bw:g} keep={keep}: raw_d={raw['dNDCG_all']:.5f} "
                f"sel_d={sel['metrics']['dNDCG_all']:.5f} sel_n={sel['selected_count']} "
                f"ov={sel['metrics']['overlap@20']}",
                flush=True,
            )
    results.sort(key=lambda r: r["selective"]["metrics"]["dNDCG_all"], reverse=True)
    best = results[0]["selective"]
    best_m = best["metrics"]
    verdict = "GO" if best_m["dNDCG_all"] >= 0.010 and best_m["all_gates_pass"] else "NO_GO"
    out = {
        "experiment": "R519 selective wide semantic",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "config": CFG,
        "pool_stats": pool_stats,
        "train_report": train_report,
        "best_selective": best,
        "results": results[:12],
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    lines = [
        "# R519 - Selective Wide Semantic",
        "",
        f"**Date:** {datetime.now():%Y-%m-%d}",
        f"**Verdict:** **{verdict}**",
        "",
        "## Best Offline Result",
        "",
        f"- policy: `{best['policy']}`",
        f"- selected: `{best['selected_count']}` / `{dev['n']}` (`{best['selected_rate']:.3f}`)",
        f"- nDCG@20: `{best_m['nDCG@20']:.6f}` vs base `{best_m['base_nDCG@20']:.6f}`",
        f"- dNDCG: `{best_m['dNDCG_all']:.6f}`",
        f"- churn top1 per 80: `{best_m['churn_top1_per80']}`",
        f"- overlap@20: `{best_m['overlap@20']}`",
        "",
        "## Interpretation",
        "",
        "This searches simple GT-independent selectors on R518 rankings using fold-held-out rule choice. "
        "It is designed to recover R518's dNDCG while restoring overlap/churn safety.",
        "",
        f"Full JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"{ts()} wrote {OUT_JSON} and {OUT_MD}", flush=True)
    print(f"{ts()} verdict={verdict} best_policy={best['policy']} dNDCG={best_m['dNDCG_all']:.6f}", flush=True)


if __name__ == "__main__":
    main()
