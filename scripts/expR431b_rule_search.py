#!/usr/bin/env python3
"""R431b rule search — find a GT-INDEPENDENT selective rule that keeps Arm-B wins.

Reads the per-case dump from expR431b_pool_only_selective.py. Baseline = Arm A
(production-equivalent). A selective deployment applies B's ordering on the cases where a
GT-independent predicate holds, and A's ordering everywhere else. We measure:
  ΔnDCG@20 all / h7, top-1 churn/80, mean overlap@20, same/diff-artist Δ.

Submit-worthiness target (spec): ΔnDCG ≥ +0.004 all OR ≥ +0.006 h7, with safe churn.

First we print the ORACLE ceiling (select where ndB>ndA — GT-dependent, NOT deployable) to
bound what any GT-independent rule could ever reach.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
PC = REPO / "exp/eval/expR431b_pool_only_percase.json"


def evaluate(rows, selected_mask):
    """selected_mask: bool array — True => use B, False => use A."""
    nd = np.array([(r["ndB"] if s else r["ndA"]) for r, s in zip(rows, selected_mask)])
    ndA = np.array([r["ndA"] for r in rows])
    h7 = np.array([r["n_prior_music"] == 7 for r in rows])
    same = np.array([bool(r["same_artist"]) for r in rows])
    # churn: a selected case churns top-1 iff B changed top-1
    churn_cases = sum(1 for r, s in zip(rows, selected_mask) if s and r["b_changed_top1"])
    churn_per80 = churn_cases / len(rows) * 80
    overlap = np.array([(r["overlap_AB"] if s else 20) for r, s in zip(rows, selected_mask)])
    return {
        "n_sel": int(np.sum(selected_mask)),
        "dAll": float(nd.mean() - ndA.mean()),
        "dH7": float(nd[h7].mean() - ndA[h7].mean()) if h7.any() else 0.0,
        "dSame": float(nd[same].mean() - ndA[same].mean()) if same.any() else 0.0,
        "dDiff": float(nd[~same].mean() - ndA[~same].mean()) if (~same).any() else 0.0,
        "churn80": churn_per80,
        "overlap20": float(overlap.mean()),
    }


def main():
    d = json.load(open(PC))
    rows = d["results"]
    n = len(rows)
    print(f"R431b rule search — {n} cases  (ucf_weight={d.get('ucf_weight')})\n")

    ndA = np.array([r["ndA"] for r in rows])
    ndB = np.array([r["ndB"] for r in rows])
    h7m = np.array([r["n_prior_music"] == 7 for r in rows])
    print(f"baseline  A all={ndA.mean():.4f}  h7={ndA[h7m].mean():.4f}")
    print(f"always-B  all={ndB.mean():.4f} (Δ{ndB.mean()-ndA.mean():+.4f})  "
          f"h7={ndB[h7m].mean():.4f} (Δ{ndB[h7m].mean()-ndA[h7m].mean():+.4f})")

    # --- ORACLE ceiling (GT-dependent, NOT deployable) ---
    for label, mask in [
        ("oracle ndB>ndA ", np.array([r["ndB"] > r["ndA"] for r in rows])),
        ("oracle ndB>=ndA", np.array([r["ndB"] >= r["ndA"] for r in rows])),
    ]:
        m = evaluate(rows, mask)
        print(f"  [{label}] n_sel={m['n_sel']:4d}  dAll={m['dAll']:+.4f}  dH7={m['dH7']:+.4f}  "
              f"churn={m['churn80']:.1f}/80")
    print()

    # --- GT-independent candidate rules ---
    rules = []
    rules.append(("always_B", lambda r: True))
    rules.append(("b_changed_any", lambda r: bool(r["b_changed_any"])))
    rules.append(("not_ucf_top1", lambda r: not r["b_top1_is_ucf"]))
    rules.append(("ucf_top1", lambda r: bool(r["b_top1_is_ucf"])))
    rules.append(("has_ucf_in_b20", lambda r: r["n_ucf_in_b20"] >= 1))
    rules.append(("no_ucf_in_b20", lambda r: r["n_ucf_in_b20"] == 0))
    for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        rules.append((f"max_ucf_cos>={thr}", (lambda r, t=thr: r["max_ucf_cos_b20"] >= t)))
        rules.append((f"ucf_top1_cos>={thr}", (lambda r, t=thr: r["b_top1_is_ucf"] and r["b_top1_ucf_cos"] >= t)))
    for thr in (5, 10, 20, 50):
        rules.append((f"best_ucf_rank<={thr}", (lambda r, t=thr: 0 < r["best_ucf_rank_b20"] <= t)))
    for thr in (0.05, 0.1, 0.2, 0.5, 1.0):
        rules.append((f"a_margin<={thr}", (lambda r, t=thr: r["a_margin"] <= t)))
    # combo: low production confidence AND user-cf surfaced a strong candidate
    for mt in (0.50, 0.60):
        for am in (0.2, 0.5):
            rules.append((f"cos>={mt}&margin<={am}",
                          (lambda r, m=mt, a=am: r["max_ucf_cos_b20"] >= m and r["a_margin"] <= a)))

    evald = []
    for name, fn in rules:
        mask = np.array([bool(fn(r)) for r in rows])
        if mask.sum() == 0:
            continue
        m = evaluate(rows, mask)
        m["rule"] = name
        evald.append(m)

    def safe(m):
        return m["churn80"] <= 30 and m["dSame"] >= -0.005 and m["overlap20"] >= 16

    print("=== rules sorted by dAll (safe-churn marked *) ===")
    print(f"{'rule':28} {'n_sel':>6} {'dAll':>9} {'dH7':>9} {'dSame':>8} {'dDiff':>8} "
          f"{'churn':>6} {'ovl':>6} safe")
    for m in sorted(evald, key=lambda x: -x["dAll"])[:18]:
        print(f"{m['rule']:28} {m['n_sel']:6d} {m['dAll']:+9.4f} {m['dH7']:+9.4f} "
              f"{m['dSame']:+8.4f} {m['dDiff']:+8.4f} {m['churn80']:6.1f} {m['overlap20']:6.2f} "
              f"{'*' if safe(m) else ''}")
    print("\n=== rules sorted by dH7 ===")
    for m in sorted(evald, key=lambda x: -x["dH7"])[:12]:
        print(f"{m['rule']:28} {m['n_sel']:6d} {m['dAll']:+9.4f} {m['dH7']:+9.4f} "
              f"{m['dSame']:+8.4f} {m['dDiff']:+8.4f} {m['churn80']:6.1f} {m['overlap20']:6.2f} "
              f"{'*' if safe(m) else ''}")

    # --- verdict ---
    passers = [m for m in evald if safe(m) and (m["dAll"] >= 0.004 or m["dH7"] >= 0.006)]
    print("\n=== VERDICT ===")
    if passers:
        best = max(passers, key=lambda x: max(x["dAll"], x["dH7"]))
        print(f"  PASS: rule '{best['rule']}' dAll={best['dAll']:+.4f} dH7={best['dH7']:+.4f} "
              f"churn={best['churn80']:.1f} -> build blind candidate")
    else:
        print("  FAIL: no GT-independent rule reaches +0.004 all or +0.006 h7 with safe churn.")
        print("  -> ARCHIVE R431b. Production stays R106 A-clean. Public user-cf fully closed.")

    out = REPO / "exp/eval/expR431b_rule_search.json"
    json.dump({"baseline_A_all": float(ndA.mean()), "rules": evald,
               "pass": bool(passers)}, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
