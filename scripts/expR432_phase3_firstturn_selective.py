#!/usr/bin/env python3
"""R432 Phase 3 — first-turn selective deployment of the goal-aware ranker (arm C).

Phase 2: first-turn (n_prior=0) dC_vs_A = +0.0115 (PASS >+0.010) but churn 32.6/80 (caution;
hard-stop 35, overlap 14.08). This searches a GT-INDEPENDENT selective rule: deploy arm-C
ordering on the first-turn cases where the rule holds, else keep production (arm A). Goal: keep
ΔnDCG@20 >= +0.010 while cutting top-1 churn under ~25/80 and overlap >= 14.

All rule signals are computable at blind inference (no GT). Baseline = arm A on first-turn.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
PC = REPO / "exp/eval/expR432_integrate_w1_percase.json"


def evaluate(rows, mask):
    nd = np.array([(r["ndC"] if m else r["ndA"]) for r, m in zip(rows, mask)])
    ndA = np.array([r["ndA"] for r in rows])
    # churn: selected case churns top-1 iff C changed it (top1_AC_eq==0)
    churn = sum(1 for r, m in zip(rows, mask) if m and not r["top1_AC_eq"]) / len(rows) * 80
    overlap = np.mean([(r["overlap_AC"] if m else 20) for r, m in zip(rows, mask)])
    Cin = np.array([r["C_in20"] for r in rows], dtype=bool)
    Ain = np.array([r["A_in20"] for r in rows], dtype=bool)
    sel = np.array(mask, dtype=bool)
    rec = int(np.sum(sel & Cin & ~Ain))
    lost = int(np.sum(sel & ~Cin & Ain))
    return {"n_sel": int(sel.sum()), "dNDCG": float(nd.mean() - ndA.mean()),
            "churn80": float(churn), "overlap20": float(overlap),
            "rec": rec, "lost": lost}


def main():
    d = json.load(open(PC))
    rows = [r for r in d["results"] if r["n_prior_music"] == 0]
    n = len(rows)
    print(f"R432 Phase 3 — first-turn selective (n={n})\n")
    ndA = np.array([r["ndA"] for r in rows]); ndC = np.array([r["ndC"] for r in rows])
    print(f"baseline A = {ndA.mean():.4f}")
    print(f"always-C : dNDCG={ndC.mean()-ndA.mean():+.4f}  churn={evaluate(rows,[True]*n)['churn80']:.1f}  ovl={evaluate(rows,[True]*n)['overlap20']:.2f}")
    orc = evaluate(rows, [r["ndC"] >= r["ndA"] for r in rows])
    print(f"oracle ndC>=ndA: n_sel={orc['n_sel']} dNDCG={orc['dNDCG']:+.4f} churn={orc['churn80']:.1f}\n")

    # GT-independent rules: revert risky goal injections to production
    rules = []
    rules.append(("always_C", lambda r: True))
    # keep C only when its top-1 was already in the base top-20 (no risky injection)
    rules.append(("top1_in_base", lambda r: not r["c_top1_base_absent"]))
    # keep C unless it injected a LOW-confidence goal candidate at top-1
    for thr in (0.55, 0.60, 0.65, 0.70, 0.75):
        rules.append((f"keep_unless_inject_lowcos<{thr}",
                      lambda r, t=thr: not (r["c_top1_base_absent"] and r["c_top1_gte_cos"] < t)))
    # keep C only when its top-1 is a confident goal candidate OR unchanged
    for thr in (0.60, 0.65, 0.70):
        rules.append((f"C_if_gtecos>={thr}_or_eqtop1",
                      lambda r, t=thr: r["top1_AC_eq"] or r["c_top1_gte_cos"] >= t))
    # keep C only when production was low-confidence (a_margin small) — let goal help weak cases
    for thr in (0.1, 0.2, 0.3, 0.5):
        rules.append((f"a_margin<={thr}", lambda r, t=thr: r["a_margin"] <= t))
    # combos
    for cos in (0.60, 0.65):
        for am in (0.3, 0.5):
            rules.append((f"gtecos>={cos}|margin<={am}|eqtop1",
                          lambda r, c=cos, a=am: r["top1_AC_eq"] or r["c_top1_gte_cos"] >= c or r["a_margin"] <= a))

    res = []
    for name, fn in rules:
        mask = [bool(fn(r)) for r in rows]
        if sum(mask) == 0:
            continue
        m = evaluate(rows, mask); m["rule"] = name
        res.append(m)

    def ok(m):
        return m["dNDCG"] >= 0.010 and m["churn80"] <= 25 and m["overlap20"] >= 14

    print(f"{'rule':36} {'n_sel':>5} {'dNDCG':>8} {'churn':>6} {'ovl':>6} {'rec/lost':>9} ok")
    for m in sorted(res, key=lambda x: (x["churn80"] - 100*(x["dNDCG"] >= 0.010))):
        print(f"{m['rule']:36} {m['n_sel']:5d} {m['dNDCG']:+8.4f} {m['churn80']:6.1f} "
              f"{m['overlap20']:6.2f} {m['rec']:3d}/{m['lost']:<3d} {'*' if ok(m) else ''}")

    passers = [m for m in res if ok(m)]
    print("\n=== VERDICT ===")
    if passers:
        best = min(passers, key=lambda x: x["churn80"])
        print(f"  PASS: '{best['rule']}' dNDCG={best['dNDCG']:+.4f} churn={best['churn80']:.1f} "
              f"ovl={best['overlap20']:.2f} -> build Blind-A candidate")
    else:
        # fall back: is always-C within the established hard-stop (churn<=35, ovl>=14)?
        ac = evaluate(rows, [True] * n)
        within = ac["dNDCG"] >= 0.010 and ac["churn80"] <= 35 and ac["overlap20"] >= 14
        print(f"  No rule hits churn<=25 with dNDCG>=+0.010.")
        print(f"  always-C within established hard-stop (churn<=35,ovl>=14,dNDCG>=+0.010): {within} "
              f"(churn {ac['churn80']:.1f}, ovl {ac['overlap20']:.2f}, dNDCG {ac['dNDCG']:+.4f})")
    json.dump({"rules": res}, open(REPO / "exp/eval/expR432_phase3_firstturn_selective.json", "w"), indent=2)


if __name__ == "__main__":
    main()
