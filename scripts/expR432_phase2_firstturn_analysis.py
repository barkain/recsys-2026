#!/usr/bin/env python3
"""R432 Phase 2 — first-turn (n_prior=0) conversion analysis of the goal-aware integration.

Reads the expR103_integrate_eval per-case dump (run with --gte-oof = goal source) and reports
A/B/C nDCG on the FIRST-TURN slice (n_prior=0 = Blind-A shape) — the gate that matters — plus
all-dev and h7, with churn + same/diff-artist canaries.

Arms: A=base (no goal), B=goal in pool only, C=goal in pool + goal features (the deployable form).
Gate (R432): first-turn dC_vs_A meaningfully positive (>=+0.010) AND all-dev dC_vs_A>=0 AND
churn<=30/80 AND same/diff-artist Δ>=-0.005.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
PC = REPO / "exp/eval/expR432_integrate_w1_percase.json"


def stat(rows):
    if not rows:
        return None
    ndA = np.array([r["ndA"] for r in rows])
    ndB = np.array([r["ndB"] for r in rows])
    ndC = np.array([r["ndC"] for r in rows])
    # churn / overlap are A-vs-C (C is the deployable arm)
    top1_eq = np.array([r["top1_AC_eq"] for r in rows])
    overlap = np.array([r["overlap_AC"] for r in rows])
    Cin = np.array([r["C_in20"] for r in rows], dtype=bool)
    Ain = np.array([r["A_in20"] for r in rows], dtype=bool)
    return {
        "n": len(rows),
        "A": float(ndA.mean()), "B": float(ndB.mean()), "C": float(ndC.mean()),
        "dB_vs_A": float(ndB.mean() - ndA.mean()),
        "dC_vs_A": float(ndC.mean() - ndA.mean()),
        "dC_vs_B": float(ndC.mean() - ndB.mean()),
        "churn80": float((1 - top1_eq.mean()) * 80),
        "overlap20": float(overlap.mean()),
        "recovered": int(np.sum(Cin & ~Ain)),
        "lost": int(np.sum(~Cin & Ain)),
    }


def main():
    d = json.load(open(PC))
    rows = d["results"]
    n0 = [r for r in rows if r["n_prior_music"] == 0]
    h7 = [r for r in rows if r["n_prior_music"] == 7]
    same = [r for r in rows if r["same_artist"]]
    diff = [r for r in rows if not r["same_artist"]]
    n0_same = [r for r in n0 if r["same_artist"]]
    n0_diff = [r for r in n0 if not r["same_artist"]]

    print(f"R432 Phase 2 — goal-aware integration, first-turn focus  (n={len(rows)})\n")
    hdr = f"{'subset':16} {'n':>5} {'A':>7} {'C':>7} {'dC-A':>8} {'dB-A':>8} {'dC-B':>8} {'churn':>6} {'ovl':>6} {'rec/lost':>9}"
    print(hdr)
    for name, rs in [("ALL", rows), ("n0 FIRST-TURN", n0), ("h7", h7),
                     ("same_artist", same), ("diff_artist", diff),
                     ("n0 same", n0_same), ("n0 diff", n0_diff)]:
        s = stat(rs)
        if s:
            print(f"{name:16} {s['n']:5d} {s['A']:7.4f} {s['C']:7.4f} {s['dC_vs_A']:+8.4f} "
                  f"{s['dB_vs_A']:+8.4f} {s['dC_vs_B']:+8.4f} {s['churn80']:6.1f} {s['overlap20']:6.2f} "
                  f"{s['recovered']:4d}/{s['lost']:<4d}")

    s0 = stat(n0); sa = stat(rows); s_same = stat(same)
    g_ft = s0["dC_vs_A"] >= 0.010
    g_all = sa["dC_vs_A"] >= 0.0
    g_churn = s0["churn80"] <= 30
    # first-turn has no prior artist => same-artist canary is the ALL-dev one
    g_same = s_same["dC_vs_A"] >= -0.005
    print("\n=== R432 GATE (first-turn = Blind-A) ===")
    print(f"  first-turn dC_vs_A >= +0.010 : {g_ft}  ({s0['dC_vs_A']:+.4f})")
    print(f"  all-dev    dC_vs_A >=  0     : {g_all}  ({sa['dC_vs_A']:+.4f})")
    print(f"  first-turn churn   <= 30/80  : {g_churn}  ({s0['churn80']:.1f})  [hard-stop 35, overlap {s0['overlap20']:.2f}]")
    print(f"  same-artist Δ      >= -0.005 : {g_same}  ({s_same['dC_vs_A']:+.4f})")
    passed = g_ft and g_all and g_churn and g_same
    print(f"\n  VERDICT: {'PASS -> build Blind-A candidate' if passed else 'FAIL -> archive R432'}")

    out = REPO / "exp/eval/expR432_phase2_firstturn.json"
    json.dump({"subsets": {k: stat(v) for k, v in
                           [("all", rows), ("n0", n0), ("h7", h7),
                            ("same_artist", same), ("diff_artist", diff)]},
               "gate_pass": bool(passed)}, open(out, "w"), indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
