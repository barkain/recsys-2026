#!/usr/bin/env python3
"""R103 selective-deployment evaluator (shared harness).

Codex constraint: the DEPLOYED object is NOT a GTE-feature LR rerank (that churns
~40/80 by construction — adding GTE feature columns reshuffles LightGBM splits
globally, independent of GTE pool weight). Instead:

    Base   = production ranking (arm A = base_r84 OOF, the established OOF prod proxy)
    Patch  = local edits on a GT-INDEPENDENT-selected subset of rows only

Two local-edit types (codex's allowed set):
  - "switch"  : on selected rows, adopt arm C's (GTE-aug LR) full ranking.
  - "promote" : on selected rows, insert arm C's top-1 candidate (topC[0]) into the
                base list at rank `prank` (2..5). Promoting to rank>=2 NEVER changes
                top-1 -> ~zero top-1 churn, overlap 19/20. This is the churn-efficient
                R84c-style edit.

The SELECTOR must be GT-independent (blind-available). Allowed fields:
  c_top1_gte_present, c_top1_gte_cos, c_top3_gte, a_margin, c_top1_diff_artist,
  c_top1_base_absent, n_prior_music, topA/topC (ids/order).
NEVER select on: gt, rA/rC, ndA/ndC, *_in20, gte_unique, same_artist, top1_AC_eq,
overlap_AC (all GT-derived) — those are for EVALUATION only.

Hard gates (codex):
  top-1 churn <= 35/80 ; top-20 overlap >= 14 ; same-artist Δ >= -0.005 ;
  diff-artist Δ > 0 ; (preferred) all Δ >= +0.006 ; (preferred) h7 Δ >= 0.

Usage:
  from expR103_selective_eval import load, evaluate
  recs = load()
  m = evaluate(recs, lambda r: r["c_top1_gte_present"]==1 and r["c_top1_base_absent"]==1,
               edit="promote", prank=2)
  print(m)            # dAll/dH7/dSame/dDiff, churn80, overlap20, n_changed, gates...

CLI self-test runs a baseline panel of rules and prints the table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DUMP = REPO / "exp" / "eval" / "expR103_dump_percase.json"

# GT-derived fields a selector must never read (defensive guard for agent rules)
GT_FIELDS = {"gt", "rA", "rB", "rC", "ndA", "ndB", "ndC", "A_in20", "B_in20",
             "C_in20", "gte_unique", "same_artist", "top1_AC_eq", "overlap_AC"}


def ndcg(rank: int) -> float:
    """1-based rank in the displayed list; relevant iff 1<=rank<=20."""
    return 1.0 / math.log2(rank + 1) if (rank is not None and 0 < rank <= 20) else 0.0


def load(path: Path = DUMP) -> list[dict]:
    d = json.load(open(path))
    return d["results"]


def _promote(r: dict, prank: int):
    """Insert topC[0] into base list at rank `prank`. Returns (nd, overlap, churn)."""
    topA = r.get("topA") or []
    c0 = (r.get("topC") or [None])[0]
    gt = r.get("gt")
    rA = r.get("rA", -1)
    if c0 is None or c0 in set(topA):          # nothing genuinely new to promote
        return r["ndA"], 20, 0
    if c0 == gt:                               # GTE surfaced the GT -> now at rank prank
        nd = ndcg(prank)
    elif rA is not None and 0 < rA <= 20:      # GT already shown; one insert may push it down 1
        nd = ndcg(rA + 1 if rA >= prank else rA)
    else:                                      # GT not in base top-20 and c0 != gt
        nd = 0.0
    churn = 1 if (prank == 1 and (not topA or c0 != topA[0])) else 0
    overlap = 19 if prank <= 20 else 20
    return nd, overlap, churn


def evaluate(recs: list[dict], select_fn, edit: str = "promote", prank: int = 2) -> dict:
    """Apply `select_fn` (GT-independent) + edit to base (arm A); return metrics vs base."""
    rows = []
    n_changed = 0
    churn_sum = 0
    overlap_sum = 0.0
    for r in recs:
        sel = bool(select_fn(r))
        if not sel:
            nd, ov, ch = r["ndA"], 20, 0
        else:
            n_changed += 1
            if edit == "switch":
                nd, ov, ch = r["ndC"], r["overlap_AC"], (1 - r["top1_AC_eq"])
            elif edit == "promote":
                nd, ov, ch = _promote(r, prank)
            else:
                raise ValueError(edit)
        churn_sum += ch
        overlap_sum += ov
        rows.append((r, nd))
    n = len(recs)

    def delta(subset):
        sub = [(r, nd) for r, nd in rows if subset(r)]
        if not sub:
            return 0.0, 0
        base = sum(r["ndA"] for r, _ in sub) / len(sub)
        new = sum(nd for _, nd in sub) / len(sub)
        return new - base, len(sub)

    dAll, _ = delta(lambda r: True)
    dH7, nH7 = delta(lambda r: r["n_prior_music"] == 7)
    dSame, _ = delta(lambda r: r["same_artist"])
    dDiff, _ = delta(lambda r: not r["same_artist"])
    churn80 = churn_sum / n * 80.0
    overlap20 = overlap_sum / n

    gates = {
        "churn_le_35": churn80 <= 35.0,
        "overlap_ge_14": overlap20 >= 14.0,
        "same_ge_-005": dSame >= -0.005,
        "diff_positive": dDiff > 0.0,
        "all_ge_006_pref": dAll >= 0.006,
        "h7_nonneg_pref": dH7 >= 0.0,
    }
    hard_pass = (gates["churn_le_35"] and gates["overlap_ge_14"]
                 and gates["same_ge_-005"] and gates["diff_positive"])
    return {
        "edit": edit, "prank": prank, "n_changed": n_changed,
        "dAll": round(dAll, 5), "dH7": round(dH7, 5),
        "dSame": round(dSame, 5), "dDiff": round(dDiff, 5),
        "churn80": round(churn80, 2), "overlap20": round(overlap20, 3),
        "gates": gates, "hard_pass": hard_pass,
    }


def _fmt(name, m):
    g = m["gates"]
    flags = "".join(k for k, v in [("C", g["churn_le_35"]), ("O", g["overlap_ge_14"]),
                                    ("S", g["same_ge_-005"]), ("D", g["diff_positive"])] if v)
    return (f"{name:34} chg={m['n_changed']:4d} dAll={m['dAll']:+.4f} dH7={m['dH7']:+.4f} "
            f"dDiff={m['dDiff']:+.4f} dSame={m['dSame']:+.4f} churn={m['churn80']:5.1f} "
            f"ovlp={m['overlap20']:5.2f} {'HARD-PASS' if m['hard_pass'] else flags}")


def main():
    recs = load()
    print(f"loaded {len(recs)} cases from {DUMP.name}\n")
    panel = [
        ("switch: all rows (= full rerank)", (lambda r: True, "switch", 2)),
        ("promote2: gte_present & base_absent", (lambda r: r["c_top1_gte_present"] == 1 and r.get("c_top1_base_absent", 0) == 1, "promote", 2)),
        ("promote2: gte_present", (lambda r: r["c_top1_gte_present"] == 1, "promote", 2)),
        ("promote2: diff_artist & gte_present", (lambda r: r.get("c_top1_diff_artist", 0) == 1 and r["c_top1_gte_present"] == 1, "promote", 2)),
        ("promote2: gte_cos>0.6 & base_absent", (lambda r: r["c_top1_gte_cos"] > 0.6 and r.get("c_top1_base_absent", 0) == 1, "promote", 2)),
        ("switch: low margin & gte_present", (lambda r: r["a_margin"] < 0.5 and r["c_top1_gte_present"] == 1, "switch", 2)),
    ]
    for name, (fn, edit, pr) in panel:
        print(_fmt(name, evaluate(recs, fn, edit=edit, prank=pr)))


if __name__ == "__main__":
    main()
