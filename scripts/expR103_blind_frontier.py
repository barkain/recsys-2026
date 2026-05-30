#!/usr/bin/env python3
"""R103 blind threshold/edit frontier vs R92 p11 (GT-independent, no slot cost).

Loads exp/eval/expR103_blind_rows.json (per-blind-case base_list, arm-C list cC,
GT-independent signals) and characterizes, for the SWITCH and PROMOTE@k edits across
selection budgets, the churn (top-1 changes /80) and top-20 overlap vs the R92 p11
base. Answers: is there ANY gate-passing config (churn<=30 AND overlap>=16), and how
many rows does it patch? (We cannot measure blind nDCG — no GT — so this is purely the
deployment-stability frontier the user's pre-submit gate is about.)
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROWS = json.load(open(REPO / "exp/eval/expR103_blind_rows.json"))["rows"]
TOP_K = 20


def score(r):
    return r["c_top1_gte_cos"] * (1.0 + r["c_top1_diff_artist"] + 2.0 * r["c_top1_base_absent"])


def patch_switch(r):
    return r["cC"][:TOP_K]


def patch_promote(r, prank=2):
    base = list(r["base_list"][:TOP_K])
    c0 = r["cC"][0] if r["cC"] else None
    if c0 is None or c0 in set(base):
        return base
    base.insert(prank - 1, c0)
    return base[:TOP_K]


def evaluate(selected_sids, edit, prank=2):
    churn = 0
    overlaps = []
    n_changed = 0
    for r in ROWS:
        base = r["base_list"][:TOP_K]
        if r["sid"] in selected_sids and r["cC"]:
            cand = patch_switch(r) if edit == "switch" else patch_promote(r, prank)
            n_changed += 1
        else:
            cand = base
        if cand and base and cand[0] != base[0]:
            churn += 1
        overlaps.append(len(set(cand[:TOP_K]) & set(base[:TOP_K])))
    ov = sum(overlaps) / len(ROWS)
    return {"n_patched": n_changed, "churn80": churn, "overlap20": round(ov, 2),
            "pass": churn <= 30 and ov >= 16}


def main():
    ranked = sorted(ROWS, key=score, reverse=True)
    print(f"R103 blind frontier (80 cases) vs R92 p11 — gate: churn<=30 AND overlap>=16\n")
    for edit in ["switch", "promote"]:
        print(f"=== edit={edit}{' (prank=2)' if edit=='promote' else ''} ===")
        print(f"  {'budget N':>8} {'patched':>8} {'churn/80':>9} {'overlap':>8}  gate")
        for N in [5, 10, 15, 20, 25, 30, 40, 50, 60, 80]:
            sel = {r["sid"] for r in ranked[:N]}
            m = evaluate(sel, edit)
            print(f"  {N:>8} {m['n_patched']:>8} {m['churn80']:>9} {m['overlap20']:>8}  "
                  f"{'PASS' if m['pass'] else ''}")
        print()
    # the as-built R103a (threshold 0.6195) and full-switch reference
    thr_sel = {r["sid"] for r in ROWS if score(r) >= 0.6195}
    print(f"as-built R103a (thr 0.6195, switch): {evaluate(thr_sel, 'switch')}")
    print(f"promote@2 on same selection: {evaluate(thr_sel, 'promote')}")


if __name__ == "__main__":
    main()
