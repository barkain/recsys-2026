#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R420 — codex's selective popularity gate, run faithfully vs the +0.02 nDCG bar.

Unlike R108 (full rerank) and R109 (LR feature), this keeps the production lr_top20 ranking
and ONLY selectively promotes a high-popularity in-pool candidate when production's own top-1
pick is much LESS popular than an available pooled candidate (margin-free proxy for "low
production confidence"). Recoverable subset = GT in top-300 pool but outside production top-20.
Sweeps the popularity-gap threshold, retrieval-rank cap, insert position, and max promotions;
reports OOF nDCG Δ + churn/canary gates. Popularity leak-free (train folds only).

PASS = OOF nDCG lift >= +0.02 with gates -> build blind candidate. Else popularity selection
is definitively closed and the +0.12 nDCG gap to vkost needs a stronger retriever, not this.
"""
from __future__ import annotations
import json, pickle, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.exp_goal65_eval import load_dev, evaluate


def main():
    dev = load_dev()
    gt, fold, lr20 = dev["gt"], dev["fold"], dev["lr_top20"]
    r54p, r84p = dev["r54pool"], dev["r84pool"]
    n = dev["n"]
    cases = pickle.load(open(REPO / "exp/eval/_R12_all_turns_payload.pkl", "rb"))["cases"]

    pool300 = []
    for i in range(n):
        p = list(dict.fromkeys((r84p[i] or [])[:300] + (r54p[i] or [])[:300]))
        pool300.append(p)
    rank_in = []  # best retrieval rank per track
    for i in range(n):
        rr = {}
        for src in (r84p[i] or [], r54p[i] or []):
            for j, t in enumerate(src):
                rr[t] = min(rr.get(t, 1e9), j)
        rank_in.append(rr)

    # leak-free popularity per fold
    popf = {}
    for f in sorted(set(fold)):
        tr = [i for i in range(n) if fold[i] != f]
        c = Counter()
        for i in tr:
            for t in cases[i].get("music_turns", []):
                c[t] += 1
        popf[f] = c

    # --- recovery ceiling on the recoverable subset ---
    miss_recoverable = [i for i in range(n) if gt[i] in set(pool300[i]) and gt[i] not in (lr20[i] or [])]
    print(f"recoverable misses (GT in top-300, outside production top-20): {len(miss_recoverable)}")
    rec_top1 = rec_top3 = 0
    for i in miss_recoverable:
        pop = popf[fold[i]]
        played = set(cases[i].get("music_turns", []))
        cands = [t for t in pool300[i] if t not in (lr20[i] or []) and t not in played]
        cands.sort(key=lambda t: -pop.get(t, 0))
        if cands:
            if cands[0] == gt[i]: rec_top1 += 1
            if gt[i] in cands[:3]: rec_top3 += 1
    print(f"  most-popular out-of-top20 in-pool candidate IS the GT: {rec_top1}/{len(miss_recoverable)} ({100*rec_top1/max(1,len(miss_recoverable)):.1f}%)")
    print(f"  GT among 3 most-popular such candidates:                {rec_top3}/{len(miss_recoverable)} ({100*rec_top3/max(1,len(miss_recoverable)):.1f}%)\n")

    def build(gap_mult, rank_cap, insert_pos, max_promo):
        rankings = []
        for i in range(n):
            base = list(lr20[i] or [])
            tail = [t for t in pool300[i] if t not in base]
            full = base + tail
            pop = popf[fold[i]]
            played = set(cases[i].get("music_turns", []))
            top1_pop = pop.get(base[0], 0) if base else 0
            cands = [t for t in pool300[i]
                     if t not in base and t not in played
                     and rank_in[i].get(t, 1e9) <= rank_cap
                     and pop.get(t, 0) >= gap_mult * max(top1_pop, 1)]
            cands.sort(key=lambda t: -pop.get(t, 0))
            for t in cands[:max_promo]:
                if t in full:
                    full.remove(t)
                full.insert(min(insert_pos, len(full)), t)
            rankings.append(full[:30])
        return rankings

    print("sweep: gap_mult x rank_cap x insert_pos x max_promo -> OOF nDCG Δ (gate-pass?)")
    best = None
    for gap_mult in (1.5, 2.0, 3.0):
        for rank_cap in (50, 100):
            for insert_pos in (0, 3):
                for max_promo in (1, 2):
                    rk = build(gap_mult, rank_cap, insert_pos, max_promo)
                    m = evaluate(dev, rk)
                    tag = f"gap{gap_mult} rcap{rank_cap} pos{insert_pos} m{max_promo}"
                    passed = m["gates"]["same_artist_ge_-005"] and m["gates"]["diff_artist_ge_-005"] and m["gates"]["churn_top1_le_30/80"]
                    print(f"  {tag:34s}: ΔnDCG={m['dNDCG_all']:+.4f} Δsame={m['dNDCG_same']:+.4f} churn={m['churn_top1_per80']:.1f} ov={m['overlap@20']:.1f} {'GATES-OK' if passed else ''}")
                    if best is None or m["dNDCG_all"] > best[1]:
                        best = (tag, m["dNDCG_all"], m)
    print(f"\nBEST: {best[0]}  ΔnDCG={best[1]:+.4f}  (codex bar = +0.02)")
    print("VERDICT:", "PASS -> build blind candidate" if best[1] >= 0.02 else
          "FAIL -> popularity selection capped below +0.02; +0.12 gap needs a stronger retriever")
    json.dump({"best_tag": best[0], "best_dNDCG": best[1], "n_recoverable": len(miss_recoverable),
               "rec_top1": rec_top1, "rec_top3": rec_top3},
              open(REPO / "exp/eval/expR420_selective_popularity.json", "w"), indent=2)


if __name__ == "__main__":
    main()
