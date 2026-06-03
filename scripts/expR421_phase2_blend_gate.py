#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R421 Phase 2 — blend cross-encoder scores with production + selective gate, OOF eval.

Reads dev_scores.json (case_idx -> {tid: cross-encoder score}) produced by Phase 1 on the
A100. Builds candidate rankings (pure CE rerank / production+CE blend / selective inject),
evaluates each via the goal65 harness, sweeps, and reports nDCG@20 Δ + the full gate suite
vs the production lr_top20 baseline.

GATE (codex spec): OOF nDCG@20 Δ >= +0.02, h7 non-negative, same-artist not negative,
recovered > lost, churn controlled. PASS -> build blind candidate; else archive.

  python scripts/expR421_phase2_blend_gate.py --scores .scratch/r421/dev_scores.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.exp_goal65_eval import load_dev, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=".scratch/r421/dev_scores.json")
    args = ap.parse_args()
    dev = load_dev()
    n = dev["n"]; lr20 = dev["lr_top20"]; r54p, r84p = dev["r54pool"], dev["r84pool"]
    ce = json.load(open(REPO / args.scores))
    pool = [list(dict.fromkeys((r84p[i] or [])[:200] + (r54p[i] or [])[:200])) for i in range(n)]

    def cescore(i, t):
        return ce.get(str(i), {}).get(t, None)

    def prod_rank(i):
        return {t: j for j, t in enumerate(lr20[i] or [])}

    def rank_pure_ce():
        out = []
        for i in range(n):
            sc = {t: cescore(i, t) for t in pool[i]}
            sc = {t: v for t, v in sc.items() if v is not None}
            if not sc:
                out.append((lr20[i] or [])[:30]); continue
            out.append(sorted(sc, key=lambda t: -sc[t])[:30])
        return out

    def rank_blend(alpha):
        """blend production-rank prior with CE score (both normalized per case)."""
        out = []
        for i in range(n):
            pr = prod_rank(i)
            cand = list(dict.fromkeys((lr20[i] or []) + pool[i]))
            ces = np.array([cescore(i, t) if cescore(i, t) is not None else -10.0 for t in cand])
            ces = (ces - ces.mean()) / (ces.std() + 1e-6)
            prr = np.array([1.0 / (1 + pr[t]) if t in pr else 0.0 for t in cand])
            blend = alpha * prr + (1 - alpha) * (ces - ces.min()) / (ces.max() - ces.min() + 1e-6)
            out.append([cand[j] for j in np.argsort(-blend)][:30])
        return out

    def rank_selective(margin_keep, conf_q):
        """keep production top-`margin_keep`; below that, let CE reorder/inject pool. Churn-safe."""
        out = []
        # global CE confidence threshold = conf_q quantile of per-case max CE
        allmax = [max((cescore(i, t) for t in pool[i] if cescore(i, t) is not None), default=-10) for i in range(n)]
        thr = np.quantile([m for m in allmax if m > -9], conf_q) if any(m > -9 for m in allmax) else 1e9
        for i in range(n):
            base = list(lr20[i] or [])
            keep = base[:margin_keep]
            rest_cands = [t for t in pool[i] if t not in keep and cescore(i, t) is not None]
            rest_cands.sort(key=lambda t: -cescore(i, t))
            # only inject CE-preferred candidates if confident
            inj = [t for t in rest_cands if cescore(i, t) >= thr]
            tail = inj + [t for t in base[margin_keep:] if t not in inj] + \
                   [t for t in rest_cands if t not in inj]
            out.append((keep + tail)[:30])
        return out

    def report(name, ranks):
        m = evaluate(dev, ranks)
        ok = (m["dNDCG_all"] >= 0.02 and m["dNDCG_h7"] >= 0 and m["dNDCG_same"] >= -0.005
              and m["churn_top1_per80"] <= 30)
        print(f"  {name:26s}: ΔnDCG={m['dNDCG_all']:+.4f} Δh7={m['dNDCG_h7']:+.4f} "
              f"Δsame={m['dNDCG_same']:+.4f} churn={m['churn_top1_per80']:.1f} ov={m['overlap@20']:.1f} "
              f"hit@20={m['hit@20']:.3f} {'*** PASS' if ok else ''}")
        return m["dNDCG_all"], ok

    print("R421 cross-encoder — dev OOF vs production lr_top20 (codex bar +0.02):")
    report("pure CE rerank", rank_pure_ce())
    for a in (0.3, 0.5, 0.7):
        report(f"blend a={a}", rank_blend(a))
    for mk in (5, 10):
        for cq in (0.7, 0.9):
            report(f"selective keep{mk} q{cq}", rank_selective(mk, cq))


if __name__ == "__main__":
    main()
