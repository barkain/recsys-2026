#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R421 Phase 2b — CE score as a FEATURE in a ranker (not a blend/reranker).

The naive blend fails because the CE is worse-than-production overall. Per our own lesson
(feedback_rerank_only_closed) the right deployment is CE-as-feature alongside the production
rank, so a learned ranker stays close to production and uses the CE only where it adds. Per-
candidate LambdaRank over the pool with features [production lr_top20 rank, retrieval ranks,
CE score], 5-fold OOF, vs production lr_top20. WITH vs WITHOUT the CE feature isolates its
marginal value. Gate: ΔnDCG ≥ +0.02, h7 ≥ 0, same-artist ≥ −0.005, churn ≤ 30/80.

  python scripts/expR421_phase2b_ce_feature.py --scores .scratch/r421/dev_scores_v2ep0.json
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
    ap.add_argument("--scores", default=".scratch/r421/dev_scores_v2ep0.json")
    args = ap.parse_args()
    import lightgbm as lgb
    dev = load_dev()
    gt, fold, lr20 = dev["gt"], dev["fold"], dev["lr_top20"]
    r54p, r84p = dev["r54pool"], dev["r84pool"]
    n = dev["n"]
    ce = json.load(open(REPO / args.scores))
    pool = [list(dict.fromkeys((r84p[i] or [])[:200] + (r54p[i] or [])[:200])) for i in range(n)]
    r84r = [{t: j for j, t in enumerate(r84p[i] or [])} for i in range(n)]
    r54r = [{t: j for j, t in enumerate(r54p[i] or [])} for i in range(n)]
    prr = [{t: j for j, t in enumerate(lr20[i] or [])} for i in range(n)]

    def feats(i, use_ce):
        sc = ce.get(str(i), {})
        cev = np.array([sc.get(t, np.nan) for t in pool[i]])
        m = np.nanmean(cev) if np.isfinite(cev).any() else 0.0
        s = np.nanstd(cev) if np.isfinite(cev).any() else 1.0
        rows = []
        for k, t in enumerate(pool[i]):
            pr = prr[i].get(t)
            base = [1.0/(1+pr) if pr is not None else 0.0, 1.0 if pr is not None else 0.0,
                    1.0/(1+r84r[i].get(t, 999)), 1.0/(1+r54r[i].get(t, 999)),
                    min(r84r[i].get(t, 999), r54r[i].get(t, 999))]
            if use_ce:
                z = (sc.get(t, m) - m) / (s + 1e-6)
                base += [z]
            rows.append(base)
        return np.array(rows, np.float32)

    def run(use_ce):
        rankings = [None]*n
        for f in sorted(set(fold)):
            tr = [i for i in range(n) if fold[i] != f and pool[i] and gt[i] in pool[i]]
            te = [i for i in range(n) if fold[i] == f]
            X, y, grp = [], [], []
            for i in tr:
                ff = feats(i, use_ce); X.append(ff)
                y.extend([1 if t == gt[i] else 0 for t in pool[i]]); grp.append(len(pool[i]))
            X = np.vstack(X)
            ds = lgb.Dataset(X, label=np.array(y), group=grp)
            model = lgb.train(dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[20],
                                   learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                                   lambda_l2=1.0, verbose=-1), ds, num_boost_round=300)
            for i in te:
                if not pool[i]:
                    rankings[i] = (lr20[i] or [])[:20]; continue
                s = model.predict(feats(i, use_ce))
                rankings[i] = [pool[i][j] for j in np.argsort(-s)][:30]
        return evaluate(dev, rankings)

    mb = run(False); mp = run(True)
    print(f"R421 CE-as-FEATURE (dev OOF vs production lr_top20=0.3159):")
    for name, m in [("base LR (prod+retrieval)", mb), ("+CE feature", mp)]:
        ok = (m["dNDCG_all"] >= 0.02 and m["dNDCG_h7"] >= 0 and m["dNDCG_same"] >= -0.005 and m["churn_top1_per80"] <= 30)
        print(f"  {name:26s}: nDCG={m['nDCG@20']:.4f} ΔnDCG={m['dNDCG_all']:+.4f} Δh7={m['dNDCG_h7']:+.4f} "
              f"Δsame={m['dNDCG_same']:+.4f} churn={m['churn_top1_per80']:.1f} {'*** PASS' if ok else ''}")
    print(f"  MARGINAL VALUE OF CE FEATURE: ΔnDCG {mp['nDCG@20']-mb['nDCG@20']:+.4f}")


if __name__ == "__main__":
    main()
