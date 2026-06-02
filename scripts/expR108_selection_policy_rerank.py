#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R108 — selection-policy reranker (does the popularity bias convert to dev OOF nDCG?).

Phase-0 found the Gemini GT is popularity-biased (GT pool-percentile 0.916; on production
MISSES pop(GT)>pop(prod top-1 FP) 64.5%), but popularity ALONE recovers GT#1 only 2.1% —
it's a tilt feature, not a ranker. This builds a LEARNED reranker over the production
candidate pool combining popularity + retrieval-rank + session/user CF + production
signals, trained per-fold OOF (popularity computed leak-free from TRAIN folds only),
and measures dev OOF nDCG@20 Δ + the full gate suite (same/diff-artist canary, churn,
overlap) vs the production lr_top20 baseline. PASS gates -> selective candidate; FAIL ->
popularity is absorbed/non-convertible, archive.
"""
from __future__ import annotations
import pickle, sys
from collections import Counter
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.exp_goal65_eval import load_dev, evaluate


def main():
    import lightgbm as lgb
    from datasets import load_dataset
    dev = load_dev()
    gt, fold, lr20 = dev["gt"], dev["fold"], dev["lr_top20"]
    r54p, r84p = dev["r54pool"], dev["r84pool"]
    n = dev["n"]
    cases = pickle.load(open(REPO / "exp/eval/_R12_all_turns_payload.pkl", "rb"))["cases"]
    meta = __import__("json").load(open(REPO / "cache/metadata/track_metadata_all_tracks.json"))

    def artist(tid):
        a = meta.get(tid, {}).get("artist_name", [])
        return a[0] if isinstance(a, list) and a else (a if isinstance(a, str) else "")

    # CF embeddings
    te = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")["all_tracks"]
    tids = [str(t) for t in te["track_id"]]
    Tcf = np.zeros((len(tids), 128), np.float32)
    for i, v in enumerate(te["cf-bpr"]):
        if v is not None and len(v) == 128:
            Tcf[i] = v
    tid2i = {t: i for i, t in enumerate(tids)}
    Tn = Tcf / np.clip(np.linalg.norm(Tcf, axis=1, keepdims=True), 1e-8, None)
    ue = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Embeddings")
    U = {str(r["user_id"]): np.array(r["cf-bpr"], np.float32) for sp in ue for r in ue[sp]}

    # candidate sets (dedup r84 then r54, cap K) + retrieval ranks
    K = 120
    cand, r54rank, r84rank = [], [], []
    for i in range(n):
        a = (r84p[i] or [])[:K]; b = (r54p[i] or [])[:K]
        seen = list(dict.fromkeys(a + b))
        cand.append(seen)
        rr54 = {t: j for j, t in enumerate((r54p[i] or []))}
        rr84 = {t: j for j, t in enumerate((r84p[i] or []))}
        r54rank.append(rr54); r84rank.append(rr84)

    # session-cf vec per case, user-cf per case
    sessvec = [None] * n; uservec = [None] * n
    for i in range(n):
        pl = [tid2i[t] for t in cases[i].get("music_turns", []) if t in tid2i]
        if pl:
            sv = Tn[pl].mean(0); nn = np.linalg.norm(sv)
            sessvec[i] = sv / nn if nn > 1e-7 else None
        u = str(cases[i]["user_id"])
        if u in U and np.linalg.norm(U[u]) > 1e-7:
            uservec[i] = U[u] / np.linalg.norm(U[u])
    played_artists = [set(artist(t) for t in cases[i].get("music_turns", [])) for i in range(n)]

    def feats_for(i, pop, apop):
        L = lr20[i] or []
        lrpos = {t: j for j, t in enumerate(L)}
        rows = []
        for t in cand[i]:
            ti = tid2i.get(t)
            r54 = r54rank[i].get(t, 999); r84 = r84rank[i].get(t, 999)
            rrf = 1.0 / (60 + r54) + 1.0 / (60 + r84)
            scf = float(Tn[ti] @ sessvec[i]) if (ti is not None and sessvec[i] is not None) else 0.0
            ucf = float(Tn[ti] @ uservec[i]) if (ti is not None and uservec[i] is not None) else 0.0
            rows.append([
                r54, r84, rrf, min(r54, r84),
                np.log1p(pop.get(t, 0)), np.log1p(apop.get(artist(t), 0)),
                scf, ucf, 1.0 if uservec[i] is not None else 0.0,
                1.0 if t in lrpos else 0.0, lrpos.get(t, 99),
                1.0 if artist(t) in played_artists[i] else 0.0,
            ])
        return np.array(rows, np.float32)

    # per-fold OOF
    folds = sorted(set(fold))
    rankings = [None] * n
    for f in folds:
        tr = [i for i in range(n) if fold[i] != f]
        te_idx = [i for i in range(n) if fold[i] == f]
        # leak-free popularity from TRAIN cases only
        pop = Counter(); apop = Counter()
        for i in tr:
            for t in cases[i].get("music_turns", []):
                pop[t] += 1; apop[artist(t)] += 1
        X, y, grp = [], [], []
        for i in tr:
            ff = feats_for(i, pop, apop)
            if not len(ff):
                continue
            X.append(ff); y.extend([1 if t == gt[i] else 0 for t in cand[i]]); grp.append(len(cand[i]))
        X = np.vstack(X); y = np.array(y)
        ds = lgb.Dataset(X, label=y, group=grp)
        params = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[20],
                      learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
                      lambda_l2=1.0, verbose=-1)
        model = lgb.train(params, ds, num_boost_round=200)
        for i in te_idx:
            ff = feats_for(i, pop, apop)
            if not len(ff):
                rankings[i] = (lr20[i] or [])[:20]; continue
            s = model.predict(ff)
            order = [cand[i][j] for j in np.argsort(-s)]
            rankings[i] = order[:30]
        print(f"  fold {f}: trained on {len(tr)} cases, scored {len(te_idx)}", flush=True)

    m = evaluate(dev, rankings)
    print("\n=== R108 selection-policy reranker — dev OOF vs production lr_top20 ===")
    print(f"nDCG@20: {m['nDCG@20']:.4f}  (base {m['base_nDCG@20']:.4f}, Δ {m['dNDCG_all']:+.4f})")
    print(f"  Δsame_artist={m['dNDCG_same']:+.4f}  Δdiff_artist={m['dNDCG_diff']:+.4f}  Δh7={m['dNDCG_h7']:+.4f}")
    print(f"  hit@1={m['hit@1']:.4f} hit@5={m['hit@5']:.4f} hit@20={m['hit@20']:.4f}")
    print(f"  churn_top1/80={m['churn_top1_per80']:.1f}  overlap@20={m['overlap@20']:.2f}")
    print(f"  GATES: {m['gates']}")
    import json as _j
    _j.dump({"metrics": {k: v for k, v in m.items() if k != "gates"}, "gates": m["gates"]},
            open(REPO / "exp/eval/expR108_selection_policy_oof.json", "w"), indent=2)
    print("saved exp/eval/expR108_selection_policy_oof.json")


if __name__ == "__main__":
    main()
