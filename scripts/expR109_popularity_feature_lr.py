#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R109 — does popularity convert as a FEATURE inside an LR (not a naked reranker)?

R108 showed a naked popularity reranker REPLACES the calibrated production LR and is far
worse. The correct test (feedback_rerank_only_closed): add popularity as a FEATURE to an
LR that already has strong ranking features, and measure its MARGINAL value. We build a
production-grade per-candidate logistic ranker from the two main retrieval sources (R21 +
R54 OOF, with scores), and compare OOF nDCG@20 + gate suite WITH vs WITHOUT popularity /
artist-pop / session-cf / user-cf features. Popularity computed leak-free (train folds
only). If +pop beats base cleanly with gates -> worth the full production pipeline; if it
adds nothing -> popularity is absorbed by ranking features, archive the lever.
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
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from datasets import load_dataset
    dev = load_dev()
    gt, fold = dev["gt"], dev["fold"]
    n = dev["n"]
    cases = pickle.load(open(REPO / "exp/eval/_R12_all_turns_payload.pkl", "rb"))["cases"]
    meta = json.load(open(REPO / "cache/metadata/track_metadata_all_tracks.json"))
    r21 = json.load(open(REPO / "cache/r21_production/dev_r21_oof_lists.json"))
    r21 = r21["lists"] if isinstance(r21, dict) else r21
    r54 = json.load(open(REPO / "cache/r54/phase2_full/oof_r54_lists.json"))["lists"]

    def artist(tid):
        a = meta.get(tid, {}).get("artist_name", [])
        return a[0] if isinstance(a, list) and a else (a if isinstance(a, str) else "")

    # per-source rank/score maps
    def rankmap(lst):
        m = {}
        for j, item in enumerate(lst or []):
            t = item[0] if isinstance(item, (list, tuple)) else item
            sc = item[1] if isinstance(item, (list, tuple)) and len(item) == 2 else 0.0
            if t not in m:
                m[t] = (j, sc)
        return m
    R21 = [rankmap(x) for x in r21]
    R54 = [rankmap(x) for x in r54]
    cand = [list(dict.fromkeys(list(R54[i].keys())[:150] + list(R21[i].keys())[:150])) for i in range(n)]

    # CF vectors
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
    sessvec, uservec = [None]*n, [None]*n
    for i in range(n):
        pl = [tid2i[t] for t in cases[i].get("music_turns", []) if t in tid2i]
        if pl:
            sv = Tn[pl].mean(0); nn = np.linalg.norm(sv)
            sessvec[i] = sv/nn if nn > 1e-7 else None
        u = str(cases[i]["user_id"])
        if u in U and np.linalg.norm(U[u]) > 1e-7:
            uservec[i] = U[u]/np.linalg.norm(U[u])

    def base_feats(i, t):
        r2 = R21[i].get(t); r5 = R54[i].get(t)
        return [
            1.0/(1+r2[0]) if r2 else 0.0, 1.0 if r2 else 0.0, r2[1] if r2 else 0.0,
            1.0/(1+r5[0]) if r5 else 0.0, 1.0 if r5 else 0.0, r5[1] if r5 else 0.0,
        ]

    def pop_feats(i, t, pop, apop):
        ti = tid2i.get(t)
        scf = float(Tn[ti] @ sessvec[i]) if (ti is not None and sessvec[i] is not None) else 0.0
        ucf = float(Tn[ti] @ uservec[i]) if (ti is not None and uservec[i] is not None) else 0.0
        return [np.log1p(pop.get(t, 0)), np.log1p(apop.get(artist(t), 0)), scf, ucf,
                1.0 if uservec[i] is not None else 0.0]

    def run(use_pop):
        rankings = [None]*n
        for f in sorted(set(fold)):
            tr = [i for i in range(n) if fold[i] != f]
            tei = [i for i in range(n) if fold[i] == f]
            pop = Counter(); apop = Counter()
            for i in tr:
                for t in cases[i].get("music_turns", []):
                    pop[t] += 1; apop[artist(t)] += 1
            X, y = [], []
            for i in tr:
                for t in cand[i]:
                    X.append(base_feats(i, t) + (pop_feats(i, t, pop, apop) if use_pop else []))
                    y.append(1 if t == gt[i] else 0)
            X = np.array(X, np.float32); y = np.array(y)
            sc = StandardScaler().fit(X)
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced").fit(sc.transform(X), y)
            for i in tei:
                if not cand[i]:
                    rankings[i] = []; continue
                XX = np.array([base_feats(i, t) + (pop_feats(i, t, pop, apop) if use_pop else []) for t in cand[i]], np.float32)
                p = clf.predict_proba(sc.transform(XX))[:, 1]
                rankings[i] = [cand[i][j] for j in np.argsort(-p)][:30]
        return evaluate(dev, rankings)

    print("Building base LR (R21+R54 features only)...", flush=True)
    mb = run(False)
    print("Building +popularity LR...", flush=True)
    mp = run(True)
    print(f"\n=== R109 LR feature-addition (dev OOF; production baseline lr_top20=0.3159) ===")
    for name, m in [("base LR (R21+R54)", mb), ("+pop/artist/cf", mp)]:
        print(f"\n{name}:")
        print(f"  nDCG@20={m['nDCG@20']:.4f} (Δ vs lr_top20 {m['dNDCG_all']:+.4f}) "
              f"hit@1={m['hit@1']:.4f} hit@20={m['hit@20']:.4f}")
        print(f"  Δsame={m['dNDCG_same']:+.4f} Δdiff={m['dNDCG_diff']:+.4f} Δh7={m['dNDCG_h7']:+.4f} "
              f"churn={m['churn_top1_per80']:.1f} ov@20={m['overlap@20']:.2f}")
    print(f"\nMARGINAL VALUE OF POPULARITY: nDCG {mp['nDCG@20']-mb['nDCG@20']:+.4f}, "
          f"hit@20 {mp['hit@20']-mb['hit@20']:+.4f}")
    json.dump({"base": {k: v for k, v in mb.items() if k != "gates"},
               "pop": {k: v for k, v in mp.items() if k != "gates"}},
              open(REPO/"exp/eval/expR109_popularity_feature_oof.json", "w"), indent=2)
    print("saved exp/eval/expR109_popularity_feature_oof.json")


if __name__ == "__main__":
    main()
