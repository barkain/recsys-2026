#!/usr/bin/env python3
"""R100 Phase 0 prep: per-dev-case UNION candidate sets for cross-encoder rerank.

The nDCG bottleneck is ranking, not recall: the complete union holds ~78% of GT
@300 but production delivers only ~48% @20. This builds the candidate set a
cross-encoder will rerank — the union of pool / R54 / R84 / R90 / R21 — capped
at top-K per case, with GT + fold for OOF training/eval. Compact JSON for upload
to Drive (the reranker notebook consumes it).

Output: exp/eval/expR100_union_candidates.json
  {meta, cases:[{case_idx, fold_idx, gt, gt_rank_in_cands(-1 if absent), candidates:[tid,...]}]}
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import pickle
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
CASE_FEATS = REPO / "cache" / "r84b" / "case_features.pkl"
R84_OOF = sorted(glob.glob(str(REPO / "cache/r84/phase1_fold*/oof_r84_lists.json"))) + \
    [str(REPO / "cache/r84/phase0b_fold0/oof_r84_lists.json")]
R90_OOF = sorted(glob.glob(str(REPO / "cache/r90/phase1_fold*_varA/oof_r84_lists.json")))
R21_OOF = REPO / "cache/r21_production/dev_r21_oof_lists.json"
OUT = REPO / "exp" / "eval" / "expR100_union_candidates.json"

TOPK_PER_SOURCE = 100   # take top-K from each source before union
CAND_CAP = 120          # final candidates per case to rerank


def ids(seq, k=None):
    out = [x[0] if isinstance(x, (list, tuple)) else x for x in (seq or [])]
    return out[:k] if k else out


def load_idx_dict(files):
    m = {}
    for f in files:
        d = json.load(open(f))
        inner = d.get("lists", d) if isinstance(d, dict) else None
        if inner is None:
            continue
        for k, v in inner.items():
            m[int(k)] = ids(v)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=CAND_CAP)
    ap.add_argument("--per-source", type=int, default=TOPK_PER_SOURCE)
    args = ap.parse_args()

    ref = pickle.load(open(REF, "rb"))
    r84 = load_idx_dict(R84_OOF)
    r90 = load_idx_dict(R90_OOF)
    r21 = {i: ids(v) for i, v in enumerate(json.load(open(R21_OOF)))}
    cf = pickle.load(open(CASE_FEATS, "rb"))
    pool = {k: list(cf[k]["pool"]) for k in cf}
    del cf
    gc.collect()

    cases = []
    gt_in = 0
    recall_at = defaultdict(int)
    for e in ref:
        i = e["case_idx"]
        gt = e["gt_track_id"]
        srcs = {
            "pool": pool.get(i, [])[: args.per_source],
            "r54": ids(e["r54_single_source_top300"], args.per_source),
            "r84": r84.get(i, [])[: args.per_source],
            "r90": r90.get(i, [])[: args.per_source],
            "r21": r21.get(i, [])[: args.per_source],
        }
        # union ordered by min rank across sources (RRF-ish), dedup, cap
        best_rank = {}
        for lst in srcs.values():
            for r, t in enumerate(lst):
                if t not in best_rank or r < best_rank[t]:
                    best_rank[t] = r
        cands = [t for t, _ in sorted(best_rank.items(), key=lambda kv: kv[1])][: args.cap]
        gr = cands.index(gt) if gt in cands else -1
        if gr >= 0:
            gt_in += 1
            for D in (20, 30, 50, 100, args.cap):
                if gr < D:
                    recall_at[D] += 1
        cases.append({
            "case_idx": i, "fold_idx": e["fold_idx"], "gt": gt,
            "gt_rank_in_cands": gr, "candidates": cands,
        })

    N = len(cases)
    out = {
        "experiment": "R100 union candidate sets for cross-encoder rerank",
        "n_cases": N, "cand_cap": args.cap, "per_source": args.per_source,
        "sources": ["pool", "r54", "r84", "r90", "r21"],
        "gt_in_candidates_rate": round(gt_in / N, 4),
        "ceiling_recall_at": {str(D): round(recall_at[D] / N, 4) for D in (20, 30, 50, 100, args.cap)},
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"))
    import os
    print(f"cases={N}  cand_cap={args.cap}")
    print(f"GT in candidate set: {out['gt_in_candidates_rate']:.4f}  (reranker ceiling = this)")
    print(f"ceiling recall@K (if reranker perfect): {out['ceiling_recall_at']}")
    print(f"(baseline production recall@20 = 0.479)")
    print(f"size: {os.path.getsize(OUT)/1e6:.1f} MB -> {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
