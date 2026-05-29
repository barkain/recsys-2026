#!/usr/bin/env python3
"""R96 Phase 0a: recall-ceiling gate (CPU, no GPU, no Codabench slots).

Before spending A100 time on a new retriever, prove that recall is actually the
bottleneck. For every dev case (8000, one GT each) measure how much ground truth
is already reachable by the UNION of current retrieval pools. If the union
already contains nearly all GT, the bottleneck is ranking, not recall, and a new
retriever cannot help. If a real fraction of GT is absent from every pool, there
is genuine recall headroom and an A100 retriever (Phase 0b) is justified.

Union sources (all 8000 dev cases, aligned by case_idx; dense/ST families that
are readily cached as dev OOF):
  - R54 single-source top-300         (reference_stats)
  - R84 BGE-large 5-fold OOF top-300  (cache/r84/phase1_fold*, phase0b_fold0)
  - R90 retuned BGE 5-fold OOF top-300 (cache/r90/phase1_fold*_varA)
  - R21 SentenceTransformer OOF top-300 (cache/r21_production/dev_r21_oof_lists.json)

This is a LOWER BOUND on the true union coverage (BM25 / ALS / CFBPR / qwen3
dev pools are not included here), so if coverage is already high, headroom is
even smaller than reported. Production recall@20 (R54c LR top-20) is the floor.
"""
from __future__ import annotations

import glob
import json
import pickle
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
R84_OOF = sorted(glob.glob(str(REPO / "cache/r84/phase1_fold*/oof_r84_lists.json"))) + \
    [str(REPO / "cache/r84/phase0b_fold0/oof_r84_lists.json")]
R90_OOF = sorted(glob.glob(str(REPO / "cache/r90/phase1_fold*_varA/oof_r84_lists.json")))
R21_OOF = REPO / "cache/r21_production/dev_r21_oof_lists.json"
OUT = REPO / "exp" / "eval" / "expR96_recall_ceiling.json"


def ids(seq, k=None):
    out = [x[0] if isinstance(x, (list, tuple)) else x for x in (seq or [])]
    return out[:k] if k else out


def load_idx_dict(files):
    merged = {}
    for f in files:
        d = json.load(open(f))
        inner = d.get("lists", d) if isinstance(d, dict) else None
        if inner is None:
            continue
        for k, v in inner.items():
            merged[int(k)] = ids(v)
    return merged


def main():
    ref = pickle.load(open(REF, "rb"))
    r84 = load_idx_dict(R84_OOF)
    r90 = load_idx_dict(R90_OOF)
    r21_list = json.load(open(R21_OOF))
    r21 = {i: ids(v) for i, v in enumerate(r21_list)}

    N = len(ref)
    DEPTHS = [20, 30, 50, 300]
    cov = {src: Counter() for src in ["R54", "R84", "R90", "R21", "UNION"]}
    in_lr20 = 0
    union300_present = 0
    union300_absent = 0
    union30_present = 0
    deep_only = 0          # in union@300 but not in union@30
    absent_from_r54_but_in_union = 0
    rescue_rank_dist = Counter()  # min rank of GT across sources, when in union@300

    for e in ref:
        i = e["case_idx"]
        gt = e["gt_track_id"]
        pools = {
            "R54": ids(e["r54_single_source_top300"]),
            "R84": r84.get(i, []),
            "R90": r90.get(i, []),
            "R21": r21.get(i, []),
        }
        lr20 = ids(e["lr_top20"], 20)
        if gt in lr20:
            in_lr20 += 1

        # per-source coverage at each depth
        for src, pool in pools.items():
            pos = pool.index(gt) if gt in pool else None
            for D in DEPTHS:
                if pos is not None and pos < D:
                    cov[src][D] += 1

        # union at each depth
        for D in DEPTHS:
            u = set()
            for pool in pools.values():
                u.update(pool[:D])
            if gt in u:
                cov["UNION"][D] += 1

        # full-depth union (300) coverage + headroom
        full_union = set()
        min_rank = None
        for pool in pools.values():
            full_union.update(pool)
            if gt in pool:
                r = pool.index(gt)
                min_rank = r if min_rank is None else min(min_rank, r)
        if gt in full_union:
            union300_present += 1
            if min_rank < 20:
                rescue_rank_dist["top20"] += 1
            elif min_rank < 30:
                rescue_rank_dist["20-30"] += 1
            elif min_rank < 50:
                rescue_rank_dist["30-50"] += 1
            elif min_rank < 100:
                rescue_rank_dist["50-100"] += 1
            else:
                rescue_rank_dist["100-300"] += 1
            union30 = set()
            for pool in pools.values():
                union30.update(pool[:30])
            if gt in union30:
                union30_present += 1
            else:
                deep_only += 1
            if gt not in set(ids(e["r54_single_source_top300"])):
                absent_from_r54_but_in_union += 1
        else:
            union300_absent += 1

    def pct(x):
        return round(x / N, 4)

    report = {
        "experiment": "R96 Phase 0a recall-ceiling gate",
        "n_dev_cases": N,
        "union_sources": ["R54", "R84", "R90", "R21"],
        "note": "LOWER BOUND on union coverage (no BM25/ALS/CFBPR/qwen3 dev pools).",
        "production_recall_at_20_r54c_lr": pct(in_lr20),
        "per_source_recall": {
            src: {str(D): pct(cov[src][D]) for D in DEPTHS}
            for src in ["R54", "R84", "R90", "R21", "UNION"]
        },
        "union_top300_coverage": pct(union300_present),
        "gt_absent_from_full_union": pct(union300_absent),
        "union_top30_coverage": pct(union30_present),
        "in_union300_but_deep_only_(30-300)": pct(deep_only),
        "gt_absent_from_r54_but_recovered_by_union": pct(absent_from_r54_but_in_union),
        "rescue_rank_distribution_min_across_sources": {
            k: pct(v) for k, v in rescue_rank_dist.items()
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)

    print(f"dev cases: {N}")
    print(f"Production recall@20 (R54c LR):        {report['production_recall_at_20_r54c_lr']:.3f}  (ranking floor)")
    print("\nPer-source GT recall by depth:")
    print(f"  {'src':6s}{'@20':>7s}{'@30':>7s}{'@50':>7s}{'@300':>7s}")
    for src in ["R54", "R84", "R90", "R21", "UNION"]:
        r = report["per_source_recall"][src]
        print(f"  {src:6s}{r['20']:>7.3f}{r['30']:>7.3f}{r['50']:>7.3f}{r['300']:>7.3f}")
    print(f"\nUNION top-300 coverage (recall ceiling): {report['union_top300_coverage']:.3f}")
    print(f"GT ABSENT from full union (headroom):    {report['gt_absent_from_full_union']:.3f}")
    print(f"UNION top-30 coverage:                   {report['union_top30_coverage']:.3f}")
    print(f"In union but deep-only (rank 30-300):    {report['in_union300_but_deep_only_(30-300)']:.3f}  (rescuable by ranking)")
    print(f"GT absent from R54 but recovered by union:{report['gt_absent_from_r54_but_recovered_by_union']:.3f}")
    print("\nMin-rank-across-sources distribution (GT in union@300):")
    for k in ["top20", "20-30", "30-50", "50-100", "100-300"]:
        if k in report["rescue_rank_distribution_min_across_sources"]:
            print(f"  {k:8s}: {report['rescue_rank_distribution_min_across_sources'][k]:.3f}")
    print(f"\nWrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
