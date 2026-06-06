#!/usr/bin/env python3
"""R431 fast conversion policies for official user-cf retrieval.

R180 showed official user cf-bpr vectors retrieve real missing GTs. This script
tests whether that signal can be used *safely* without rebuilding the full 12M-row
LR feature matrix:

1. blend_top20: reorder the existing production top-20 using user-cf rank.
2. inject: insert one high-ranked user-cf candidate into production top-20.

This is intentionally a fast go/no-go. If these fail, full LR integration is
unlikely to be worth running on the Mac; if they pass, move the full integration
to Colab/A100 or optimize the R103 harness.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
W0 = REPO / "exp/eval/expR68_r54_reference_stats.pkl"
R12 = REPO / "exp/eval/_R12_all_turns_payload.pkl"

SRC_DEFAULT = REPO / "cache/r431_user_cf/user_cf_oof_lists.json"
OUT_JSON = REPO / "exp/eval/expR431_fast_policy_eval.json"
OUT_MD = REPO / "docs/r431_user_cf_fast_policy.md"


def ndcg_at_k(rank: int, k: int = 20) -> float:
    return 0.0 if rank <= 0 or rank > k else 1.0 / math.log2(rank + 1)


def rank_of(row: list[str], gt: str) -> int:
    try:
        return row.index(gt) + 1
    except ValueError:
        return -1


def load_light_dev() -> dict[str, Any]:
    """Load only what the policy sweep needs; avoid heavy R54/R84 pool JSONs."""
    w0 = sorted(pickle.load(open(W0, "rb")), key=lambda r: r["case_idx"])
    payload = pickle.load(open(R12, "rb"))
    cases = payload["cases"]
    track_artist = payload["track_artist"]
    gt = [str(r["gt_track_id"]) for r in w0]
    base = [list(r.get("lr_top20") or [])[:20] for r in w0]
    n_prior = [int(c.get("n_prior_music") or len(c.get("music_turns") or [])) for c in cases]
    same_artist = []
    for i, c in enumerate(cases):
        ga = track_artist.get(gt[i], "")
        played_artists = {track_artist.get(t, "") for t in (c.get("music_turns") or [])} - {""}
        same_artist.append(bool(ga and ga in played_artists))
    return {"n": len(gt), "gt": gt, "lr_top20": base, "n_prior": n_prior, "same_artist": same_artist}


def evaluate_light(dev: dict[str, Any], rankings: list[list[str]]) -> dict[str, Any]:
    n = dev["n"]
    gt = dev["gt"]
    base = dev["lr_top20"]
    nd = np.zeros(n, dtype=np.float32)
    bnd = np.zeros(n, dtype=np.float32)
    churn_top1 = 0
    overlap20 = 0.0
    hit1 = hit5 = hit20 = 0
    for i in range(n):
        r = rank_of(rankings[i], gt[i])
        br = rank_of(base[i], gt[i])
        nd[i] = ndcg_at_k(r)
        bnd[i] = ndcg_at_k(br)
        hit1 += r == 1
        hit5 += 0 < r <= 5
        hit20 += 0 < r <= 20
        if rankings[i] and base[i] and rankings[i][0] != base[i][0]:
            churn_top1 += 1
        overlap20 += len(set(rankings[i][:20]) & set(base[i][:20]))

    def delta(mask: list[bool]) -> tuple[float, int]:
        idx = np.asarray(mask, dtype=bool)
        if not bool(idx.any()):
            return 0.0, 0
        return float(np.mean(nd[idx] - bnd[idx])), int(idx.sum())

    d_same, n_same = delta(dev["same_artist"])
    d_diff, n_diff = delta([not x for x in dev["same_artist"]])
    d_h7, n_h7 = delta([p == 7 for p in dev["n_prior"]])
    d_all = float(np.mean(nd - bnd))
    churn80 = churn_top1 / n * 80.0
    overlap = overlap20 / n
    return {
        "nDCG@20": float(np.mean(nd)),
        "base_nDCG@20": float(np.mean(bnd)),
        "dNDCG_all": d_all,
        "dNDCG_same": d_same,
        "dNDCG_diff": d_diff,
        "dNDCG_h7": d_h7,
        "hit@1": hit1 / n,
        "hit@5": hit5 / n,
        "hit@20": hit20 / n,
        "churn_top1_per80": round(churn80, 2),
        "overlap@20": round(overlap, 3),
        "n_same": n_same,
        "n_diff": n_diff,
        "n_h7": n_h7,
        "gates": {
            "same_artist_ge_-005": d_same >= -0.005,
            "diff_artist_ge_-005": d_diff >= -0.005,
            "churn_top1_le_30/80": churn80 <= 30.0,
            "overlap20_ge_16": overlap >= 16.0,
            "ndcg_lift_positive": d_all > 0,
        },
    }


def load_source(path: Path) -> tuple[list[list[str]], list[dict[str, int]], list[dict[str, float]]]:
    raw = json.load(open(path))
    lists_raw = raw.get("lists", raw)
    n = max(int(k) for k in lists_raw) + 1
    lists = [[] for _ in range(n)]
    ranks: list[dict[str, int]] = [{} for _ in range(n)]
    scores: list[dict[str, float]] = [{} for _ in range(n)]
    for ci_s, pairs in lists_raw.items():
        ci = int(ci_s)
        row = [str(t) for t, _ in pairs]
        lists[ci] = row
        ranks[ci] = {t: r + 1 for r, t in enumerate(row)}
        scores[ci] = {str(t): float(s) for t, s in pairs}
    return lists, ranks, scores


def uniq20(row: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in row:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) == 20:
            break
    return out


def blend_top20(base: list[list[str]], ucf_ranks: list[dict[str, int]], alpha: float) -> list[list[str]]:
    out: list[list[str]] = []
    for i, row in enumerate(base):
        def score(pair: tuple[int, str]) -> float:
            pos, tid = pair
            base_s = 1.0 / (pos + 1)
            u_rank = ucf_ranks[i].get(tid)
            u_s = 0.0 if u_rank is None else 1.0 / u_rank
            return base_s + alpha * u_s

        ordered = [tid for _, tid in sorted(enumerate(row), key=score, reverse=True)]
        out.append(uniq20(ordered))
    return out


def inject_one(
    base: list[list[str]],
    ucf_lists: list[list[str]],
    *,
    max_ucf_rank: int,
    insert_pos: int,
    min_score: float | None,
    ucf_scores: list[dict[str, float]],
) -> tuple[list[list[str]], int]:
    out: list[list[str]] = []
    changed = 0
    for i, row in enumerate(base):
        row_set = set(row)
        cand = None
        for r, tid in enumerate(ucf_lists[i][:max_ucf_rank], start=1):
            if tid in row_set:
                continue
            if min_score is not None and ucf_scores[i].get(tid, -999.0) < min_score:
                continue
            cand = tid
            break
        if cand is None:
            out.append(list(row))
            continue
        pos = max(0, min(insert_pos - 1, 19))
        new = list(row)
        new.insert(pos, cand)
        new = uniq20(new)
        if new != row:
            changed += 1
        out.append(new)
    return out, changed


def top1_precision(dev: dict[str, Any], base: list[list[str]], ucf_lists: list[list[str]], max_ucf_rank: int) -> dict[str, Any]:
    gt = dev["gt"]
    considered = 0
    hits = 0
    hit_positions: list[int] = []
    for i, row in enumerate(base):
        row_set = set(row)
        for r, tid in enumerate(ucf_lists[i][:max_ucf_rank], start=1):
            if tid in row_set:
                continue
            considered += 1
            if tid == gt[i]:
                hits += 1
                hit_positions.append(r)
            break
    return {
        "max_ucf_rank": max_ucf_rank,
        "considered": considered,
        "hits": hits,
        "precision": hits / considered if considered else 0.0,
        "hit_rank_median": float(np.median(hit_positions)) if hit_positions else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=SRC_DEFAULT)
    ap.add_argument("--out-json", type=Path, default=OUT_JSON)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    source = args.source if args.source.is_absolute() else REPO / args.source
    out_json = args.out_json if args.out_json.is_absolute() else REPO / args.out_json
    out_md = args.out_md if args.out_md.is_absolute() else REPO / args.out_md

    dev = load_light_dev()
    base = dev["lr_top20"]
    ucf_lists, ucf_ranks, ucf_scores = load_source(source)
    assert len(ucf_lists) == dev["n"], (len(ucf_lists), dev["n"])

    baseline = evaluate_light(dev, base)
    rows: list[dict[str, Any]] = []

    for alpha in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        rankings = blend_top20(base, ucf_ranks, alpha)
        m = evaluate_light(dev, rankings)
        rows.append({"policy": "blend_top20", "alpha": alpha, **m})

    for cap in [1, 3, 5, 10, 20, 50, 100]:
        for pos in [1, 3, 5, 10, 20]:
            rankings, changed = inject_one(
                base,
                ucf_lists,
                max_ucf_rank=cap,
                insert_pos=pos,
                min_score=None,
                ucf_scores=ucf_scores,
            )
            m = evaluate_light(dev, rankings)
            rows.append({"policy": "inject", "max_ucf_rank": cap, "insert_pos": pos, "changed": changed, **m})

    # Score-threshold injection. The raw cf-bpr scores are cosine similarities.
    all_top_scores = []
    for i, row in enumerate(base):
        row_set = set(row)
        for tid in ucf_lists[i][:20]:
            if tid not in row_set:
                all_top_scores.append(ucf_scores[i].get(tid, -999.0))
                break
    quantiles = [float(np.quantile(all_top_scores, q)) for q in [0.5, 0.7, 0.8, 0.9, 0.95]] if all_top_scores else []
    for thr in quantiles:
        rankings, changed = inject_one(
            base,
            ucf_lists,
            max_ucf_rank=20,
            insert_pos=20,
            min_score=thr,
            ucf_scores=ucf_scores,
        )
        m = evaluate_light(dev, rankings)
        rows.append({"policy": "inject_score_thr", "max_ucf_rank": 20, "insert_pos": 20, "score_thr": thr, "changed": changed, **m})

    precision = [top1_precision(dev, base, ucf_lists, cap) for cap in [1, 3, 5, 10, 20, 50, 100]]
    best = sorted(rows, key=lambda r: (r["dNDCG_all"], r["dNDCG_h7"], -r["churn_top1_per80"]), reverse=True)[:12]
    out = {
        "experiment": "R431 fast policy conversion eval",
        "source": str(source),
        "baseline": baseline,
        "injection_precision": precision,
        "best_rows": best,
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    lines = [
        "# R431 User-CF Fast Policy Eval",
        "",
        f"Baseline internal dev nDCG@20: `{baseline['nDCG@20']:.4f}`",
        "",
        "## Injection Precision",
        "",
        "| max user-cf rank | considered | GT hits | precision |",
        "|---:|---:|---:|---:|",
    ]
    for p in precision:
        lines.append(f"| {p['max_ucf_rank']} | {p['considered']} | {p['hits']} | {p['precision']:.4f} |")
    lines += [
        "",
        "## Best Policies",
        "",
        "| policy | params | dNDCG | dH7 | dSame | dDiff | churn/80 | overlap@20 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in best[:10]:
        params = []
        for k in ["alpha", "max_ucf_rank", "insert_pos", "score_thr", "changed"]:
            if k in r:
                v = r[k]
                params.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
        lines.append(
            f"| {r['policy']} | {', '.join(params)} | {r['dNDCG_all']:+.4f} | "
            f"{r['dNDCG_h7']:+.4f} | {r['dNDCG_same']:+.4f} | {r['dNDCG_diff']:+.4f} | "
            f"{r['churn_top1_per80']:.1f} | {r['overlap@20']:.2f} |"
        )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n")

    print(f"wrote {out_json.relative_to(REPO)}")
    print(f"wrote {out_md.relative_to(REPO)}")
    print("best:")
    for r in best[:5]:
        print(
            r["policy"],
            {k: r[k] for k in ("alpha", "max_ucf_rank", "insert_pos", "score_thr", "changed") if k in r},
            f"dNDCG={r['dNDCG_all']:+.4f}",
            f"h7={r['dNDCG_h7']:+.4f}",
            f"same={r['dNDCG_same']:+.4f}",
            f"diff={r['dNDCG_diff']:+.4f}",
        )


if __name__ == "__main__":
    main()
