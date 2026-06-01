#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""GOAL-0.65 reusable dev OOF evaluation harness for nDCG@20 schemes.

Any candidate scheme that produces a per-case ranked list of track_ids plugs in here
to get the standard metrics + gates vs the production-family baseline (lr_top20):
  nDCG@20, hit@{1,5,20}, recall@{20,100,300} (vs the candidate's own pool),
  same-artist / diff-artist nDCG deltas (the canary), h7 subset, and churn
  (top-1 changed, top-20 overlap) vs the production top-20.

Baseline (lr_top20, the production R54/R39 LR dev top-20): dev nDCG@20 = 0.3159.
Gates (from project memory): same-artist Δ >= -0.005, diff-artist Δ >= -0.005,
churn top-1 <= 30/80-equiv (here reported as a rate), top-20 overlap >= 16/20-equiv.

Usage:
    from exp_goal65_eval import load_dev, evaluate
    dev = load_dev()
    m = evaluate(dev, my_rankings)   # my_rankings: list[list[str]] len 8000, each top-K ids
    print(m["nDCG@20"], m["gates"])
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

W0 = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
R12 = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R54_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"
TOP_K = 20


def ndcg_at_k(rank: int, k: int = TOP_K) -> float:
    return 1.0 / math.log2(rank + 1) if (rank is not None and 0 < rank <= k) else 0.0


def _r84_path(fold: int) -> Path:
    return (REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json" if fold == 0
            else REPO / "cache" / "r84" / f"phase1_fold{fold}" / "oof_r84_lists.json")


def load_dev() -> dict:
    """Load the canonical dev OOF artifacts once."""
    w0 = pickle.load(open(W0, "rb"))
    w0 = sorted(w0, key=lambda r: r["case_idx"])
    payload = pickle.load(open(R12, "rb"))
    cases = payload["cases"]
    track_artist = payload["track_artist"]
    n = len(cases)

    gt = [r["gt_track_id"] for r in w0]
    fold = [int(r["fold_idx"]) for r in w0]
    lr_top20 = [list(r.get("lr_top20") or [])[:TOP_K] for r in w0]
    n_prior = [int(c["n_prior_music"]) for c in cases]
    played = [c["music_turns"] for c in cases]

    # same-artist flag per case (GT artist appears among played artists)
    same_artist = []
    for i in range(n):
        ga = track_artist.get(gt[i], "")
        pa = {track_artist.get(t, "") for t in played[i]} - {""}
        same_artist.append(bool(ga and ga in pa))

    # candidate pools for recall (R54 ∪ R84 OOF, top-300)
    r54 = json.load(open(R54_OOF))["lists"]
    r54pool = [[t for t, _ in L] for L in r54]
    r84_by_fold = {}
    for k in range(5):
        raw = json.load(open(_r84_path(k)))
        r84_by_fold[k] = {int(ci): [t for t, _ in v] for ci, v in raw.items()}
    r84pool = [r84_by_fold[fold[i]].get(i, []) for i in range(n)]

    return {"n": n, "gt": gt, "fold": fold, "lr_top20": lr_top20, "n_prior": n_prior,
            "same_artist": same_artist, "track_artist": track_artist,
            "r54pool": r54pool, "r84pool": r84pool}


def _rank_of(ranked: list[str], gt: str) -> int:
    try:
        return ranked.index(gt) + 1
    except ValueError:
        return -1


def evaluate(dev: dict, rankings: list[list[str]], baseline: list[list[str]] | None = None) -> dict:
    """rankings: per-case ranked track_id lists (>=20). Returns metrics vs GT + churn vs baseline."""
    n = dev["n"]
    assert len(rankings) == n, f"need {n} rankings, got {len(rankings)}"
    base = baseline if baseline is not None else dev["lr_top20"]
    gt = dev["gt"]

    nd = np.zeros(n); base_nd = np.zeros(n)
    hit1 = hit5 = hit20 = 0
    churn_top1 = 0; overlap20 = 0.0
    rec20 = rec100 = rec300 = 0
    for i in range(n):
        r = _rank_of(rankings[i], gt[i])
        nd[i] = ndcg_at_k(r)
        base_nd[i] = ndcg_at_k(_rank_of(base[i], gt[i]))
        hit1 += r == 1; hit5 += 0 < r <= 5; hit20 += 0 < r <= 20
        pool = set(dev["r54pool"][i][:300]) | set(dev["r84pool"][i][:300])
        rec300 += gt[i] in pool
        rec100 += gt[i] in (set(dev["r54pool"][i][:100]) | set(dev["r84pool"][i][:100]))
        rec20 += gt[i] in (set(dev["r54pool"][i][:20]) | set(dev["r84pool"][i][:20]))
        if rankings[i] and base[i] and rankings[i][0] != base[i][0]:
            churn_top1 += 1
        overlap20 += len(set(rankings[i][:20]) & set(base[i][:20]))

    def subset_delta(mask):
        idx = [i for i in range(n) if mask[i]]
        if not idx:
            return 0.0, 0
        return float(np.mean([nd[i] - base_nd[i] for i in idx])), len(idx)

    d_same, n_same = subset_delta(dev["same_artist"])
    d_diff, n_diff = subset_delta([not s for s in dev["same_artist"]])
    d_h7, n_h7 = subset_delta([p == 7 for p in dev["n_prior"]])
    d_all = float(np.mean(nd - base_nd))
    churn80 = churn_top1 / n * 80.0
    ov20 = overlap20 / n

    gates = {
        "same_artist_ge_-005": d_same >= -0.005,
        "diff_artist_ge_-005": d_diff >= -0.005,
        "churn_top1_le_30/80": churn80 <= 30.0,
        "overlap20_ge_16": ov20 >= 16.0,
        "ndcg_lift_positive": d_all > 0,
    }
    return {
        "nDCG@20": float(np.mean(nd)), "base_nDCG@20": float(np.mean(base_nd)),
        "dNDCG_all": d_all, "dNDCG_same": d_same, "dNDCG_diff": d_diff, "dNDCG_h7": d_h7,
        "hit@1": hit1 / n, "hit@5": hit5 / n, "hit@20": hit20 / n,
        "recall@20": rec20 / n, "recall@100": rec100 / n, "recall@300": rec300 / n,
        "churn_top1_per80": round(churn80, 2), "overlap@20": round(ov20, 3),
        "n_same": n_same, "n_diff": n_diff, "n_h7": n_h7,
        "gates": gates, "all_gates_pass": all(gates.values()),
    }


def main():
    dev = load_dev()
    # self-test: evaluating the baseline against itself -> nDCG reproduces, deltas 0, churn 0
    m = evaluate(dev, dev["lr_top20"])
    print("self-test (baseline vs baseline):")
    for k in ["nDCG@20", "hit@1", "hit@5", "hit@20", "recall@20", "recall@100", "recall@300",
              "dNDCG_all", "churn_top1_per80", "overlap@20"]:
        print(f"  {k:18} {m[k]}")
    print(f"  expected nDCG@20≈0.3159, recall@300≈0.717, churn 0, overlap 20")


if __name__ == "__main__":
    main()
