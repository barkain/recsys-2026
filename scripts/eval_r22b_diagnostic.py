#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R22b diagnostic: overlap analysis + ensemble test (R21 ∪ R22b).

Answers:
1. Where do R22b gains/losses vs R21 concentrate?
2. Does union R21+R22b improve pool_hit@300?
3. Gate: pool_hit@300 >= 0.620 for ensemble?
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"

import json
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R22B_LISTS = REPO / "cache" / "r22b" / "dev_r22b_lists.json"
V3_POOLS = REPO / "cache" / "r21_production" / "v3_pools.json"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    print(f"{ts()} Loading data...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)

    with open(R21_OOF) as f:
        r21_lists = json.load(f)
    with open(R22B_LISTS) as f:
        r22b_lists = json.load(f)
    with open(V3_POOLS) as f:
        v3_pools = [set(p) for p in json.load(f)]

    # Train tracks for seen/unseen
    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_tracks = set()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                train_tracks.add(str(c["content"]).strip())

    # ===== PART 1: Gain/Loss Analysis =====
    print(f"\n{'='*60}")
    print("PART 1: R22b vs R21 — Gain/Loss Analysis")
    print(f"{'='*60}")

    gained = []  # R22b finds, R21 misses
    lost = []    # R21 finds, R22b misses
    both = []    # both find
    neither = [] # neither finds

    for i in range(n):
        gt = cases[i]["gt"]
        in_r21 = gt in r21_lists[i][:200]
        in_r22b = gt in r22b_lists[i][:200]
        if in_r22b and not in_r21:
            gained.append(i)
        elif in_r21 and not in_r22b:
            lost.append(i)
        elif in_r21 and in_r22b:
            both.append(i)
        else:
            neither.append(i)

    print(f"\nOverlap at hit@200:")
    print(f"  Both find:    {len(both)}")
    print(f"  R22b gains:   {len(gained)}")
    print(f"  R22b loses:   {len(lost)}")
    print(f"  Neither:      {len(neither)}")

    # Slice by seen/unseen
    def slice_stats(indices, label):
        unseen = sum(1 for i in indices if cases[i]["gt"] not in train_tracks)
        seen = len(indices) - unseen
        hist0 = sum(1 for i in indices if len(cases[i]["music_turns"]) == 0)
        in_v3 = sum(1 for i in indices if cases[i]["gt"] in v3_pools[i])
        hist_depths = [len(cases[i]["music_turns"]) for i in indices]
        avg_hist = np.mean(hist_depths) if hist_depths else 0
        return {
            "n": len(indices), "unseen": unseen, "seen": seen,
            "hist0": hist0, "avg_hist": f"{avg_hist:.1f}", "in_v3": in_v3,
        }

    print(f"\nSlice breakdown:")
    print(f"  {'Category':<14} {'N':>6} {'Unseen':>7} {'Seen':>6} {'Hist0':>6} {'AvgHist':>8} {'InV3':>6}")
    print(f"  {'-'*55}")
    for label, indices in [("Gained", gained), ("Lost", lost), ("Both", both), ("Neither", neither)]:
        s = slice_stats(indices, label)
        print(f"  {label:<14} {s['n']:>6} {s['unseen']:>7} {s['seen']:>6} "
              f"{s['hist0']:>6} {s['avg_hist']:>8} {s['in_v3']:>6}")

    # Gains/losses by history depth bucket
    print(f"\nGain/loss by history depth:")
    for depth in range(8):
        g = sum(1 for i in gained if cases[i]["n_prior_music"] == depth)
        l = sum(1 for i in lost if cases[i]["n_prior_music"] == depth)
        total = sum(1 for i in range(n) if cases[i]["n_prior_music"] == depth)
        print(f"  hist_{depth}: gained={g:>4} lost={l:>4} total={total:>5} "
              f"net={g-l:>+4}")

    # ===== PART 2: Ensemble R21 ∪ R22b =====
    print(f"\n{'='*60}")
    print("PART 2: Ensemble R21 ∪ R22b — Pool Hit Simulation")
    print(f"{'='*60}")

    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector

    print(f"\n{ts()} Building ALS...")
    als_factors, als_track_ids_als, als_track_to_idx = build_als()
    als_source = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids_als[j] for j in top_idx])
        else:
            als_source.append([])

    # Test multiple configurations
    configs = {
        "R21 only": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0},
        "R22b only": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R22b": 1.0},
        "R21+R22b (both w=1.0)": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R22b": 1.0},
        "R21+R22b (R22b w=0.5)": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R22b": 0.5},
        "R21+R22b (R22b w=0.3)": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0, "R22b": 0.3},
    }

    print(f"\n{ts()} Computing pool_hit@300 for each config...")
    for config_name, sw in configs.items():
        pool_hit = 0
        for i in range(n):
            gt = cases[i]["gt"]
            src = {
                "A": payload["src_a"][i], "B": payload["src_b"][i],
                "C": payload["src_c"][i], "D": payload["src_d"][i],
                "F": payload["src_f"][i], "ALS": als_source[i],
            }
            if "R21" in sw:
                src["R21"] = r21_lists[i][:200]
            if "R22b" in sw:
                src["R22b"] = r22b_lists[i][:200]
            pool = set(weighted_rrf(src, sw, topk=300, k=20))
            if gt in pool:
                pool_hit += 1

        ph = pool_hit / n
        gate = "PASS" if ph >= 0.620 else "FAIL"
        strong = "STRONG" if ph >= 0.650 else ""
        predicted = 0.4628 * ph + 0.1894
        print(f"  {config_name:<30} pool_hit={ph:.4f} [{gate}] {strong} "
              f"predicted_blind={predicted:.4f}")

    # ===== PART 3: Union hit analysis =====
    print(f"\n{'='*60}")
    print("PART 3: Union Analysis")
    print(f"{'='*60}")

    union_hit200 = sum(1 for i in range(n)
                       if cases[i]["gt"] in r21_lists[i][:200]
                       or cases[i]["gt"] in r22b_lists[i][:200])
    r21_only_hit = sum(1 for i in range(n)
                       if cases[i]["gt"] in r21_lists[i][:200])
    r22b_only_hit = sum(1 for i in range(n)
                        if cases[i]["gt"] in r22b_lists[i][:200])

    print(f"\n  R21 hit@200:         {r21_only_hit} ({r21_only_hit/n:.1%})")
    print(f"  R22b hit@200:        {r22b_only_hit} ({r22b_only_hit/n:.1%})")
    print(f"  Union hit@200:       {union_hit200} ({union_hit200/n:.1%})")
    print(f"  Union ceiling gain:  +{union_hit200 - r21_only_hit} vs R21")

    # Save results
    results = {
        "overlap": {
            "both": len(both), "gained": len(gained),
            "lost": len(lost), "neither": len(neither),
        },
        "union_hit200": union_hit200,
        "created_at": datetime.now().isoformat(),
    }
    out_path = REPO / "exp" / "eval" / "r22b_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{ts()} Saved: {out_path}")


if __name__ == "__main__":
    main()
