"""R89 Phase 0 — Mac-side comparison vs R84 5-fold OOF.

Loads cached R89 outputs (from Colab via Drive) + existing R84 5-fold OOF
lists on Mac. Computes unique h7 top-30/top-300 recoveries + canaries.

Reads:
- cache/r89/phase0_fold0/oof_r89_lists.json
- cache/r84/phase0b_fold0/oof_r84_lists.json     (fold-0 R84 OOF)
- exp/eval/_R12_all_turns_payload.pkl

Writes:
- exp/eval/expR89_phase0.json (final gate report)
- docs/r89_phase0_result.md
"""
from __future__ import annotations

import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R89_LISTS = REPO / "cache" / "r89" / "phase0_fold0" / "oof_r89_lists.json"
R84_FOLD0_LISTS = REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json"
OUT_JSON = REPO / "exp" / "eval" / "expR89_phase0.json"
OUT_MD = REPO / "docs" / "r89_phase0_result.md"

GATE = {
    "h7_delta_ndcg_ge": 0.005,
    "min_unique_h7_top30": 10,
    "ambig_min_unique_h7_top30": 5,
    "same_artist_delta_ge": -0.005,
    "diff_artist_delta_ge": -0.005,
}


def ts(): return f"[{datetime.now():%H:%M:%S}]"
def head_sha():
    g = shutil.which("git")
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip() if g else "no-git"


def ndcg_at_k(rank, k):
    return 1.0 / math.log2(rank + 1) if 0 < rank <= k else 0.0


def main():
    print(f"{ts()} R89 Phase 0 — Mac-side compare vs R84 OOF")
    print("=" * 70)

    # Load payload
    print(f"\n{ts()} Loading dev payload + R84 + R89 lists...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload.get("track_artist", {})
    if not ta:
        # Build from R12 if missing
        ta = payload.get("track_artist") or {}

    from scripts.expR84_phase0a_census import grouped_session_folds
    sessions = [c["session_id"] for c in cases]
    folds = grouped_session_folds(sessions, seed=0)
    fold0_idx = folds[0].tolist()
    val_cases_local = [cases[i] for i in fold0_idx]
    n_h7 = sum(1 for c in val_cases_local if c.get("n_prior_music") == 7)
    print(f"  fold-0: {len(val_cases_local)} cases (h7={n_h7})")

    # Load R89 + R84 lists
    with open(R89_LISTS) as f:
        r89_lists = {int(k): v for k, v in json.load(f).items()}
    with open(R84_FOLD0_LISTS) as f:
        r84_lists = json.load(f)
        r84_lists = {int(k): v for k, v in r84_lists.items()}
    print(f"  r89 cases: {len(r89_lists)}, r84 cases: {len(r84_lists)}")

    # Per-fold-0 case: compute unique recoveries
    h7_local = [i for i, c in enumerate(val_cases_local) if c.get("n_prior_music") == 7]
    same_artist_h7 = []
    for li in h7_local:
        c = val_cases_local[li]
        gt = c["gt"]
        gt_artist = ta.get(gt, "")
        played_artists = {ta.get(t, "") for t in c.get("music_turns", [])}
        same_artist_h7.append(bool(gt_artist) and gt_artist in played_artists)

    # Per-K hits
    def hit_at(lists, indices, K, source_lookup):
        hit = 0
        for li in indices:
            ci = fold0_idx[li]
            gt = val_cases_local[li]["gt"]
            top = set(t for t, _ in source_lookup[ci][:K])
            if gt in top:
                hit += 1
        return hit / max(1, len(indices))

    all_local = list(range(len(val_cases_local)))
    h7_same = [li for li, s in zip(h7_local, same_artist_h7) if s]
    h7_diff = [li for li, s in zip(h7_local, same_artist_h7) if not s]

    metrics = {
        "r89_h7_hit@20": hit_at(r89_lists, h7_local, 20, r89_lists),
        "r89_h7_hit@30": hit_at(r89_lists, h7_local, 30, r89_lists),
        "r89_h7_hit@300": hit_at(r89_lists, h7_local, 300, r89_lists),
        "r84_h7_hit@20": hit_at(r84_lists, h7_local, 20, r84_lists),
        "r84_h7_hit@30": hit_at(r84_lists, h7_local, 30, r84_lists),
        "r84_h7_hit@300": hit_at(r84_lists, h7_local, 300, r84_lists),
    }
    print(f"\n  R89 h7 hits:  @20={metrics['r89_h7_hit@20']:.3f}  @30={metrics['r89_h7_hit@30']:.3f}  "
          f"@300={metrics['r89_h7_hit@300']:.3f}")
    print(f"  R84 h7 hits:  @20={metrics['r84_h7_hit@20']:.3f}  @30={metrics['r84_h7_hit@30']:.3f}  "
          f"@300={metrics['r84_h7_hit@300']:.3f}")

    # Unique recoveries
    unique_h7_top20 = 0
    unique_h7_top30 = 0
    lost_h7_top30 = 0
    unique_h7_top300 = 0
    unique_h7_top30_same = 0
    unique_h7_top30_diff = 0
    for li, is_same in zip(h7_local, same_artist_h7):
        ci = fold0_idx[li]
        gt = val_cases_local[li]["gt"]
        r89_top20 = set(t for t, _ in r89_lists[ci][:20])
        r89_top30 = set(t for t, _ in r89_lists[ci][:30])
        r89_top300 = set(t for t, _ in r89_lists[ci][:300])
        r84_top20 = set(t for t, _ in r84_lists[ci][:20])
        r84_top30 = set(t for t, _ in r84_lists[ci][:30])
        r84_top300 = set(t for t, _ in r84_lists[ci][:300])
        if gt in r89_top20 and gt not in r84_top20:
            unique_h7_top20 += 1
        if gt in r89_top30 and gt not in r84_top30:
            unique_h7_top30 += 1
            if is_same:
                unique_h7_top30_same += 1
            else:
                unique_h7_top30_diff += 1
        if gt in r84_top30 and gt not in r89_top30:
            lost_h7_top30 += 1
        if gt in r89_top300 and gt not in r84_top300:
            unique_h7_top300 += 1

    print(f"\n  Unique h7 top-20 (R89 surfaces, R84 misses):  {unique_h7_top20}")
    print(f"  Unique h7 top-30 (R89 surfaces, R84 misses):  {unique_h7_top30} "
          f"(same={unique_h7_top30_same}, diff={unique_h7_top30_diff})")
    print(f"  Lost h7 top-30  (R84 had, R89 misses):        {lost_h7_top30}")
    print(f"  Net h7 top-30:                                  {unique_h7_top30 - lost_h7_top30:+d}")
    print(f"  Unique h7 top-300:                              {unique_h7_top300}")

    # nDCG@20 deltas (source-alone vs R84 source-alone)
    r89_ndcgs = {"h7": [], "all": [], "same_artist": [], "diff_artist": []}
    r84_ndcgs = {"h7": [], "all": [], "same_artist": [], "diff_artist": []}
    for li in range(len(val_cases_local)):
        ci = fold0_idx[li]
        gt = val_cases_local[li]["gt"]
        r89_top20 = [t for t, _ in r89_lists[ci][:20]]
        r84_top20 = [t for t, _ in r84_lists[ci][:20]]
        r89_rank = r89_top20.index(gt) + 1 if gt in r89_top20 else -1
        r84_rank = r84_top20.index(gt) + 1 if gt in r84_top20 else -1
        n89 = ndcg_at_k(r89_rank, 20)
        n84 = ndcg_at_k(r84_rank, 20)
        r89_ndcgs["all"].append(n89)
        r84_ndcgs["all"].append(n84)
        if val_cases_local[li].get("n_prior_music") == 7:
            r89_ndcgs["h7"].append(n89)
            r84_ndcgs["h7"].append(n84)
        gt_artist = ta.get(gt, "")
        played_artists = {ta.get(t, "") for t in val_cases_local[li].get("music_turns", [])}
        if gt_artist and gt_artist in played_artists:
            r89_ndcgs["same_artist"].append(n89)
            r84_ndcgs["same_artist"].append(n84)
        else:
            r89_ndcgs["diff_artist"].append(n89)
            r84_ndcgs["diff_artist"].append(n84)

    print(f"\n  source-alone nDCG@20 (R89 vs R84):")
    for k in ["h7", "all", "same_artist", "diff_artist"]:
        m89 = float(np.mean(r89_ndcgs[k]))
        m84 = float(np.mean(r84_ndcgs[k]))
        delta = m89 - m84
        print(f"    {k:14}  R89={m89:.4f}  R84={m84:.4f}  Δ={delta:+.4f}")

    h7_delta = float(np.mean(r89_ndcgs["h7"]) - np.mean(r84_ndcgs["h7"]))
    same_delta = float(np.mean(r89_ndcgs["same_artist"]) - np.mean(r84_ndcgs["same_artist"]))
    diff_delta = float(np.mean(r89_ndcgs["diff_artist"]) - np.mean(r84_ndcgs["diff_artist"]))

    # Gate eval
    cond_A1 = h7_delta >= GATE["h7_delta_ndcg_ge"]
    cond_A2 = (unique_h7_top30 >= GATE["min_unique_h7_top30"]) and (lost_h7_top30 <= unique_h7_top30)
    cond_A3 = (h7_delta >= -0.003) and (unique_h7_top30 >= GATE["ambig_min_unique_h7_top30"])
    canary_same = same_delta >= GATE["same_artist_delta_ge"]
    canary_diff = diff_delta >= GATE["diff_artist_delta_ge"]
    gate_pass = (cond_A1 or cond_A2 or cond_A3) and canary_same and canary_diff
    verdict = "PROCEED_TO_PHASE_0B_LORA" if gate_pass else "ARCHIVE_LEARNED_MM"

    print(f"\n  Gate evaluation:")
    print(f"    A1 (h7 Δ ≥ +0.005):                     {cond_A1}  ({h7_delta:+.4f})")
    print(f"    A2 (≥10 unique h7 top-30 AND rec≥lost): {cond_A2}  "
          f"({unique_h7_top30} unique, {lost_h7_top30} lost)")
    print(f"    A3 (h7 Δ ≥ -0.003 AND ≥5 unique):        {cond_A3}")
    print(f"    Canary same-artist Δ ≥ -0.005:           {canary_same}  ({same_delta:+.4f})")
    print(f"    Canary diff-artist Δ ≥ -0.005:           {canary_diff}  ({diff_delta:+.4f})")
    print(f"\n  VERDICT: {verdict}")

    out = {
        "experiment": "R89 Phase 0 fold-0 — learned multimodal vs R84 OOF",
        "created_at": datetime.now().isoformat(),
        "head_sha": head_sha(),
        "fold": 0, "n_val": len(val_cases_local), "n_h7": n_h7,
        "verdict": verdict,
        "h7_hits_R89_vs_R84": {
            k: metrics[k] for k in metrics
        },
        "unique_recoveries": {
            "h7_top20_r89_only": unique_h7_top20,
            "h7_top30_r89_only": unique_h7_top30,
            "h7_top300_r89_only": unique_h7_top300,
            "h7_top30_lost_to_r84": lost_h7_top30,
            "h7_top30_net": unique_h7_top30 - lost_h7_top30,
            "h7_top30_same_artist": unique_h7_top30_same,
            "h7_top30_diff_artist": unique_h7_top30_diff,
        },
        "ndcg20_source_alone": {
            k: {"r89": float(np.mean(r89_ndcgs[k])),
                 "r84": float(np.mean(r84_ndcgs[k])),
                 "delta": float(np.mean(r89_ndcgs[k]) - np.mean(r84_ndcgs[k]))}
            for k in ["h7", "all", "same_artist", "diff_artist"]
        },
        "gates": {
            "A1_h7_delta_ge_p005": {"value": h7_delta, "pass": cond_A1},
            "A2_unique_ge_10_recov": {"value": [unique_h7_top30, lost_h7_top30], "pass": cond_A2},
            "A3_ambig_positive": {"pass": cond_A3},
            "canary_same": {"value": same_delta, "pass": canary_same},
            "canary_diff": {"value": diff_delta, "pass": canary_diff},
            "overall_pass": gate_pass,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
