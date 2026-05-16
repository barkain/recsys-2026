#!/usr/bin/env python3
# ruff: noqa: E402,T201,S301
"""R55 post-LR top-20 comparison vs R54b submission (cache-backed, resumable).

Uses the prebuilt blind source cache (from expR55_blind_source_cache.py) so
the slow retriever path (BM25, CFBPR, track-sim, A', ALS, R21) runs zero
times here. Substitutes R55 in the R54 slot, applies the existing LR model,
writes per-session top-20 to disk as we go so a stall can be resumed.

This is the correct churn metric: post-LR R55 top-1/top-20 vs R54b post-LR
top-1/top-20, both produced by the same LR/feature pipeline.

Inputs:
  cache/blind_a/source_cache.pkl              (prebuilt source cache; run
                                                expR55_blind_source_cache.py first)
  cache/r55_production/blind_r55_lists.json
  cache/r54_phase3_lr_model.txt
  cache/r54_phase3_track_pop.json
  cache/r54_phase3_payload_maps.pkl
  cache/r54_phase3_als.npz                    (only for factors used in feats)
  exp/inference/blind_a/r54b_aligned_submission.json

Outputs:
  cache/r55_compare/per_session/<sid>.json    (incremental, resumable)
  cache/r55_compare/r55_top20.json            (consolidated)
  exp/eval/expR55_post_lr_compare.json        (churn report)

Decision gates (same as before):
  top-1 changed vs R54b: < 25 / 80 soft, > 35 / 80 hard stop
  top-20 overlap median: >= 14 / 20
Exit codes: 0 PASS / 1 SOFT_FAIL / 2 HARD_FAIL.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expR54_phase3_blind_submission import (  # noqa: E402
    POOL_K,
    RRF_K,
    SW,
    _featurize_row,
    load_track_albums,
)

SRC_CACHE = REPO / "cache" / "blind_a" / "source_cache.pkl"
R55_BLIND_LISTS = REPO / "cache" / "r55_production" / "blind_r55_lists.json"
R54B_SUBMISSION = REPO / "exp" / "inference" / "blind_a" / "r54b_aligned_submission.json"
LR_MODEL_PATH = REPO / "cache" / "r54_phase3_lr_model.txt"
ALS_CACHE = REPO / "cache" / "r54_phase3_als.npz"
POP_CACHE = REPO / "cache" / "r54_phase3_track_pop.json"
MAPS_CACHE = REPO / "cache" / "r54_phase3_payload_maps.pkl"

PER_SESSION_DIR = REPO / "cache" / "r55_compare" / "per_session"
CONSOLIDATED = REPO / "cache" / "r55_compare" / "r55_top20.json"
REPORT_OUT = REPO / "exp" / "eval" / "expR55_post_lr_compare.json"

SOFT_TOP1 = 25
HARD_TOP1 = 35
TOP20_OVERLAP_MIN = 14


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def percentile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return float(s[f])
    return float(s[f] + (s[c] - s[f]) * (k - f))


def rank_one(rec, variant_list, variant_score_map, ranker, valid_catalog,
             ta, tt, ttl, tat, tmt, als_factors, als_to_idx,
             track_pop, max_pop, track_album):
    variant_rank_map = {tid: r + 1 for r, tid in enumerate(variant_list)}
    als_vec = np.array(rec["als_vec"], dtype=np.float32) if rec["als_vec"] is not None else None

    src_lists = {
        "A": rec["src_a"], "B": rec["src_b"], "C": rec["src_c"],
        "D": rec["src_d"], "F": rec["src_f"],
        "ALS": rec["als_tracks"], "R21": rec["r21_list"], "R54": variant_list,
    }
    pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
    feats = _featurize_row(
        pool, src_lists, rec["r21_rank_map"], variant_rank_map, variant_score_map,
        rec["user_query"], rec["history"], rec["music_turns"], set(rec["music_turns"]),
        ta, tt, ttl, tat, tmt,
        als_factors, als_to_idx, als_vec, track_pop, max_pop, track_album,
    )
    scores = ranker.predict(feats)
    ranked = np.argsort(-scores)
    top20 = []
    seen = set()
    for j in ranked:
        tid = pool[int(j)]
        if tid in valid_catalog and tid not in seen:
            top20.append(tid)
            seen.add(tid)
            if len(top20) >= 20:
                break
    return top20, pool


def compute_r55_top20(force=False):
    import lightgbm as lgb  # type: ignore[reportMissingImports]

    if not SRC_CACHE.exists():
        print(f"FAIL: source cache {SRC_CACHE} not found. "
              f"Run expR55_blind_source_cache.py first.")
        sys.exit(3)
    if not R55_BLIND_LISTS.exists():
        print(f"FAIL: {R55_BLIND_LISTS} not found.")
        sys.exit(3)

    print(f"{ts()} Loading source cache...")
    with open(SRC_CACHE, "rb") as f:
        src_cache = pickle.load(f)
    print(f"  {len(src_cache)} sessions")

    print(f"{ts()} Loading R55 blind lists...")
    with open(R55_BLIND_LISTS) as f:
        r55_blind = json.load(f)["lists"]
    print(f"  {len(r55_blind)} sessions")

    print(f"{ts()} Loading LR model + supporting caches...")
    lr_model = lgb.Booster(model_file=str(LR_MODEL_PATH))
    track_album = load_track_albums()
    valid_catalog = set(track_album.keys())
    with open(POP_CACHE) as f:
        track_pop = json.load(f)
    als_data = np.load(ALS_CACHE, allow_pickle=True)
    als_factors = als_data["factors"]
    als_ids = als_data["track_ids"].tolist()
    als_to_idx = {tid: i for i, tid in enumerate(als_ids)}
    with open(MAPS_CACHE, "rb") as f:
        maps = pickle.load(f)
    ta = maps["track_artist"]
    tt = maps["track_tags"]
    ttl = maps["track_title_toks"]
    tat = maps["track_artist_toks"]
    tmt = maps["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    PER_SESSION_DIR.mkdir(parents=True, exist_ok=True)
    top20_by_sid = {}

    # Resumable: skip sessions whose per-session JSON already exists
    sids = sorted(src_cache.keys())
    t0 = time.time()
    pending = []
    for sid in sids:
        path = PER_SESSION_DIR / f"{sid}.json"
        if not force and path.exists():
            with open(path) as f:
                top20_by_sid[sid] = json.load(f)["top20"]
        else:
            pending.append(sid)
    print(f"  Pending: {len(pending)}/{len(sids)}  (force={force})")

    for i, sid in enumerate(pending):
        rec = src_cache[sid]
        if sid not in r55_blind:
            print(f"  WARN: sid={sid} missing from R55 blind lists; skipping")
            continue
        r55_pairs = r55_blind[sid]
        r55_list = [t for t, _ in r55_pairs[:300]]
        r55_score_map = {t: float(s) for t, s in r55_pairs}

        top20, _pool = rank_one(rec, r55_list, r55_score_map, lr_model, valid_catalog,
                                  ta, tt, ttl, tat, tmt,
                                  als_factors, als_to_idx,
                                  track_pop, max_pop, track_album)
        top20_by_sid[sid] = top20

        # Per-session write
        tmp_path = PER_SESSION_DIR / f"{sid}.json.tmp"
        final_path = PER_SESSION_DIR / f"{sid}.json"
        with open(tmp_path, "w") as f:
            json.dump({"session_id": sid, "top20": top20}, f)
        tmp_path.rename(final_path)

        if (i + 1) % 10 == 0 or i == len(pending) - 1:
            elapsed = time.time() - t0
            eta = (len(pending) - i - 1) * elapsed / max(i + 1, 1)
            print(f"  [{i + 1}/{len(pending)}]  elapsed={elapsed:.1f}s  eta={eta:.1f}s",
                  flush=True)

    # Consolidate
    CONSOLIDATED.parent.mkdir(parents=True, exist_ok=True)
    with open(CONSOLIDATED, "w") as f:
        json.dump(top20_by_sid, f, indent=2)
    print(f"  Consolidated: {CONSOLIDATED}")
    return top20_by_sid


def compare(top20_by_sid):
    print(f"\n{ts()} Comparing R55 post-LR top-20 vs R54b submission")
    with open(R54B_SUBMISSION) as f:
        r54b = json.load(f)
    r54b_by_sid = {r["session_id"]: r["predicted_track_ids"] for r in r54b}

    common = set(top20_by_sid) & set(r54b_by_sid)
    print(f"  Common: {len(common)}/{max(len(top20_by_sid), len(r54b_by_sid))}")

    n = len(common)
    top1_change = 0
    top20_overlap = []
    r55_top1_in_r54b_top20 = 0
    r54b_top1_in_r55_top20 = 0
    top20_exact_same = 0
    changed_sids = []

    for sid in sorted(common):
        r55_t = top20_by_sid[sid]
        r54b_t = r54b_by_sid[sid]
        if r55_t[0] != r54b_t[0]:
            top1_change += 1
            changed_sids.append(sid)
        top20_overlap.append(len(set(r55_t) & set(r54b_t)))
        if r55_t[0] in r54b_t:
            r55_top1_in_r54b_top20 += 1
        if r54b_t[0] in r55_t:
            r54b_top1_in_r55_top20 += 1
        if r55_t == r54b_t:
            top20_exact_same += 1

    top20_median = percentile(top20_overlap, 0.5)
    top20_mean = sum(top20_overlap) / n if n else 0

    print(f"\n=== R55 post-LR vs R54b submission (n={n}) ===")
    print(f"  top-1 changed:                {top1_change}/{n}")
    print(f"  top-20 exact same:            {top20_exact_same}/{n}")
    print(f"  top-20 overlap:               mean={top20_mean:.2f}  median={top20_median:.1f}  "
          f"min={min(top20_overlap)}  max={max(top20_overlap)}")
    print(f"  R55 top-1 is in R54b top-20:  {r55_top1_in_r54b_top20}/{n}")
    print(f"  R54b top-1 is in R55 top-20:  {r54b_top1_in_r55_top20}/{n}")

    # Track ID validity
    track_album = load_track_albums()
    valid = set(track_album.keys())
    invalid = 0
    dup = 0
    for sid, t20 in top20_by_sid.items():
        if len(t20) != 20:
            invalid += 1
        if any(tid not in valid for tid in t20):
            invalid += 1
        if len(set(t20)) != 20:
            dup += 1
    print(f"  Track ID validation:          {n - invalid}/{n} valid (20 unique IDs), "
          f"{dup} with duplicates")

    print(f"\n=== Submission gates ===")
    print(f"  top-1 churn:           {top1_change}/{n}  "
          f"(soft <{SOFT_TOP1}, hard >{HARD_TOP1})")
    print(f"  top-20 overlap median: {top20_median:.1f}/20  (require >= {TOP20_OVERLAP_MIN})")

    status = "PASS"
    reasons = []
    if top1_change > HARD_TOP1:
        status = "HARD_FAIL"
        reasons.append(f"top-1 churn {top1_change} > hard stop {HARD_TOP1}")
    elif top1_change >= SOFT_TOP1:
        status = "SOFT_FAIL"
        reasons.append(f"top-1 churn {top1_change} >= soft threshold {SOFT_TOP1}")
    if top20_median < TOP20_OVERLAP_MIN:
        if status == "PASS":
            status = "SOFT_FAIL"
        reasons.append(f"top-20 overlap median {top20_median:.1f} < {TOP20_OVERLAP_MIN}")

    print(f"\n  GATE STATUS: {status}")
    for r in reasons:
        print(f"    - {r}")

    report = {
        "comparison": "R55 post-LR top-20 vs R54b submission",
        "n_sessions": n,
        "top1_change": top1_change,
        "top20_exact_same": top20_exact_same,
        "changed_sids": changed_sids,
        "top20_overlap_mean": top20_mean,
        "top20_overlap_median": top20_median,
        "top20_overlap_min": min(top20_overlap),
        "top20_overlap_max": max(top20_overlap),
        "r55_top1_in_r54b_top20": r55_top1_in_r54b_top20,
        "r54b_top1_in_r55_top20": r54b_top1_in_r55_top20,
        "invalid_rows": invalid,
        "duplicate_rows": dup,
        "gates": {
            "status": status, "fail_reasons": reasons,
            "soft_top1": SOFT_TOP1, "hard_top1": HARD_TOP1,
            "top20_overlap_min": TOP20_OVERLAP_MIN,
        },
        "created_at": datetime.now().isoformat(),
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report: {REPORT_OUT}")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="Rebuild all per-session R55 top-20 caches.")
    args = ap.parse_args()
    t0 = time.time()
    top20 = compute_r55_top20(force=args.force)
    print(f"\n  R55 top-20 build elapsed: {time.time() - t0:.1f}s")
    status = compare(top20)
    if status == "HARD_FAIL":
        sys.exit(2)
    if status == "SOFT_FAIL":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
