#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54 Phase 2 full 5-fold OOF standalone aggregate report.

Reads OOF R54 retrieval lists (assembled from all 5 folds).
Compares against R21 OOF.

Reports:
- hit@20/100/200/300 overall
- h7 hit@200/300
- same/diff h7 hit@300
- unique vs R21 OOF
- lost vs R21 OOF
- Bucket D/E retrieved
- fold-by-fold hit@200
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R54_OOF = REPO / "cache" / "r54" / "phase2_full" / "oof_r54_lists.json"

RRF_K = 20
POOL_K = 300
SW_BASE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def hit_at_k(retrieval, gt, k):
    return gt in set(retrieval[:k])


def main():
    t0 = time.time()
    print(f"{ts()} R54 Phase 2 full 5-fold standalone aggregate")
    print("=" * 70)

    print(f"{ts()} Loading payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)
    ta = payload["track_artist"]

    print(f"{ts()} Loading R21 OOF...")
    r21_lists = json.load(open(R21_OOF))

    print(f"{ts()} Loading R54 OOF...")
    r54_data = json.load(open(R54_OOF))
    r54_with_scores = r54_data["lists"]

    # --- Validation: format consistency, length, score finiteness ---
    print(f"{ts()} Validating OOF list format...")
    assert len(r54_with_scores) == n, f"OOF list length {len(r54_with_scores)} != cases {n}"
    catalog_size_seen = set()
    list_lengths = []
    score_min, score_max = float("inf"), float("-inf")
    for case_idx, case_lists in enumerate(r54_with_scores):
        assert case_lists is not None, f"case {case_idx}: R54 list is None"
        assert len(case_lists) > 0, f"case {case_idx}: R54 list is empty"
        assert isinstance(case_lists, list), f"case {case_idx}: R54 entry is not a list"
        for item_idx, item in enumerate(case_lists):
            assert isinstance(item, (list, tuple)) and len(item) == 2, \
                f"case {case_idx} item {item_idx}: format != (tid, score), got {type(item).__name__}"
            tid, score = item
            assert isinstance(tid, str), f"case {case_idx} item {item_idx}: tid not str"
            assert isinstance(score, (int, float)), f"case {case_idx} item {item_idx}: score not numeric"
            assert np.isfinite(score), f"case {case_idx} item {item_idx}: score not finite ({score})"
            catalog_size_seen.add(tid)
            score_min = min(score_min, score)
            score_max = max(score_max, score)
        list_lengths.append(len(case_lists))
    print(f"  Validation passed: all {n} cases have valid (tid, score) format")
    print(f"  List lengths: min={min(list_lengths)} max={max(list_lengths)} median={int(np.median(list_lengths))}")
    print(f"  Unique tracks across OOF: {len(catalog_size_seen)}")
    print(f"  Score range: [{score_min:.4f}, {score_max:.4f}]")

    r54_lists = [[t for t, _ in case_lists] for case_lists in r54_with_scores]
    n_with_r54 = sum(1 for x in r54_lists if x)
    print(f"  R54 lists: {n_with_r54}/{n} cases")

    # Build folds
    folds = grouped_session_folds(sessions, seed=0)

    # Classify h7 cases
    h7_idx = [i for i in range(n) if cases[i]["n_prior_music"] == 7]
    h7_same = [i for i in h7_idx if ta.get(cases[i]["gt"], "") and
               ta.get(cases[i]["gt"], "") in {ta.get(t, "") for t in cases[i]["music_turns"]}]
    h7_diff = [i for i in h7_idx if i not in set(h7_same)]
    print(f"  h7 cases: {len(h7_idx)}  same={len(h7_same)}  diff={len(h7_diff)}")

    # Build ALS source for union (for bucket classification)
    print(f"{ts()} Building ALS...")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            sc = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    sc[als_track_to_idx[t]] = -np.inf
            top = np.argpartition(-sc, 200)[:200]
            top = top[np.argsort(-sc[top])]
            als_source.append([als_track_ids[j] for j in top])
        else:
            als_source.append([])

    # Bucket classification on h7 cases (R39 baseline pool)
    print(f"{ts()} Classifying buckets on h7...")
    buckets = {}
    for i in h7_idx:
        gt = cases[i]["gt"]
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_lists[i],
        }
        pool = weighted_rrf(src_lists, SW_BASE, topk=POOL_K, k=RRF_K)
        if gt in pool:
            buckets[i] = "in_pool"
        else:
            union = set()
            for sl in src_lists.values():
                union.update(sl[:300])
            if gt in union:
                buckets[i] = "D"
            else:
                buckets[i] = "E"
    bucket_counts = Counter(buckets.values())
    print(f"  Baseline (R39): {dict(bucket_counts)}")

    # ============================================================
    # Aggregate hits — R54 OOF
    # ============================================================
    print(f"\n{ts()} === R54 OOF AGGREGATE ===")
    r54_hits = {"@20": 0, "@100": 0, "@200": 0, "@300": 0}
    r21_hits = {"@20": 0, "@100": 0, "@200": 0, "@300": 0}
    for i in range(n):
        gt = cases[i]["gt"]
        for k_label, k in [("@20", 20), ("@100", 100), ("@200", 200), ("@300", 300)]:
            if hit_at_k(r54_lists[i], gt, k):
                r54_hits[k_label] += 1
            if hit_at_k(r21_lists[i], gt, k):
                r21_hits[k_label] += 1

    print(f"  {'k':<8} {'R54':>10} {'R21':>10} {'Δ':>8}")
    for k in ["@20", "@100", "@200", "@300"]:
        d = r54_hits[k] - r21_hits[k]
        print(f"  hit{k:<6} {r54_hits[k]:>5}/{n} ({r54_hits[k]/n:.4f})  "
              f"{r21_hits[k]:>5}/{n} ({r21_hits[k]/n:.4f})  {d:>+8d}")

    # h7-specific
    print(f"\n{ts()} === h7 HITS ===")
    h7_metrics = {}
    for label, idx_list in [("h7", h7_idx), ("h7_same", h7_same), ("h7_diff", h7_diff)]:
        r54_h = {"@200": 0, "@300": 0}
        r21_h = {"@200": 0, "@300": 0}
        for i in idx_list:
            gt = cases[i]["gt"]
            for k_label, k in [("@200", 200), ("@300", 300)]:
                if hit_at_k(r54_lists[i], gt, k):
                    r54_h[k_label] += 1
                if hit_at_k(r21_lists[i], gt, k):
                    r21_h[k_label] += 1
        n_idx = len(idx_list)
        d200 = r54_h["@200"] - r21_h["@200"]
        d300 = r54_h["@300"] - r21_h["@300"]
        print(f"  {label}: R54 hit@200={r54_h['@200']}/{n_idx} ({r54_h['@200']/max(n_idx,1):.4f})  "
              f"R21={r21_h['@200']}/{n_idx} ({r21_h['@200']/max(n_idx,1):.4f})  Δ={d200:+d}")
        print(f"  {label}: R54 hit@300={r54_h['@300']}/{n_idx} ({r54_h['@300']/max(n_idx,1):.4f})  "
              f"R21={r21_h['@300']}/{n_idx} ({r21_h['@300']/max(n_idx,1):.4f})  Δ={d300:+d}")
        h7_metrics[label] = {"n": n_idx, "r54": r54_h, "r21": r21_h, "d200": d200, "d300": d300}

    # ============================================================
    # Unique / lost vs R21
    # ============================================================
    print(f"\n{ts()} === UNIQUE / LOST vs R21 (@300) ===")
    both = r54_only = r21_only = neither = 0
    for i in range(n):
        gt = cases[i]["gt"]
        in_r54 = hit_at_k(r54_lists[i], gt, 300)
        in_r21 = hit_at_k(r21_lists[i], gt, 300)
        if in_r54 and in_r21:
            both += 1
        elif in_r54:
            r54_only += 1
        elif in_r21:
            r21_only += 1
        else:
            neither += 1
    net = r54_only - r21_only
    print(f"  both={both}  R54-only(unique)={r54_only}  R21-only(lost)={r21_only}  neither={neither}")
    print(f"  net (unique - lost): {net:+d}")

    # ============================================================
    # Bucket D/E recovered by R54 (in top-300)
    # ============================================================
    print(f"\n{ts()} === BUCKET D/E RECOVERY by R54 ===")
    d_total = sum(1 for v in buckets.values() if v == "D")
    e_total = sum(1 for v in buckets.values() if v == "E")
    d_recovered = sum(1 for i, b in buckets.items() if b == "D" and hit_at_k(r54_lists[i], cases[i]["gt"], 300))
    e_recovered = sum(1 for i, b in buckets.items() if b == "E" and hit_at_k(r54_lists[i], cases[i]["gt"], 300))
    print(f"  Bucket D recovered: {d_recovered}/{d_total} ({d_recovered/max(d_total,1):.2%})")
    print(f"  Bucket E recovered: {e_recovered}/{e_total} ({e_recovered/max(e_total,1):.2%})")

    # ============================================================
    # Fold-by-fold hit@200
    # ============================================================
    print(f"\n{ts()} === FOLD-BY-FOLD hit@200 ===")
    fold_hits = {}
    for fi, fold in enumerate(folds):
        idx_list = fold.tolist()
        r54_h = sum(1 for i in idx_list if hit_at_k(r54_lists[i], cases[i]["gt"], 200))
        r21_h = sum(1 for i in idx_list if hit_at_k(r21_lists[i], cases[i]["gt"], 200))
        d = r54_h - r21_h
        print(f"  fold {fi}: R54={r54_h}/{len(idx_list)} ({r54_h/len(idx_list):.4f})  "
              f"R21={r21_h}/{len(idx_list)} ({r21_h/len(idx_list):.4f})  Δ={d:+d}")
        fold_hits[fi] = {"r54": r54_h, "r21": r21_h, "n": len(idx_list), "delta": d}

    # ============================================================
    # Aggregate manifest (consolidates per-fold info)
    # ============================================================
    print(f"\n{ts()} Writing aggregate OOF manifest...")
    manifest_path = REPO / "cache" / "r54" / "phase2_full" / "oof_manifest.json"
    manifest = {
        "experiment": "R54 Phase 2 (structured query)",
        "model": "BAAI/bge-base-en-v1.5",
        "query_format": "structured: [QUERY] current  [HISTORY] last-3 user  [CONTEXT] last-5 played",
        "track_text_format": "r21_exact: '{name} by {artist}. Album: {album}. Tags: {tags[:10]}'",
        "training_hyperparams": {
            "epochs": 2, "batch_size": 32, "lr": 2e-5, "tau": 0.05,
            "max_seq_len": 256, "loss": "in-batch InfoNCE",
            "negatives": "in-batch only (no hard negatives)",
        },
        "data": {
            "train_split": "dev folds excluding val fold",
            "no_train_split_data": True,
            "no_same_session_positives": True,
            "no_enriched_metadata": True,
        },
        "folds": {
            str(fi): {
                "fold": fi,
                "model_dir": str(REPO / "cache" / "r54" / "phase2_full" / f"fold_{fi}" / "model"),
                "lists_path": str(REPO / "cache" / "r54" / "phase2_full" / f"fold_{fi}" / "oof_lists.json"),
                "n_train_cases": 6400,
                "n_val_cases": int(len(folds[fi])),
                "val_indices_sample": folds[fi][:5].tolist(),
                "hit_at_200": fold_hits[fi]["r54"],
                "hit_at_200_rate": fold_hits[fi]["r54"] / fold_hits[fi]["n"],
            }
            for fi in range(5)
        },
        "aggregate": {
            "n_cases": n,
            "hit_at_20": r54_hits["@20"],
            "hit_at_100": r54_hits["@100"],
            "hit_at_200": r54_hits["@200"],
            "hit_at_300": r54_hits["@300"],
        },
        "fold_split": {
            "function": "scripts.expS2_lambdarank_grouped.grouped_session_folds",
            "seed": 0,
            "k": 5,
        },
        "created_at": datetime.now().isoformat(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest: {manifest_path}")

    # Save aggregate metrics
    out = {
        "r54_hits": r54_hits,
        "r21_hits": r21_hits,
        "h7_metrics": h7_metrics,
        "comparison_at_300": {
            "both": both, "r54_only": r54_only, "r21_only": r21_only,
            "neither": neither, "net": net,
        },
        "bucket_recovery": {
            "D_total": d_total, "D_recovered_by_r54": d_recovered,
            "E_total": e_total, "E_recovered_by_r54": e_recovered,
        },
        "baseline_buckets": dict(bucket_counts),
        "fold_hits": fold_hits,
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    out_path = REPO / "exp" / "eval" / "expR54_phase2_full5fold_standalone.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{ts()} Standalone aggregate complete. Elapsed: {time.time() - t0:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
