#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201
"""R517: learned diagonal semantic metric over R516 deployment pools.

R516 proved semantic scalar features help but only by ~+0.0048 dNDCG.  This
script tests a stronger-but-still-CPU-safe model: a linear classifier over
elementwise query*track embedding interactions plus the R516 scalar features.

This is effectively a learned diagonal bilinear retriever/reranker, evaluated
OOF on all dev cases.  It is still deployment-faithful:
  - production top20 + natural R480 insertion candidates only
  - no GT injection
  - all 8000 rows, not a miss-only slice
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sklearn.linear_model import SGDClassifier  # type: ignore[reportMissingImports]
from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_goal65_eval import evaluate, load_dev  # noqa: E402
from scripts.expR516_semantic_residual_reranker import (  # noqa: E402
    FEATURE_NAMES,
    QUERY_EMBS,
    TRACK_EMBS,
    TRACK_IDS,
    build_case_pools,
    build_matrix,
    load_payload_bits,
    load_r480_rows,
)

OUT_JSON = REPO / "exp/eval/expR517_diagonal_semantic_sgd.json"
OUT_MD = REPO / "docs/r517_diagonal_semantic_sgd.md"

TOP_K = 20
DEPTH = 80
BATCH = 8192
EPOCHS = 5
BASE_WEIGHTS = [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.5, 5.0, 8.0]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def load_tid_to_idx() -> dict[str, int]:
    ids = json.load(open(TRACK_IDS))
    return {tid: i for i, tid in enumerate(ids)}


def row_maps(pools: list[list[str]], tid_to_idx: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    n_rows = sum(len(p) for p in pools)
    row_case = np.empty(n_rows, dtype=np.int32)
    row_tid_idx = np.empty(n_rows, dtype=np.int32)
    pos = 0
    for ci, pool in enumerate(pools):
        for tid in pool:
            row_case[pos] = ci
            row_tid_idx[pos] = tid_to_idx.get(tid, -1)
            pos += 1
    return row_case, row_tid_idx


def case_rows(starts: np.ndarray, counts: np.ndarray, cases: list[int]) -> np.ndarray:
    total = sum(int(counts[i]) for i in cases)
    out = np.empty(total, dtype=np.int64)
    pos = 0
    for i in cases:
        start = int(starts[i])
        count = int(counts[i])
        out[pos:pos + count] = np.arange(start, start + count, dtype=np.int64)
        pos += count
    return out


def make_features(
    rows: np.ndarray,
    X_scalar: np.ndarray,
    row_case: np.ndarray,
    row_tid_idx: np.ndarray,
    query_embs: np.ndarray,
    track_embs: np.ndarray,
) -> np.ndarray:
    cases = row_case[rows]
    tids = row_tid_idx[rows]
    valid = tids >= 0
    prod = np.zeros((len(rows), query_embs.shape[1]), dtype=np.float32)
    if valid.any():
        prod[valid] = query_embs[cases[valid]] * track_embs[tids[valid]]
    return np.hstack([prod, X_scalar[rows]]).astype(np.float32, copy=False)


def train_oof_sgd(
    dev: dict[str, Any],
    pools: list[list[str]],
    X_scalar: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    tid_to_idx = load_tid_to_idx()
    row_case, row_tid_idx = row_maps(pools, tid_to_idx)
    query_embs = np.load(QUERY_EMBS, mmap_mode="r")
    track_embs = np.load(TRACK_EMBS, mmap_mode="r")
    scores = [np.zeros(int(c), dtype=np.float32) for c in counts]
    fold_reports = []
    classes = np.asarray([0.0, 1.0], dtype=np.float32)

    for fold in sorted(set(dev["fold"])):
        val_cases = [i for i, f in enumerate(dev["fold"]) if f == fold]
        train_cases = [i for i, f in enumerate(dev["fold"]) if f != fold]
        train_idx = case_rows(starts, counts, train_cases)
        val_idx = case_rows(starts, counts, val_cases)
        pos = train_idx[y[train_idx] > 0.5]
        neg = train_idx[y[train_idx] <= 0.5]
        print(
            f"{ts()} fold {fold}: train_rows={len(train_idx):,} pos={len(pos):,} "
            f"val_rows={len(val_idx):,}",
            flush=True,
        )

        scaler = StandardScaler(copy=False)
        # Fit scaler on scalar features only; product features are already tiny normalized values.
        scaler.fit(X_scalar[train_idx])

        clf = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            alpha=2e-5,
            l1_ratio=0.05,
            learning_rate="optimal",
            average=True,
            random_state=517 + int(fold),
            n_jobs=1,
        )
        rng = np.random.default_rng(517 + int(fold))
        for epoch in range(EPOCHS):
            # Use all positives and a fresh 20x negative sample each epoch.  This
            # focuses learning on discrimination while avoiding a 1:180 class skew.
            neg_sample = rng.choice(neg, size=min(len(neg), max(len(pos) * 20, 60000)), replace=False)
            epoch_idx = np.concatenate([pos, neg_sample])
            rng.shuffle(epoch_idx)
            for off in range(0, len(epoch_idx), BATCH):
                rows = epoch_idx[off:off + BATCH]
                Xb = make_features(rows, X_scalar, row_case, row_tid_idx, query_embs, track_embs)
                Xb[:, 768:] = scaler.transform(Xb[:, 768:])
                yb = y[rows]
                sample_weight = np.where(yb > 0.5, 20.0, 1.0).astype(np.float32)
                if epoch == 0 and off == 0:
                    clf.partial_fit(Xb, yb, classes=classes, sample_weight=sample_weight)
                else:
                    clf.partial_fit(Xb, yb, sample_weight=sample_weight)
            print(f"{ts()} fold {fold}: epoch {epoch + 1}/{EPOCHS}", flush=True)

        pred_all = np.empty(len(val_idx), dtype=np.float32)
        for off in range(0, len(val_idx), BATCH):
            rows = val_idx[off:off + BATCH]
            Xb = make_features(rows, X_scalar, row_case, row_tid_idx, query_embs, track_embs)
            Xb[:, 768:] = scaler.transform(Xb[:, 768:])
            pred_all[off:off + len(rows)] = clf.decision_function(Xb).astype(np.float32)

        offset = 0
        fold_hits20 = 0
        for case_idx in val_cases:
            count = int(counts[case_idx])
            scores[case_idx] = pred_all[offset:offset + count].copy()
            order = np.argsort(-scores[case_idx], kind="mergesort")
            gt_rank = -1
            start = int(starts[case_idx])
            for rank, local in enumerate(order[:TOP_K], 1):
                if y[start + int(local)] > 0.5:
                    gt_rank = rank
                    break
            fold_hits20 += gt_rank > 0
            offset += count
        fold_reports.append({"fold": int(fold), "hit20_model": fold_hits20 / max(len(val_cases), 1)})
        del train_idx, val_idx, pos, neg, scaler, clf, pred_all
        gc.collect()

    return scores, {
        "epochs": EPOCHS,
        "batch": BATCH,
        "feature_names": [f"prod_{i}" for i in range(768)] + FEATURE_NAMES,
        "fold_reports": fold_reports,
    }


def model_rankings(pools: list[list[str]], scores: list[np.ndarray], base: list[list[str]], keep_top1: bool) -> list[list[str]]:
    out = []
    for pool, sc, b in zip(pools, scores, base, strict=True):
        ranked = [pool[int(j)] for j in np.argsort(-sc, kind="mergesort")]
        if keep_top1 and b:
            ranked = [b[0]] + [t for t in ranked if t != b[0]]
        out.append(ranked[:TOP_K])
    return out


def blended_rankings(
    pools: list[list[str]],
    scores: list[np.ndarray],
    base: list[list[str]],
    base_weight: float,
    keep_top1: bool,
) -> list[list[str]]:
    out = []
    for pool, sc, b in zip(pools, scores, base, strict=True):
        br = {t: r for r, t in enumerate(b, 1)}
        vals = np.asarray([
            float(sc[j]) + (base_weight / br[tid] if tid in br else 0.0)
            for j, tid in enumerate(pool)
        ], dtype=np.float32)
        ranked = [pool[int(j)] for j in np.argsort(-vals, kind="mergesort")]
        if keep_top1 and b:
            ranked = [b[0]] + [t for t in ranked if t != b[0]]
        out.append(ranked[:TOP_K])
    return out


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R517 diagonal semantic SGD", flush=True)
    dev = load_dev()
    r480_rows = load_r480_rows()
    played, track_artist = load_payload_bits()
    pools, pool_stats = build_case_pools(dev, r480_rows, DEPTH)
    X_scalar, y, starts, counts = build_matrix(dev, pools, r480_rows, played, track_artist)
    print(f"{ts()} matrix scalar={X_scalar.shape} positives={int(y.sum())}", flush=True)
    scores, train_report = train_oof_sgd(dev, pools, X_scalar, y, starts, counts)

    policies = []
    for name, rankings in [
        ("pure_model", model_rankings(pools, scores, dev["lr_top20"], keep_top1=False)),
        ("keep_top1_model", model_rankings(pools, scores, dev["lr_top20"], keep_top1=True)),
    ]:
        policies.append({"policy": name, **evaluate(dev, rankings)})
    for keep in [False, True]:
        for bw in BASE_WEIGHTS:
            rankings = blended_rankings(pools, scores, dev["lr_top20"], bw, keep)
            policies.append({"policy": f"blend_bw{bw:g}_keep{int(keep)}", "base_weight": bw, "keep_top1": keep, **evaluate(dev, rankings)})
    policies.sort(key=lambda r: r["dNDCG_all"], reverse=True)
    best = policies[0]
    verdict = "GO" if best["dNDCG_all"] >= 0.010 and best["all_gates_pass"] else "NO_GO"
    out = {
        "experiment": "R517 diagonal semantic SGD",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "depth": DEPTH,
        "pool_stats": pool_stats,
        "train_report": train_report,
        "policies": policies[:24],
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    lines = [
        "# R517 - Diagonal Semantic SGD",
        "",
        f"**Date:** {datetime.now():%Y-%m-%d}",
        f"**Verdict:** **{verdict}**",
        "",
        "## Best Offline Result",
        "",
        f"- policy: `{best['policy']}`",
        f"- nDCG@20: `{best['nDCG@20']:.6f}` vs base `{best['base_nDCG@20']:.6f}`",
        f"- dNDCG: `{best['dNDCG_all']:.6f}`",
        f"- same/diff/h7 delta: `{best['dNDCG_same']:.6f}` / `{best['dNDCG_diff']:.6f}` / `{best['dNDCG_h7']:.6f}`",
        f"- churn top1 per 80: `{best['churn_top1_per80']}`",
        f"- overlap@20: `{best['overlap@20']}`",
        "",
        "## Interpretation",
        "",
        "This tests whether a learned diagonal semantic metric over q*track embeddings "
        "can convert beyond R516's scalar cosine features. It remains all-dev OOF and "
        "deployment-faithful; no Blind-A submission is justified unless it clears the "
        "+0.010 dNDCG gate.",
        "",
        f"Full JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"{ts()} wrote {OUT_JSON} and {OUT_MD}", flush=True)
    print(f"{ts()} verdict={verdict} best_policy={best['policy']} dNDCG={best['dNDCG_all']:.6f}", flush=True)


if __name__ == "__main__":
    main()
