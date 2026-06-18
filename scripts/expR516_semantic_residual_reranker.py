#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201
"""R516: semantic residual reranker over deployment-faithful blind-style pools.

R515 showed that rank/source-only residual fusion does not convert.  This test
adds the missing signal class: R21 query-track and history-track semantic
similarities, evaluated all-dev OOF against the production top20.

Candidate pool per dev case:
  production lr_top20 + R480 natural insertion candidates

No GT injection, no selected hit/miss slice.  A Blind-A candidate is justified
only if the best all-dev policy clears a material dNDCG gain with sane churn.
"""
from __future__ import annotations

import gc
import gzip
import json
import math
import os
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.exp_goal65_eval import evaluate, load_dev  # noqa: E402

R12 = REPO / "exp/eval/_R12_all_turns_payload.pkl"
R480 = REPO / "exp/eval/r480_alldev_sim.json.gz"
QUERY_EMBS = REPO / "cache/r33c/query_embs.npy"
TRACK_EMBS = REPO / "cache/r21_production/track_embeddings.npy"
TRACK_IDS = REPO / "cache/r21_production/track_ids.json"
OUT_JSON = REPO / "exp/eval/expR516_semantic_residual_reranker.json"
OUT_MD = REPO / "docs/r516_semantic_residual_reranker.md"

TOP_K = 20
POOL_DEPTHS = [40, 80]
NUM_BOOST_ROUND = 180


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def ndcg_rank(rank: int) -> float:
    return 1.0 / math.log2(rank + 1) if 0 < rank <= TOP_K else 0.0


def unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def load_payload_bits() -> tuple[list[list[str]], dict[str, str]]:
    payload = pickle.load(open(R12, "rb"))
    cases = payload["cases"]
    return [list(c["music_turns"]) for c in cases], payload["track_artist"]


def load_r480_rows() -> list[dict[str, Any]]:
    with gzip.open(R480, "rt") as f:
        rows = json.load(f)
    rows = sorted(rows, key=lambda r: int(r["case_idx"]))
    if len(rows) != 8000:
        raise RuntimeError(f"expected 8000 R480 rows, got {len(rows)}")
    return rows


def rank_features(rank: int, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_rank": float(rank if rank > 0 else 999.0),
        f"{prefix}_rank_inv": float(1.0 / rank if rank > 0 else 0.0),
        f"{prefix}_top1": float(rank == 1),
        f"{prefix}_top3": float(0 < rank <= 3),
        f"{prefix}_top5": float(0 < rank <= 5),
        f"{prefix}_top10": float(0 < rank <= 10),
        f"{prefix}_top20": float(0 < rank <= 20),
        f"{prefix}_top40": float(0 < rank <= 40),
        f"{prefix}_top80": float(0 < rank <= 80),
        f"{prefix}_missing": float(rank <= 0),
    }


FEATURE_NAMES = [
    "base_rank",
    "base_rank_inv",
    "base_top1",
    "base_top3",
    "base_top5",
    "base_top10",
    "base_top20",
    "base_missing",
    "ins_rank",
    "ins_rank_inv",
    "ins_top3",
    "ins_top5",
    "ins_top10",
    "ins_top20",
    "ins_top40",
    "ins_top80",
    "ins_missing",
    "r54_rank",
    "r54_rank_inv",
    "r54_top5",
    "r54_top20",
    "r54_top80",
    "r54_missing",
    "r84_rank",
    "r84_rank_inv",
    "r84_top5",
    "r84_top20",
    "r84_top80",
    "r84_missing",
    "n_sources_present",
    "best_rank",
    "best_rank_inv",
    "rank_spread",
    "not_in_base_any_source",
    "n_prior_music",
    "last_artist_match",
    "any_artist_match",
    "artist_history_count",
    "qcos",
    "qcos_rank_pool",
    "qcos_z_pool",
    "qcos_minus_base1",
    "hist_mean_cos",
    "hist_last_cos",
    "hist_max_cos",
    "hist_mean_minus_qcos",
    "hist_last_minus_qcos",
]


def build_case_pools(dev: dict[str, Any], r480_rows: list[dict[str, Any]], depth: int) -> tuple[list[list[str]], dict[str, float]]:
    pools: list[list[str]] = []
    stats = Counter()
    for i in range(dev["n"]):
        ins = list(r480_rows[i]["cands"])[:depth]
        pool = unique_keep_order(dev["lr_top20"][i] + ins)
        pools.append(pool)
        gt = dev["gt"][i]
        if gt in pool:
            stats["pool_hit"] += 1
        if gt in dev["lr_top20"][i]:
            stats["base_hit"] += 1
        if gt not in dev["lr_top20"][i] and gt in pool:
            stats["new_reachable"] += 1
    return pools, {
        "depth": float(depth),
        "mean_pool_size": float(np.mean([len(p) for p in pools])),
        "pool_hit": stats["pool_hit"] / dev["n"],
        "base_hit": stats["base_hit"] / dev["n"],
        "new_reachable": stats["new_reachable"] / dev["n"],
    }


def _track_idx_map() -> dict[str, int]:
    ids = json.load(open(TRACK_IDS))
    return {tid: i for i, tid in enumerate(ids)}


def _case_semantics(
    case_idx: int,
    pool: list[str],
    played: list[str],
    tid_to_idx: dict[str, int],
    query_embs: np.ndarray,
    track_embs: np.ndarray,
) -> dict[str, dict[str, float]]:
    q = np.asarray(query_embs[case_idx], dtype=np.float32)
    pool_idx = [tid_to_idx.get(t, -1) for t in pool]
    valid = np.array([j >= 0 for j in pool_idx], dtype=bool)
    qcos = np.zeros(len(pool), dtype=np.float32)
    if valid.any():
        emb = track_embs[np.array([j for j in pool_idx if j >= 0], dtype=np.int64)]
        qcos[valid] = emb @ q
    order = np.argsort(-qcos, kind="mergesort")
    qrank = np.empty(len(pool), dtype=np.float32)
    for r, j in enumerate(order, 1):
        qrank[j] = float(r)
    mu = float(qcos.mean()) if len(qcos) else 0.0
    sd = float(qcos.std()) if len(qcos) else 1.0
    if sd < 1e-6:
        sd = 1.0
    base1 = float(qcos[0]) if len(qcos) else 0.0

    hist_ids = [tid_to_idx[t] for t in played if t in tid_to_idx]
    hist_mean = np.zeros_like(qcos)
    hist_last = np.zeros_like(qcos)
    hist_max = np.zeros_like(qcos)
    if hist_ids and valid.any():
        hist_embs = track_embs[np.asarray(hist_ids, dtype=np.int64)]
        pool_embs = track_embs[np.array([j for j in pool_idx if j >= 0], dtype=np.int64)]
        sims = pool_embs @ hist_embs.T
        hist_mean[valid] = sims.mean(axis=1)
        hist_max[valid] = sims.max(axis=1)
        last_idx = hist_ids[-1]
        hist_last[valid] = pool_embs @ track_embs[last_idx]

    out: dict[str, dict[str, float]] = {}
    for j, tid in enumerate(pool):
        out[tid] = {
            "qcos": float(qcos[j]),
            "qcos_rank_pool": float(qrank[j]),
            "qcos_z_pool": float((qcos[j] - mu) / sd),
            "qcos_minus_base1": float(qcos[j] - base1),
            "hist_mean_cos": float(hist_mean[j]),
            "hist_last_cos": float(hist_last[j]),
            "hist_max_cos": float(hist_max[j]),
            "hist_mean_minus_qcos": float(hist_mean[j] - qcos[j]),
            "hist_last_minus_qcos": float(hist_last[j] - qcos[j]),
        }
    return out


def build_matrix(
    dev: dict[str, Any],
    pools: list[list[str]],
    r480_rows: list[dict[str, Any]],
    played: list[list[str]],
    track_artist: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    query_embs = np.load(QUERY_EMBS, mmap_mode="r")
    track_embs = np.load(TRACK_EMBS, mmap_mode="r")
    tid_to_idx = _track_idx_map()

    n_rows = sum(len(p) for p in pools)
    X = np.zeros((n_rows, len(FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros(n_rows, dtype=np.float32)
    starts = np.zeros(len(pools), dtype=np.int64)
    counts = np.zeros(len(pools), dtype=np.int32)
    name_to_idx = {name: j for j, name in enumerate(FEATURE_NAMES)}
    row = 0
    for i, pool in enumerate(pools):
        starts[i] = row
        counts[i] = len(pool)
        gt = dev["gt"][i]
        base_rank = {t: r for r, t in enumerate(dev["lr_top20"][i], 1)}
        ins_rank = {t: r for r, t in enumerate(r480_rows[i]["cands"], 1)}
        r54_rank = {t: r for r, t in enumerate(dev["r54pool"][i], 1)}
        r84_rank = {t: r for r, t in enumerate(dev["r84pool"][i], 1)}
        hist_artists = [track_artist.get(t, "") for t in played[i]]
        hist_counter = Counter(a for a in hist_artists if a)
        last_artist = hist_artists[-1] if hist_artists else ""
        semantic = _case_semantics(i, pool, played[i], tid_to_idx, query_embs, track_embs)
        for tid in pool:
            vals: dict[str, float] = {}
            br = base_rank.get(tid, -1)
            ir = ins_rank.get(tid, -1)
            r54r = r54_rank.get(tid, -1)
            r84r = r84_rank.get(tid, -1)
            for k, v in rank_features(br, "base").items():
                if k in name_to_idx:
                    vals[k] = v
            for k, v in rank_features(ir, "ins").items():
                if k in name_to_idx:
                    vals[k] = v
            for k, v in rank_features(r54r, "r54").items():
                if k in name_to_idx:
                    vals[k] = v
            for k, v in rank_features(r84r, "r84").items():
                if k in name_to_idx:
                    vals[k] = v
            present = [r for r in (br, ir, r54r, r84r) if r > 0]
            best = min(present) if present else 999
            worst = max(present) if present else 999
            vals.update({
                "n_sources_present": float(len(present)),
                "best_rank": float(best),
                "best_rank_inv": float(1.0 / best if best > 0 and best < 999 else 0.0),
                "rank_spread": float(worst - best if present else 0.0),
                "not_in_base_any_source": float(br <= 0 and any(r > 0 for r in (ir, r54r, r84r))),
                "n_prior_music": float(dev["n_prior"][i]),
            })
            artist = track_artist.get(tid, "")
            vals["last_artist_match"] = float(bool(artist and artist == last_artist))
            vals["any_artist_match"] = float(bool(artist and artist in hist_counter))
            vals["artist_history_count"] = float(hist_counter.get(artist, 0)) if artist else 0.0
            vals.update(semantic[tid])
            for name, val in vals.items():
                if name in name_to_idx:
                    X[row, name_to_idx[name]] = val
            if tid == gt:
                y[row] = 1.0
            row += 1
    return X, y, starts, counts


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


def train_oof(dev: dict[str, Any], X: np.ndarray, y: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> tuple[list[np.ndarray], dict[str, Any]]:
    folds = sorted(set(dev["fold"]))
    scores = [np.zeros(int(c), dtype=np.float32) for c in counts]
    importances = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    fold_reports = []
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [20],
        "label_gain": [0, 1],
        "num_leaves": 31,
        "learning_rate": 0.035,
        "min_data_in_leaf": 35,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbose": -1,
        "num_threads": 4,
        "force_col_wise": True,
        "deterministic": True,
        "seed": 516,
    }
    for fold in folds:
        val_cases = [i for i, f in enumerate(dev["fold"]) if f == fold]
        train_cases = [i for i, f in enumerate(dev["fold"]) if f != fold]
        train_idx = case_rows(starts, counts, train_cases)
        val_idx = case_rows(starts, counts, val_cases)
        group_train = [int(counts[i]) for i in train_cases]
        group_val = [int(counts[i]) for i in val_cases]
        print(
            f"{ts()} fold {fold}: train_rows={len(train_idx):,} val_rows={len(val_idx):,} "
            f"train_pos={int(y[train_idx].sum())} val_pos={int(y[val_idx].sum())}",
            flush=True,
        )
        dtrain = lgb.Dataset(X[train_idx], label=y[train_idx], group=group_train, feature_name=FEATURE_NAMES)
        dval = lgb.Dataset(X[val_idx], label=y[val_idx], group=group_val, reference=dtrain, feature_name=FEATURE_NAMES)
        model = lgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND, valid_sets=[dval], callbacks=[lgb.log_evaluation(0)])
        pred = model.predict(X[val_idx]).astype(np.float32)
        importances += model.feature_importance(importance_type="gain")
        offset = 0
        fold_nd = []
        for case_idx in val_cases:
            count = int(counts[case_idx])
            case_pred = pred[offset:offset + count].copy()
            offset += count
            scores[case_idx] = case_pred
            order = np.argsort(-case_pred, kind="mergesort")
            gt_local = -1
            for j, local in enumerate(order[:TOP_K], 1):
                row_idx = int(starts[case_idx] + local)
                if y[row_idx] > 0:
                    gt_local = j
                    break
            fold_nd.append(ndcg_rank(gt_local))
        fold_reports.append({"fold": int(fold), "model_nDCG@20": float(np.mean(fold_nd))})
        del dtrain, dval, model, pred, train_idx, val_idx
        gc.collect()
    return scores, {
        "params": params,
        "num_boost_round": NUM_BOOST_ROUND,
        "fold_reports": fold_reports,
        "feature_importance": {
            name: float(val / max(len(folds), 1))
            for name, val in sorted(zip(FEATURE_NAMES, importances, strict=True), key=lambda x: -x[1])
        },
    }


def model_rankings(pools: list[list[str]], scores: list[np.ndarray], base: list[list[str]], keep_top1: bool) -> list[list[str]]:
    out: list[list[str]] = []
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
    out: list[list[str]] = []
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


def run_depth(
    dev: dict[str, Any],
    r480_rows: list[dict[str, Any]],
    played: list[list[str]],
    track_artist: dict[str, str],
    depth: int,
) -> dict[str, Any]:
    print(f"{ts()} depth={depth}: building pools/features", flush=True)
    pools, pool_stats = build_case_pools(dev, r480_rows, depth)
    X, y, starts, counts = build_matrix(dev, pools, r480_rows, played, track_artist)
    print(f"{ts()} depth={depth}: matrix {X.shape}, positives={int(y.sum())}", flush=True)
    scores, train_report = train_oof(dev, X, y, starts, counts)
    policies = []
    for name, rankings in [
        ("pure_model", model_rankings(pools, scores, dev["lr_top20"], keep_top1=False)),
        ("keep_top1_model", model_rankings(pools, scores, dev["lr_top20"], keep_top1=True)),
    ]:
        policies.append({"policy": name, **evaluate(dev, rankings)})
    for keep in [False, True]:
        for bw in [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.5, 5.0, 8.0]:
            rankings = blended_rankings(pools, scores, dev["lr_top20"], base_weight=bw, keep_top1=keep)
            policies.append({"policy": f"blend_bw{bw:g}_keep{int(keep)}", "base_weight": bw, "keep_top1": keep, **evaluate(dev, rankings)})
    policies.sort(key=lambda r: r["dNDCG_all"], reverse=True)
    best = policies[0]
    print(
        f"{ts()} depth={depth}: best {best['policy']} dNDCG={best['dNDCG_all']:.5f} "
        f"nDCG={best['nDCG@20']:.5f} churn80={best['churn_top1_per80']} overlap={best['overlap@20']}",
        flush=True,
    )
    del X, y, starts, counts, scores
    gc.collect()
    return {"depth": depth, "pool_stats": pool_stats, "train_report": train_report, "policies": policies[:16]}


def main() -> None:
    t0 = time.time()
    print(f"{ts()} R516 semantic residual reranker", flush=True)
    dev = load_dev()
    r480_rows = load_r480_rows()
    played, track_artist = load_payload_bits()
    base_metrics = evaluate(dev, dev["lr_top20"])
    results = []
    for depth in POOL_DEPTHS:
        results.append(run_depth(dev, r480_rows, played, track_artist, depth))
    best_rows = []
    for result in results:
        for policy in result["policies"]:
            best_rows.append({"depth": result["depth"], **policy})
    best_rows.sort(key=lambda r: r["dNDCG_all"], reverse=True)
    best = best_rows[0]
    verdict = "GO" if best["dNDCG_all"] >= 0.010 and best["all_gates_pass"] else "NO_GO"
    out = {
        "experiment": "R516 semantic residual reranker",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "base_metrics": base_metrics,
        "best_overall": best_rows[:24],
        "depth_results": results,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    lines = [
        "# R516 - Semantic Residual Reranker",
        "",
        f"**Date:** {datetime.now():%Y-%m-%d}",
        f"**Verdict:** **{verdict}**",
        "",
        "## Best Offline Result",
        "",
        f"- depth: `{best['depth']}`",
        f"- policy: `{best['policy']}`",
        f"- nDCG@20: `{best['nDCG@20']:.6f}` vs base `{best['base_nDCG@20']:.6f}`",
        f"- dNDCG: `{best['dNDCG_all']:.6f}`",
        f"- same/diff/h7 delta: `{best['dNDCG_same']:.6f}` / `{best['dNDCG_diff']:.6f}` / `{best['dNDCG_h7']:.6f}`",
        f"- churn top1 per 80: `{best['churn_top1_per80']}`",
        f"- overlap@20: `{best['overlap@20']}`",
        "",
        "## Interpretation",
        "",
        "This is an all-dev OOF test over production top20 plus natural R480 insertion candidates, "
        "with R21 query-track and history-track semantic features. It is deployment-faithful: no GT "
        "injection and no miss-only selection. A blind build requires roughly +0.010 dNDCG and sane "
        "churn/overlap; otherwise this path is not strong enough for the 0.55 target.",
        "",
        f"Full JSON: `{OUT_JSON}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"{ts()} wrote {OUT_JSON} and {OUT_MD}", flush=True)
    print(f"{ts()} verdict={verdict} best_depth={best['depth']} best_policy={best['policy']} dNDCG={best['dNDCG_all']:.6f}", flush=True)


if __name__ == "__main__":
    main()
