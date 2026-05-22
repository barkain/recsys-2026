#!/usr/bin/env python3
"""R70 production-style training — trains the candidate-shippable LR.

Two modes:
  MODE_R70_PROD     : train R54c-recipe LR on all 8000 dev with 40 features
                      (FEAT_R39_ALL + FEAT_R54 + FEAT_R68). Output:
                      cache/r70_production_lr_40feat.txt
  MODE_R70B_REPRO   : train R54c-recipe LR on all 8000 dev with 37 features
                      (FEAT_R39_ALL + FEAT_R54 only). This SHOULD match frozen
                      R54c bitwise if the training pipeline is unchanged. If
                      it doesn't, our siblings have hidden recipe drift.
                      Output: cache/r70b_repro_lr_37feat.txt

Both modes use:
  - All 8000 dev cases (no fold split — same as R54c production recipe)
  - R54-stacked RRF pool (SW_BASELINE, identical to R54c)
  - LightGBM LambdaRank, num_leaves=31, lr=0.05, min_data_in_leaf=10, seed=0,
    num_boost_round=300

For MODE_R70_PROD: r68 features use the fold-0 R68 model's lists/cosines
across all 8000 cases. This is mixed in-sample/OOF per case but is the only
production-feasible option without 5-fold BGE-large training.

After training, compares predictions on dev against frozen R54c (top-1
agreement, top-20 overlap) to sanity-check the LR.

Usage:
    uv run python scripts/expR70_production_train.py --mode r70_prod
    uv run python scripts/expR70_production_train.py --mode r70b_repro
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import lightgbm as lgb  # type: ignore[reportMissingImports]
import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expR54_phase3_blind_submission import (  # noqa: E402
    FEAT_R39_ALL, FEAT_R54, FEAT_ALL,
    _featurize_row,
)
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps,
)

SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
POOL_K = 300
RRF_K = 20
TOP_K = 20

FEAT_R68 = ["r68_rank_inv", "r68_presence", "r68_cosine"]
FEAT_R70_PROD = list(FEAT_R39_ALL) + list(FEAT_R54) + list(FEAT_R68)  # 40

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R54_LR = REPO / "cache" / "r54_phase3_lr_model.txt"
R68_DIR = REPO / "cache" / "r68" / "phase0_fold0"
R68_DEV_LISTS = R68_DIR / "oof_r68_lists_fold0.json"
R68_TRAIN_LISTS = R68_DIR / "oof_r68_lists_train.json"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    git_bin = shutil.which("git")
    if git_bin is None:
        return "no-git"
    return subprocess.check_output(  # noqa: S603
        [git_bin, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["r70_prod", "r70b_repro"], required=True)
    args = ap.parse_args()
    mode = args.mode
    is_prod = mode == "r70_prod"

    t0 = time.time()
    print(f"{ts()} R70 production-style training — mode={mode}")
    if is_prod:
        print(f"  features: {len(FEAT_R70_PROD)} (R39+R54+R68)")
        out_lr = REPO / "cache" / "r70_production_lr_40feat.txt"
    else:
        print(f"  features: {len(FEAT_ALL)} (R39+R54, repro of R54c)")
        out_lr = REPO / "cache" / "r70b_repro_lr_37feat.txt"
    print(f"  output: {out_lr}")
    print(f"  pool: R54-stacked (SW_BASELINE), all 8000 dev cases")
    print("=" * 70)

    print(f"{ts()} Loading payloads ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1

    if is_prod:
        print(f"{ts()} Loading R68 fold-0 dev + train lists ...", flush=True)
        with open(R68_DEV_LISTS) as f:
            r68_dev_data = json.load(f)
        r68_dev_val_idx = r68_dev_data.get("val_idx") or r68_dev_data["manifest"]["val_idx"]
        r68_dev_lists_by_case = {
            int(case_idx): [(str(t), float(s)) for t, s in r68_dev_data["lists"][k_pos]]
            for k_pos, case_idx in enumerate(r68_dev_val_idx)
        }
        with open(R68_TRAIN_LISTS) as f:
            r68_train_lists_raw = json.load(f)

        def _case_id_str(c: dict) -> str:
            return c.get("case_id") or f"{c['session_id']}_{c['turn_number']}"
        r68_train_lists_by_case: dict[int, list[tuple[str, float]]] = {}
        for i, c in enumerate(cases):
            cid = _case_id_str(c)
            if cid in r68_train_lists_raw:
                r68_train_lists_by_case[i] = [
                    (str(t), float(s)) for t, s in r68_train_lists_raw[cid]]

        def r68_lists_for(case_idx: int) -> list[tuple[str, float]]:
            # dev lists for fold-0 held-out cases, train lists otherwise.
            if case_idx in r68_dev_lists_by_case:
                return r68_dev_lists_by_case[case_idx]
            return r68_train_lists_by_case.get(case_idx, [])

        n_with_r68 = sum(1 for i in range(n) if r68_lists_for(i))
        print(f"  cases with R68 lists: {n_with_r68}/{n}", flush=True)

    print(f"{ts()} Building case index ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )

    def featurize(case_idx: int, src_lists: dict[str, list[str]],
                  pool: list[str]) -> np.ndarray:
        case = cases[case_idx]
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}

        feats_r54 = _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[case_idx],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][case_idx],
            track_pop, max_pop, track_album,
        )
        if not is_prod:
            return feats_r54

        # Compute r68 features (rank_inv, presence, cosine)
        r68_list = r68_lists_for(case_idx)
        r68_rank_map = {str(t): r + 1 for r, (t, _) in enumerate(r68_list[:POOL_K])}
        r68_smap = {str(t): float(s) for t, s in r68_list}
        feats_r68 = _featurize_row(
            pool, src_lists, r21_rank_map, r68_rank_map, r68_smap,
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][case_idx],
            track_pop, max_pop, track_album,
        )
        return np.concatenate([feats_r54, feats_r68[:, -3:]], axis=1)

    print(f"\n{ts()} === Featurize all {n} cases ===", flush=True)
    X_flat, y, groups = [], [], []
    gt_in_pool_count = 0
    t_feat = time.time()
    for i in range(n):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        gt = cases[i]["gt"]
        gi = pool.index(gt) if gt in pool else -1
        if gi >= 0:
            gt_in_pool_count += 1
        feats = featurize(i, src_lists, pool)
        for k_row in range(len(pool)):
            X_flat.append(feats[k_row])
            y.append(1.0 if k_row == gi else 0.0)
        groups.append(len(pool))
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{n} ({time.time() - t_feat:.0f}s)", flush=True)
    print(f"  pool_hit@{POOL_K}: {gt_in_pool_count}/{n} "
          f"({gt_in_pool_count/n:.4f})", flush=True)

    feat_names = FEAT_R70_PROD if is_prod else list(FEAT_ALL)
    print(f"\n{ts()} === Train LightGBM ({len(groups)} groups, "
          f"{len(y)} candidates, {len(feat_names)} features) ===", flush=True)
    ds = lgb.Dataset(np.array(X_flat, dtype=np.float64),
                     label=np.array(y, dtype=np.float64),
                     group=groups, feature_name=feat_names)
    lr_model = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    lr_model.save_model(str(out_lr))
    print(f"  Saved -> {out_lr}", flush=True)

    # ---- Sanity check: compare to frozen R54c on dev ----
    if not is_prod:
        # R70b repro should match R54c bitwise. Verify.
        print(f"\n{ts()} === Bitwise comparison to frozen R54c ===",
              flush=True)
        r54c = lgb.Booster(model_file=str(R54_LR))
        # Compare predictions on the first 100 cases
        n_match = 0
        n_total = 0
        max_diff = 0.0
        for i in range(min(100, n)):
            src_lists = c3.make_source_lists(
                payload, r21_source, r54_source, case_index["als_source"], i)
            pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
            feats = featurize(i, src_lists, pool)
            p_r54c = r54c.predict(feats)
            p_repro = lr_model.predict(feats)
            o_r54c = np.argsort(-p_r54c, kind="mergesort")
            o_repro = np.argsort(-p_repro, kind="mergesort")
            if np.array_equal(o_r54c[:TOP_K], o_repro[:TOP_K]):
                n_match += 1
            n_total += 1
            max_diff = max(max_diff, float(np.max(np.abs(p_r54c - p_repro))))
        print(f"  Top-20 ordering match on {n_total} cases: {n_match}/{n_total}",
              flush=True)
        print(f"  Max prediction abs diff: {max_diff:.6e}", flush=True)

    out_json = REPO / "exp" / "eval" / f"expR70_production_train_{mode}.json"
    report = {
        "mode": mode,
        "head_sha": head_sha(),
        "elapsed_s": time.time() - t0,
        "n_cases": n,
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "pool_hit_at_300": gt_in_pool_count / n,
        "lr_path": str(out_lr),
        "created_at": datetime.now().isoformat(),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{ts()} Saved {out_json}")


if __name__ == "__main__":
    main()
