"""R84c Blind-A candidate builder.

Pipeline:
1. Load 5-fold R84 blind retrieval lists, ensemble into single top-300
   per blind sid (avg cosine where present, 0 elsewhere).
2. Train production R84-LR on all 8000 dev cases (analog of frozen R54c,
   but with R84 features substituted).
3. For each of 80 Blind-A cases:
   - Build R54-stacked RRF pool (= R78/R54c production pool).
   - Score with frozen R54c LR → R54-rank.
   - Compute R54c LR top-1 margin.
   - Score with production R84-LR (R84 features substituted).
   - Route: use R84-LR top-20 if margin < 0.5 OR margin >= 2.0, else R54c top-20.
4. Audit vs R78 blind submission:
   - top-1 churn /80
   - top-20 overlap mean
   - sessions changed (top-1 + any 6+ swaps)
5. Hard stop if churn > 35/80 OR overlap < 14/20.
6. If passes, write blind candidate JSON + report; do NOT auto-submit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
RRF_K = 20
POOL_K = 300
TOP_K = 20
N_FOLDS = 5

# R84c routing rule (raw thresholds from R84c margin transfer)
ROUTE_LOW = 0.5
ROUTE_HIGH = 2.0

# Churn audit constraints (hard stops)
CHURN_MAX_PER_80 = 35
OVERLAP_MIN_PER_20 = 14.0

BLIND_SRC = REPO / "cache" / "blind_a" / "source_cache.pkl"
R54_LR_PATH = REPO / "cache" / "r54_phase3_lr_model.txt"
R54_BLIND_LISTS = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
R78_SUB = REPO / "exp" / "inference" / "blind_a" / "r78_llm_polish_submission.zip"
FEAT_CACHE = REPO / "cache" / "r84b" / "case_features.pkl"

OUT_DIR = REPO / "exp" / "inference" / "blind_a"
OUT_R84C_TRACKS_JSON = OUT_DIR / "r84c_blind_track_lists.json"
OUT_R84C_AUDIT = REPO / "exp" / "eval" / "expR84c_blind_audit.json"
OUT_PROD_LR_PATH = REPO / "cache" / "r84c_production_lr.txt"
OUT_ENSEMBLE_PATH = REPO / "cache" / "r84_production" / "blind_r84_ensemble_lists.json"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R84 = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    g = shutil.which("git")
    if g is None:
        return "no-git"
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def ensemble_5fold_blind() -> dict[str, list[tuple[str, float]]]:
    """Average cosine across 5 folds where present, sort, take top-300 per sid."""
    print(f"{ts()} Loading 5 fold blind lists and ensembling...", flush=True)
    fold_lists = []
    for fold in range(N_FOLDS):
        path = REPO / f"cache/r84/blind_fold{fold}/blind_r84_lists.json"
        with open(path) as f:
            fold_lists.append(json.load(f))
    all_sids = set()
    for fl in fold_lists:
        all_sids.update(fl.keys())
    print(f"  unique blind sids: {len(all_sids)}")

    ensemble = {}
    union_sizes = []
    for sid in sorted(all_sids):
        cumulative: dict[str, list[float]] = defaultdict(list)
        for fl in fold_lists:
            if sid in fl:
                for entry in fl[sid]:
                    tid, score = entry[0], float(entry[1])
                    cumulative[tid].append(score)
        union_sizes.append(len(cumulative))
        scored = [(tid, sum(s_list) / N_FOLDS) for tid, s_list in cumulative.items()]
        scored.sort(key=lambda x: -x[1])
        ensemble[sid] = scored[:POOL_K]
    print(f"  union pool sizes: min={min(union_sizes)} median="
          f"{int(np.median(union_sizes))} max={max(union_sizes)}")
    return ensemble


def build_blind_features(blind, r54_blind_by_sid, r84_ensemble,
                          maps, track_pop, max_pop, track_album,
                          als_factors, als_to_idx):
    """Per blind case: build R54-stacked pool, featurize R54 (37) and R84-substituted (37)."""
    print(f"{ts()} Building blind features...", flush=True)

    # Reuse R3-det / F1 / S source builders from the existing inference scripts —
    # the blind source_cache already has src_a/b/c/d/f/als/r21 + r21_rank_map.
    out = {}
    for sid in sorted(blind.keys()):
        case = blind[sid]
        # Construct case dict compatible with _featurize_row (uses user_query, history,
        # music_turns; needs als_session_vec which we have in case["als_vec"]).
        src_lists = {
            "A": case.get("src_a", []),
            "B": case.get("src_b", []),
            "C": case.get("src_c", []),
            "D": case.get("src_d", []),
            "F": case.get("src_f", []),
            "ALS": case.get("als_tracks", []),
            "R21": case.get("r21_list", []),
            "R54": [t for t, _ in r54_blind_by_sid.get(sid, [])][:POOL_K],
        }
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        r21_rank_map = {t: r + 1 for r, t in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {t: r + 1 for r, t in enumerate(src_lists["R54"][:POOL_K])}
        r54_score_map = {t: float(s) for t, s in r54_blind_by_sid.get(sid, [])}
        als_vec = case.get("als_vec")
        feats_r54 = _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_score_map,
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, als_vec,
            track_pop, max_pop, track_album,
        )
        # R84 features: rank_inv, presence, cosine from ensemble lists
        r84_pairs = r84_ensemble.get(sid, [])
        r84_rank_map = {t: r + 1 for r, t in enumerate(r84_pairs[:POOL_K])}
        r84_score_map = {t: float(s) for t, s in r84_pairs}
        feats_r84 = feats_r54.copy()
        for k_row, tid in enumerate(pool):
            feats_r84[k_row, N_R39 + 0] = (1.0 / r84_rank_map[tid]) if tid in r84_rank_map else 0.0
            feats_r84[k_row, N_R39 + 1] = 1.0 if tid in r84_rank_map else 0.0
            feats_r84[k_row, N_R39 + 2] = r84_score_map.get(tid, 0.0)
        out[sid] = {
            "session_id": sid,
            "turn_number": case["turn_number"],
            "user_query": case["user_query"],
            "played_set": set(case["music_turns"]),
            "pool": pool,
            "feats_r54": feats_r54,
            "feats_r84": feats_r84,
        }
    print(f"  {len(out)} blind cases featurized")
    return out


def train_production_r84_lr(case_features_dev):
    """Train production R84-LR on all 8000 dev cases with R84 features."""
    print(f"{ts()} Training production R84-LR on all 8000 dev (in-sample)...",
          flush=True)
    X, y, gt = [], [], []
    for i, cf in case_features_dev.items():
        pool_len = len(cf["pool"])
        for k_row in range(pool_len):
            X.append(cf["feats_r84_only"][k_row])
            y.append(1.0 if k_row == cf["gt_pos"] else 0.0)
        gt.append(pool_len)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    ds = lgb.Dataset(X, label=y, group=gt, feature_name=FEAT_NAMES_R84)
    lr = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    lr.save_model(str(OUT_PROD_LR_PATH))
    print(f"  saved -> {OUT_PROD_LR_PATH}")
    return lr


def load_r78_top1(r78_path):
    import zipfile
    with zipfile.ZipFile(r78_path) as z:
        items = json.loads(z.read("prediction.json"))
    return {
        (i["session_id"], int(i["turn_number"])): i["predicted_track_ids"]
        for i in items
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-fail", action="store_true",
                        help="Continue past hard-stop churn/overlap gates (audit only)")
    args = parser.parse_args()
    t0 = time.time()
    print(f"{ts()} R84c Blind-A candidate builder")
    print("=" * 70)

    # --- 1. Ensemble ---
    r84_ensemble = ensemble_5fold_blind()
    OUT_ENSEMBLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_ENSEMBLE_PATH, "w") as f:
        json.dump({"lists": {sid: [[t, float(s)] for t, s in pairs]
                              for sid, pairs in r84_ensemble.items()},
                   "method": "avg cosine across 5 R84 folds where present, default 0"},
                  f)
    print(f"  saved ensemble -> {OUT_ENSEMBLE_PATH}")

    # --- 2. Load blind data + production support ---
    print(f"\n{ts()} Loading blind sources + maps + R54 blind + frozen R54c LR...")
    with open(BLIND_SRC, "rb") as f:
        blind = pickle.load(f)
    with open(R54_BLIND_LISTS) as f:
        r54_blind = json.load(f)["lists"]
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    r54c_lr = lgb.Booster(model_file=str(R54_LR_PATH))

    # --- 3. Build blind features ---
    blind_feats = build_blind_features(
        blind, r54_blind, r84_ensemble,
        maps, track_pop, max_pop, track_album,
        als_factors, als_to_idx,
    )

    # --- 4. Train production R84-LR ---
    print(f"\n{ts()} Loading case_features cache for production R84-LR training...")
    with open(FEAT_CACHE, "rb") as f:
        case_features_dev = pickle.load(f)
    print(f"  {len(case_features_dev)} dev cases loaded")
    prod_r84_lr = train_production_r84_lr(case_features_dev)

    # --- 5. Score 80 blind, apply routing, build candidate ---
    print(f"\n{ts()} Scoring blind + routing per R84c rule "
          f"(margin < {ROUTE_LOW} OR margin >= {ROUTE_HIGH})...")
    r78_top1 = load_r78_top1(R78_SUB)
    r84c_track_lists = []
    audit_rows = []
    n_routed_r84 = 0
    n_routed_r54 = 0
    for sid in sorted(blind.keys()):
        bf = blind_feats[sid]
        s_r54 = r54c_lr.predict(bf["feats_r54"])
        s_r84 = prod_r84_lr.predict(bf["feats_r84"])
        sorted_r54 = np.sort(s_r54)[::-1]
        margin = float(sorted_r54[0] - sorted_r54[1]) if len(sorted_r54) >= 2 else 0.0
        use_r84 = (margin < ROUTE_LOW) or (margin >= ROUTE_HIGH)
        if use_r84:
            n_routed_r84 += 1
            order = np.argsort(-s_r84, kind="mergesort")
        else:
            n_routed_r54 += 1
            order = np.argsort(-s_r54, kind="mergesort")

        # Filter out played tracks (production constraint)
        played = bf["played_set"]
        top20 = []
        for idx in order:
            tid = bf["pool"][int(idx)]
            if tid in played:
                continue
            top20.append(tid)
            if len(top20) == TOP_K:
                break

        r78_list = r78_top1.get((sid, bf["turn_number"]), [])
        r78_set = set(r78_list)
        r84c_set = set(top20)
        overlap = len(r78_set & r84c_set)
        top1_change = (top20[0] != r78_list[0]) if (top20 and r78_list) else False
        change_count = len(r78_set - r84c_set)

        audit_rows.append({
            "session_id": sid, "turn_number": bf["turn_number"],
            "r54_margin": margin, "routed_r84": use_r84,
            "r78_top1": r78_list[0] if r78_list else None,
            "r84c_top1": top20[0] if top20 else None,
            "top1_changed": top1_change,
            "overlap_20": overlap,
            "tracks_changed": change_count,
        })
        r84c_track_lists.append({
            "session_id": sid,
            "turn_number": bf["turn_number"],
            "predicted_track_ids": top20,
            "_routed_r84": use_r84,
            "_r54_margin": margin,
        })

    # --- 6. Audit ---
    n_top1_changed = sum(1 for r in audit_rows if r["top1_changed"])
    churn_per_80 = n_top1_changed  # since n=80
    overlap_mean = float(np.mean([r["overlap_20"] for r in audit_rows]))
    n_changed_6plus = sum(1 for r in audit_rows if r["tracks_changed"] >= 6)
    routed_r84_rate = n_routed_r84 / len(blind)

    print(f"\n{ts()} === AUDIT ===", flush=True)
    print(f"  Routed: R84={n_routed_r84} / R54={n_routed_r54} "
          f"({routed_r84_rate:.1%} R84)")
    print(f"  Top-1 churn vs R78: {n_top1_changed}/80 = {churn_per_80}/80")
    print(f"  Top-20 overlap mean: {overlap_mean:.2f}/20")
    print(f"  Sessions with >=6 changed tracks: {n_changed_6plus}/80")
    churn_ok = churn_per_80 <= CHURN_MAX_PER_80
    overlap_ok = overlap_mean >= OVERLAP_MIN_PER_20
    print(f"\n  GATES:")
    print(f"    churn <= {CHURN_MAX_PER_80}/80:  "
          f"{'PASS' if churn_ok else 'FAIL'} ({churn_per_80})")
    print(f"    overlap >= {OVERLAP_MIN_PER_20}/20: "
          f"{'PASS' if overlap_ok else 'FAIL'} ({overlap_mean:.2f})")
    pass_audit = churn_ok and overlap_ok

    audit = {
        "experiment": "R84c Blind-A candidate audit",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "routing_rule": f"use R84 if margin < {ROUTE_LOW} or >= {ROUTE_HIGH}",
        "n_blind_cases": len(blind),
        "n_routed_r84": n_routed_r84,
        "n_routed_r54": n_routed_r54,
        "routed_r84_rate": routed_r84_rate,
        "top1_churn_per_80": churn_per_80,
        "top20_overlap_mean": overlap_mean,
        "sessions_changed_6plus": n_changed_6plus,
        "gates": {
            "churn_le_35": {"value": churn_per_80, "pass": churn_ok},
            "overlap_ge_14": {"value": overlap_mean, "pass": overlap_ok},
        },
        "audit_passes": pass_audit,
        "per_case_audit": audit_rows,
    }
    OUT_R84C_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_R84C_AUDIT, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  Audit saved -> {OUT_R84C_AUDIT}")

    # Save track lists regardless (for inspection)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_R84C_TRACKS_JSON, "w") as f:
        json.dump(r84c_track_lists, f, indent=2)
    print(f"  Tracks saved -> {OUT_R84C_TRACKS_JSON}")

    # Hard stop if audit fails
    if not pass_audit and not args.allow_fail:
        print(f"\n  HARD STOP: audit failed. Re-run with --allow-fail to override.")
        sys.exit(2)

    print(f"\n{ts()} === AUDIT PASSED — ready for response regeneration step ===")
    print(f"  Next: regen R78-style responses for the {n_top1_changed} sessions where "
          f"top-1 changed.")


if __name__ == "__main__":
    main()
