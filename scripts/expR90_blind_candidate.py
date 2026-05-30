"""R90 Blind-A candidate builder.

Adapted from expR84c_blind_candidate.py. Builds R90 5-fold ensemble blind
candidate using the same production pattern that won R84c on Blind-A:

1. Load 5-fold R90 blind retrieval lists (from cache/r90/blind_fold{0..4}/),
   ensemble into single top-300 per blind sid (avg cosine where present).
2. Train production R90-LR on all 8000 dev cases (with R90 features
   substituted per case using each fold's R90 OOF list).
3. For each of 80 Blind-A cases:
   - Build R54-stacked RRF pool (= R78/R84c production pool).
   - Score with frozen R54c LR (-> R54-rank + R54-LR-margin for routing).
   - Score with production R90-LR.
   - Route: use R90-LR top-20 if R54c margin < ROUTE_LOW OR >= ROUTE_HIGH,
     else R54c top-20 (= the current R84c-style routing rule).
4. Audit vs R84c (current Blind-A production):
   - top-1 churn /80
   - top-20 overlap mean
   - sessions with >=6 changed tracks
5. Hard stop if churn > 35/80 OR overlap < 14/20.
6. Output: candidate track lists JSON + audit; do NOT auto-submit.

Default routing thresholds: 0.25 / 2.0 (the dev-best per R84c sweep + R90
LR-conversion test).

NOTE: this script reuses cache/r90/blind_fold{N}/blind_r84_lists.json paths
(the encoding helper expR84_blind_encode.py writes under that filename).
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

# R90 routing rule (default = dev-best from LR conversion test + R84c sweep)
DEFAULT_ROUTE_LOW = 0.25
DEFAULT_ROUTE_HIGH = 2.0

# Churn audit gates (same as R84c)
CHURN_MAX_PER_80 = 35
OVERLAP_MIN_PER_20 = 14.0

BLIND_SRC = REPO / "cache" / "blind_a" / "source_cache.pkl"
R54_LR_PATH = REPO / "cache" / "r54_phase3_lr_model.txt"
R54_BLIND_LISTS = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
R84C_TRACKS = REPO / "exp" / "inference" / "blind_a" / "r84c_blind_track_lists.json"
R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"

R90_FOLD_BLIND_DIRS = [REPO / f"cache/r90/blind_fold{f}" for f in range(N_FOLDS)]
R90_FOLD_OOF_DIRS = {f: REPO / f"cache/r90/phase1_fold{f}_varA" for f in range(N_FOLDS)}

OUT_DIR = REPO / "exp" / "inference" / "blind_a"
OUT_R90_TRACKS_JSON = OUT_DIR / "r90_blind_track_lists.json"
OUT_R90_AUDIT = REPO / "exp" / "eval" / "expR90_blind_audit.json"
OUT_PROD_LR_PATH = REPO / "cache" / "r90_production_lr.txt"
OUT_ENSEMBLE_PATH = REPO / "cache" / "r90_production" / "blind_r90_ensemble_lists.json"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R90 = list(FEAT_R39_ALL) + ["ret_rank_inv", "ret_presence", "ret_cosine"]
assert len(FEAT_NAMES_R90) == len(FEAT_ALL), \
    f"feature count mismatch: {len(FEAT_NAMES_R90)} vs {len(FEAT_ALL)}"


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    g = shutil.which("git")
    if g is None:
        return "no-git"
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def ensemble_5fold_blind(fold_dirs: list[Path]) -> dict[str, list[tuple[str, float]]]:
    """Average cosine across 5 folds where present, sort, take top-300 per sid."""
    print(f"{ts()} Loading 5 R90 fold blind lists and ensembling...", flush=True)
    fold_lists = []
    for fold, d in enumerate(fold_dirs):
        p = d / "blind_r84_lists.json"  # filename inherited from encoder
        if not p.exists():
            raise FileNotFoundError(f"R90 blind fold-{fold} missing: {p}")
        with open(p) as f:
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


def load_5fold_oof_r90_maps() -> dict[int, dict]:
    """Load all 5 R90 OOF lists -> {case_idx: {ranks, scores, lists}}."""
    out: dict[int, dict] = {}
    for fold, fold_dir in R90_FOLD_OOF_DIRS.items():
        p = fold_dir / "oof_r84_lists.json"
        if not p.exists():
            raise FileNotFoundError(f"R90 OOF fold-{fold} missing: {p}")
        with open(p) as f:
            raw = json.load(f)
        for case_idx_str, pairs in raw.items():
            ci = int(case_idx_str)
            tids = [t for t, _ in pairs]
            out[ci] = {
                "ranks": {t: r + 1 for r, t in enumerate(tids)},
                "scores": {t: float(s) for t, s in pairs},
            }
    return out


def build_dev_case_features(payload, r21_source, r54_source, r54_scores,
                              als_factors, als_to_idx, case_index, maps,
                              track_pop, max_pop, track_album,
                              r90_oof_maps) -> dict[int, dict]:
    """Per dev case: build R54-stacked pool, featurize with R90 features substituted."""
    cases = payload["cases"]
    print(f"{ts()} Building dev case features ({len(cases)} cases)...", flush=True)
    out = {}
    t0 = time.time()
    for ci, case in enumerate(cases):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], ci)
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}
        feats_r54 = _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[ci],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][ci],
            track_pop, max_pop, track_album,
        )
        # Substitute R90 features (last 3 cols) using THIS case's fold OOF
        r90_ranks = r90_oof_maps[ci]["ranks"]
        r90_scores = r90_oof_maps[ci]["scores"]
        feats_r90 = feats_r54.copy()
        for k_row, tid in enumerate(pool):
            feats_r90[k_row, N_R39 + 0] = (1.0 / r90_ranks[tid]) if tid in r90_ranks else 0.0
            feats_r90[k_row, N_R39 + 1] = 1.0 if tid in r90_ranks else 0.0
            feats_r90[k_row, N_R39 + 2] = r90_scores.get(tid, 0.0)
        gt = case["gt"]
        gt_pos = pool.index(gt) if gt in pool else -1
        out[ci] = {
            "pool": pool,
            "feats_r90": feats_r90,
            "gt_pos": gt_pos,
        }
        if (ci + 1) % 1000 == 0:
            print(f"  {ci + 1}/{len(cases)} ({time.time() - t0:.0f}s)")
    print(f"  done in {time.time() - t0:.0f}s")
    return out


def train_production_r90_lr(case_features_dev):
    """Train production R90-LR on all 8000 dev cases."""
    print(f"{ts()} Training production R90-LR on all 8000 dev (in-sample)...",
          flush=True)
    X, y, gt = [], [], []
    for ci, cf in case_features_dev.items():
        pool_len = len(cf["pool"])
        for k_row in range(pool_len):
            X.append(cf["feats_r90"][k_row])
            y.append(1.0 if k_row == cf["gt_pos"] else 0.0)
        gt.append(pool_len)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    ds = lgb.Dataset(X, label=y, group=gt, feature_name=FEAT_NAMES_R90)
    lr = lgb.train(LR_PARAMS, ds, num_boost_round=LR_NUM_BOOST_ROUND)
    lr.save_model(str(OUT_PROD_LR_PATH))
    print(f"  saved -> {OUT_PROD_LR_PATH.relative_to(REPO)}")
    return lr


def build_blind_features(blind, r54_blind_by_sid, r90_ensemble,
                          maps, track_pop, max_pop, track_album,
                          als_factors, als_to_idx):
    """Per blind case: build R54-stacked pool, featurize R54 and R90-substituted."""
    print(f"{ts()} Building blind features...", flush=True)
    out = {}
    for sid in sorted(blind.keys()):
        case = blind[sid]
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
        r90_pairs = r90_ensemble.get(sid, [])
        r90_rank_map = {t: r + 1 for r, t in enumerate(r90_pairs[:POOL_K])}
        r90_score_map = {t: float(s) for t, s in r90_pairs}
        feats_r90 = feats_r54.copy()
        for k_row, tid in enumerate(pool):
            feats_r90[k_row, N_R39 + 0] = (1.0 / r90_rank_map[tid]) if tid in r90_rank_map else 0.0
            feats_r90[k_row, N_R39 + 1] = 1.0 if tid in r90_rank_map else 0.0
            feats_r90[k_row, N_R39 + 2] = r90_score_map.get(tid, 0.0)
        out[sid] = {
            "session_id": sid,
            "turn_number": case["turn_number"],
            "user_query": case["user_query"],
            "played_set": set(case["music_turns"]),
            "pool": pool,
            "feats_r54": feats_r54,
            "feats_r90": feats_r90,
        }
    print(f"  {len(out)} blind cases featurized")
    return out


def load_r84c_top20(r84c_path: Path) -> dict[tuple[str, int], list[str]]:
    """Load R84c current Blind-A production track lists."""
    with open(r84c_path) as f:
        items = json.load(f)
    return {(i["session_id"], int(i["turn_number"])): i["predicted_track_ids"]
            for i in items}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-low", type=float, default=DEFAULT_ROUTE_LOW,
                    help=f"Default {DEFAULT_ROUTE_LOW} (dev-best per R84c sweep + "
                         f"R90 LR conversion).")
    ap.add_argument("--route-high", type=float, default=DEFAULT_ROUTE_HIGH,
                    help=f"Default {DEFAULT_ROUTE_HIGH}.")
    ap.add_argument("--allow-fail", action="store_true",
                    help="Continue past churn/overlap hard stops (audit only).")
    args = ap.parse_args()
    route_low, route_high = args.route_low, args.route_high

    t0 = time.time()
    print(f"{ts()} R90 Blind-A candidate builder")
    print(f"  routing: margin < {route_low} OR >= {route_high} -> R90")
    print("=" * 70)

    # --- 1. Ensemble R90 blind ---
    OUT_ENSEMBLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    r90_ensemble = ensemble_5fold_blind(R90_FOLD_BLIND_DIRS)
    with open(OUT_ENSEMBLE_PATH, "w") as f:
        json.dump({"lists": {sid: [[t, float(s)] for t, s in pairs]
                              for sid, pairs in r90_ensemble.items()},
                   "method": "avg cosine across 5 R90 folds where present, default 0",
                   "route_low": route_low, "route_high": route_high},
                  f)
    print(f"  saved ensemble -> {OUT_ENSEMBLE_PATH.relative_to(REPO)}")

    # --- 2. Load everything ---
    print(f"\n{ts()} Loading payload + R21/R54 OOF + supports + 5-fold R90 OOF...")
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )
    r54c_lr = lgb.Booster(model_file=str(R54_LR_PATH))
    r90_oof_maps = load_5fold_oof_r90_maps()
    assert len(r90_oof_maps) == n, f"R90 OOF missing for {n - len(r90_oof_maps)} cases"
    print(f"  n dev cases: {n}  R90 OOF maps: {len(r90_oof_maps)}")

    # --- 3. Build dev case features (R90-substituted) and train production R90-LR ---
    dev_features = build_dev_case_features(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_to_idx, case_index, maps, track_pop, max_pop,
        track_album, r90_oof_maps,
    )
    prod_r90_lr = train_production_r90_lr(dev_features)

    # --- 4. Load blind + featurize ---
    print(f"\n{ts()} Loading blind data...")
    with open(BLIND_SRC, "rb") as f:
        blind = pickle.load(f)
    with open(R54_BLIND_LISTS) as f:
        r54_blind = json.load(f)["lists"]
    blind_feats = build_blind_features(
        blind, r54_blind, r90_ensemble, maps, track_pop, max_pop, track_album,
        als_factors, als_to_idx,
    )

    # --- 5. Load R84c current production for audit ---
    print(f"\n{ts()} Loading R84c current production track lists for audit...")
    if not R84C_TRACKS.exists():
        print(f"FATAL: R84c track lists missing at {R84C_TRACKS}")
        sys.exit(1)
    r84c_top20 = load_r84c_top20(R84C_TRACKS)
    print(f"  R84c entries: {len(r84c_top20)}")

    # --- 6. Score blind + apply routing ---
    print(f"\n{ts()} Scoring blind + routing per R90 rule "
          f"(margin < {route_low} OR >= {route_high})...")
    r90_track_lists = []
    audit_rows = []
    n_routed_r90 = 0
    n_routed_r54 = 0
    for sid in sorted(blind.keys()):
        bf = blind_feats[sid]
        s_r54 = r54c_lr.predict(bf["feats_r54"])
        s_r90 = prod_r90_lr.predict(bf["feats_r90"])
        sorted_r54 = np.sort(s_r54)[::-1]
        margin = float(sorted_r54[0] - sorted_r54[1]) if len(sorted_r54) >= 2 else 0.0
        use_r90 = (margin < route_low) or (margin >= route_high)
        if use_r90:
            n_routed_r90 += 1
            order = np.argsort(-s_r90, kind="mergesort")
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

        r84c_list = r84c_top20.get((sid, bf["turn_number"]), [])
        overlap = len(set(top20) & set(r84c_list))
        top1_change = (top20[0] != r84c_list[0]) if (top20 and r84c_list) else False
        change_count = len(set(r84c_list) - set(top20))

        audit_rows.append({
            "session_id": sid, "turn_number": bf["turn_number"],
            "r54_margin": margin, "routed_r90": use_r90,
            "r84c_top1": r84c_list[0] if r84c_list else None,
            "r90_top1": top20[0] if top20 else None,
            "top1_changed": top1_change,
            "overlap_20": overlap,
            "tracks_changed": change_count,
        })
        r90_track_lists.append({
            "session_id": sid,
            "turn_number": bf["turn_number"],
            "predicted_track_ids": top20,
            "_routed_r90": use_r90,
            "_r54_margin": margin,
        })

    # --- 7. Audit ---
    n_top1_changed = sum(1 for r in audit_rows if r["top1_changed"])
    churn_per_80 = n_top1_changed
    overlap_mean = float(np.mean([r["overlap_20"] for r in audit_rows]))
    n_changed_6plus = sum(1 for r in audit_rows if r["tracks_changed"] >= 6)
    routed_r90_rate = n_routed_r90 / len(blind)

    print(f"\n{ts()} === AUDIT ===", flush=True)
    print(f"  Routed: R90={n_routed_r90} / R54={n_routed_r54} "
          f"({routed_r90_rate:.1%} R90)")
    print(f"  Top-1 churn vs R84c: {n_top1_changed}/80 = {churn_per_80}/80")
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
        "experiment": "R90 Blind-A candidate audit",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "route_low": route_low, "route_high": route_high,
        "routing_rule": f"use R90 if margin < {route_low} or >= {route_high}",
        "n_blind_cases": len(blind),
        "n_routed_r90": n_routed_r90,
        "n_routed_r54": n_routed_r54,
        "routed_r90_rate": routed_r90_rate,
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
    OUT_R90_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_R90_AUDIT, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  Audit saved -> {OUT_R90_AUDIT.relative_to(REPO)}")

    # --- 8. Save tracks (no submission yet — Opus regen + packaging is a separate step) ---
    if pass_audit or args.allow_fail:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(OUT_R90_TRACKS_JSON, "w") as f:
            json.dump(r90_track_lists, f, indent=2)
        print(f"  Tracks saved -> {OUT_R90_TRACKS_JSON.relative_to(REPO)}")
        print(f"\n  NEXT STEPS:")
        print(f"  1. Inspect {OUT_R90_AUDIT.relative_to(REPO)} per-case audit")
        print(f"  2. Identify changed-top1 sessions ({sum(1 for r in audit_rows if r['top1_changed'])} of 80)")
        print(f"  3. Regenerate responses for changed-top1 only using R78-style prompt")
        print(f"  4. Package as submission ZIP (do NOT auto-upload)")
        print(f"  5. Manual upload via Codabench")
    else:
        print(f"\n  HARD STOP — audit FAILED. NOT writing track lists.")
        sys.exit(1)

    print(f"\n{ts()} Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
