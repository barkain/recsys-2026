"""R84c blind replay — parameterized R84c pipeline for Blind-A (dry-run) or Blind-B.

Stages (all Mac-side; Colab work documented in docs/blind_b_r84c_runbook.md):

  1. Load blind source cache + R54 blind ensemble + R84 5-fold blind lists.
  2. Build R84 ensemble (avg cosine where present).
  3. Featurize blind cases on R54-stacked pool.
  4. Score with frozen R54c LR + production R84c LR.
  5. Apply R84c selective routing: R54c margin < 0.5 OR >= 2.0 -> R84 LR.
  6. Extract top-20 (drop played tracks).
  7. Generate responses (R78-style prompt for changed-top1 only on Blind-A;
     all responses fresh on Blind-B since R78 has no Blind-B baseline).
  8. Package submission ZIP + audit JSON + diff doc.

Dry-run check: replay --blind-name blind_a should produce track lists
bytewise identical to r84c_selective_submission.zip's tracks. Responses
differ (Opus non-deterministic) but track hash MUST match.

Usage:
  uv run python scripts/expR84c_blind_replay.py --blind-name blind_a --tracks-only   # dry-run hash check
  uv run python scripts/expR84c_blind_replay.py --blind-name blind_b                  # full production
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
import zipfile
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
    FEAT_R39_ALL, FEAT_R54, FEAT_ALL, _featurize_row,
)
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps,
)

# Constants from R84c
SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
RRF_K = 20
POOL_K = 300
TOP_K = 20
N_FOLDS = 5
ROUTE_LOW = 0.5
ROUTE_HIGH = 2.0
CHURN_MAX_PER_80 = 35  # only relevant if comparing to a prior baseline submission
OVERLAP_MIN_PER_20 = 14.0
N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R84 = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]

R54_LR_PATH = REPO / "cache" / "r54_phase3_lr_model.txt"
R84C_LR_PATH = REPO / "cache" / "r84c_production_lr.txt"


def ts(): return f"[{datetime.now():%H:%M:%S}]"


def head_sha():
    g = shutil.which("git")
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip() if g else "no-git"


def resolve_paths(blind_name: str) -> dict:
    """Map blind_name -> input/output paths."""
    paths = {
        "blind_src": REPO / "cache" / blind_name / "source_cache.pkl",
        "r54_blind_lists": (REPO / "cache" / "r54_production" /
                              ("blind_r54_lists.json" if blind_name == "blind_a"
                                else f"{blind_name}_r54_lists.json")),
        "r84_fold_lists": [
            (REPO / "cache" / "r84" /
              (f"blind_fold{k}" if blind_name == "blind_a" else f"{blind_name}_blind_fold{k}") /
              "blind_r84_lists.json")
            for k in range(N_FOLDS)
        ],
        "r78_sub": (REPO / "exp" / "inference" / "blind_a" /
                     "r78_llm_polish_submission.zip"),
        "out_dir": REPO / "exp" / "inference" / blind_name,
        "out_zip": REPO / "exp" / "inference" / blind_name /
                    f"r84c_replay_{blind_name}_submission.zip",
        "out_audit": REPO / "exp" / "eval" /
                      f"expR84c_replay_{blind_name}_audit.json",
        "out_track_lists": REPO / "exp" / "inference" / blind_name /
                            f"r84c_replay_{blind_name}_track_lists.json",
        "out_metadata": REPO / "exp" / "inference" / blind_name /
                         f"r84c_replay_{blind_name}_submission.metadata.json",
        "r84_ensemble_path": (REPO / "cache" / "r84_production" /
                                (f"blind_r84_ensemble_lists.json" if blind_name == "blind_a"
                                  else f"{blind_name}_r84_ensemble_lists.json")),
    }
    return paths


def ensemble_5fold(fold_paths: list[Path]) -> dict[str, list[tuple[str, float]]]:
    print(f"{ts()} Ensembling 5-fold R84 blind lists...")
    fold_lists = []
    for fp in fold_paths:
        with open(fp) as f:
            fold_lists.append(json.load(f))
    all_sids = set()
    for fl in fold_lists:
        all_sids.update(fl.keys())
    ensemble = {}
    for sid in sorted(all_sids):
        cum: dict[str, list[float]] = defaultdict(list)
        for fl in fold_lists:
            if sid in fl:
                for entry in fl[sid]:
                    cum[entry[0]].append(float(entry[1]))
        scored = [(t, sum(s) / N_FOLDS) for t, s in cum.items()]
        scored.sort(key=lambda x: -x[1])
        ensemble[sid] = scored[:POOL_K]
    return ensemble


def build_blind_features(blind, r54_blind, r84_ensemble, maps, track_pop, max_pop,
                          track_album, als_factors, als_to_idx):
    print(f"{ts()} Building blind features...")
    out = {}
    for sid in sorted(blind.keys()):
        case = blind[sid]
        src_lists = {
            "A": case.get("src_a", []), "B": case.get("src_b", []),
            "C": case.get("src_c", []), "D": case.get("src_d", []),
            "F": case.get("src_f", []),
            "ALS": case.get("als_tracks", []),
            "R21": case.get("r21_list", []),
            "R54": [t for t, _ in r54_blind.get(sid, [])][:POOL_K],
        }
        pool = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        r21_rank_map = {t: r + 1 for r, t in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {t: r + 1 for r, t in enumerate(src_lists["R54"][:POOL_K])}
        r54_score_map = {t: float(s) for t, s in r54_blind.get(sid, [])}
        als_vec = case.get("als_vec")
        feats_r54 = _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_score_map,
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, als_vec, track_pop, max_pop, track_album,
        )
        r84_pairs = r84_ensemble.get(sid, [])
        r84_rank_map = {t: r + 1 for r, t in enumerate(r84_pairs[:POOL_K])}
        r84_score_map = {t: float(s) for t, s in r84_pairs}
        feats_r84 = feats_r54.copy()
        for k, tid in enumerate(pool):
            feats_r84[k, N_R39 + 0] = (1.0 / r84_rank_map[tid]) if tid in r84_rank_map else 0.0
            feats_r84[k, N_R39 + 1] = 1.0 if tid in r84_rank_map else 0.0
            feats_r84[k, N_R39 + 2] = r84_score_map.get(tid, 0.0)
        out[sid] = {
            "session_id": sid, "turn_number": case["turn_number"],
            "user_query": case["user_query"], "played_set": set(case["music_turns"]),
            "pool": pool, "feats_r54": feats_r54, "feats_r84": feats_r84,
        }
    return out


def load_r78_top_responses(r78_sub_path):
    if not r78_sub_path.exists():
        return {}
    with zipfile.ZipFile(r78_sub_path) as z:
        items = json.loads(z.read("prediction.json"))
    return {(i["session_id"], int(i["turn_number"])): i for i in items}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blind-name", default="blind_a",
                   choices=["blind_a", "blind_b"])
    p.add_argument("--tracks-only", action="store_true",
                   help="Skip response generation; just emit track lists JSON")
    p.add_argument("--skip-package", action="store_true",
                   help="Don't write a ZIP; useful for hash-check dry-runs")
    p.add_argument("--allow-no-r78", action="store_true",
                   help="Permit Blind-A run without R78 (responses will all be regen)")
    p.add_argument("--route-low", type=float, default=None,
                   help="Lower R54c LR margin threshold for selective routing. "
                        "If margin < route_low -> use R84 (low-confidence). "
                        "Default 0.5 (R84c-shipped predeclared rule).")
    p.add_argument("--route-high", type=float, default=None,
                   help="Upper R54c LR margin threshold for selective routing. "
                        "If margin >= route_high -> use R84 (high-confidence). "
                        "Default 2.0 (R84c-shipped predeclared rule).")
    args = p.parse_args()
    route_low = args.route_low if args.route_low is not None else ROUTE_LOW
    route_high = args.route_high if args.route_high is not None else ROUTE_HIGH

    t0 = time.time()
    print(f"{ts()} R84c blind replay — target: {args.blind_name}")
    print("=" * 70)
    paths = resolve_paths(args.blind_name)

    # --- 0. Sanity check inputs ---
    print(f"\n{ts()} Sanity checking inputs...")
    if not paths["blind_src"].exists():
        print(f"FATAL: source cache missing: {paths['blind_src']}")
        sys.exit(1)
    if not paths["r54_blind_lists"].exists():
        print(f"FATAL: R54 blind lists missing: {paths['r54_blind_lists']}")
        sys.exit(1)
    missing_r84 = [p for p in paths["r84_fold_lists"] if not p.exists()]
    if missing_r84:
        print(f"FATAL: {len(missing_r84)} R84 fold blind lists missing:")
        for p in missing_r84:
            print(f"  {p}")
        print(f"\nRun the Colab R84 5-fold ensemble pipeline first. "
              f"See docs/blind_b_r84c_runbook.md.")
        sys.exit(1)
    print(f"  all inputs present")

    # --- 1. Ensemble R84 5-fold ---
    r84_ensemble = ensemble_5fold(paths["r84_fold_lists"])
    paths["r84_ensemble_path"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["r84_ensemble_path"], "w") as f:
        json.dump({"lists": {sid: [[t, float(s)] for t, s in pairs]
                              for sid, pairs in r84_ensemble.items()},
                   "method": "avg cosine where present, default 0",
                   "blind_name": args.blind_name},
                  f)
    print(f"  ensemble saved -> {paths['r84_ensemble_path'].name}")

    # --- 2. Load inputs ---
    print(f"\n{ts()} Loading blind source cache + supports...")
    with open(paths["blind_src"], "rb") as f:
        blind = pickle.load(f)
    with open(paths["r54_blind_lists"]) as f:
        r54_blind_raw = json.load(f)
    r54_blind = r54_blind_raw.get("lists", r54_blind_raw)
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    r54c_lr = lgb.Booster(model_file=str(R54_LR_PATH))
    r84c_lr = lgb.Booster(model_file=str(R84C_LR_PATH))
    print(f"  {len(blind)} blind cases, R54c LR + R84c LR loaded")

    # --- 3. Featurize ---
    blind_feats = build_blind_features(
        blind, r54_blind, r84_ensemble, maps, track_pop, max_pop,
        track_album, als_factors, als_to_idx,
    )

    # --- 4. Score + route + extract top-20 ---
    print(f"\n{ts()} Scoring + routing per R84c rule (margin < {route_low} OR >= {route_high})...")
    track_lists = []
    audit_rows = []
    n_r84 = n_r54 = 0
    for sid in sorted(blind.keys()):
        bf = blind_feats[sid]
        s_r54 = r54c_lr.predict(bf["feats_r54"])
        s_r84 = r84c_lr.predict(bf["feats_r84"])
        s_r54_sorted = np.sort(s_r54)[::-1]
        margin = float(s_r54_sorted[0] - s_r54_sorted[1]) if len(s_r54_sorted) >= 2 else 0.0
        use_r84 = (margin < route_low) or (margin >= route_high)
        if use_r84:
            n_r84 += 1
            order = np.argsort(-s_r84, kind="mergesort")
        else:
            n_r54 += 1
            order = np.argsort(-s_r54, kind="mergesort")
        played = bf["played_set"]
        top20 = []
        for idx in order:
            tid = bf["pool"][int(idx)]
            if tid in played:
                continue
            top20.append(tid)
            if len(top20) == TOP_K:
                break
        track_lists.append({
            "session_id": sid, "turn_number": bf["turn_number"],
            "predicted_track_ids": top20,
            "_routed_r84": use_r84, "_r54_margin": margin,
        })
        audit_rows.append({
            "session_id": sid, "turn_number": bf["turn_number"],
            "r54_margin": margin, "routed_r84": use_r84,
            "top1": top20[0] if top20 else None,
            "top20_size": len(top20),
        })

    n = len(blind)
    routed_r84_rate = n_r84 / n
    print(f"\n  Routed: R84={n_r84} / R54={n_r54} ({routed_r84_rate:.1%} R84)")

    # Track hash
    track_hash_payload = sorted(
        (t["session_id"], int(t["turn_number"]), tuple(t["predicted_track_ids"]))
        for t in track_lists
    )
    track_hash = hashlib.sha256(json.dumps(track_hash_payload, sort_keys=True).encode()).hexdigest()
    print(f"  Track hash: {track_hash[:16]}...")

    # Save track lists JSON (always)
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["out_track_lists"], "w") as f:
        json.dump(track_lists, f, indent=2)
    print(f"  saved {paths['out_track_lists'].name}")

    if args.tracks_only:
        print(f"\n[tracks-only mode] exiting after track-list dump.")
        # Print audit summary
        audit = {
            "experiment": f"R84c blind replay tracks-only ({args.blind_name})",
            "created_at": datetime.now().isoformat(),
            "head_sha": head_sha(),
            "blind_name": args.blind_name,
            "n_cases": n,
            "route_low": route_low,
            "route_high": route_high,
            "n_routed_r84": n_r84,
            "n_routed_r54": n_r54,
            "routed_r84_rate": routed_r84_rate,
            "track_hash_sha256": track_hash,
            "elapsed_s": time.time() - t0,
        }
        with open(paths["out_audit"], "w") as f:
            json.dump(audit, f, indent=2)
        print(f"  audit saved -> {paths['out_audit']}")
        return

    # --- 5. Response generation ---
    if args.blind_name == "blind_a" and paths["r78_sub"].exists():
        r78_by_key = load_r78_top_responses(paths["r78_sub"])
        print(f"\n  R78 baseline available for Blind-A: {len(r78_by_key)} entries")
        # Identify changed-top1 cases
        changed = []
        for tl in track_lists:
            key = (tl["session_id"], tl["turn_number"])
            r78 = r78_by_key.get(key)
            if r78 and tl["predicted_track_ids"][0] != r78["predicted_track_ids"][0]:
                changed.append(tl)
        print(f"  Changed-top1 vs R78: {len(changed)}")
        regen_mode = "changed_top1_only"
    else:
        r78_by_key = {}
        changed = list(track_lists)  # regen ALL
        regen_mode = "all_fresh"
        print(f"\n  No R78 baseline (Blind-B or --allow-no-r78). Will regen ALL {len(changed)} responses.")

    if not changed:
        print(f"\n  No regeneration needed. Skipping response stage.")
        final_items = [
            {"session_id": tl["session_id"], "turn_number": int(tl["turn_number"]),
             "predicted_track_ids": tl["predicted_track_ids"],
             "predicted_response": r78_by_key[(tl["session_id"], tl["turn_number"])]["predicted_response"]}
            for tl in track_lists
        ]
    else:
        print(f"\n{ts()} === RESPONSE GENERATION ({regen_mode}, {len(changed)} cases) ===")
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
        if not api_key:
            print(f"  WARNING: no API key — emitting submission with empty responses.")
            print(f"  Use scripts/expR84c_response_regen.py (Blind-A) OR write a new "
                  f"regen for Blind-B and re-run packaging.")
            final_items = [
                {"session_id": tl["session_id"], "turn_number": int(tl["turn_number"]),
                 "predicted_track_ids": tl["predicted_track_ids"],
                 "predicted_response": (r78_by_key.get((tl["session_id"], tl["turn_number"]), {}).get("predicted_response", ""))}
                for tl in track_lists
            ]
        else:
            # Generate via existing R87-style pipeline (best current prompt)
            # For simplicity here, fall back to R84c's R78-style call_opus loop
            # imported from expR84c_response_regen if available
            print(f"  Response generation script: scripts/expR84c_response_regen.py "
                  f"(Blind-A) or scripts/expR87_llm_push.py-like for fresh.")
            print(f"  NOT calling Opus from this orchestrator. Run regen separately:")
            print(f"")
            print(f"    For Blind-A dry-run:  ANTHROPIC_RECSYS_API_KEY=... "
                  f"uv run python scripts/expR84c_response_regen.py")
            print(f"")
            print(f"    For Blind-B: write Blind-B-aware regen using the tracks output above.")
            final_items = [
                {"session_id": tl["session_id"], "turn_number": int(tl["turn_number"]),
                 "predicted_track_ids": tl["predicted_track_ids"],
                 "predicted_response": (r78_by_key.get((tl["session_id"], tl["turn_number"]), {}).get("predicted_response", ""))}
                for tl in track_lists
            ]

    # --- 6. Package ---
    if not args.skip_package:
        payload_str = json.dumps(final_items, indent=2)
        paths["out_zip"].parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(paths["out_zip"], "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("prediction.json", payload_str)
        sha = hashlib.sha256(open(paths["out_zip"], "rb").read()).hexdigest()
        print(f"\n  Wrote {paths['out_zip']} ({paths['out_zip'].stat().st_size} bytes)")
        print(f"  sha256: {sha}")

        meta = {
            "experiment": f"R84c blind replay ({args.blind_name})",
            "created_at": datetime.now().isoformat(),
            "head_sha": head_sha(),
            "blind_name": args.blind_name,
            "n_cases": n,
            "n_routed_r84": n_r84,
            "n_routed_r54": n_r54,
            "track_hash_sha256": track_hash,
            "submission_sha256": sha,
            "regen_mode": regen_mode,
            "n_responses_filled": sum(1 for it in final_items if it["predicted_response"]),
            "n_responses_empty": sum(1 for it in final_items if not it["predicted_response"]),
        }
        with open(paths["out_metadata"], "w") as f:
            json.dump(meta, f, indent=2)

    # Audit
    audit = {
        "experiment": f"R84c blind replay audit ({args.blind_name})",
        "created_at": datetime.now().isoformat(),
        "head_sha": head_sha(),
        "blind_name": args.blind_name,
        "n_cases": n,
        "route_low": route_low,
        "route_high": route_high,
        "n_routed_r84": n_r84,
        "n_routed_r54": n_r54,
        "routed_r84_rate": routed_r84_rate,
        "track_hash_sha256": track_hash,
        "elapsed_s": time.time() - t0,
        "per_case_audit": audit_rows,
    }
    with open(paths["out_audit"], "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  Audit saved -> {paths['out_audit']}")

    print(f"\n{ts()} Done. Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
