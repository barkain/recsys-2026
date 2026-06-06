#!/usr/bin/env python3
"""R431b — pool-admission-only (Arm B) user-CF, per-case dump for selective deployment.

R431 found Arm C (user-cf pool + user-cf FEATURES) fails the +0.010 nDCG gate, and that the
user-cf features actively hurt h7 (ΔC-vs-B = -0.0038 h7). Arm B (user-cf expands the pool, LR
ranks with the existing 37 features only) was cleaner: ΔB-vs-A all +0.00132, h7 +0.00422.

This script recomputes ONLY arms A and B (drops the expensive arm C), and dumps per-case A/B
records plus **B-side GT-INDEPENDENT signals** so we can search for a selective rule that keeps
B's wins while limiting churn. The selective rule at submit time may use ONLY GT-independent
fields (everything except gt/rA/rB/ndA/ndB/*_in20, which are for offline scoring only).

Arms (5-fold OOF, OOF-vs-OOF, never vs in-sample R54c):
  A  base      : pool = R54-stacked RRF (no user-cf);   feats = R39+r84 (37)
  B  aug-pool  : pool = R54-stacked RRF + user-cf source; feats = R39+r84 (37)
Production-equivalent baseline is A. A selective deployment applies B's ordering on a chosen
GT-independent subset and A's ordering elsewhere.

Output:
  --out (json)                : aggregate A/B metrics (all/h7/same/diff)
  <out>_percase.json          : per-case records (input to expR431b_rule_search.py)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]
import lightgbm as lgb  # type: ignore[reportMissingImports]

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)
from scripts.expR54_phase3_blind_submission import _featurize_row, FEAT_R39_ALL  # noqa: E402
from scripts.expR103_integrate_eval import (  # noqa: E402
    SW_BASELINE, RRF_K, POOL_K, TOP_K, N_FOLDS, LR_PARAMS, LR_NUM_BOOST_ROUND,
    FEAT_NAMES_R84, N_R39, W0_STATS, ndcg_at_k, load_r84_lists_for_fold,
    load_gte_oof, overwrite_r84, ts, head_sha,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ucf-oof", default=str(REPO / "cache/r431_user_cf/user_cf_oof_lists.json"))
    ap.add_argument("--ucf-weight", type=float, default=1.0)
    ap.add_argument("--ucf-rank-cap", type=int, default=POOL_K)
    ap.add_argument("--out", default=str(REPO / "exp/eval/expR431b_pool_only.json"))
    args = ap.parse_args()
    out_json = Path(args.out)
    SW_AUG = {**SW_BASELINE, "GTE": args.ucf_weight}

    t0 = time.time()
    print(f"{ts()} R431b — user-cf pool-admission only (arms A,B), 5-fold OOF")
    print(f"  pool: R54-stacked RRF (+user-cf w={args.ucf_weight}, cap {args.ucf_rank_cap})")
    print("=" * 72)

    ucf_path = Path(args.ucf_oof)
    if not ucf_path.exists():
        sys.exit(f"user-cf OOF lists not found: {ucf_path}")

    print(f"\n{ts()} Loading payload + R21/R54/ALS + maps ...", flush=True)
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    als_factors, als_track_ids, als_to_idx = c3.load_als_cache()
    maps, track_pop, track_album = load_supporting_maps()
    max_pop = max(track_pop.values()) if track_pop else 1

    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = [-1] * n
    for row in w0_stats:
        case_fold[row["case_idx"]] = int(row["fold_idx"])
    fold_to_idx: dict[int, list[int]] = {k: [] for k in range(N_FOLDS)}
    for i in range(n):
        fold_to_idx[case_fold[i]].append(i)

    print(f"{ts()} Loading R84 OOF (5 folds) + user-cf OOF ...", flush=True)
    r84_per_fold = {k: load_r84_lists_for_fold(k) for k in range(N_FOLDS)}
    ucf = load_gte_oof(ucf_path)

    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx)

    def featurize37(i, src_lists, pool):
        case = cases[i]
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R21"][:POOL_K])}
        r54_rank_map = {tid: r + 1 for r, tid in enumerate(src_lists["R54"][:POOL_K])}
        return _featurize_row(
            pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[i],
            case["user_query"], case["history"], case["music_turns"],
            set(case["music_turns"]),
            maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
            maps["track_artist_toks"], maps["track_meta_toks"],
            als_factors, als_to_idx, case_index["als_session_vecs"][i],
            track_pop, max_pop, track_album).astype(np.float32)

    # --- Pre-build per-case pools + A/B feature matrices ---
    print(f"\n{ts()} === Pre-building pools + A/B features ({n} cases) ===", flush=True)
    cf: dict[int, dict] = {}
    t_feat = time.time()
    hit_base = hit_aug = 0
    for i in range(n):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        umaps = ucf.get(i, {"ids": [], "ranks": {}, "scores": {}})
        uids = umaps["ids"]

        pool_base = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        src_aug = {**src_lists, "GTE": uids[:args.ucf_rank_cap]}
        pool_aug = weighted_rrf(src_aug, SW_AUG, topk=POOL_K, k=RRF_K)

        gt = cases[i]["gt"]
        gp_base = pool_base.index(gt) if gt in pool_base else -1
        gp_aug = pool_aug.index(gt) if gt in pool_aug else -1
        hit_base += gp_base >= 0
        hit_aug += gp_aug >= 0

        owner = case_fold[i]
        r84m = r84_per_fold[owner][i]
        feats_A = overwrite_r84(featurize37(i, src_lists, pool_base), pool_base,
                                r84m["ranks"], r84m["scores"])
        feats_B = overwrite_r84(featurize37(i, src_lists, pool_aug), pool_aug,
                                r84m["ranks"], r84m["scores"])

        ucf_unique = set(pool_aug) - set(pool_base)   # admitted ONLY via user-cf
        cf[i] = {
            "pool_base": pool_base, "pool_aug": pool_aug,
            "feats_A": feats_A, "feats_B": feats_B,
            "gp_base": gp_base, "gp_aug": gp_aug, "gt": gt,
            "ucf_unique": ucf_unique,
            "ucf_ranks": umaps["ranks"], "ucf_scores": umaps["scores"],
        }
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{n} ({time.time() - t_feat:.0f}s) "
                  f"pool_hit base={hit_base} aug={hit_aug}", flush=True)
    print(f"  pool_hit: base={hit_base}/{n}  aug(+ucf)={hit_aug}/{n}  Δ={hit_aug-hit_base:+d}", flush=True)
    print(f"  feat-build elapsed: {time.time() - t_feat:.0f}s", flush=True)

    # --- 5-fold OOF: train A and B sibling LRs ---
    results: list[dict] = []
    for fk in range(N_FOLDS):
        print(f"\n{ts()} === FOLD {fk} held out ===", flush=True)
        eval_idx = fold_to_idx[fk]
        train_idx = [i for i in range(n) if case_fold[i] != fk]

        def stack(feat_key, pool_key, gp_key):
            X, y, grp = [], [], []
            for i in train_idx:
                c = cf[i]
                P = c[pool_key]; F = c[feat_key]; gp = c[gp_key]
                for r in range(len(P)):
                    X.append(F[r]); y.append(1.0 if r == gp else 0.0)
                grp.append(len(P))
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), grp

        print("  train A (base pool, 37) ...", flush=True)
        XA, yA, gA = stack("feats_A", "pool_base", "gp_base")
        lrA = lgb.train(LR_PARAMS, lgb.Dataset(XA, label=yA, group=gA,
                        feature_name=FEAT_NAMES_R84), num_boost_round=LR_NUM_BOOST_ROUND)
        del XA, yA
        print("  train B (aug pool, 37) ...", flush=True)
        XB, yB, gB = stack("feats_B", "pool_aug", "gp_aug")
        lrB = lgb.train(LR_PARAMS, lgb.Dataset(XB, label=yB, group=gB,
                        feature_name=FEAT_NAMES_R84), num_boost_round=LR_NUM_BOOST_ROUND)
        del XB, yB

        print(f"  scoring fold {fk} ({len(eval_idx)} cases) ...", flush=True)
        for i in eval_idx:
            c = cf[i]
            pb, pa = c["pool_base"], c["pool_aug"]
            gpb, gpa = c["gp_base"], c["gp_aug"]
            sA = lrA.predict(c["feats_A"])
            sB = lrB.predict(c["feats_B"])
            oA = np.argsort(-sA, kind="mergesort")
            oB = np.argsort(-sB, kind="mergesort")

            def rank_of(order, gp):
                if gp < 0:
                    return -1
                w = np.where(order == gp)[0]
                return int(w[0]) + 1 if len(w) else -1

            rA = rank_of(oA, gpb)
            rB = rank_of(oB, gpa)
            topA = [pb[int(j)] for j in oA[:TOP_K]]
            topB = [pa[int(j)] for j in oB[:TOP_K]]

            # --- B-side GT-INDEPENDENT signals (legal at submit time) ---
            uniq = c["ucf_unique"]
            n_ucf_in_b20 = sum(1 for t in topB if t in uniq)
            b_top1 = topB[0] if topB else None
            b_top1_is_ucf = int(b_top1 in uniq) if b_top1 else 0
            b_top1_ucf_rank = int(c["ucf_ranks"].get(b_top1, 0)) if b_top1 else 0
            b_top1_ucf_cos = float(c["ucf_scores"].get(b_top1, 0.0)) if b_top1 else 0.0
            # strongest user-cf candidate that B surfaced into top-20 (cosine + best rank)
            ucf_in_b20 = [t for t in topB if t in uniq]
            max_ucf_cos_b20 = max((c["ucf_scores"].get(t, 0.0) for t in ucf_in_b20), default=0.0)
            best_ucf_rank_b20 = min((c["ucf_ranks"].get(t, 10**9) for t in ucf_in_b20),
                                    default=0)
            a_margin = float(sA[int(oA[0])] - sA[int(oA[1])]) if len(oA) > 1 else 0.0
            overlap_AB = len(set(topA) & set(topB))
            b_changed_top1 = int(bool(topA) and bool(topB) and topA[0] != topB[0])
            b_changed_any = int(topA != topB)

            results.append({
                "case_idx": i, "fold": fk,
                "n_prior_music": int(cases[i]["n_prior_music"]),
                "same_artist": int(same_artist_case(cases[i], maps["track_artist"])),
                # offline-scoring only (NOT usable by the rule):
                "rA": rA, "rB": rB,
                "ndA": ndcg_at_k(rA, TOP_K), "ndB": ndcg_at_k(rB, TOP_K),
                "A_in20": int(0 < rA <= TOP_K), "B_in20": int(0 < rB <= TOP_K),
                # GT-INDEPENDENT signals (rule may use these):
                "n_ucf_in_b20": n_ucf_in_b20,
                "b_top1_is_ucf": b_top1_is_ucf,
                "b_top1_ucf_rank": b_top1_ucf_rank,
                "b_top1_ucf_cos": b_top1_ucf_cos,
                "max_ucf_cos_b20": max_ucf_cos_b20,
                "best_ucf_rank_b20": best_ucf_rank_b20,
                "a_margin": a_margin,
                "overlap_AB": overlap_AB,
                "b_changed_top1": b_changed_top1,
                "b_changed_any": b_changed_any,
            })
        del lrA, lrB

    # --- Aggregate A/B ---
    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    h7 = [r for r in results if r["n_prior_music"] == 7]
    same = [r for r in results if r["same_artist"]]
    diff = [r for r in results if not r["same_artist"]]
    metrics = {}
    for name, rows in [("all", results), ("h7", h7), ("same_artist", same), ("diff_artist", diff)]:
        A, B = avg(rows, "ndA"), avg(rows, "ndB")
        metrics[name] = {"n": len(rows), "A": A, "B": B, "dB_vs_A": B - A}

    n_changed = sum(r["b_changed_any"] for r in results)
    n_changed_top1 = sum(r["b_changed_top1"] for r in results)
    print(f"\n{ts()} === AGGREGATE (5-fold OOF, A vs B) ===")
    for name, m in metrics.items():
        print(f"  {name:12} n={m['n']:5d} A={m['A']:.4f} B={m['B']:.4f} ΔB-A={m['dB_vs_A']:+.4f}")
    print(f"  cases where B changed top-20: {n_changed}/{n}  (top-1 changed: {n_changed_top1})")

    out = {
        "experiment": "R431b — user-cf pool-admission only (A vs B), 5-fold OOF",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0, "head_sha": head_sha(),
        "ucf_oof": str(ucf_path), "ucf_weight": args.ucf_weight,
        "pool_hit": {"base": hit_base, "aug": hit_aug, "delta": hit_aug - hit_base, "n": n},
        "metrics": metrics,
        "n_changed_any": n_changed, "n_changed_top1": n_changed_top1,
    }
    def _ser(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return float(o)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=2, default=_ser)
    pc_path = out_json.with_name(out_json.stem + "_percase.json")
    json.dump({"ucf_weight": args.ucf_weight, "n": len(results), "results": results},
              open(pc_path, "w"), default=_ser)
    print(f"\n{ts()} Saved {out_json}\n{ts()} Dumped per-case ({len(results)}) -> {pc_path}")


if __name__ == "__main__":
    main()
