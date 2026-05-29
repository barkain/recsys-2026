#!/usr/bin/env python3
"""R103 — integrate GTE-7B as an ADDED retrieval source (R84 playbook), 5-fold OOF.

Tests whether GTE-7B's orthogonal recall (the ~32/1600 fold-0 top-30 GTs the union
misses, R102) CONVERTS to nDCG@20 through the sibling LR — the exact mechanism R84
used for its blind win. Compares OOF-vs-OOF (NOT against frozen in-sample R54c, per
feedback_lr_wall_was_artifact).

Three OOF sibling-LR arms (per held-out fold, LR trained on the other 4 folds):
  A  base_r84      : pool = R54-stacked RRF (no GTE);  feats = R39 + r84  (37)
                     -> OOF analog of the current best system (R84c / R92 family).
  B  base_r84_aug  : pool = R54-stacked RRF + GTE source; feats = R39 + r84 (37)
                     -> isolates the POOL effect (do GTE candidates help even when the
                        LR cannot score them? expect ~0: the pool-broadening wall).
  C  r103_gte      : pool = R54-stacked RRF + GTE source; feats = R39 + r84 + gte (40)
                     -> the full R103 treatment (GTE in pool AND as LR features).

GTE enters the pool as an RRF source so its uniques can actually be ranked (R84
Phase-1 kept a fixed pool and could never rank R84's uniques — the documented wall).

Primary comparison C vs A. Reports also C vs B (feature effect) and B vs A (pool effect).
Gates (spec): (h7 Δ>=+0.005 OR h7 rec>lost) AND same-artist Δ>=-0.005 AND
diff-artist Δ>=-0.005 AND top-20 overlap safe. Plus: GTE-unique CONVERSION (of GTs
that are in the pool ONLY because of GTE, how many land in top-20 under C), the
orthogonal-recall audit (GTE top-30 GTs absent from the R54-stacked pool, per fold),
and the arm-C LR feature importance for the gte_* columns (stop condition: LR ignores
GTE features).

Input:  cache/r103_gte/oof_gte_lists.json  (from expR103_gte_oof.py; {case_idx:[[tid,score],...]})
Output: exp/eval/expR103_integrate.json + docs/r103_result.md
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
    FEAT_R39_ALL, FEAT_ALL, _featurize_row,
)
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
RRF_K = 20
POOL_K = 300
TOP_K = 20
N_FOLDS = 5

R54_LR = REPO / "cache" / "r54_phase3_lr_model.txt"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
GTE_OOF_DEFAULT = REPO / "cache" / "r103_gte" / "oof_gte_lists.json"
OUT_JSON = REPO / "exp" / "eval" / "expR103_integrate.json"
OUT_MD = REPO / "docs" / "r103_result.md"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

N_R39 = len(FEAT_R39_ALL)
FEAT_NAMES_R84 = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
FEAT_NAMES_R103 = FEAT_NAMES_R84 + ["gte_rank_inv", "gte_presence", "gte_cosine"]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    g = shutil.which("git")
    return "no-git" if g is None else subprocess.check_output(
        [g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def ndcg_at_k(gt_rank: int, k: int) -> float:
    return 0.0 if (gt_rank <= 0 or gt_rank > k) else 1.0 / math.log2(gt_rank + 1)


def load_r84_lists_for_fold(fold: int) -> dict[int, dict]:
    path = (REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json" if fold == 0
            else REPO / "cache" / "r84" / f"phase1_fold{fold}" / "oof_r84_lists.json")
    if not path.exists():
        raise FileNotFoundError(f"R84 OOF lists missing for fold {fold}: {path}")
    raw = json.load(open(path))
    out = {}
    for ci_str, pairs in raw.items():
        tids = [t for t, _ in pairs]
        out[int(ci_str)] = {"ranks": {t: r + 1 for r, t in enumerate(tids)},
                            "scores": {t: float(s) for t, s in pairs}}
    return out


def load_gte_oof(path: Path) -> dict[int, dict]:
    """case_idx -> {ranks:{tid:rank1based}, scores:{tid:cosine}, ids:[tid,...]}"""
    d = json.load(open(path))
    lists = d.get("lists", d)
    out = {}
    for ci_str, pairs in lists.items():
        ids = [p[0] for p in pairs]
        out[int(ci_str)] = {
            "ids": ids,
            "ranks": {t: r + 1 for r, t in enumerate(ids)},
            "scores": {p[0]: float(p[1]) for p in pairs},
        }
    return out


def overwrite_r84(feats: np.ndarray, pool: list[str],
                  r84_ranks: dict[str, int], r84_scores: dict[str, float]) -> np.ndarray:
    for i, tid in enumerate(pool):
        feats[i, N_R39 + 0] = (1.0 / r84_ranks[tid]) if tid in r84_ranks else 0.0
        feats[i, N_R39 + 1] = 1.0 if tid in r84_ranks else 0.0
        feats[i, N_R39 + 2] = r84_scores.get(tid, 0.0)
    return feats


def gte_three_cols(pool: list[str], gte_ranks: dict[str, int],
                   gte_scores: dict[str, float]) -> np.ndarray:
    g = np.zeros((len(pool), 3), dtype=np.float32)
    for i, tid in enumerate(pool):
        if tid in gte_ranks:
            g[i, 0] = 1.0 / gte_ranks[tid]
            g[i, 1] = 1.0
            g[i, 2] = gte_scores.get(tid, 0.0)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gte-oof", default=str(GTE_OOF_DEFAULT))
    ap.add_argument("--gte-weight", type=float, default=1.0,
                    help="RRF weight for GTE source in the augmented pool")
    ap.add_argument("--gte-rank-cap", type=int, default=POOL_K,
                    help="how deep into GTE list to use as an RRF source")
    args = ap.parse_args()
    SW_AUG = {**SW_BASELINE, "GTE": args.gte_weight}

    t0 = time.time()
    print(f"{ts()} R103 — GTE as added source, 5-fold OOF sibling LR")
    print(f"  pool: R54-stacked RRF (+GTE w={args.gte_weight}, cap {args.gte_rank_cap}), K={POOL_K}")
    print(f"  arms: A=base_r84(no GTE)  B=base_r84+augpool  C=r103_gte(+feats)")
    print("=" * 72)

    gte_path = Path(args.gte_oof)
    if not gte_path.exists():
        sys.exit(f"GTE OOF lists not found: {gte_path}\n"
                 f"Run scripts/expR103_gte_oof.py on the A100 first and download the aggregate.")

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

    print(f"{ts()} Loading R84 OOF (5 folds) + GTE OOF ...", flush=True)
    r84_per_fold = {k: load_r84_lists_for_fold(k) for k in range(N_FOLDS)}
    gte = load_gte_oof(gte_path)
    miss_gte = [i for i in range(n) if i not in gte]
    if miss_gte:
        print(f"  WARNING: {len(miss_gte)} cases missing GTE lists (first {miss_gte[:3]}) "
              f"— treated as empty GTE.", flush=True)

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

    # --- Pre-build per-case pools + feature matrices ---
    print(f"\n{ts()} === Pre-building pools + features (8000 cases) ===", flush=True)
    cf: dict[int, dict] = {}
    t_feat = time.time()
    hit_base = hit_aug = ortho30 = 0
    for i in range(n):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        gmaps = gte.get(i, {"ids": [], "ranks": {}, "scores": {}})
        gids = gmaps["ids"]

        pool_base = weighted_rrf(src_lists, SW_BASELINE, topk=POOL_K, k=RRF_K)
        src_aug = {**src_lists, "GTE": gids[:args.gte_rank_cap]}
        pool_aug = weighted_rrf(src_aug, SW_AUG, topk=POOL_K, k=RRF_K)

        gt = cases[i]["gt"]
        gp_base = pool_base.index(gt) if gt in pool_base else -1
        gp_aug = pool_aug.index(gt) if gt in pool_aug else -1
        hit_base += gp_base >= 0
        hit_aug += gp_aug >= 0
        # orthogonal recall: GT in GTE top-30 but NOT in the R54-stacked pool.
        # window tied to the admission cap so a "unique" can always actually enter the aug pool.
        gt_in_gte30 = gt in gids[:min(30, args.gte_rank_cap)]
        gte_unique = gt_in_gte30 and (gp_base < 0)
        ortho30 += gte_unique

        owner = case_fold[i]
        r84m = r84_per_fold[owner][i]

        feats_A = overwrite_r84(featurize37(i, src_lists, pool_base), pool_base,
                                r84m["ranks"], r84m["scores"])
        feats_B = overwrite_r84(featurize37(i, src_lists, pool_aug), pool_aug,
                                r84m["ranks"], r84m["scores"])
        # GTE feature depth == admission depth (gte_rank_cap), so source and feature
        # levers are gated identically -> clean B-vs-C decomposition under any cap.
        cap_ids = gids[:args.gte_rank_cap]
        cap_ranks = {t: r + 1 for r, t in enumerate(cap_ids)}
        cap_scores = {t: gmaps["scores"][t] for t in cap_ids if t in gmaps["scores"]}
        gte3 = gte_three_cols(pool_aug, cap_ranks, cap_scores)
        feats_C = np.concatenate([feats_B, gte3], axis=1)

        cf[i] = {
            "pool_base": pool_base, "pool_aug": pool_aug,
            "feats_A": feats_A, "feats_C": feats_C,            # B = feats_C[:, :37]
            "gp_base": gp_base, "gp_aug": gp_aug,
            "gt": gt, "gte_unique": gte_unique,
        }
        if (i + 1) % 1000 == 0:
            print(f"    {i + 1}/{n} ({time.time() - t_feat:.0f}s) "
                  f"pool_hit base={hit_base} aug={hit_aug} ortho30={ortho30}", flush=True)
    print(f"  pool_hit: base={hit_base}/{n} ({hit_base/n:.4f})  "
          f"aug(+GTE)={hit_aug}/{n} ({hit_aug/n:.4f})  Δ={hit_aug-hit_base:+d}", flush=True)
    print(f"  GTE orthogonal recall (GT in GTE@30, absent from R54-stacked pool): "
          f"{ortho30}/{n}", flush=True)
    print(f"  feat-build elapsed: {time.time() - t_feat:.0f}s", flush=True)

    # --- 5-fold OOF: train 3 sibling LRs, score held-out fold ---
    results: list[dict] = []
    gte_imp_accum = defaultdict(float)
    for fk in range(N_FOLDS):
        print(f"\n{ts()} === FOLD {fk} held out ===", flush=True)
        eval_idx = fold_to_idx[fk]
        train_idx = [i for i in range(n) if case_fold[i] != fk]

        def stack(feat_key, pool_key, gp_key, ncol):
            X, y, grp = [], [], []
            for i in train_idx:
                c = cf[i]
                P = c[pool_key]; F = c[feat_key]
                gp = c[gp_key]
                for r in range(len(P)):
                    X.append(F[r, :ncol]); y.append(1.0 if r == gp else 0.0)
                grp.append(len(P))
            return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32), grp

        # Arm A
        print("  train A (base_r84, R54 pool, 37) ...", flush=True)
        XA, yA, gA = stack("feats_A", "pool_base", "gp_base", 37)
        lrA = lgb.train(LR_PARAMS, lgb.Dataset(XA, label=yA, group=gA,
                        feature_name=FEAT_NAMES_R84), num_boost_round=LR_NUM_BOOST_ROUND)
        del XA, yA
        # Arm B
        print("  train B (base_r84, aug pool, 37) ...", flush=True)
        XB, yB, gB = stack("feats_C", "pool_aug", "gp_aug", 37)
        lrB = lgb.train(LR_PARAMS, lgb.Dataset(XB, label=yB, group=gB,
                        feature_name=FEAT_NAMES_R84), num_boost_round=LR_NUM_BOOST_ROUND)
        del XB, yB
        # Arm C
        print("  train C (r103_gte, aug pool, 40) ...", flush=True)
        XC, yC, gC = stack("feats_C", "pool_aug", "gp_aug", 40)
        lrC = lgb.train(LR_PARAMS, lgb.Dataset(XC, label=yC, group=gC,
                        feature_name=FEAT_NAMES_R103), num_boost_round=LR_NUM_BOOST_ROUND)
        del XC, yC
        for fn, im in zip(FEAT_NAMES_R103, lrC.feature_importance(importance_type="gain")):
            gte_imp_accum[fn] += float(im)

        print(f"  scoring fold {fk} ({len(eval_idx)} cases) ...", flush=True)
        for i in eval_idx:
            c = cf[i]
            pb, pa = c["pool_base"], c["pool_aug"]
            gpb, gpa = c["gp_base"], c["gp_aug"]
            sA = lrA.predict(c["feats_A"])
            sB = lrB.predict(c["feats_C"][:, :37])
            sC = lrC.predict(c["feats_C"])
            oA = np.argsort(-sA, kind="mergesort")
            oB = np.argsort(-sB, kind="mergesort")
            oC = np.argsort(-sC, kind="mergesort")

            def rank_of(order, gp):
                if gp < 0:
                    return -1
                w = np.where(order == gp)[0]
                return int(w[0]) + 1 if len(w) else -1

            rA = rank_of(oA, gpb); rB = rank_of(oB, gpa); rC = rank_of(oC, gpa)
            topA = [pb[int(j)] for j in oA[:TOP_K]]
            topC = [pa[int(j)] for j in oC[:TOP_K]]

            results.append({
                "case_idx": i, "fold": fk,
                "n_prior_music": int(cases[i]["n_prior_music"]),
                "same_artist": same_artist_case(cases[i], maps["track_artist"]),
                "gte_unique": c["gte_unique"],
                "rA": rA, "rB": rB, "rC": rC,
                "ndA": ndcg_at_k(rA, TOP_K), "ndB": ndcg_at_k(rB, TOP_K),
                "ndC": ndcg_at_k(rC, TOP_K),
                "A_in20": 0 < rA <= TOP_K, "B_in20": 0 < rB <= TOP_K, "C_in20": 0 < rC <= TOP_K,
                "overlap_AC": len(set(topA) & set(topC)),
                "top1_AC_eq": int(bool(topA) and bool(topC) and topA[0] == topC[0]),
            })
        del lrA, lrB, lrC

    # --- Aggregate ---
    def avg(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    h7 = [r for r in results if r["n_prior_music"] == 7]
    same = [r for r in results if r["same_artist"]]
    diff = [r for r in results if not r["same_artist"]]

    metrics = {}
    for name, rows in [("all", results), ("h7", h7), ("same_artist", same), ("diff_artist", diff)]:
        A, B, C = avg(rows, "ndA"), avg(rows, "ndB"), avg(rows, "ndC")
        metrics[name] = {
            "n": len(rows), "A_base_r84": A, "B_augpool": B, "C_r103_gte": C,
            "dC_vs_A": C - A, "dC_vs_B": C - B, "dB_vs_A": B - A,
        }

    h7_rec = sum(1 for r in h7 if r["C_in20"] and not r["A_in20"])
    h7_lost = sum(1 for r in h7 if r["A_in20"] and not r["C_in20"])
    overlap_AC = avg(results, "overlap_AC")
    churn = (1 - avg(results, "top1_AC_eq")) * 80

    # GTE-unique conversion: of GTE-unique cases, how many land in top-20 under C
    uniq = [r for r in results if r["gte_unique"]]
    uniq_h7 = [r for r in uniq if r["n_prior_music"] == 7]
    uniq_conv = sum(1 for r in uniq if r["C_in20"])
    uniq_conv_h7 = sum(1 for r in uniq_h7 if r["C_in20"])

    by_fold_h7 = {}
    for k in range(N_FOLDS):
        rows = [r for r in h7 if r["fold"] == k]
        if rows:
            by_fold_h7[k] = {"n": len(rows), "A": avg(rows, "ndA"), "C": avg(rows, "ndC"),
                             "dC_vs_A": avg(rows, "ndC") - avg(rows, "ndA")}

    gte_imp = {k: gte_imp_accum[k] for k in ["gte_rank_inv", "gte_presence", "gte_cosine"]}
    tot_imp = sum(gte_imp_accum.values()) or 1.0
    gte_imp_frac = sum(gte_imp.values()) / tot_imp

    h7_d = metrics["h7"]["dC_vs_A"]
    sa_d = metrics["same_artist"]["dC_vs_A"]
    diff_d = metrics["diff_artist"]["dC_vs_A"]
    A1 = h7_d >= 0.005
    A2 = h7_rec > h7_lost
    B1 = sa_d >= -0.005
    B2 = diff_d >= -0.005
    B3 = overlap_AC >= 8.0
    gte_used = gte_imp_frac > 0.01            # stop condition: LR ignores GTE feats

    if (A1 or A2) and B1 and B2 and B3 and gte_used:
        verdict = "PROCEED_TO_BLIND"
    elif not B1 or not B2:
        verdict = "ARCHIVE_SPRINT"
    elif not gte_used:
        verdict = "ARCHIVE_GTE_IGNORED"
    elif h7_d < 0.002 and not A2:
        verdict = "ARCHIVE_NO_LIFT"
    else:
        verdict = "INVESTIGATE"

    # --- Report ---
    print(f"\n{ts()} === AGGREGATE (5-fold OOF) ===")
    print(f"  {'subset':12} {'n':>5} {'A_base':>8} {'B_aug':>8} {'C_gte':>8} "
          f"{'ΔC-A':>8} {'ΔC-B':>8} {'ΔB-A':>8}")
    for name, m in metrics.items():
        print(f"  {name:12} {m['n']:5d} {m['A_base_r84']:8.4f} {m['B_augpool']:8.4f} "
              f"{m['C_r103_gte']:8.4f} {m['dC_vs_A']:+8.4f} {m['dC_vs_B']:+8.4f} {m['dB_vs_A']:+8.4f}")
    print(f"\n  h7 recovered (C vs A): {h7_rec}   lost: {h7_lost}   net: {h7_rec - h7_lost:+d}")
    print(f"  top-20 overlap A vs C: {overlap_AC:.2f}/20   top-1 churn: {churn:.1f}/80")
    print(f"  GTE orthogonal-unique cases (all): {len(uniq)} | h7: {len(uniq_h7)}")
    print(f"  GTE-unique CONVERTED to top-20 under C: {uniq_conv}/{len(uniq)} (all) "
          f"| {uniq_conv_h7}/{len(uniq_h7)} (h7)")
    print(f"  arm-C LR gain importance: gte feats = {sum(gte_imp.values()):.0f} "
          f"({gte_imp_frac*100:.2f}% of total)  {gte_imp}")
    print(f"\n  per-fold h7 ΔC-A:")
    for k, m in by_fold_h7.items():
        print(f"    fold {k}: n={m['n']} A={m['A']:.4f} C={m['C']:.4f} Δ={m['dC_vs_A']:+.4f}")
    print(f"\n  Gates:")
    print(f"    A1 h7 Δ≥+0.005:          {A1}  ({h7_d:+.4f})")
    print(f"    A2 h7 rec>lost:          {A2}  ({h7_rec}>{h7_lost})")
    print(f"    B1 same-artist Δ≥-0.005: {B1}  ({sa_d:+.4f})")
    print(f"    B2 diff-artist Δ≥-0.005: {B2}  ({diff_d:+.4f})")
    print(f"    B3 overlap≥8/20:         {B3}  ({overlap_AC:.2f})")
    print(f"    GTE feats used (>1%):    {gte_used}  ({gte_imp_frac*100:.2f}%)")
    print(f"\n  VERDICT: {verdict}", flush=True)

    out = {
        "experiment": "R103 — GTE-7B added source (R84 playbook), 5-fold OOF",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "gte_oof": str(gte_path), "gte_weight": args.gte_weight,
        "verdict": verdict,
        "feature_stacks": {"A_B": FEAT_NAMES_R84, "C": FEAT_NAMES_R103},
        "pool_hit": {"base": hit_base, "aug": hit_aug, "delta": hit_aug - hit_base, "n": n},
        "gte_orthogonal_recall_at30": ortho30,
        "metrics": metrics,
        "metrics_per_fold_h7": by_fold_h7,
        "h7_recovery_C_vs_A": {"recovered": h7_rec, "lost": h7_lost, "net": h7_rec - h7_lost},
        "churn": {"overlap_AC": overlap_AC, "top1_churn_per_80": churn},
        "gte_unique_conversion": {
            "n_unique_all": len(uniq), "converted_top20_all": uniq_conv,
            "n_unique_h7": len(uniq_h7), "converted_top20_h7": uniq_conv_h7},
        "gte_feature_importance_gain": gte_imp, "gte_feature_importance_frac": gte_imp_frac,
        "gates": {"A1": [h7_d, A1], "A2": [[h7_rec, h7_lost], A2], "B1": [sa_d, B1],
                  "B2": [diff_d, B2], "B3": [overlap_AC, B3], "gte_used": [gte_imp_frac, gte_used]},
    }
    def _ser(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return float(o)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2, default=_ser)
    print(f"\n{ts()} Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
