"""R85 Phase 1 — multimodal integration vs R84c OOF baseline (Mac-only).

Two integration approaches, both 5-fold OOF sibling-LR test:

- R85b: LR FEATURE ADDITION (cheapest, pool stable)
  Adds image_siglip + attributes_qwen rank_inv/presence/cosine features to
  the R84-feature LR. 37 → 43 columns. Pool stays R54-stacked POOL_K=300.

- R85a: RRF SOURCE ADDITION (pool grows by 2)
  Adds image_siglip + attributes_qwen as RRF sources 9 and 10 with low
  weights (default 0.5). Pool changes; features (R39 + R84) recomputed
  per fold on the new pool.

- R85c: SELECTIVE ROUTING (if R85a/R85b miss but segment-positive)
  Build observable rule analogous to R84c using R54c margin AND
  image_siglip top-1 strength.

Baseline: R84c sibling-R84 5-fold OOF LR (= R84c production on dev =
sibling_r84 from prior R84b/R84c sweep).

Outputs:
- exp/eval/expR85_phase1_sweep.json
- docs/r85_phase1_result.md
"""
from __future__ import annotations

import argparse
import hashlib
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
    FEAT_R39_ALL, FEAT_R54, FEAT_ALL, _featurize_row,
)
import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

# --- Constants ---
SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}
RRF_K = 20
POOL_K = 300
TOP_K = 20
N_FOLDS = 5
ANCHOR_K = 3  # max-of-last-3 played anchor (same as R85 Phase 0)
MODALITY_TOP_K = 300

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
FEAT_CACHE = REPO / "cache" / "r84b" / "case_features.pkl"
META_QWEN_DIR = REPO / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b"
MULTIMOD_LISTS_DIR = REPO / "cache" / "r85" / "multimodal_lists"

OUT_JSON = REPO / "exp" / "eval" / "expR85_phase1_sweep.json"

LR_PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
    "num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 10,
    "verbose": -1, "seed": 0,
}
LR_NUM_BOOST_ROUND = 300

N_R39 = len(FEAT_R39_ALL)
# 37-col R84 sibling schema (replace R54 with R84)
FEAT_NAMES_R84_ONLY = list(FEAT_R39_ALL) + ["r84_rank_inv", "r84_presence", "r84_cosine"]
# 43-col R85b schema (R84 + IMG + META)
FEAT_NAMES_R85B = FEAT_NAMES_R84_ONLY + [
    "img_rank_inv", "img_presence", "img_cosine",
    "meta_rank_inv", "meta_presence", "meta_cosine",
]

GATE = {
    "h7_delta_ge": 0.005, "all_delta_ge": 0.0,
    "same_artist_delta_ge": -0.005, "diff_artist_delta_ge": 0.0,
    "recov_ge_lost": True, "overlap_ge": 14.0,
}


def ts(): return f"[{datetime.now():%H:%M:%S}]"
def head_sha():
    g = shutil.which("git")
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip() if g else "no-git"


def ndcg_at_k(gt_rank, k):
    return 1.0 / math.log2(gt_rank + 1) if 0 < gt_rank <= k else 0.0


def load_modality_emb(name):
    """Load image_siglip or attributes_qwen as L2-normalized matrix + tids."""
    if name == "attributes_qwen":
        emb = np.load(META_QWEN_DIR / "vectors.npy").astype(np.float32)
        tids = json.load(open(META_QWEN_DIR / "track_ids.json"))
    else:  # image_siglip / audio_clap / lyrics_qwen via HF
        from datasets import DownloadConfig, load_dataset  # type: ignore
        try:
            ds = load_dataset(
                "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                download_config=DownloadConfig(local_files_only=True),
            )["all_tracks"]
        except Exception:
            ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")["all_tracks"]
        col_map = {
            "image_siglip": "image-siglip2",
            "audio_clap": "audio-laion_clap",
            "lyrics_qwen": "lyrics-qwen3_embedding_0.6b",
        }
        col = col_map[name]
        tids = []
        vecs = []
        dim = None
        for item in ds:
            tids.append(str(item["track_id"]))
            v = item.get(col)
            if v is not None and dim is None and len(v) > 0:
                dim = len(v)
            vecs.append(v)
        emb = np.zeros((len(vecs), dim), dtype=np.float32)
        for i, v in enumerate(vecs):
            if v is not None and len(v) == dim:
                emb[i] = v
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(norms > 0, norms, 1.0), tids


def retrieve_max_recent_for_all(cases, emb, tids, recent_k=ANCHOR_K, topk=MODALITY_TOP_K):
    """Returns {case_idx: [(tid, score), ...]} per case."""
    tid_to_idx = {tid: i for i, tid in enumerate(tids)}
    out = {}
    for i, case in enumerate(cases):
        played = case.get("music_turns", [])
        recent = played[-recent_k:]
        recent_idx = [tid_to_idx[t] for t in recent if t in tid_to_idx]
        if not recent_idx:
            out[i] = []
            continue
        played_set = {tid_to_idx[t] for t in played if t in tid_to_idx}
        recent_embs = emb[recent_idx]
        sims = emb @ recent_embs.T
        max_sims = sims.max(axis=1)
        for pi in played_set:
            max_sims[pi] = -np.inf
        if topk >= len(max_sims):
            order = np.argsort(-max_sims)
        else:
            top_idx = np.argpartition(-max_sims, topk)[:topk]
            order = top_idx[np.argsort(-max_sims[top_idx])]
        out[i] = [(tids[int(j)], float(max_sims[int(j)])) for j in order]
    return out


def load_or_compute_modality_lists(name, cases):
    cache_path = MULTIMOD_LISTS_DIR / f"{name}_top{MODALITY_TOP_K}.json"
    if cache_path.exists():
        print(f"  loading cached {name} lists from {cache_path.name}")
        with open(cache_path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    print(f"  computing {name} top-{MODALITY_TOP_K} for {len(cases)} cases...")
    t0 = time.time()
    emb, tids = load_modality_emb(name)
    print(f"    embeddings: {emb.shape}")
    out = retrieve_max_recent_for_all(cases, emb, tids)
    print(f"    retrieval done in {time.time() - t0:.0f}s")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in out.items()}, f)
    print(f"    cached -> {cache_path.name}")
    return out


def train_sibling_lr(case_features, train_idx, feat_key, feat_names, params=LR_PARAMS):
    X, y, gt = [], [], []
    for i in train_idx:
        cf = case_features[i]
        pool_len = len(cf["pool"])
        for k_row in range(pool_len):
            X.append(cf[feat_key][k_row])
            y.append(1.0 if k_row == cf["gt_pos"] else 0.0)
        gt.append(pool_len)
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    ds = lgb.Dataset(X, label=y, group=gt, feature_name=feat_names)
    return lgb.train(params, ds, num_boost_round=LR_NUM_BOOST_ROUND)


def extend_features_with_modality(case_features, img_lists, meta_lists):
    """Builds feats_r85b (43 col) per case by appending IMG + META features
    to feats_r84_only (37 cols)."""
    for i, cf in case_features.items():
        pool = cf["pool"]
        n_pool = len(pool)
        img = img_lists.get(i, [])
        meta = meta_lists.get(i, [])
        img_ranks = {t: r + 1 for r, (t, _) in enumerate(img[:MODALITY_TOP_K])}
        img_scores = {t: float(s) for t, s in img}
        meta_ranks = {t: r + 1 for r, (t, _) in enumerate(meta[:MODALITY_TOP_K])}
        meta_scores = {t: float(s) for t, s in meta}
        extra = np.zeros((n_pool, 6), dtype=np.float64)
        for k_row, tid in enumerate(pool):
            extra[k_row, 0] = (1.0 / img_ranks[tid]) if tid in img_ranks else 0.0
            extra[k_row, 1] = 1.0 if tid in img_ranks else 0.0
            extra[k_row, 2] = img_scores.get(tid, 0.0)
            extra[k_row, 3] = (1.0 / meta_ranks[tid]) if tid in meta_ranks else 0.0
            extra[k_row, 4] = 1.0 if tid in meta_ranks else 0.0
            extra[k_row, 5] = meta_scores.get(tid, 0.0)
        cf["feats_r85b"] = np.concatenate([cf["feats_r84_only"], extra], axis=1)


def build_rrf_pool_with_modality(cases, payload, r21_source, r54_source, case_index,
                                   img_lists, meta_lists, weights):
    """For each case, build R54-stacked + modality RRF pool with given weights."""
    pool_per_case = {}
    sw = {**SW_BASELINE, "IMG": weights.get("IMG", 0.5), "META": weights.get("META", 0.5)}
    for i in range(len(cases)):
        src_lists = c3.make_source_lists(
            payload, r21_source, r54_source, case_index["als_source"], i)
        img_tids = [t for t, _ in img_lists.get(i, [])][:MODALITY_TOP_K]
        meta_tids = [t for t, _ in meta_lists.get(i, [])][:MODALITY_TOP_K]
        src_lists["IMG"] = img_tids
        src_lists["META"] = meta_tids
        pool = weighted_rrf(src_lists, sw, topk=POOL_K, k=RRF_K)
        pool_per_case[i] = pool
    return pool_per_case


def metrics_from_scores(cases, case_features, scores_per_case_dict, maps,
                         score_key, fold_for_case, pool_override=None):
    rows = []
    for i, sd in scores_per_case_dict.items():
        if isinstance(sd, dict):
            s = sd[score_key]
        else:
            s = sd
        cf = case_features[i]
        pool = pool_override.get(i, cf["pool"]) if pool_override else cf["pool"]
        gt_pos = pool.index(cases[i]["gt"]) if cases[i]["gt"] in pool else -1
        order = np.argsort(-s, kind="mergesort")
        rank = -1
        if gt_pos >= 0:
            p = np.where(order == gt_pos)[0]
            if len(p):
                rank = int(p[0]) + 1
        rows.append({
            "case_idx": i, "fold": fold_for_case[i],
            "n_prior_music": int(cases[i]["n_prior_music"]),
            "same_artist": same_artist_case(cases[i], maps["track_artist"]),
            "rank": rank, "ndcg20": ndcg_at_k(rank, TOP_K),
            "in_top20": rank > 0 and rank <= TOP_K,
            "top20": [pool[int(j)] for j in order[:TOP_K]],
        })
    return rows


def metrics_summary(rows_test, rows_baseline):
    def avg(rows, key): return float(np.mean([r[key] for r in rows])) if rows else 0.0
    h7 = [r for r in rows_test if r["n_prior_music"] == 7]
    h7_b = [r for r in rows_baseline if r["n_prior_music"] == 7]
    same = [r for r in rows_test if r["same_artist"]]
    same_b = [r for r in rows_baseline if r["same_artist"]]
    diff = [r for r in rows_test if not r["same_artist"]]
    diff_b = [r for r in rows_baseline if not r["same_artist"]]
    out = {}
    for name, (rt, rb) in [("h7", (h7, h7_b)), ("all", (rows_test, rows_baseline)),
                            ("same_artist", (same, same_b)), ("diff_artist", (diff, diff_b))]:
        out[name] = {"n": len(rt), "test": avg(rt, "ndcg20"),
                     "baseline": avg(rb, "ndcg20"),
                     "delta": avg(rt, "ndcg20") - avg(rb, "ndcg20")}
    # recovery (top-20 in/out)
    h7_b_in = {r["case_idx"]: r["in_top20"] for r in h7_b}
    h7_t_in = {r["case_idx"]: r["in_top20"] for r in h7}
    recov = sum(1 for cid, t in h7_t_in.items() if t and not h7_b_in.get(cid, False))
    lost = sum(1 for cid, b in h7_b_in.items() if b and not h7_t_in.get(cid, False))
    out["h7_recovery"] = {"recovered": recov, "lost": lost, "net": recov - lost}
    # overlap
    b_top = {r["case_idx"]: set(r["top20"]) for r in rows_baseline}
    overlaps = [len(set(r["top20"]) & b_top.get(r["case_idx"], set())) for r in rows_test]
    out["overlap_mean"] = float(np.mean(overlaps)) if overlaps else 0.0
    return out


def gate_eval(s):
    h7_d, all_d = s["h7"]["delta"], s["all"]["delta"]
    sa_d, diff_d = s["same_artist"]["delta"], s["diff_artist"]["delta"]
    recov, lost = s["h7_recovery"]["recovered"], s["h7_recovery"]["lost"]
    overlap = s["overlap_mean"]
    g = {
        "h7": (h7_d >= GATE["h7_delta_ge"], h7_d),
        "all": (all_d >= GATE["all_delta_ge"], all_d),
        "same": (sa_d >= GATE["same_artist_delta_ge"], sa_d),
        "diff": (diff_d >= GATE["diff_artist_delta_ge"], diff_d),
        "recov": (recov >= lost, [recov, lost]),
        "overlap": (overlap >= GATE["overlap_ge"], overlap),
    }
    return all(v[0] for v in g.values()), g


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skip-r85a", action="store_true",
                   help="Run only R85b (LR feature addition); skip R85a pool extension")
    args = p.parse_args()

    t0 = time.time()
    print(f"{ts()} R85 Phase 1 — multimodal integration sweep")
    print("=" * 70)

    # Load fundamentals
    print(f"\n{ts()} Loading payload + R21/R54 OOF + maps...")
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
    fold_for_case = {i: case_fold[i] for i in range(n)}
    fold_to_idx = {k: [i for i in range(n) if case_fold[i] == k] for k in range(N_FOLDS)}

    # Modality lists
    print(f"\n{ts()} === Loading/computing modality top-300 lists ===")
    img_lists = load_or_compute_modality_lists("image_siglip", cases)
    meta_lists = load_or_compute_modality_lists("attributes_qwen", cases)

    # Load case_features cache (R54-stacked pool)
    print(f"\n{ts()} Loading case_features cache ({FEAT_CACHE.stat().st_size/1e6:.0f} MB)...")
    with open(FEAT_CACHE, "rb") as f:
        case_features = pickle.load(f)
    print(f"  {len(case_features)} cases loaded")

    # Extend with R85b features (43 cols)
    print(f"\n{ts()} Extending features with IMG + META → 43 cols (R85b)")
    extend_features_with_modality(case_features, img_lists, meta_lists)

    # --- BASELINE: R84c sibling (37 cols, R84 features) on R54-stacked pool ---
    print(f"\n{ts()} === Training BASELINE R84c sibling (5-fold OOF) ===")
    all_scores_baseline = {}
    for fold_k in range(N_FOLDS):
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = fold_to_idx[fold_k]
        lr = train_sibling_lr(case_features, train_idx, "feats_r84_only",
                                FEAT_NAMES_R84_ONLY)
        for i in eval_idx:
            all_scores_baseline[i] = lr.predict(case_features[i]["feats_r84_only"])
    rows_baseline = metrics_from_scores(cases, case_features, all_scores_baseline,
                                          maps, score_key=None,  # unused since scores are arrays
                                          fold_for_case=fold_for_case)
    # baseline "summary" against itself — for displaying h7 baseline
    baseline_h7_ndcg = float(np.mean([r["ndcg20"] for r in rows_baseline if r["n_prior_music"] == 7]))
    baseline_all_ndcg = float(np.mean([r["ndcg20"] for r in rows_baseline]))
    print(f"  baseline (R84c sibling): h7 nDCG={baseline_h7_ndcg:.4f}  "
          f"all nDCG={baseline_all_ndcg:.4f}")

    # --- R85b: LR FEATURE ADDITION (43 col) ---
    print(f"\n{ts()} === R85b: LR feature addition (43 col, pool unchanged) ===")
    all_scores_r85b = {}
    for fold_k in range(N_FOLDS):
        train_idx = [i for i in range(n) if case_fold[i] != fold_k]
        eval_idx = fold_to_idx[fold_k]
        t_lr = time.time()
        lr = train_sibling_lr(case_features, train_idx, "feats_r85b", FEAT_NAMES_R85B)
        for i in eval_idx:
            all_scores_r85b[i] = lr.predict(case_features[i]["feats_r85b"])
        print(f"  fold {fold_k}: {time.time() - t_lr:.0f}s")
    rows_r85b = metrics_from_scores(cases, case_features, all_scores_r85b,
                                      maps, score_key=None, fold_for_case=fold_for_case)
    summary_r85b = metrics_summary(rows_r85b, rows_baseline)
    pass_r85b, gates_r85b = gate_eval(summary_r85b)
    print(f"\n  R85b vs R84c baseline:")
    for k in ["h7", "all", "same_artist", "diff_artist"]:
        m = summary_r85b[k]
        print(f"    {k:14}  n={m['n']:5d}  base={m['baseline']:.4f}  "
              f"r85b={m['test']:.4f}  Δ={m['delta']:+.4f}")
    rec = summary_r85b["h7_recovery"]
    print(f"    recov/lost = {rec['recovered']}/{rec['lost']}  net={rec['net']:+d}")
    print(f"    overlap mean = {summary_r85b['overlap_mean']:.2f}/20")
    print(f"    GATE: {'PASS' if pass_r85b else 'fail'}")

    # --- R85a: RRF SOURCE ADDITION (pool grows by 2 sources at weight 0.5) ---
    if args.skip_r85a:
        print(f"\n{ts()} === R85a SKIPPED (--skip-r85a) ===")
        summary_r85a = None
        pass_r85a = False
    else:
        print(f"\n{ts()} === R85a: RRF source addition (IMG + META @ w=0.5) ===")
        print(f"  Building 10-source pool with IMG/META weights = 0.5...")
        t_pool = time.time()
        case_index = c3.build_case_index(
            payload, r21_source, r54_source, r54_scores,
            als_factors, als_track_ids, als_to_idx,
        )
        case_index["als_to_idx"] = als_to_idx
        pool_per_case = build_rrf_pool_with_modality(
            cases, payload, r21_source, r54_source, case_index,
            img_lists, meta_lists, weights={"IMG": 0.5, "META": 0.5},
        )
        print(f"  pool built in {time.time() - t_pool:.0f}s")

        # Re-featurize per case on new pool (37 cols R84-substituted)
        print(f"\n  Re-featurizing on new pool (R39 + R84-substituted, 37 cols)...")
        t_feat = time.time()
        new_case_features = {}
        for i in range(n):
            src_lists = c3.make_source_lists(
                payload, r21_source, r54_source, case_index["als_source"], i)
            img_tids = [t for t, _ in img_lists.get(i, [])][:MODALITY_TOP_K]
            meta_tids = [t for t, _ in meta_lists.get(i, [])][:MODALITY_TOP_K]
            src_lists["IMG"] = img_tids
            src_lists["META"] = meta_tids
            pool = pool_per_case[i]
            gt_pos = pool.index(cases[i]["gt"]) if cases[i]["gt"] in pool else -1
            r21_rank_map = {t: r + 1 for r, t in enumerate(src_lists["R21"][:POOL_K])}
            r54_rank_map = {t: r + 1 for r, t in enumerate(src_lists["R54"][:POOL_K])}
            case = cases[i]
            feats_r54_new = _featurize_row(
                pool, src_lists, r21_rank_map, r54_rank_map, r54_scores[i],
                case["user_query"], case["history"], case["music_turns"],
                set(case["music_turns"]),
                maps["track_artist"], maps["track_tags"], maps["track_title_toks"],
                maps["track_artist_toks"], maps["track_meta_toks"],
                als_factors, als_to_idx, case_index["als_session_vecs"][i],
                track_pop, max_pop, track_album,
            )
            # R84-substituted: swap last 3 cols with R84 ranks/scores
            owning_fold = case_fold[i]
            r84_path = (REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json"
                         if owning_fold == 0 else
                         REPO / f"cache/r84/phase1_fold{owning_fold}/oof_r84_lists.json")
            # Load once and cache to avoid repeated JSON reads
            if not hasattr(main, "_r84_per_fold"):
                main._r84_per_fold = {}
                for k in range(N_FOLDS):
                    path = (REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json"
                             if k == 0 else
                             REPO / f"cache/r84/phase1_fold{k}/oof_r84_lists.json")
                    raw = json.load(open(path))
                    main._r84_per_fold[k] = {
                        int(cidx): {"ranks": {t: r + 1 for r, (t, _) in enumerate(v)},
                                     "scores": {t: float(s) for t, s in v}}
                        for cidx, v in raw.items()
                    }
            r84_data = main._r84_per_fold[owning_fold].get(i, {"ranks": {}, "scores": {}})
            feats_r84 = feats_r54_new.copy()
            for k_row, tid in enumerate(pool):
                feats_r84[k_row, N_R39 + 0] = (1.0 / r84_data["ranks"][tid]) if tid in r84_data["ranks"] else 0.0
                feats_r84[k_row, N_R39 + 1] = 1.0 if tid in r84_data["ranks"] else 0.0
                feats_r84[k_row, N_R39 + 2] = r84_data["scores"].get(tid, 0.0)
            new_case_features[i] = {
                "pool": pool, "gt_pos": gt_pos, "feats_r84_only": feats_r84,
            }
            if (i + 1) % 1000 == 0:
                print(f"    feat {i + 1}/{n} ({time.time() - t_feat:.0f}s)")

        print(f"  feature rebuild done in {time.time() - t_feat:.0f}s")

        # Per-fold sibling LR + score
        print(f"\n  R85a per-fold sibling LR (37 col on new pool)...")
        all_scores_r85a = {}
        for fold_k in range(N_FOLDS):
            train_idx = [i for i in range(n) if case_fold[i] != fold_k]
            eval_idx = fold_to_idx[fold_k]
            t_lr = time.time()
            lr = train_sibling_lr(new_case_features, train_idx, "feats_r84_only",
                                    FEAT_NAMES_R84_ONLY)
            for i in eval_idx:
                all_scores_r85a[i] = lr.predict(new_case_features[i]["feats_r84_only"])
            print(f"    fold {fold_k}: {time.time() - t_lr:.0f}s")

        # Build rows on new pool for R85a; baseline is still old-pool rows (cross-pool overlap is fuzzy)
        rows_r85a = metrics_from_scores(cases, new_case_features, all_scores_r85a,
                                          maps, score_key=None, fold_for_case=fold_for_case)
        summary_r85a = metrics_summary(rows_r85a, rows_baseline)
        pass_r85a, _ = gate_eval(summary_r85a)
        print(f"\n  R85a vs R84c baseline (cross-pool — overlap is approximate):")
        for k in ["h7", "all", "same_artist", "diff_artist"]:
            m = summary_r85a[k]
            print(f"    {k:14}  n={m['n']:5d}  base={m['baseline']:.4f}  "
                  f"r85a={m['test']:.4f}  Δ={m['delta']:+.4f}")
        rec = summary_r85a["h7_recovery"]
        print(f"    recov/lost = {rec['recovered']}/{rec['lost']}  net={rec['net']:+d}")
        print(f"    overlap mean = {summary_r85a['overlap_mean']:.2f}/20")
        print(f"    GATE: {'PASS' if pass_r85a else 'fail'}")

    # --- Verdict + summary ---
    print(f"\n{ts()} === VERDICT ===")
    if pass_r85b:
        winner = "R85b"
        print(f"  R85b PASSES gate. Recommend candidate prep on R85b.")
    elif (summary_r85a is not None) and pass_r85a:
        winner = "R85a"
        print(f"  R85a PASSES gate (R85b did not). Recommend candidate prep on R85a.")
    else:
        winner = None
        print(f"  Neither global variant passes. Consider R85c (selective routing).")

    # Persist
    out = {
        "experiment": "R85 Phase 1 — multimodal integration sweep",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "head_sha": head_sha(),
        "winner": winner,
        "n_cases": n,
        "baseline_h7_ndcg": baseline_h7_ndcg,
        "baseline_all_ndcg": baseline_all_ndcg,
        "R85b": {
            "summary": summary_r85b, "passes_gate": pass_r85b,
        },
        "R85a": {
            "summary": summary_r85a, "passes_gate": pass_r85a,
        } if summary_r85a else None,
        "gate_definition": GATE,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
