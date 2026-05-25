"""R85 Phase 0 — multimodal headroom inventory vs R84c.

Mac-only. Measures whether any of {audio CLAP, lyrics Qwen, attributes Qwen,
image SigLIP} produces unique h7 top-30 recoveries that R84 doesn't already
surface, with acceptable same-artist behavior.

Pipeline:
1. Load/download 4 modality embeddings via HF Track-Embeddings dataset (cached).
2. For each modality, build per-dev-case top-300 via max-of-last-3-played
   (R36 anchor strategy).
3. Compare to R84 source-alone 5-fold OOF top-30:
   - unique h7 GTs in modality top-30 NOT in R84 top-30
   - unique h7 GTs in modality top-300 NOT in R84 top-300 (pool-level)
4. Segment by same-vs-diff artist + R84c sibling-R54 LR margin (low/mid/high).
5. Gate per modality: >=5 unique h7 top-30 recoveries, with same-artist not
   destroyed (>=20% of recoveries on same-artist subset, mirroring base rate).

Outputs:
- exp/eval/expR85_phase0_inventory.json
- docs/r85_phase0_inventory_result.md
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
W0_STATS = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
META_QWEN_DIR = REPO / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b"

# R84 5-fold OOF lists (from Phase 0B fold 0 + Phase 1 folds 1-4)
R84_FOLD_LISTS = {
    0: REPO / "cache" / "r84" / "phase0b_fold0" / "oof_r84_lists.json",
    1: REPO / "cache" / "r84" / "phase1_fold1" / "oof_r84_lists.json",
    2: REPO / "cache" / "r84" / "phase1_fold2" / "oof_r84_lists.json",
    3: REPO / "cache" / "r84" / "phase1_fold3" / "oof_r84_lists.json",
    4: REPO / "cache" / "r84" / "phase1_fold4" / "oof_r84_lists.json",
}

OUT_JSON = REPO / "exp" / "eval" / "expR85_phase0_inventory.json"
OUT_MD = REPO / "docs" / "r85_phase0_inventory_result.md"
CACHE_DIR = REPO / "cache" / "r85"

POOL_K = 300
TOP_K_GATES = [20, 30, 300]
ANCHOR_K = 3  # last-3 played tracks
MIN_UNIQUE_H7_TOP30_FOR_GATE = 5


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_modality_embeddings(quick=False):
    """Returns {modality_name: {emb: np.ndarray (n, dim) L2-normalized, tids: list}}.

    Uses local metadata-qwen cache where available, downloads others from HF.
    """
    out = {}

    # 1. metadata-qwen3 — local on disk
    if (META_QWEN_DIR / "vectors.npy").exists():
        tids = json.load(open(META_QWEN_DIR / "track_ids.json"))
        emb = np.load(META_QWEN_DIR / "vectors.npy")
        emb = emb.astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms > 0, norms, 1.0)
        out["attributes_qwen"] = {"emb": emb, "tids": tids, "source": "local"}
        print(f"  attributes_qwen: {emb.shape} (local)")

    if quick:
        return out

    # 2/3/4. audio CLAP, lyrics Qwen, image SigLIP — HF Track-Embeddings
    from datasets import DownloadConfig, load_dataset  # type: ignore
    try:
        ds = load_dataset(
            "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            download_config=DownloadConfig(local_files_only=True),
        )["all_tracks"]
        print(f"  Track-Embeddings: local HF cache ({len(ds)} rows)")
    except Exception:
        print(f"  Track-Embeddings: not in HF cache, downloading...")
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings")["all_tracks"]
        print(f"  Downloaded ({len(ds)} rows)")

    col_map = {
        "audio-laion_clap": "audio_clap",
        "lyrics-qwen3_embedding_0.6b": "lyrics_qwen",
        "image-siglip2": "image_siglip",
    }
    track_ids_hf = []
    cols: dict[str, list] = {k: [] for k in col_map.values()}
    for item in ds:
        track_ids_hf.append(str(item["track_id"]))
        for hf_col, key in col_map.items():
            v = item.get(hf_col)
            cols[key].append(v)

    for key, vecs in cols.items():
        dim = None
        for v in vecs:
            if v is not None and len(v) > 0:
                dim = len(v)
                break
        if dim is None:
            print(f"  {key}: NO VECTORS available")
            continue
        arr = np.zeros((len(vecs), dim), dtype=np.float32)
        n_valid = 0
        for i, v in enumerate(vecs):
            if v is not None and len(v) == dim:
                arr[i] = v
                n_valid += 1
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.where(norms > 0, norms, 1.0)
        out[key] = {"emb": arr, "tids": track_ids_hf, "source": "hf"}
        print(f"  {key}: {arr.shape} valid={n_valid}/{len(vecs)}")

    return out


def retrieve_max_recent(played, emb, tid_to_idx, recent_k=ANCHOR_K, topk=POOL_K):
    """R36-style max-of-last-K anchored retrieval."""
    recent = played[-recent_k:]
    recent_idx = [tid_to_idx[t] for t in recent if t in tid_to_idx]
    if not recent_idx:
        return []
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
    return order.tolist()


def load_r84_oof_per_case_top30_sets():
    """Returns {case_idx: set(r84 top-30 tids from that case's held-out fold)}."""
    with open(W0_STATS, "rb") as f:
        w0_stats = pickle.load(f)
    case_fold = {row["case_idx"]: int(row["fold_idx"]) for row in w0_stats}

    fold_data = {}
    for fold_idx, path in R84_FOLD_LISTS.items():
        with open(path) as f:
            fold_data[fold_idx] = json.load(f)

    top30_per_case = {}
    top300_per_case = {}
    for case_idx, fold_idx in case_fold.items():
        fold_lists = fold_data[fold_idx]
        entries = fold_lists.get(str(case_idx), [])
        top30_per_case[case_idx] = set(t for t, _ in entries[:30])
        top300_per_case[case_idx] = set(t for t, _ in entries[:300])
    return top30_per_case, top300_per_case, case_fold


def same_artist_case(case, ta):
    """Is GT's artist also in played?"""
    gt_artist = ta.get(case["gt"], "")
    if not gt_artist:
        return False
    played_artists = {ta.get(t, "") for t in case.get("music_turns", [])}
    return gt_artist in played_artists


def evaluate_modality(mod_name, mod_data, cases, r84_top30, r84_top300,
                       ta, h7_indices, oof_r54_margins=None):
    """Returns dict of metrics for one modality."""
    emb = mod_data["emb"]
    tids = mod_data["tids"]
    tid_to_idx = {tid: i for i, tid in enumerate(tids)}
    n = len(cases)

    print(f"\n  {mod_name}: computing top-{POOL_K} for all 8000 cases...")
    t0 = time.time()
    per_case_lists = {}
    for i, case in enumerate(cases):
        played = case.get("music_turns", [])
        idx_list = retrieve_max_recent(played, emb, tid_to_idx)
        per_case_lists[i] = [tids[j] for j in idx_list]
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{n} ({time.time() - t0:.0f}s)")
    print(f"    retrieval done in {time.time() - t0:.0f}s")

    # Hit rates per K + unique vs R84
    metrics = {"name": mod_name}
    for K in TOP_K_GATES:
        hit_all = sum(1 for i in range(n)
                       if cases[i]["gt"] in set(per_case_lists[i][:K]))
        hit_h7 = sum(1 for i in h7_indices
                      if cases[i]["gt"] in set(per_case_lists[i][:K]))
        metrics[f"hit_all_at_{K}"] = hit_all / n
        metrics[f"hit_h7_at_{K}"] = hit_h7 / max(1, len(h7_indices))

    # Unique recoveries vs R84 5-fold OOF
    unique_h7_at_30 = []
    unique_h7_at_300 = []
    for i in h7_indices:
        gt = cases[i]["gt"]
        mod_top30 = set(per_case_lists[i][:30])
        mod_top300 = set(per_case_lists[i][:300])
        if gt in mod_top30 and gt not in r84_top30.get(i, set()):
            unique_h7_at_30.append(i)
        if gt in mod_top300 and gt not in r84_top300.get(i, set()):
            unique_h7_at_300.append(i)
    metrics["unique_h7_top30_vs_r84"] = len(unique_h7_at_30)
    metrics["unique_h7_top300_vs_r84"] = len(unique_h7_at_300)
    metrics["unique_h7_top30_indices"] = unique_h7_at_30
    metrics["unique_h7_top300_indices"] = unique_h7_at_300

    # Segment unique recoveries by same/diff artist
    same_30 = sum(1 for i in unique_h7_at_30 if same_artist_case(cases[i], ta))
    diff_30 = len(unique_h7_at_30) - same_30
    metrics["unique_h7_top30_same_artist"] = same_30
    metrics["unique_h7_top30_diff_artist"] = diff_30
    n_h7_same = sum(1 for i in h7_indices if same_artist_case(cases[i], ta))
    n_h7_diff = len(h7_indices) - n_h7_same
    metrics["h7_same_artist_base_rate"] = n_h7_same / max(1, len(h7_indices))
    metrics["unique_h7_top30_same_artist_rate"] = (same_30 / max(1, len(unique_h7_at_30))) if unique_h7_at_30 else 0.0

    # Optional: segment by R84c sibling-R54 LR margin
    if oof_r54_margins is not None:
        low = mid = high = 0
        for i in unique_h7_at_30:
            m = oof_r54_margins.get(i, 0.5)
            if m < 0.5:
                low += 1
            elif m < 2.0:
                mid += 1
            else:
                high += 1
        metrics["unique_h7_top30_by_r84c_margin"] = {
            "low": low, "mid": mid, "high": high,
        }

    # Gate verdict
    same_artist_acceptable = (
        (metrics["unique_h7_top30_same_artist_rate"]
         >= 0.5 * metrics["h7_same_artist_base_rate"])
        if unique_h7_at_30 else False
    )
    gate_pass = (
        metrics["unique_h7_top30_vs_r84"] >= MIN_UNIQUE_H7_TOP30_FOR_GATE
        and same_artist_acceptable
    )
    metrics["gate_pass"] = gate_pass
    metrics["gate_same_artist_acceptable"] = same_artist_acceptable

    print(f"    hit h7 @20={metrics['hit_h7_at_20']:.3f} "
          f"@30={metrics['hit_h7_at_30']:.3f} "
          f"@300={metrics['hit_h7_at_300']:.3f}")
    print(f"    unique h7 vs R84: @30={metrics['unique_h7_top30_vs_r84']} "
          f"@300={metrics['unique_h7_top300_vs_r84']}")
    print(f"    unique@30 same_artist={same_30}  diff_artist={diff_30}  "
          f"(base h7 same-artist rate={metrics['h7_same_artist_base_rate']:.3f})")
    if oof_r54_margins is not None:
        seg = metrics["unique_h7_top30_by_r84c_margin"]
        print(f"    unique@30 by R84c margin: low={seg['low']} mid={seg['mid']} "
              f"high={seg['high']}")
    print(f"    gate (>=5 unique@30 + same-artist ok): "
          f"{'PASS' if gate_pass else 'fail'}")

    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Only load local metadata-qwen, skip HF download")
    args = p.parse_args()

    t0 = time.time()
    print(f"{ts()} R85 Phase 0 — multimodal headroom inventory vs R84c")
    print("=" * 70)

    print(f"\n{ts()} Loading payload + R84 5-fold OOF top-K sets...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload["track_artist"]
    n = len(cases)
    h7_indices = [i for i in range(n) if cases[i].get("n_prior_music") == 7]
    print(f"  cases: {n}, h7: {len(h7_indices)}")

    r84_top30, r84_top300, case_fold = load_r84_oof_per_case_top30_sets()
    r84_h7_hit_at_30 = sum(1 for i in h7_indices if cases[i]["gt"] in r84_top30.get(i, set()))
    r84_h7_hit_at_300 = sum(1 for i in h7_indices if cases[i]["gt"] in r84_top300.get(i, set()))
    print(f"  R84 5-fold OOF h7 hit@30: {r84_h7_hit_at_30/len(h7_indices):.3f} "
          f"({r84_h7_hit_at_30}/{len(h7_indices)})")
    print(f"  R84 5-fold OOF h7 hit@300: {r84_h7_hit_at_300/len(h7_indices):.3f} "
          f"({r84_h7_hit_at_300}/{len(h7_indices)})")

    # Load R84c sibling-R54 margins (from prior R84c sweep cache, if exists)
    margin_cache = REPO / "exp" / "eval" / "expR84c_margin_transfer.json"
    oof_r54_margins = None
    if margin_cache.exists():
        # We don't have per-case margins saved in that file (only stats).
        # Re-derive from case_features cache if available.
        feat_cache = REPO / "cache" / "r84b" / "case_features.pkl"
        if feat_cache.exists():
            print(f"  Loading case_features for margin segments ({feat_cache.stat().st_size/1e6:.0f} MB)...")
            with open(feat_cache, "rb") as f:
                case_features = pickle.load(f)
            # Compute sibling-R54 margins via per-fold sibling LR (mirrors R84c)
            import lightgbm as lgb  # type: ignore
            from scripts.expR54_phase3_blind_submission import FEAT_ALL
            LR_PARAMS = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
                          "num_leaves": 31, "learning_rate": 0.05,
                          "min_data_in_leaf": 10, "verbose": -1, "seed": 0}
            oof_r54_margins = {}
            for fold_idx in range(5):
                print(f"    margin fold {fold_idx}...")
                train_idx = [i for i in range(n) if case_fold[i] != fold_idx]
                X, y, gt = [], [], []
                for i in train_idx:
                    cf = case_features[i]
                    pool_len = len(cf["pool"])
                    for k_row in range(pool_len):
                        X.append(cf["feats_r54"][k_row])
                        y.append(1.0 if k_row == cf["gt_pos"] else 0.0)
                    gt.append(pool_len)
                X = np.array(X, dtype=np.float64)
                y = np.array(y, dtype=np.float64)
                ds = lgb.Dataset(X, label=y, group=gt, feature_name=list(FEAT_ALL))
                lr = lgb.train(LR_PARAMS, ds, num_boost_round=300)
                eval_idx = [i for i in range(n) if case_fold[i] == fold_idx]
                for i in eval_idx:
                    cf = case_features[i]
                    scores = lr.predict(cf["feats_r54"])
                    s_sorted = np.sort(scores)[::-1]
                    oof_r54_margins[i] = float(s_sorted[0] - s_sorted[1]) if len(s_sorted) >= 2 else 0.0
            print(f"    computed sibling-R54 margins for {len(oof_r54_margins)} cases")

    print(f"\n{ts()} Loading modality embeddings...")
    modalities = load_modality_embeddings(quick=args.quick)
    if not modalities:
        print("ERROR: No modalities loaded.")
        sys.exit(1)
    print(f"  modalities ready: {list(modalities.keys())}")

    print(f"\n{ts()} === Per-modality evaluation ===")
    results = {}
    for mod_name, mod_data in modalities.items():
        results[mod_name] = evaluate_modality(
            mod_name, mod_data, cases, r84_top30, r84_top300,
            ta, h7_indices, oof_r54_margins,
        )

    # Summary
    print(f"\n{ts()} === SUMMARY ===")
    print(f"  R84 5-fold OOF h7 hit@30: {r84_h7_hit_at_30}/{len(h7_indices)} "
          f"({r84_h7_hit_at_30/len(h7_indices):.3f})")
    print()
    print(f"  {'modality':18}  {'hit@30':>8}  {'unique@30':>10}  "
          f"{'unique@300':>11}  {'same/diff':>10}  gate")
    for mod_name, m in results.items():
        print(f"  {mod_name:18}  {m['hit_h7_at_30']:>8.3f}  "
              f"{m['unique_h7_top30_vs_r84']:>10d}  "
              f"{m['unique_h7_top300_vs_r84']:>11d}  "
              f"{m['unique_h7_top30_same_artist']}/{m['unique_h7_top30_diff_artist']:>4}  "
              f"{'PASS' if m['gate_pass'] else 'fail'}")

    passing = [m for m in results.values() if m["gate_pass"]]
    verdict = "PROCEED_TO_PHASE_1" if passing else "ARCHIVE_PHASE0"
    print(f"\n  Modalities passing gate (>=5 unique h7 top-30 + "
          f"same-artist acceptable): {len(passing)}")
    print(f"  VERDICT: {verdict}")
    if passing:
        print(f"  Strongest: " + ", ".join(
            f"{m['name']} ({m['unique_h7_top30_vs_r84']} unique)"
            for m in sorted(passing, key=lambda x: -x['unique_h7_top30_vs_r84'])
        ))

    out = {
        "experiment": "R85 Phase 0 — multimodal headroom inventory vs R84c",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "verdict": verdict,
        "n_cases": n,
        "n_h7": len(h7_indices),
        "r84_h7_hit_at_30": r84_h7_hit_at_30 / len(h7_indices),
        "r84_h7_hit_at_300": r84_h7_hit_at_300 / len(h7_indices),
        "modalities_loaded": list(modalities.keys()),
        "modalities": {
            k: {kk: vv for kk, vv in v.items()
                 if not isinstance(vv, list) or len(vv) <= 50}
            for k, v in results.items()
        },
        "passing_count": len(passing),
        "passing_modalities": [m["name"] for m in passing],
        "gate_definition": {
            "min_unique_h7_top30_vs_r84": MIN_UNIQUE_H7_TOP30_FOR_GATE,
            "same_artist_acceptable": ">=50% of h7 same-artist base rate among recoveries",
        },
        "anchor_strategy": f"max of last-{ANCHOR_K} played tracks",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
