#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R29: Cross-encoder reranker diagnostic.

Rerank R21 top-100 candidates with (conversation, track_metadata) cross-encoder.
Train on dev cases where GT is in top-100. Evaluate hist_7/last_turn.

Primary gate: hist_7 >= R21 +0.005. No blind until gate passes.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import argparse
import gc
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# R29 repeatedly OOM-crashed this Mac on MPS. Keep this diagnostic CPU-only.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
CE_CACHE = REPO / "cache" / "r29"
CE_MODEL = "BAAI/bge-reranker-base"
CE_DEVICE = "cpu"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_track_metadata_pyarrow():
    """Load track metadata via pyarrow (no datasets import)."""
    import pyarrow as pa
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    if not matches:
        raise FileNotFoundError("all_tracks arrow not found")
    with pa.memory_map(str(matches[-1]), "r") as source:
        table = pa.ipc.open_stream(source).read_all()
    cols = {col: table.column(col).to_pylist() for col in table.column_names}
    meta = {}
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        names = cols.get("track_name", [[]])[i]
        artists = cols.get("artist_name", [[]])[i]
        album = cols.get("album_name", [[]])[i]
        tags = cols.get("tag_list", [[]])[i]
        name = names[0] if isinstance(names, list) and names else str(names)
        artist = ", ".join(artists) if isinstance(artists, list) else str(artists)
        alb = album[0] if isinstance(album, list) and album else str(album)
        tag_str = ", ".join(str(t) for t in tags[:10]) if isinstance(tags, list) else str(tags)
        meta[tid] = f"{name} by {artist}. Album: {alb}. Tags: {tag_str}"
    return meta


def build_query_text(case):
    """Build conversation text for cross-encoder query side."""
    parts = []
    for h in case["history"]:
        if h["role"] == "user":
            parts.append(str(h["content"]))
    parts.append(case["user_query"])
    return " ".join(parts[-3:])


def score_cross_encoder_streaming(
    model,
    query_texts,
    candidate_texts_per_query,
    *,
    batch_size: int,
    pair_chunk: int,
):
    """Score (query, candidate) pairs without materializing a large pair list.

    CrossEncoder.predict tokenizes its input list before scoring. Keeping the
    list bounded avoids the MPS/CPU memory spikes that crashed previous runs.
    """
    results: list[np.ndarray] = []
    for qi, query in enumerate(query_texts):
        cands = candidate_texts_per_query[qi]
        query_scores = []
        for start in range(0, len(cands), pair_chunk):
            chunk = cands[start:start + pair_chunk]
            pairs = [(query, cand) for cand in chunk]
            if not pairs:
                continue
            scores = model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            query_scores.append(np.asarray(scores, dtype=np.float32))
            del pairs, scores
        if query_scores:
            results.append(np.concatenate(query_scores).astype(np.float32, copy=False))
        else:
            results.append(np.array([], dtype=np.float32))
        if (qi + 1) % 25 == 0:
            print(f"    streamed {qi + 1}/{len(query_texts)} queries", flush=True)
            gc.collect()
    return results


def ndcg_at_k(ranked_list, gt, k=20):
    """Compute nDCG@k for a single query."""
    if gt not in ranked_list[:k]:
        return 0.0
    pos = ranked_list.index(gt)
    if pos >= k:
        return 0.0
    return 1.0 / np.log2(pos + 2)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hist-min", type=int, default=7,
                        help="Minimum n_prior_music slice to score. Default: 7 (Blind-A last-turn proxy).")
    parser.add_argument("--depth", type=int, default=50,
                        help="R21 candidate depth to rerank. Default: 50 for safe diagnostic.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="CrossEncoder inference batch size. Keep small on laptops.")
    parser.add_argument("--pair-chunk", type=int, default=128,
                        help="Max pair list passed to CrossEncoder.predict at once.")
    parser.add_argument("--case-chunk", type=int, default=25,
                        help="Number of cases processed before GC/progress logging.")
    parser.add_argument("--max-cases", type=int, default=0,
                        help="Optional smoke cap after hist filtering. 0 means all.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cached scores and overwrite them.")
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow Hugging Face network downloads. Default is offline/cache-only.")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    print(f"{ts()} R29: Cross-Encoder Reranker Diagnostic")
    print("=" * 70)
    if not args.allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if CE_DEVICE != "cpu":
        raise RuntimeError("R29 is CPU-only. MPS is disabled because it repeatedly OOM-crashed this Mac.")
    if args.depth <= 0 or args.batch_size <= 0 or args.pair_chunk <= 0 or args.case_chunk <= 0:
        raise ValueError("depth, batch-size, pair-chunk, and case-chunk must be positive")

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    print(f"{ts()} Loading track metadata...")
    track_meta = load_track_metadata_pyarrow()
    print(f"  {len(track_meta)} tracks")

    # Filter to diagnostic slice if set
    if args.hist_min > 0:
        diag_indices = [i for i, c in enumerate(cases) if c["n_prior_music"] >= args.hist_min]
        print(f"  Diagnostic slice: hist>={args.hist_min}, {len(diag_indices)} cases")
    else:
        diag_indices = list(range(len(cases)))
    if args.max_cases > 0:
        diag_indices = diag_indices[:args.max_cases]
        print(f"  Smoke cap: {len(diag_indices)} cases")

    # Build query texts and candidate sets (only for diag slice)
    print(f"{ts()} Building query/candidate pairs...")
    query_texts = []
    candidate_lists = []
    candidate_texts = []
    diag_cases = []

    for i in diag_indices:
        c = cases[i]
        qt = build_query_text(c)
        query_texts.append(qt)
        cands = r21_source[i][:args.depth]
        candidate_lists.append(cands)
        candidate_texts.append([track_meta.get(tid, tid) for tid in cands])
        diag_cases.append(c)

    n_pairs = sum(len(ct) for ct in candidate_texts)
    print(f"  {len(diag_cases)} queries, {n_pairs} total pairs ({n_pairs/len(diag_cases):.0f} avg)")

    # Score with cross-encoder (zero-shot, no fine-tuning)
    CE_CACHE.mkdir(parents=True, exist_ok=True)
    scores_path = CE_CACHE / (
        f"zeroshot_scores_{CE_DEVICE}_hist{args.hist_min}_d{args.depth}"
        f"_bs{args.batch_size}_pc{args.pair_chunk}_n{len(diag_cases)}.npy"
    )

    if scores_path.exists() and not args.no_cache:
        print(f"{ts()} Loading cached zero-shot scores...")
        all_scores_flat = np.load(scores_path, allow_pickle=True)
        ce_scores = []
        offset = 0
        for cands in candidate_lists:
            ce_scores.append(all_scores_flat[offset:offset + len(cands)])
            offset += len(cands)
    else:
        print(f"{ts()} Scoring with {CE_MODEL} on {CE_DEVICE} (chunked)...")
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(
            CE_MODEL,
            device=CE_DEVICE,
            max_length=512,
            local_files_only=not args.allow_download,
        )
        n_diag = len(diag_cases)
        ce_scores: list[np.ndarray] = [np.array([]) for _ in range(n_diag)]
        for chunk_start in range(0, n_diag, args.case_chunk):
            chunk_end = min(chunk_start + args.case_chunk, n_diag)
            chunk_q = query_texts[chunk_start:chunk_end]
            chunk_ct = candidate_texts[chunk_start:chunk_end]
            chunk_scores = score_cross_encoder_streaming(
                model,
                chunk_q,
                chunk_ct,
                batch_size=args.batch_size,
                pair_chunk=args.pair_chunk,
            )
            for j, sc in enumerate(chunk_scores):
                ce_scores[chunk_start + j] = sc
            gc.collect()
            print(f"  {chunk_end}/{n_diag} scored", flush=True)
        all_flat = np.concatenate(ce_scores)
        np.save(scores_path, all_flat)
        print(f"  Cached {len(all_flat)} scores to {scores_path}")
        del model
        gc.collect()

    # Evaluate zero-shot reranking
    print(f"\n{ts()} Evaluating zero-shot cross-encoder reranking...")
    n_diag = len(diag_cases)
    baseline_ndcg = np.zeros(n_diag)
    ce_ndcg = np.zeros(n_diag)

    for i, c in enumerate(diag_cases):
        gt = c["gt"]
        cands = candidate_lists[i]

        baseline_ndcg[i] = ndcg_at_k(cands, gt, k=20)

        if len(ce_scores[i]) > 0:
            reranked_idx = np.argsort(-ce_scores[i])
            reranked = [cands[j] for j in reranked_idx]
            ce_ndcg[i] = ndcg_at_k(reranked, gt, k=20)

    # Blended scores
    blend_alphas = [0.1, 0.3, 0.5, 0.7]
    blend_ndcgs = {a: np.zeros(n_diag) for a in blend_alphas}

    for i in range(n_diag):
        gt = diag_cases[i]["gt"]
        cands = candidate_lists[i]
        k = len(cands)
        if k == 0:
            continue
        lr_scores = np.array([1.0 / (r + 1) for r in range(k)], dtype=np.float32)
        lr_scores /= lr_scores.max() + 1e-9
        ce_s = ce_scores[i]
        if len(ce_s) == 0:
            continue
        ce_norm = (ce_s - ce_s.min()) / (ce_s.max() - ce_s.min() + 1e-9)

        for alpha in blend_alphas:
            blended = (1 - alpha) * lr_scores + alpha * ce_norm
            reranked_idx = np.argsort(-blended)
            reranked = [cands[j] for j in reranked_idx]
            blend_ndcgs[alpha][i] = ndcg_at_k(reranked, gt, k=20)

    # Report by depth slice
    configs = {"LR_baseline": baseline_ndcg, "CE_zeroshot": ce_ndcg}
    for a in blend_alphas:
        configs[f"blend_{a}"] = blend_ndcgs[a]

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"R29 CROSS-ENCODER RERANKER — ZERO-SHOT (hist>={args.hist_min}, depth={args.depth})")
    print(sep)
    print(f"  {'Config':<20} {'mean':>10} {'hist_7':>10} {'hist_6':>10} {'hist_5':>10} "
          f"{'D_h7':>10}")
    print(f"  {'-'*70}")

    base_h7 = 0.0
    for name, ndcg_arr in configs.items():
        slices = {}
        for depth in range(8):
            idx = [i for i in range(n_diag) if diag_cases[i]["n_prior_music"] == depth]
            if idx:
                slices[f"hist_{depth}"] = float(np.mean([ndcg_arr[i] for i in idx]))
        mean_all = float(np.mean(ndcg_arr))
        h7 = slices.get("hist_7", 0)
        h6 = slices.get("hist_6", 0)
        h5 = slices.get("hist_5", 0)
        if name == "LR_baseline":
            base_h7 = h7
        dl = h7 - base_h7
        print(f"  {name:<20} {mean_all:>10.5f} {h7:>10.5f} {h6:>10.5f} {h5:>10.5f} {dl:>+10.5f}")

    # Gate
    print(f"\n{sep}")
    print("GATE CHECK (hist_7 >= +0.005)")
    best_name = None
    best_dl = 0.0
    for name, ndcg_arr in configs.items():
        if name == "LR_baseline":
            continue
        idx7 = [i for i in range(n_diag) if diag_cases[i]["n_prior_music"] == 7]
        h7 = float(np.mean([ndcg_arr[i] for i in idx7])) if idx7 else 0
        dl = h7 - base_h7
        g = dl >= 0.005
        print(f"  {name:<20} hist_7={dl:+.5f} {'PASS ***' if g else 'FAIL'}")
        if g and dl > best_dl:
            best_dl = dl
            best_name = name

    if best_name:
        print(f"\n  BEST: {best_name} (Δ hist_7={best_dl:+.5f})")
        print("  → Proceed to fine-tuned cross-encoder training")
    else:
        print("\n  Zero-shot cross-encoder does not improve hist_7.")
        print("  If zero-shot fails, fine-tuned may still help — but signal is weak.")

    out_path = REPO / "exp" / "eval" / "expR29_cross_encoder.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, ndcg_arr in configs.items():
        idx7 = [i for i in range(n_diag) if diag_cases[i]["n_prior_music"] == 7]
        hist57 = list(range(n_diag))
        results[name] = {
            "mean_hist57": float(np.mean(ndcg_arr)),
            "hist_7": float(np.mean([ndcg_arr[i] for i in idx7])),
            "hist_57": float(np.mean([ndcg_arr[i] for i in hist57])),
        }
    with open(out_path, "w") as f:
        json.dump({"configs": results, "best": best_name, "rerank_depth": args.depth,
                    "diag_hist_min": args.hist_min}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
