#!/usr/bin/env python3
"""R69 Phase 0 — Cross-encoder rerank of R54c LR pool (baseline RRF top-300) on fold-0.

Tests whether BAAI/bge-reranker-v2-m3 (frozen, zero-shot) beats the LightGBM
LambdaRank ("LR") ranker on the SAME R54-stacked RRF top-300 pool. This isolates
the ranker upgrade from any retrieval change.

For each fold-0 case (n=1600):
  1. Build R54-stacked RRF top-300 (baseline pool, SW_BASELINE weights).
  2. Build (query_text, candidate_text) pairs for each of 300 candidates.
  3. Score with bge-reranker-v2-m3.
  4. Take cross-encoder top-20 → r69_top20.
  5. Compare nDCG@20 / same-artist / diff-artist vs R54c LR top-20 (baseline).

Phase 0 gate (per the predeclared sprint rules):
  - h7 nDCG Δ ≥ +0.005 vs R63c-repair baseline (= R54c LR for retrieval/LR)
  - same-artist Δ ≥ -0.002
  - top-1 churn pro-rated ≤ 25/80
  - recovered > lost
"""
from __future__ import annotations
import json, pickle, sys, time, math
import pathlib as _p
import numpy as np

REPO = _p.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.expR59_c3_pool_admission_diagnostic as c3
from scripts.expR59_c3_phase2_frozen_lr_conversion import (
    load_supporting_maps, metric_block, same_artist_case,
)
from scripts.expF1_cfbpr_retrieval import weighted_rrf

R12_CACHE = REPO / "exp/eval/_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache/r21_production/dev_r21_oof_lists.json"
R54_OOF = REPO / "cache/r54/phase2_full/oof_r54_lists.json"
ALS_NPZ = REPO / "cache/r54_phase3_als.npz"
LR_TOP20_PKL = REPO / "exp/eval/expR68_r54_reference_stats.pkl"  # has fold-0 LR top-20
META_JSON = REPO / "cache/metadata/track_metadata_all_tracks.json"

OUT_JSON = REPO / "exp/eval/expR69_phase0_cross_encoder.json"
OUT_MD = REPO / "docs/r69_phase0_cross_encoder_result.md"

POOL_K = 300
TOP_K = 20
RRF_K = 20
SW_BASELINE = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0,
               "ALS": 1.0, "R21": 1.0, "R54": 1.0}

RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_BATCH = 32   # MPS-safe; larger on CUDA
RERANKER_MAX_LEN = 384  # short context — query+candidate text fits comfortably

DEVICE = None  # auto


def ts() -> str:
    from datetime import datetime
    return f"[{datetime.now():%H:%M:%S}]"


def build_candidate_text(track_id: str, meta: dict) -> str:
    m = meta.get(track_id, {})
    title = (m.get("track_name") or ["unknown"])[0]
    artist = (m.get("artist_name") or ["unknown"])[0]
    album = (m.get("album_name") or [""])[0] if m.get("album_name") else ""
    tags = m.get("tag_list") or []
    rel = m.get("release_date") or ""
    year = rel[:4] if len(rel) >= 4 else ""
    tag_str = ", ".join(tags[:5]) if tags else ""
    parts = [f"Title: {title}", f"Artist: {artist}"]
    if album:
        parts.append(f"Album: {album}")
    if tag_str:
        parts.append(f"Genre/Tags: {tag_str}")
    if year:
        parts.append(f"Year: {year}")
    return " | ".join(parts)


def build_query_text(case: dict) -> str:
    # Use last user turn + brief conversation context.
    user_turns = (case.get("history") or [])
    last_user = case.get("user_query") or ""
    # last 1-2 prior turns (truncated)
    prior = []
    for t in reversed(user_turns[-3:]):
        if isinstance(t, str):
            prior.append(t[:200])
    prior_str = " // ".join(prior) if prior else ""
    if prior_str:
        return f"User request: {last_user}\nConversation so far: {prior_str}"
    return f"User request: {last_user}"


def ndcg_at(top_list: list[str], gt: str, k: int = TOP_K) -> float:
    for i, t in enumerate(top_list[:k]):
        if t == gt:
            return 1.0 / math.log2(i + 2)
    return 0.0


def main() -> None:
    print(f"{ts()} R69 Phase 0 cross-encoder rerank", flush=True)
    print("=" * 70, flush=True)

    # Load payload + R21/R54 OOF + ALS
    print(f"{ts()} Loading R12 payload + sources ...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)

    with open(R21_OOF) as f:
        r21_source = json.load(f)
    with open(R54_OOF) as f:
        r54_data = json.load(f)
    r54_source = [[t for t, _ in lst] for lst in r54_data["lists"]]
    r54_scores = [{t: float(s) for t, s in lst} for lst in r54_data["lists"]]

    als_data = np.load(ALS_NPZ, allow_pickle=True)
    als_factors = np.asarray(als_data["factors"], dtype=np.float32)
    als_track_ids = [str(t) for t in als_data["track_ids"].tolist()]
    als_to_idx = {t: i for i, t in enumerate(als_track_ids)}

    print(f"{ts()} Loading W0 fold-0 ref stats (lr_top20 baseline) ...", flush=True)
    with open(LR_TOP20_PKL, "rb") as f:
        w0 = pickle.load(f)
    fold0_idx = [i for i in range(n) if w0[i]["fold_idx"] == 0]
    lr_top20_by_case = {w0[i]["case_id"]: list(w0[i]["lr_top20"]) for i in range(n) if w0[i]["fold_idx"] == 0}
    case_id_by_idx = {i: w0[i]["case_id"] for i in range(n)}
    print(f"  fold-0 cases: {len(fold0_idx)}")

    print(f"{ts()} Loading track metadata for candidate text ...", flush=True)
    meta = json.loads(META_JSON.read_text())
    print(f"  {len(meta)} tracks indexed")

    print(f"{ts()} Building ALS source lists + RRF baseline pools ...", flush=True)
    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        als_factors, als_track_ids, als_to_idx,
    )
    pools = {}
    t0 = time.time()
    for k, i in enumerate(fold0_idx):
        src = c3.make_source_lists(payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src, SW_BASELINE, topk=POOL_K, k=RRF_K)
        pools[i] = pool
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(fold0_idx)} pools  ({time.time()-t0:.0f}s)")
    print(f"  done. total pool candidates: {sum(len(p) for p in pools.values())}")

    # Build (query, candidate) pair lists per case
    print(f"{ts()} Building (query, candidate) pairs ...", flush=True)
    pair_query_idx = []
    pair_case_idx = []
    pair_cand_idx_in_pool = []
    queries = []
    for ki, i in enumerate(fold0_idx):
        q = build_query_text(cases[i])
        queries.append(q)
        pool = pools[i]
        for p_pos, tid in enumerate(pool):
            pair_query_idx.append(ki)
            pair_case_idx.append(i)
            pair_cand_idx_in_pool.append(p_pos)
    n_pairs = len(pair_query_idx)
    print(f"  total pairs: {n_pairs}  ({len(fold0_idx)} cases × ~{POOL_K} cands)")

    # Score with cross-encoder
    print(f"{ts()} Loading cross-encoder: {RERANKER_MODEL} ...", flush=True)
    from sentence_transformers import CrossEncoder
    import torch
    device = DEVICE
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"  device={device}")

    ce = CrossEncoder(RERANKER_MODEL, device=device, max_length=RERANKER_MAX_LEN)

    print(f"{ts()} Scoring {n_pairs} pairs (batch={RERANKER_BATCH}) ...", flush=True)
    # Build batched pair list ordered by (case, pool_pos)
    text_pairs = []
    for ki, ci, pi in zip(pair_query_idx, pair_case_idx, pair_cand_idx_in_pool):
        text_pairs.append((queries[ki], build_candidate_text(pools[ci][pi], meta)))

    t0 = time.time()
    scores = ce.predict(text_pairs, batch_size=RERANKER_BATCH, show_progress_bar=True,
                        convert_to_numpy=True)
    print(f"  scoring done in {time.time()-t0:.1f}s  ({n_pairs/(time.time()-t0):.0f} pairs/s)")

    # Reshape per case: list of (track_id, score) and pick top-20
    print(f"{ts()} Computing R69 top-20 per case + metrics ...", flush=True)
    rerank_top20 = {}
    j = 0
    for i in fold0_idx:
        pool = pools[i]
        npool = len(pool)
        case_scores = scores[j:j+npool]
        j += npool
        order = np.argsort(-case_scores)
        rerank_top20[i] = [pool[int(o)] for o in order[:TOP_K]]

    # Compute metrics — h7 / same_artist / diff_artist / all on fold-0
    # Need track_artist for same_artist filter
    maps = load_supporting_maps()
    track_artist = maps["track_artist"]

    def slice_metrics(idx_list, top_by_case, name):
        # nDCG@20 = relevant track at first hit
        ndcgs = []
        for i in idx_list:
            gt = cases[i]["gt"]
            ndcgs.append(ndcg_at(top_by_case[i], gt, TOP_K))
        return float(np.mean(ndcgs)) if ndcgs else 0.0

    h7_idx = [i for i in fold0_idx if int(cases[i]["n_prior_music"]) == 7]
    same_idx = [i for i in fold0_idx if same_artist_case(cases[i], track_artist)]
    diff_idx = [i for i in fold0_idx if not same_artist_case(cases[i], track_artist)]
    h7_same_idx = [i for i in h7_idx if same_artist_case(cases[i], track_artist)]
    h7_diff_idx = [i for i in h7_idx if not same_artist_case(cases[i], track_artist)]

    # Build lr_top20 dict keyed by idx (using case_id mapping)
    lr_top20_by_idx = {i: lr_top20_by_case[case_id_by_idx[i]] for i in fold0_idx}

    base = {
        "all_fold0": slice_metrics(fold0_idx, lr_top20_by_idx, "all"),
        "h7":        slice_metrics(h7_idx, lr_top20_by_idx, "h7"),
        "same_artist": slice_metrics(same_idx, lr_top20_by_idx, "same"),
        "diff_artist": slice_metrics(diff_idx, lr_top20_by_idx, "diff"),
        "h7_same":   slice_metrics(h7_same_idx, lr_top20_by_idx, "h7_same"),
        "h7_diff":   slice_metrics(h7_diff_idx, lr_top20_by_idx, "h7_diff"),
    }
    r69 = {
        "all_fold0": slice_metrics(fold0_idx, rerank_top20, "all"),
        "h7":        slice_metrics(h7_idx, rerank_top20, "h7"),
        "same_artist": slice_metrics(same_idx, rerank_top20, "same"),
        "diff_artist": slice_metrics(diff_idx, rerank_top20, "diff"),
        "h7_same":   slice_metrics(h7_same_idx, rerank_top20, "h7_same"),
        "h7_diff":   slice_metrics(h7_diff_idx, rerank_top20, "h7_diff"),
    }
    deltas = {k: r69[k] - base[k] for k in base}

    # Recovered / lost h7
    recovered_h7 = 0
    lost_h7 = 0
    for i in h7_idx:
        gt = cases[i]["gt"]
        in_r69 = gt in rerank_top20[i]
        in_lr = gt in lr_top20_by_idx[i]
        if in_r69 and not in_lr:
            recovered_h7 += 1
        elif in_lr and not in_r69:
            lost_h7 += 1

    # Top-1 churn (per 80 cases pro-rated)
    top1_changed = sum(1 for i in fold0_idx
                       if rerank_top20[i][0] != lr_top20_by_idx[i][0])
    churn_per_80 = top1_changed / len(fold0_idx) * 80
    top20_overlap_mean = float(np.mean([
        len(set(rerank_top20[i][:TOP_K]) & set(lr_top20_by_idx[i][:TOP_K]))
        for i in fold0_idx
    ]))

    print(f"\n  metrics on fold-0 (n={len(fold0_idx)}):")
    for k in ["all_fold0", "h7", "same_artist", "diff_artist", "h7_same", "h7_diff"]:
        print(f"    {k:14}  baseline={base[k]:.4f}  R69={r69[k]:.4f}  Δ={deltas[k]:+.4f}")
    print(f"  recovered_h7={recovered_h7}  lost_h7={lost_h7}  net={recovered_h7-lost_h7:+d}")
    print(f"  top1_changed={top1_changed}/{len(fold0_idx)}  churn_per_80={churn_per_80:.2f}")
    print(f"  top20_overlap_mean={top20_overlap_mean:.2f}/20")

    # Gate evaluation
    gate_h7 = deltas["h7"] >= 0.005
    gate_same = deltas["same_artist"] >= -0.002
    gate_churn = churn_per_80 <= 25.0
    gate_recovered = recovered_h7 > lost_h7
    passes = gate_h7 and gate_same and gate_churn and gate_recovered
    verdict = "PROCEED" if passes else "ARCHIVE_PHASE_0"

    print(f"\n{ts()} === Gates ===")
    print(f"  gate_h7    (Δh7 ≥ +0.005):       {gate_h7}  (Δ={deltas['h7']:+.4f})")
    print(f"  gate_same  (Δsame ≥ -0.002):     {gate_same}  (Δ={deltas['same_artist']:+.4f})")
    print(f"  gate_churn (churn ≤ 25/80):      {gate_churn}  ({churn_per_80:.2f})")
    print(f"  gate_recov (rec > lost):         {gate_recovered}  ({recovered_h7} > {lost_h7})")
    print(f"  VERDICT: {verdict}")

    report = {
        "experiment": "R69 Phase 0 cross-encoder rerank (bge-reranker-v2-m3, R54-RRF pool baseline)",
        "model": RERANKER_MODEL,
        "device": device,
        "verdict": verdict,
        "n_fold0": len(fold0_idx),
        "n_h7": len(h7_idx),
        "n_pairs_scored": n_pairs,
        "pool_size_per_case": POOL_K,
        "topk_eval": TOP_K,
        "baseline_metrics": base,
        "r69_metrics": r69,
        "deltas_vs_baseline_lr": deltas,
        "recovered_h7": recovered_h7,
        "lost_h7": lost_h7,
        "net_h7": recovered_h7 - lost_h7,
        "top1_changed": top1_changed,
        "churn_per_80": churn_per_80,
        "top20_overlap_mean": top20_overlap_mean,
        "gates": {
            "h7_delta_ge_005":   bool(gate_h7),
            "same_artist_ge_neg002": bool(gate_same),
            "churn_le_25_80":     bool(gate_churn),
            "recovered_gt_lost":  bool(gate_recovered),
        },
        "notes": "Cross-encoder reranks the R54-stacked RRF top-300 pool (same input as LR). "
                 "Tests whether cross-encoder beats LR on identical pool. Frozen, zero-shot.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")

    # Markdown summary
    lines = [
        "# R69 Phase 0 cross-encoder rerank result",
        "",
        f"Model: `{RERANKER_MODEL}`",
        f"Device: `{device}`  pool_size: {POOL_K}  eval_top: {TOP_K}",
        f"Cases: fold-0  n={len(fold0_idx)}  h7={len(h7_idx)}  same_artist={len(same_idx)}",
        "",
        f"## Verdict: **{verdict}**",
        "",
        "## Gates",
        "",
        "| Gate | Rule | Value | Pass |",
        "|---|---|---|:---:|",
        f"| h7 nDCG | Δh7 ≥ +0.005 | Δ={deltas['h7']:+.4f} | {'✅' if gate_h7 else '❌'} |",
        f"| same-artist | Δsame ≥ -0.002 | Δ={deltas['same_artist']:+.4f} | {'✅' if gate_same else '❌'} |",
        f"| churn | ≤ 25/80 | {churn_per_80:.2f} | {'✅' if gate_churn else '❌'} |",
        f"| recovered | rec > lost h7 | {recovered_h7} > {lost_h7} | {'✅' if gate_recovered else '❌'} |",
        "",
        "## nDCG@20 (fold-0)",
        "",
        "| Subset | n | Baseline LR | R69 cross-encoder | Δ |",
        "|---|---:|---:|---:|---:|",
        f"| all_fold0 | {len(fold0_idx)} | {base['all_fold0']:.4f} | {r69['all_fold0']:.4f} | {deltas['all_fold0']:+.4f} |",
        f"| h7 | {len(h7_idx)} | {base['h7']:.4f} | {r69['h7']:.4f} | {deltas['h7']:+.4f} |",
        f"| same_artist | {len(same_idx)} | {base['same_artist']:.4f} | {r69['same_artist']:.4f} | {deltas['same_artist']:+.4f} |",
        f"| diff_artist | {len(diff_idx)} | {base['diff_artist']:.4f} | {r69['diff_artist']:.4f} | {deltas['diff_artist']:+.4f} |",
        f"| h7_same | {len(h7_same_idx)} | {base['h7_same']:.4f} | {r69['h7_same']:.4f} | {deltas['h7_same']:+.4f} |",
        f"| h7_diff | {len(h7_diff_idx)} | {base['h7_diff']:.4f} | {r69['h7_diff']:.4f} | {deltas['h7_diff']:+.4f} |",
        "",
        "## Notes",
        "",
        "- Cross-encoder reranks R54-stacked RRF top-300 (same input as LR baseline).",
        "- Frozen, zero-shot bge-reranker-v2-m3.",
        f"- Pairs scored: {n_pairs}.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"{ts()} Saved {OUT_MD}")


if __name__ == "__main__":
    main()
