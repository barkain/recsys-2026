#!/usr/bin/env python3
"""R69 Phase 0 — Mac-feasible smoke test of cross-encoder reranking.

Scope reduction for Mac MPS practicality:
- ms-marco-MiniLM-L-6-v2 (22M params, ~30 pairs/s on M-series MPS)
- fold-0 h7 cases only (n=200)
- POOL_K=100 (instead of 300)
→ 200 × 100 = 20K pairs, expected ~10-15 min on Mac MPS

Directional signal only. If positive, escalate to bge-reranker-v2-m3 + full pool on
A100. If negative, the cross-encoder rerank hypothesis is closed cheaply.
"""
from __future__ import annotations
import json, pickle, sys, time, math
import pathlib as _p
import numpy as np

REPO = _p.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.expR59_c3_pool_admission_diagnostic as c3
from scripts.expR59_c3_phase2_frozen_lr_conversion import load_supporting_maps, same_artist_case
from scripts.expF1_cfbpr_retrieval import weighted_rrf

R12_CACHE = REPO / "exp/eval/_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache/r21_production/dev_r21_oof_lists.json"
R54_OOF = REPO / "cache/r54/phase2_full/oof_r54_lists.json"
ALS_NPZ = REPO / "cache/r54_phase3_als.npz"
LR_TOP20_PKL = REPO / "exp/eval/expR68_r54_reference_stats.pkl"
META_JSON = REPO / "cache/metadata/track_metadata_all_tracks.json"

OUT_JSON = REPO / "exp/eval/expR69_phase0_smoke.json"
OUT_MD = REPO / "docs/r69_phase0_smoke_result.md"

POOL_K = 100   # reduced from 300 for Mac feasibility
TOP_K = 20
RRF_K = 20
SW_BASELINE = {"A":1.0,"B":1.0,"C":1.0,"D":0.5,"F":1.0,"ALS":1.0,"R21":1.0,"R54":1.0}

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # 22M, MPS-friendly
RERANKER_BATCH = 64
RERANKER_MAX_LEN = 256


def ts():
    from datetime import datetime
    return f"[{datetime.now():%H:%M:%S}]"


def build_candidate_text(track_id, meta):
    m = meta.get(track_id, {})
    title = (m.get("track_name") or ["unknown"])[0]
    artist = (m.get("artist_name") or ["unknown"])[0]
    tags = m.get("tag_list") or []
    rel = m.get("release_date") or ""
    year = rel[:4] if len(rel) >= 4 else ""
    tag_str = ", ".join(tags[:5]) if tags else ""
    parts = [f"Title: {title}", f"Artist: {artist}"]
    if tag_str: parts.append(f"Genre/Tags: {tag_str}")
    if year: parts.append(f"Year: {year}")
    return " | ".join(parts)


def build_query_text(case):
    user_turns = case.get("history") or []
    last_user = case.get("user_query") or ""
    prior = []
    for t in reversed(user_turns[-2:]):
        if isinstance(t, str):
            prior.append(t[:150])
    prior_str = " // ".join(prior) if prior else ""
    if prior_str:
        return f"User request: {last_user}\nContext: {prior_str}"
    return f"User request: {last_user}"


def ndcg_at(top_list, gt, k=TOP_K):
    for i, t in enumerate(top_list[:k]):
        if t == gt:
            return 1.0 / math.log2(i + 2)
    return 0.0


def main():
    print(f"{ts()} R69 Phase 0 SMOKE (small cross-encoder + h7 subset)", flush=True)
    print(f"  model={RERANKER_MODEL}  pool_K={POOL_K}", flush=True)

    print(f"{ts()} Loading payload + sources ...", flush=True)
    payload = pickle.load(open(R12_CACHE, "rb"))
    cases = payload["cases"]
    n = len(cases)

    r21_source = json.load(open(R21_OOF))
    r54_data = json.load(open(R54_OOF))
    r54_source = [[t for t,_ in lst] for lst in r54_data["lists"]]
    r54_scores = [{t:float(s) for t,s in lst} for lst in r54_data["lists"]]

    als_data = np.load(ALS_NPZ, allow_pickle=True)
    als_factors = np.asarray(als_data["factors"], dtype=np.float32)
    als_track_ids = [str(t) for t in als_data["track_ids"].tolist()]
    als_to_idx = {t:i for i,t in enumerate(als_track_ids)}

    print(f"{ts()} Loading W0 ref stats ...", flush=True)
    w0 = pickle.load(open(LR_TOP20_PKL, "rb"))
    fold0_idx = [i for i in range(n) if w0[i]["fold_idx"] == 0]
    h7_fold0_idx = [i for i in fold0_idx if int(cases[i]["n_prior_music"]) == 7]
    lr_top20_by_idx = {i: list(w0[i]["lr_top20"]) for i in h7_fold0_idx}
    print(f"  fold-0 h7 cases: {len(h7_fold0_idx)}")

    meta = json.loads(META_JSON.read_text())
    print(f"  metadata: {len(meta)} tracks")

    print(f"{ts()} Building RRF pools (top-{POOL_K}) for h7 cases ...", flush=True)
    case_index = c3.build_case_index(payload, r21_source, r54_source, r54_scores,
                                     als_factors, als_track_ids, als_to_idx)
    pools = {}
    for i in h7_fold0_idx:
        src = c3.make_source_lists(payload, r21_source, r54_source, case_index["als_source"], i)
        pool = weighted_rrf(src, SW_BASELINE, topk=POOL_K, k=RRF_K)
        pools[i] = pool
    print(f"  {len(pools)} pools built")

    print(f"{ts()} Building pairs ...", flush=True)
    text_pairs = []
    pair_case_idx = []
    pair_pos_in_pool = []
    for i in h7_fold0_idx:
        q = build_query_text(cases[i])
        for p_pos, tid in enumerate(pools[i]):
            text_pairs.append((q, build_candidate_text(tid, meta)))
            pair_case_idx.append(i)
            pair_pos_in_pool.append(p_pos)
    n_pairs = len(text_pairs)
    print(f"  pairs: {n_pairs}")

    print(f"{ts()} Loading cross-encoder ...", flush=True)
    from sentence_transformers import CrossEncoder
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  device={device}")
    ce = CrossEncoder(RERANKER_MODEL, device=device, max_length=RERANKER_MAX_LEN)

    print(f"{ts()} Scoring {n_pairs} pairs ...", flush=True)
    t0 = time.time()
    scores = ce.predict(text_pairs, batch_size=RERANKER_BATCH, show_progress_bar=True,
                        convert_to_numpy=True)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s  ({n_pairs/elapsed:.0f} pairs/s)")

    # Top-20 per case
    rerank_top20 = {}
    j = 0
    for i in h7_fold0_idx:
        npool = len(pools[i])
        case_scores = scores[j:j+npool]
        j += npool
        order = np.argsort(-case_scores)
        rerank_top20[i] = [pools[i][int(o)] for o in order[:TOP_K]]

    # Metrics on h7 only (since we restricted)
    maps, _track_pop, _track_album = load_supporting_maps()
    track_artist = maps["track_artist"]
    h7_same_idx = [i for i in h7_fold0_idx if same_artist_case(cases[i], track_artist)]
    h7_diff_idx = [i for i in h7_fold0_idx if not same_artist_case(cases[i], track_artist)]

    def avg_ndcg(idx_list, by_case):
        return float(np.mean([ndcg_at(by_case[i], cases[i]["gt"]) for i in idx_list])) if idx_list else 0.0

    base = {
        "h7":      avg_ndcg(h7_fold0_idx, lr_top20_by_idx),
        "h7_same": avg_ndcg(h7_same_idx, lr_top20_by_idx),
        "h7_diff": avg_ndcg(h7_diff_idx, lr_top20_by_idx),
    }
    r69 = {
        "h7":      avg_ndcg(h7_fold0_idx, rerank_top20),
        "h7_same": avg_ndcg(h7_same_idx, rerank_top20),
        "h7_diff": avg_ndcg(h7_diff_idx, rerank_top20),
    }
    deltas = {k: r69[k] - base[k] for k in base}

    recovered = sum(1 for i in h7_fold0_idx if cases[i]["gt"] in rerank_top20[i] and cases[i]["gt"] not in lr_top20_by_idx[i])
    lost = sum(1 for i in h7_fold0_idx if cases[i]["gt"] in lr_top20_by_idx[i] and cases[i]["gt"] not in rerank_top20[i])
    top1_changed = sum(1 for i in h7_fold0_idx if rerank_top20[i][0] != lr_top20_by_idx[i][0])
    churn_per_80 = top1_changed / len(h7_fold0_idx) * 80
    overlap_mean = float(np.mean([len(set(rerank_top20[i][:TOP_K]) & set(lr_top20_by_idx[i][:TOP_K])) for i in h7_fold0_idx]))

    print(f"\n  h7 metrics (n={len(h7_fold0_idx)}):")
    for k in ("h7","h7_same","h7_diff"):
        print(f"    {k:10}  baseline={base[k]:.4f}  R69={r69[k]:.4f}  d={deltas[k]:+.4f}")
    print(f"  recovered={recovered}  lost={lost}  net={recovered-lost:+d}")
    print(f"  top1_churn_per_80={churn_per_80:.2f}  overlap_mean={overlap_mean:.2f}/20")

    # Smoke gates — looser (this is a directional smoke, not the formal sprint gate)
    gate_h7 = deltas["h7"] >= 0.005
    gate_same = deltas.get("h7_same", 0) >= -0.005  # tolerate slightly more for smoke
    gate_recov = recovered > lost
    smoke_pass = gate_h7 and gate_same and gate_recov
    verdict = "PROCEED_TO_FULL" if smoke_pass else "SMOKE_NEGATIVE"

    print(f"\n  gates (smoke):")
    print(f"    dh7 >= +0.005:        {gate_h7}  ({deltas['h7']:+.4f})")
    print(f"    dh7_same >= -0.005:   {gate_same}  ({deltas.get('h7_same',0):+.4f})")
    print(f"    rec > lost:           {gate_recov}  ({recovered} > {lost})")
    print(f"  VERDICT: {verdict}")

    report = {
        "experiment": "R69 Phase 0 SMOKE (MiniLM-L-6 + h7 fold-0 + POOL_K=100)",
        "model": RERANKER_MODEL,
        "device": device,
        "verdict": verdict,
        "scope_note": "Reduced scope for Mac MPS feasibility. Directional signal only.",
        "pool_size": POOL_K,
        "n_h7_fold0": len(h7_fold0_idx),
        "n_pairs_scored": n_pairs,
        "scoring_elapsed_s": elapsed,
        "scoring_pairs_per_sec": n_pairs / elapsed,
        "baseline_metrics": base,
        "r69_metrics": r69,
        "deltas": deltas,
        "recovered": recovered, "lost": lost, "net": recovered - lost,
        "top1_changed": top1_changed,
        "churn_per_80": churn_per_80,
        "top20_overlap_mean": overlap_mean,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{ts()} Saved {OUT_JSON}")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# R69 Phase 0 SMOKE — cross-encoder rerank (Mac MPS reduced scope)",
        "",
        f"Model: `{RERANKER_MODEL}`  Device: `{device}`",
        f"Scope: fold-0 h7 only (n={len(h7_fold0_idx)})  Pool: top-{POOL_K}",
        f"Pairs scored: {n_pairs}  Throughput: {n_pairs/elapsed:.0f} pairs/s",
        "",
        f"## Verdict: {verdict}",
        "",
        "| Subset | n | Baseline LR | R69 cross-enc | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, label, idx_list in (("h7","h7",h7_fold0_idx),("h7_same","h7_same",h7_same_idx),("h7_diff","h7_diff",h7_diff_idx)):
        md.append(f"| {label} | {len(idx_list)} | {base[k]:.4f} | {r69[k]:.4f} | {deltas[k]:+.4f} |")
    md += [
        "",
        f"- recovered={recovered}  lost={lost}  net={recovered-lost:+d}",
        f"- top1_churn_per_80={churn_per_80:.2f}",
        "",
        "## Caveats",
        "",
        "- This is a SMOKE test with reduced scope (MiniLM-L-6 + POOL_K=100 + h7 only).",
        "- A positive result here should be re-tested with bge-reranker-v2-m3 + POOL_K=300 + full fold-0 on A100.",
        "- A negative result closes the cross-encoder-rerank hypothesis cheaply.",
    ]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved {OUT_MD}")


if __name__ == "__main__":
    main()
