#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R11: Strong LLM semantic rerank over ABCDFG candidate pool.

Tests whether Sonnet 4.6 can rerank heterogeneous top-50 candidates
better than 8-feature Powell. Haiku failed (R7); this is the stronger gate.

Candidate indices (not UUIDs). Strict parsing. Cached calls.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3_confirm_400_deterministic import (
    build_or_load_payload,
)
from scripts.tune_postrank_v23 import tokens
from scripts.expF1_cfbpr_retrieval import (
    build_cfbpr_index,
    cfbpr_max_recent,
    weighted_rrf,
)
from scripts.expR5_sequential_retrieval import SessionTransitionGraph
from offline_retrieval_sweep import load_track_metadata
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from mcrs.utils import call_llm_api

POOL_K = 50
RRF_K = 20
MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
TOPK = 20

# ---- Prompt ---- #

SYSTEM_PROMPT = """\
You are a music recommendation expert for a conversational music service. \
A user is chatting with a music bot. After each user message, the bot recommends \
one track. Your job: given the conversation so far and a list of candidate tracks, \
predict which track the bot ACTUALLY recommended next.

Think about:
1. What is the user asking for RIGHT NOW? (not earlier in the conversation)
2. If the user named a specific artist, song, or album — the answer is almost certainly that exact match
3. If the user asked for "something similar" or "more like that" — look for same genre/artist/mood
4. If the user asked for something different — look for a contrast to what was just played
5. Tracks already played (shown as "Music:" turns) are unlikely to be recommended again

Return exactly ONE JSON array of {topk} candidate numbers (integers 1..{n_candidates}). \
Your rank-1 pick should be the single track most likely to have been the bot's actual recommendation. \
No explanations, no prose — just the JSON array."""

USER_TEMPLATE = """\
The user just said: "{last_user_message}"

Conversation so far:
{conversation}

Which of these {n_candidates} tracks did the bot recommend next?

{candidates_text}

Return a JSON array of {topk} integers (1..{n_candidates}), most likely first."""


# ---- Formatting ---- #

_USEFUL_KEYS = {"track_name", "artist_name", "tag_list", "release_year", "album_name"}


def format_conversation(history: list[dict], metadata: dict) -> str:
    lines = []
    for msg in history:
        role = msg["role"].capitalize()
        content = msg["content"]
        if role == "Music":
            tid = str(content).strip()
            meta = metadata.get(tid, {})
            name = meta.get("track_name", "?")
            artist = meta.get("artist_name", "?")
            content = f"{name} by {artist}"
        lines.append(f"{role}: {content}")
    return "\n".join(lines[-20:]) if lines else "(no prior conversation)"


def format_candidates(pool: list[str], metadata: dict) -> str:
    lines = []
    for i, tid in enumerate(pool, 1):
        meta = metadata.get(tid, {})
        parts = []
        for k in ["track_name", "artist_name", "album_name", "release_year"]:
            v = meta.get(k)
            if v:
                parts.append(f"{k}: {v}")
        tags = meta.get("tag_list", [])
        if isinstance(tags, list) and tags:
            parts.append(f"tags: {', '.join(str(t) for t in tags[:6])}")
        lines.append(f"{i}. {' | '.join(parts) if parts else tid}")
    return "\n".join(lines)


# ---- Parsing ---- #

def parse_index_response(raw: str, n_candidates: int, topk: int) -> list[int] | None:
    """Parse JSON array of 1-based candidate indices. Returns 0-based indices or None."""
    arrays = re.findall(r"\[[\s\S]*?\]", raw)
    for arr_str in reversed(arrays):
        try:
            arr = json.loads(arr_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(arr, list):
            continue
        result = []
        seen = set()
        for item in arr:
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                idx = int(item)
                if 1 <= idx <= n_candidates and idx not in seen:
                    result.append(idx - 1)  # 0-based
                    seen.add(idx)
            if len(result) >= topk:
                break
        if len(result) >= topk * 0.8:  # allow slightly short
            return result[:topk]
    return None


# ---- nDCG ---- #

def ndcg_at_k(predicted_tids: list[str], gt_tid: str, k: int = 20) -> float:
    for i, tid in enumerate(predicted_tids[:k]):
        if tid == gt_tid:
            return 1.0 / math.log2(i + 2)
    return 0.0


# ---- Main ---- #

def main():
    t0 = time.time()

    # Load payload and metadata
    payload = build_or_load_payload()
    cases = payload["cases"]
    metadata = load_track_metadata()

    # Build all sources
    print("Computing A' (qwen3)...", flush=True)
    qwen_sim = TrackSimilarityRetriever(cache_dir="./cache")
    src_a = []
    for c in cases:
        played = c["music_turns"]
        a_idxs = [qwen_sim._id_to_idx.get(str(t)) for t in played[-5:]]
        a_idxs = [i for i in a_idxs if i is not None]
        if a_idxs:
            anchor_vecs = qwen_sim.vectors[a_idxs]
            sims = qwen_sim.vectors @ anchor_vecs.T
            scores_a = sims.max(axis=1)
            exclude_a = {qwen_sim._id_to_idx[t] for t in played if t in qwen_sim._id_to_idx}
            cap = min(len(scores_a), 200 + len(exclude_a))
            cand = np.argpartition(-scores_a, cap - 1)[:cap]
            cand = cand[np.argsort(-scores_a[cand])]
            out = [qwen_sim.track_ids[int(ii)] for ii in cand if int(ii) not in exclude_a][:200]
            src_a.append(out)
        else:
            src_a.append([])

    print("Computing CF-BPR...", flush=True)
    cf_ids, cf_vecs, cf_idx = build_cfbpr_index()
    src_f = []
    for c in cases:
        played = c["music_turns"]
        src_f.append(cfbpr_max_recent(played, cf_vecs, cf_idx, cf_ids, 5, 200) if played else [])

    print("Building session transition graph...", flush=True)
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    session_sequences = {}
    for item in ds["train"]:
        sid = str(item["session_id"])
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        tracks = [str(c["content"]).strip() for c in convs if c["role"] == "music"]
        session_sequences[sid] = tracks
    graph = SessionTransitionGraph(session_sequences, metadata)
    src_g = []
    for c in cases:
        played = c["music_turns"]
        src_g.append(graph.g_session_cooccur(played, 200) if played else [])

    # Build ABCDFG pools
    print("Building ABCDFG pools...", flush=True)
    pools = []
    for i, c in enumerate(cases):
        sources_dict = {
            "A": src_a[i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": src_f[i], "G": src_g[i],
        }
        w = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "G": 0.5}
        pool = weighted_rrf(sources_dict, w, topk=POOL_K, k=RRF_K)
        pools.append(pool)

    n = len(cases)

    # =====================================================================
    # LLM Rerank — small sample first
    # =====================================================================
    # Use first 80 cases (deterministic subset)
    sample_size = min(80, n)
    print(f"\n{'='*70}")
    print(f"R7: LLM SEMANTIC RERANK — {sample_size} cases, model={MODEL}")
    print(f"{'='*70}")

    llm_results = []
    baseline_results = []
    n_parse_fail = 0
    n_retry = 0
    n_success = 0

    for i in tqdm(range(sample_size), desc="rerank"):
        c = cases[i]
        pool = pools[i]
        gt = c["gt"]

        # Baseline: pool order (RRF rank)
        baseline_ndcg = ndcg_at_k(pool, gt)
        baseline_results.append(baseline_ndcg)

        # Format prompt
        conv_text = format_conversation(c["history"], metadata)
        last_user = c["user_query"]
        cand_text = format_candidates(pool, metadata)

        sys_prompt = SYSTEM_PROMPT.format(
            topk=TOPK, n_candidates=len(pool),
        )
        user_prompt = USER_TEMPLATE.format(
            last_user_message=last_user,
            conversation=conv_text,
            n_candidates=len(pool),
            candidates_text=cand_text,
            topk=TOPK,
        )

        # Call LLM (cached)
        raw = call_llm_api(sys_prompt, user_prompt, MODEL, max_tokens=MAX_TOKENS)
        if raw is None:
            n_parse_fail += 1
            llm_results.append(baseline_ndcg)
            continue

        # Parse
        indices = parse_index_response(raw, len(pool), TOPK)
        if indices is None:
            # Retry once
            n_retry += 1
            raw2 = call_llm_api(
                sys_prompt,
                user_prompt + "\n\nPrevious response was invalid. Return ONLY a JSON array of integers.",
                MODEL, max_tokens=MAX_TOKENS,
            )
            if raw2:
                indices = parse_index_response(raw2, len(pool), TOPK)
            if indices is None:
                n_parse_fail += 1
                llm_results.append(baseline_ndcg)
                continue

        n_success += 1
        reranked = [pool[idx] for idx in indices]
        # Pad if short
        if len(reranked) < TOPK:
            seen = set(reranked)
            for tid in pool:
                if tid not in seen:
                    reranked.append(tid)
                    seen.add(tid)
                    if len(reranked) >= TOPK:
                        break

        llm_ndcg = ndcg_at_k(reranked, gt)
        llm_results.append(llm_ndcg)

    # =====================================================================
    # Analysis
    # =====================================================================
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"  Sample size: {sample_size}")
    print(f"  LLM success: {n_success}/{sample_size}")
    print(f"  Parse failures: {n_parse_fail}")
    print(f"  Retries: {n_retry}")

    bl_mean = float(np.mean(baseline_results))
    llm_mean = float(np.mean(llm_results))
    bl_hits = sum(1 for x in baseline_results if x > 0)
    llm_hits = sum(1 for x in llm_results if x > 0)

    print(f"\n  Baseline (RRF order):")
    print(f"    nDCG@20: {bl_mean:.4f}")
    print(f"    hits@20: {bl_hits}/{sample_size}")

    print(f"\n  LLM rerank:")
    print(f"    nDCG@20: {llm_mean:.4f}")
    print(f"    hits@20: {llm_hits}/{sample_size}")
    print(f"    Δ nDCG: {llm_mean - bl_mean:+.4f}")

    # Per-case analysis
    wins = losses = ties = 0
    for i in range(sample_size):
        if llm_results[i] > baseline_results[i] + 1e-6:
            wins += 1
        elif llm_results[i] < baseline_results[i] - 1e-6:
            losses += 1
        else:
            ties += 1

    print(f"\n  Per-case: wins={wins} losses={losses} ties={ties}")
    print(f"  Win rate: {wins/(wins+losses):.1%}" if wins + losses > 0 else "")

    # Also compare with F1 8f Powell baseline
    # We need to compute the F1 8f Powell nDCG for the same cases
    from scripts.r3_confirm_400_deterministic import fit_weights, vec_ndcg
    from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS

    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]
    track_title_toks = payload["track_title_toks"]
    track_artist_toks = payload["track_artist_toks"]
    track_meta_toks = payload["track_meta_toks"]

    def ensure_meta(tids):
        for tid in tids:
            if tid in track_artist:
                continue
            meta = metadata.get(str(tid), {})
            artist = str(meta.get("artist_name", "")).lower().strip()
            raw_tags = meta.get("tag_list") or []
            tags = {str(t).lower().strip() for t in raw_tags if str(t).strip()} if isinstance(raw_tags, list) else set()
            track_artist[tid] = artist
            track_tags[tid] = tags

    # Build 8f features for full 400 cases on the ABCDFG pool, fit Powell, then
    # compute per-case nDCG for the sample
    X_8f = np.zeros((n, POOL_K, len(FEATURE_NAMES)), dtype=np.float64)
    gt_8f = np.full(n, -1, dtype=np.int64)
    sz_8f = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        pool = pools[i]
        ensure_meta(pool)
        sz_8f[i] = len(pool)
        if c["gt"] in pool:
            gt_8f[i] = pool.index(c["gt"])

        user_msgs = [str(r["content"]) for r in c["history"] if r["role"] == "user"] + [c["user_query"]]
        played = c["music_turns"]
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = track_artist.get(played[-1], "") if played else ""
        l_tags = track_tags.get(played[-1], set()) if played else set()
        prior = [(1.0/(j+1), track_artist.get(t,""), track_tags.get(t,set()))
                 for j,t in enumerate(reversed(played))]
        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = track_artist.get(tid, "")
            ct = track_tags.get(tid, set())
            row = X_8f[i, rank-1]
            row[0] = 1.0/rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(track_artist_toks.get(tid, set()) & now_tok))
            row[4] = float(len(track_title_toks.get(tid, set()) & now_tok))
            row[5] = float(len(track_meta_toks.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec

    # Fit on full 400, compute per-case scores for sample
    all_idx = np.arange(n, dtype=np.int64)
    w_fit, train_ndcg = fit_weights(X_8f, gt_8f, sz_8f, all_idx)

    # Per-case 8f Powell nDCG on sample
    powell_results = []
    for i in range(sample_size):
        pool = pools[i]
        gt = cases[i]["gt"]
        feat_row = X_8f[i, :len(pool)]
        scores = feat_row @ w_fit
        order = np.argsort(-scores)
        reranked = [pool[j] for j in order[:TOPK]]
        powell_results.append(ndcg_at_k(reranked, gt))

    powell_mean = float(np.mean(powell_results))
    powell_hits = sum(1 for x in powell_results if x > 0)
    print(f"\n  8f Powell (train on full 400):")
    print(f"    nDCG@20: {powell_mean:.4f}")
    print(f"    hits@20: {powell_hits}/{sample_size}")

    # LLM vs Powell per-case
    llm_vs_powell_wins = sum(1 for i in range(sample_size) if llm_results[i] > powell_results[i] + 1e-6)
    llm_vs_powell_losses = sum(1 for i in range(sample_size) if llm_results[i] < powell_results[i] - 1e-6)
    print(f"\n  LLM vs 8f Powell: wins={llm_vs_powell_wins} losses={llm_vs_powell_losses}")
    print(f"  LLM Δ vs Powell: {llm_mean - powell_mean:+.4f}")

    # Gate
    delta = llm_mean - powell_mean
    if delta >= 0.015:
        verdict = "PASS"
    elif delta >= 0.010:
        verdict = "PROMISING"
    elif delta <= 0.005:
        verdict = "FAIL"
    else:
        verdict = "WEAK"

    print(f"\nGATE VERDICT: {verdict}")

    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expR11_strong_llm_rerank.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "meta": {"model": MODEL, "sample_size": sample_size, "pool_k": POOL_K,
                 "elapsed_sec": elapsed},
        "baseline_rrf": {"ndcg_mean": bl_mean, "hits": bl_hits},
        "llm_rerank": {"ndcg_mean": llm_mean, "hits": llm_hits,
                       "parse_fails": n_parse_fail, "retries": n_retry,
                       "success": n_success},
        "powell_8f": {"ndcg_mean": powell_mean, "hits": powell_hits},
        "llm_vs_baseline": {"delta": llm_mean - bl_mean, "wins": wins, "losses": losses},
        "llm_vs_powell": {"delta": delta, "wins": llm_vs_powell_wins, "losses": llm_vs_powell_losses},
        "verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
