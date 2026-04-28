"""F1 blind-set inference driver: cfg0209 + CF-BPR F_max_recent5.

Extends cfg0209 deterministic pipeline with CF-BPR collaborative retrieval:
  1. A' = max(cosine_sim, qwen3) over last 5 played → top 200
  2. B  = last_music_meta BM25 @ 500
  3. C  = full_history BM25 @ 500
  4. D  = track_neighbors (qwen3) @ 200
  5. F  = CF-BPR max_recent_5 @ 200   ← NEW
  6. Weighted-RRF fuse (A=1, B=1, C=1, D=0.5, F=1.0, k=20) → top-50
  7. 8-feature Powell postrank → top-20
  8. Optional Haiku response generation

CF-BPR source uses pre-computed 128-dim collaborative filtering embeddings
from talkpl-ai/TalkPlayData-Challenge-Track-Embeddings (cf-bpr column).
Cached at cache/track_sim/cf-bpr/.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from run_inference_blind_r3_det import (
    APrimeMaxRecent,
    A_PRIME_K,
    A_PRIME_RECENT_K,
    A_PRIME_VARIANT,
    BM25_K,
    POOL_K,
    POWELL_FIT_PKL,
    RRF_K,
    TOP_K,
    TRACK_NEIGHBOR_K,
    TRACK_SIM_DIR,
    _atomic_write_json,
    _ensure_meta_maps,
    _result_key,
    build_row_features,
    build_session_memory_for_response,
    fit_powell_weights,
    parse_all_turns,
    parse_last_turn,
    postrank_topk,
    weighted_rrf,
)
from offline_retrieval_sweep import CachedBM25, load_track_metadata
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from scripts.tune_postrank_v23 import FEATURE_NAMES

# F1 CF-BPR constants
CFBPR_CACHE_DIR = REPO_ROOT / "cache" / "track_sim" / "cf-bpr"
CFBPR_DEPTH = 200
CFBPR_RECENT_K = 5
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}

log = logging.getLogger(__name__)


def load_cfbpr_index() -> tuple[list[str], np.ndarray, dict[str, int]]:
    """Load cached cf-bpr index. Must be pre-built by expF1."""
    ids_path = CFBPR_CACHE_DIR / "track_ids.json"
    vecs_path = CFBPR_CACHE_DIR / "vectors.npy"
    if not ids_path.exists() or not vecs_path.exists():
        raise FileNotFoundError(
            f"CF-BPR cache not found at {CFBPR_CACHE_DIR}. "
            "Run scripts/expF1_cfbpr_retrieval.py first to build it."
        )
    with open(ids_path) as f:
        track_ids = json.load(f)
    vectors = np.load(vecs_path)
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    assert len(id_to_idx) == len(track_ids), "duplicate track IDs in cf-bpr cache"
    return track_ids, vectors, id_to_idx


class CFBPRMaxRecent:
    """CF-BPR max-recent-K retriever. Same interface as APrimeMaxRecent."""

    def __init__(self, track_ids: list[str], vectors: np.ndarray,
                 id_to_idx: dict[str, int], recent_k: int = CFBPR_RECENT_K):
        self.recent_k = recent_k
        self.track_ids = track_ids
        self.vectors = vectors
        self.id_to_idx = id_to_idx

    def topn(self, played: list[str], topn: int) -> list[str]:
        idxs = [self.id_to_idx.get(str(tid)) for tid in played[-self.recent_k:]]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            return []
        anchors = self.vectors[idxs]
        sims = self.vectors @ anchors.T
        scores = sims.max(axis=1)
        exclude = set(idxs)
        cap = min(len(scores), topn + len(exclude))
        cand = np.argpartition(-scores, cap - 1)[:cap]
        cand = cand[np.argsort(-scores[cand])]
        out = []
        for i in cand:
            if int(i) in exclude:
                continue
            out.append(self.track_ids[int(i)])
            if len(out) >= topn:
                break
        return out


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    require_cache = bool(os.environ.get("MCRS_REQUIRE_LLM_CACHE"))
    if args.skip_response_generation and not require_cache:
        os.environ["MCRS_REQUIRE_LLM_CACHE"] = "1"
        require_cache = True

    log.info("=" * 70)
    log.info("F1 blind inference driver — cfg0209 + CF-BPR F_max_recent5")
    log.info("Sources: A'(qwen3) + B(bm25) + C(bm25) + D(qwen3) + F(cf-bpr)")
    log.info("Weights: A=%g B=%g C=%g D=%g F=%g  RRF_k=%d  pool_k=%d",
             SOURCE_WEIGHTS["A"], SOURCE_WEIGHTS["B"], SOURCE_WEIGHTS["C"],
             SOURCE_WEIGHTS["D"], SOURCE_WEIGHTS["F"], RRF_K, POOL_K)
    log.info("CF-BPR: depth=%d, recent_k=%d", CFBPR_DEPTH, CFBPR_RECENT_K)
    log.info("=" * 70)

    # ----- Powell fit ----- #
    weights, fit_info = fit_powell_weights()
    log.info("Powell fitted weights:")
    for name, val in zip(FEATURE_NAMES, weights):
        log.info("  %-36s % .6f", name, val)
    log.info("Powell train nDCG@20: %.4f", fit_info["train_ndcg20"])

    # ----- Build retrievers ----- #
    log.info("Loading track metadata")
    metadata = load_track_metadata()

    log.info("Loading BM25 cache")
    bm25 = CachedBM25()

    log.info("Loading qwen3 track similarity retriever")
    track_sim = TrackSimilarityRetriever(cache_dir=str(REPO_ROOT / "cache"))
    a_prime = APrimeMaxRecent(track_sim, recent_k=A_PRIME_RECENT_K)

    log.info("Loading CF-BPR index")
    cf_ids, cf_vecs, cf_idx = load_cfbpr_index()
    cfbpr = CFBPRMaxRecent(cf_ids, cf_vecs, cf_idx, recent_k=CFBPR_RECENT_K)
    log.info("CF-BPR index: %d tracks, dim=%d", len(cf_ids), cf_vecs.shape[1])

    # ----- Sanity check CF-BPR ----- #
    log.info("CF-BPR sanity check...")
    test_played = [cf_ids[0], cf_ids[100], cf_ids[200]]
    test_neighbors = cfbpr.topn(test_played, topn=10)
    assert len(test_neighbors) == 10, f"expected 10 neighbors, got {len(test_neighbors)}"
    assert all(n in cf_idx for n in test_neighbors), "neighbor not in cf-bpr index"
    assert not set(test_played) & set(test_neighbors), "played track in neighbors"
    log.info("CF-BPR sanity check: PASS (%d neighbors, no self-match)", len(test_neighbors))

    # ----- Load blind dataset ----- #
    blind_dataset = args.blind_dataset
    log.info("Loading blind dataset: %s", blind_dataset)
    db = load_dataset(blind_dataset, split="test")
    log.info("Blind dataset: %d sessions", len(db))

    if args.sample_size:
        db = db.select(range(min(args.sample_size, len(db))))
        log.info("Selected first %d sessions (smoke)", len(db))

    # ----- Build per-row context ----- #
    rows: list[dict[str, Any]] = []
    for item in db:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        if args.last_turn_only:
            turn_num, user_query, history, music_turns = parse_last_turn(item)
            rows.append({
                "session_id": sid, "user_id": uid,
                "turn_number": turn_num, "user_query": user_query,
                "history": history, "music_turns": music_turns,
            })
        else:
            for turn_num, user_query, history, music_turns in parse_all_turns(item):
                rows.append({
                    "session_id": sid, "user_id": uid,
                    "turn_number": turn_num, "user_query": user_query,
                    "history": history, "music_turns": music_turns,
                })

    log.info("Total turns to predict: %d", len(rows))

    output_tid = args.output_tid
    out_dir = REPO_ROOT / "exp" / "inference" / "blind_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{output_tid}.json"
    out_zip = out_dir / f"{output_tid}_submission.zip"
    out_meta = out_dir / f"{output_tid}_metadata.json"

    # ----- Resume support ----- #
    inference_results: list[dict[str, Any]] = []
    completed: set[tuple[str, str | None, int]] = set()
    if args.resume and out_json.exists():
        with open(out_json, encoding="utf-8") as f:
            inference_results = json.load(f)
        completed = {_result_key(r) for r in inference_results}
        log.info("Resuming: %d completed turns", len(completed))

    pending = []
    for r in rows:
        key = (str(r["session_id"]),
               None if r["user_id"] is None else str(r["user_id"]),
               int(r["turn_number"]))
        if key in completed:
            continue
        pending.append(r)

    if completed:
        log.info("Skipping %d completed; %d pending", len(completed), len(pending))

    # ----- Maps for postrank features ----- #
    maps = {
        "track_artist": {}, "track_tags": {},
        "track_title_toks": {}, "track_artist_toks": {}, "track_meta_toks": {},
    }

    # ----- Retrieval ----- #
    t_retrieve = time.time()
    queries_b, queries_c = [], []
    for r in pending:
        from offline_retrieval_sweep import query_parts
        q_b = " ".join(query_parts(r["history"], r["user_query"], metadata, "last_music_meta"))
        q_c = " ".join(query_parts(r["history"], r["user_query"], metadata, "full"))
        queries_b.append(q_b or r["user_query"])
        queries_c.append(q_c or r["user_query"])

    log.info("BM25 retrieval (B@%d, C@%d)", BM25_K, BM25_K)
    src_b = bm25.retrieve_batch(queries_b, topk=BM25_K) if pending else []
    src_c = bm25.retrieve_batch(queries_c, topk=BM25_K) if pending else []

    log.info("Building A' + D + F for %d rows", len(pending))
    src_a: list[list[str]] = []
    src_d: list[list[str]] = []
    src_f: list[list[str]] = []
    for r in pending:
        played = r["music_turns"]
        src_a.append(a_prime.topn(played, topn=A_PRIME_K) if played else [])
        anchor = played[-1] if played else None
        src_d.append(track_sim.track_id_to_neighbors(anchor, topk=TRACK_NEIGHBOR_K) if anchor else [])
        src_f.append(cfbpr.topn(played, topn=CFBPR_DEPTH) if played else [])

    log.info("Retrieval done in %.2fs", time.time() - t_retrieve)

    # ----- Fusion + postrank ----- #
    log.info("Fusing ABCDF + postrank for %d rows", len(pending))
    pending_outputs: list[dict[str, Any]] = []
    fallback_zero_pool = 0

    # Smoke validation counters
    n_f_empty = 0
    n_f_nonempty = 0

    for i, r in enumerate(tqdm(pending, desc="rank", disable=not pending)):
        sources = {
            "A": src_a[i], "B": src_b[i], "C": src_c[i],
            "D": src_d[i], "F": src_f[i],
        }
        pool = weighted_rrf(sources, SOURCE_WEIGHTS, topk=POOL_K, k=RRF_K)

        if src_f[i]:
            n_f_nonempty += 1
        else:
            n_f_empty += 1

        if not pool:
            log.warning("Empty pool for %s/turn%d — fallback bm25", r["session_id"], r["turn_number"])
            pool = bm25.retrieve(r["user_query"], topk=POOL_K)
            fallback_zero_pool += 1

        if len(pool) < POOL_K:
            seen = set(pool)
            for tid in src_b[i] + src_c[i]:
                if tid not in seen:
                    pool.append(tid)
                    seen.add(tid)
                    if len(pool) >= POOL_K:
                        break

        _ensure_meta_maps(pool + r["music_turns"], metadata, maps)
        user_messages = [str(h["content"]) for h in r["history"] if h["role"] == "user"] + [r["user_query"]]
        feats = build_row_features(pool, user_messages, r["music_turns"], maps)
        top20 = postrank_topk(pool, feats, weights, k=TOP_K)

        if len(top20) < TOP_K:
            seen = set(top20)
            for tid in pool + src_b[i] + src_c[i]:
                if tid not in seen:
                    top20.append(tid)
                    seen.add(tid)
                    if len(top20) >= TOP_K:
                        break
        top20 = top20[:TOP_K]

        # Smoke validation
        assert len(top20) == TOP_K, f"row {i}: only {len(top20)} predictions"
        assert len(set(top20)) == TOP_K, f"row {i}: duplicate predictions"

        pending_outputs.append({
            "session_id": r["session_id"],
            "user_id": r["user_id"],
            "turn_number": r["turn_number"],
            "predicted_track_ids": top20,
            "predicted_response": "",
            "_pool_for_response": pool,
        })

    log.info("CF-BPR source: %d nonempty, %d empty", n_f_nonempty, n_f_empty)
    log.info("Fallback zero-pool: %d", fallback_zero_pool)

    # ----- Response generation ----- #
    if not args.skip_response_generation and pending_outputs:
        if not os.environ.get("ANTHROPIC_RECSYS_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_RECSYS_API_KEY not set. Pass --skip_response_generation."
            )
        from mcrs.db_item.music_catalog import MusicCatalogDB
        from mcrs.lm_modules.claude import ClaudeModule

        log.info("Loading MusicCatalogDB for response generation")
        item_db = MusicCatalogDB(
            dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
            split_types=["all_tracks"],
        )
        prompts_dir = REPO_ROOT / "mcrs" / "system_prompts"
        sys_prompt = (
            (prompts_dir / "roleplay.txt").read_text(encoding="utf-8")
            + "\n"
            + (prompts_dir / "response_generation.txt").read_text(encoding="utf-8")
        )
        haiku = ClaudeModule(model="claude-haiku-4-5-20251001")

        log.info("Generating responses for %d turns", len(pending_outputs))
        t_resp = time.time()
        for r, out in zip(pending, tqdm(pending_outputs, desc="response")):
            top_id = out["predicted_track_ids"][0]
            try:
                top_item = item_db.id_to_metadata(top_id)
            except KeyError:
                top_item = f"track_id: {top_id}"
            session_memory = build_session_memory_for_response(r["history"], r["user_query"], item_db)
            response = haiku.response_generation(sys_prompt, session_memory, top_item)
            out["predicted_response"] = response or ""
        log.info("Response generation done in %.2fs", time.time() - t_resp)
    else:
        log.info("Skipping response generation")

    # ----- Save ----- #
    for out in pending_outputs:
        out.pop("_pool_for_response", None)
    inference_results.extend(pending_outputs)
    _atomic_write_json(str(out_json), inference_results)
    log.info("Wrote %d results → %s", len(inference_results), out_json)

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(out_json), "prediction.json")
    log.info("Submission zip: %s", out_zip)

    runtime_meta = {
        "output_tid": output_tid,
        "driver": "run_inference_blind_f1.py",
        "config": "cfg0209 + CF-BPR F_max_recent5",
        "blind_dataset": blind_dataset,
        "sample_size": args.sample_size,
        "last_turn_only": args.last_turn_only,
        "skip_response_generation": args.skip_response_generation,
        "n_results": len(inference_results),
        "n_pending_this_run": len(pending),
        "fallback_zero_pool": fallback_zero_pool,
        "cf_source_nonempty": n_f_nonempty,
        "cf_source_empty": n_f_empty,
        "source_weights": SOURCE_WEIGHTS,
        "cfbpr_depth": CFBPR_DEPTH,
        "cfbpr_recent_k": CFBPR_RECENT_K,
        "a_prime_variant": A_PRIME_VARIANT,
        "a_prime_recent_k": A_PRIME_RECENT_K,
        "a_prime_topn": A_PRIME_K,
        "bm25_k": BM25_K,
        "track_neighbor_k": TRACK_NEIGHBOR_K,
        "pool_k": POOL_K,
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "powell_fit": fit_info,
        "local_cv5_estimate": "0.1700 (5-seed robust, F1 expF1_cfbpr_retrieval.json)",
        "no_sonnet_rerank": True,
        "no_13f_features": True,
        "no_reserved_slots": True,
    }
    _atomic_write_json(str(out_meta), runtime_meta)
    log.info("Metadata: %s", out_meta)
    log.info("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_tid", type=str, required=True)
    parser.add_argument("--blind_dataset", type=str,
                        default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--last_turn_only", action="store_true")
    parser.add_argument("--skip_response_generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
