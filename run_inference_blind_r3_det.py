"""Deterministic R3 blind-set inference driver (no Sonnet rerank).

Pipeline per session:
  1. A' = max(cosine_sim(candidate, played_i)) over last 5 played tracks → top-50
  2. B  = last_music_meta BM25 @ 500
  3. C  = full_history BM25 @ 500
  4. D  = track_neighbors @ 200 (anchor = last played track)
  5. Weighted-RRF fuse  (A'=2, B=2, C=0.5, D=1)  → top-50 pool
  6. 8-feature postrank, weights fit by Powell (single-fit) on
     exp/eval/_r3_confirm_400_deterministic.pkl using B=2, C=0.5, D=1.
  7. Final scoring: scores = X @ weights → top-20.
  8. Optional Haiku response generation per session.

Reuses helpers from offline_retrieval_sweep, mcrs.retrieval_modules.track_sim,
scripts.r3_confirm_400_deterministic (build_pool_features) and
scripts.tune_postrank_v23 (FEATURE_NAMES, INIT_WEIGHTS, fit_weights). No
Sonnet rerank, no use of pre-existing v23 candidate pools.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from offline_retrieval_sweep import CachedBM25, load_track_metadata, query_parts
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from scripts.r3_confirm_400_deterministic import build_pool_features, fit_weights
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens

# Pipeline constants — cfg0209 fusion (v2 submission)
#   RRF k:           60 → 20
#   A' depth:        50 → 200
#   Source weights:  (A',B,C,D) = (2,2,0.5,1) → (1,1,1,0.5)
#   Base 8-feature postrank unchanged.
A_PRIME_VARIANT = "A_max_recent_5"
A_PRIME_RECENT_K = 5
A_PRIME_K = 200
BM25_K = 500
TRACK_NEIGHBOR_K = 200
POOL_K = 50
TOP_K = 20
RRF_K = 20
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5}
POWELL_FIT_PKL = "exp/eval/_r3_confirm_400_deterministic.pkl"
TRACK_SIM_DIR = "cache/track_sim/metadata-qwen3_embedding_0.6b"

log = logging.getLogger(__name__)


# ---------- IO helpers ---------------------------------------------------- #

def _atomic_write_json(path: str, data) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _result_key(result: dict) -> tuple[str, str | None, int]:
    user_id = result.get("user_id")
    return str(result["session_id"]), None if user_id is None else str(user_id), int(result["turn_number"])


# ---------- Powell fit ---------------------------------------------------- #

def fit_powell_weights() -> tuple[np.ndarray, dict[str, Any]]:
    """Single deterministic Powell fit on the cached devset pickle (full data).

    Returns the fitted weight vector aligned with FEATURE_NAMES along with a
    metadata dict describing the fit.
    """
    pkl_path = REPO_ROOT / POWELL_FIT_PKL
    if not pkl_path.exists():
        raise FileNotFoundError(f"Powell fit cache missing: {pkl_path}")

    t0 = time.time()
    log.info("Loading Powell fit pickle %s", pkl_path)
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)  # noqa: S301 - local cache
    n_cases = len(payload["cases"])

    bcd = (SOURCE_WEIGHTS["B"], SOURCE_WEIGHTS["C"], SOURCE_WEIGHTS["D"])
    log.info(
        "Building pool features (n=%d, B=%g, C=%g, D=%g)",
        n_cases, *bcd,
    )
    X, gt_idx, sizes = build_pool_features(payload, bcd)

    log.info("Running Powell fit on full data (no holdout)")
    all_idx = np.arange(X.shape[0], dtype=np.int64)
    weights, train_ndcg = fit_weights(X, gt_idx, sizes, all_idx)
    elapsed = time.time() - t0

    init_w = np.asarray([INIT_WEIGHTS[name] for name in FEATURE_NAMES], dtype=np.float64)
    info = {
        "fit_source": str(pkl_path.relative_to(REPO_ROOT)),
        "n_train": int(n_cases),
        "feature_names": list(FEATURE_NAMES),
        "init_weights": dict(zip(FEATURE_NAMES, init_w.tolist())),
        "fitted_weights": dict(zip(FEATURE_NAMES, weights.tolist())),
        "train_ndcg20": float(train_ndcg),
        "fit_seconds": float(elapsed),
        "source_weights_BCD_for_fit": list(bcd),
        "method": "Powell",
        "options": {"xtol": 1e-3, "ftol": 1e-3, "maxiter": 500},
    }
    return weights, info


# ---------- Session parsing ----------------------------------------------- #

def parse_last_turn(item: dict[str, Any]) -> tuple[int, str, list[dict], list[str]]:
    """Return (turn_number, user_query, history, music_turns) for the LAST user turn.

    music_turns are the played track_ids (strings) up to but not including the
    target turn. history contains all conversation rows strictly before the
    target turn (preserves role).
    """
    df = pd.DataFrame(item["conversations"]).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    if user_rows.empty:
        raise ValueError(f"Session {item.get('session_id')} has no user turns")
    last_user = user_rows.iloc[-1]
    turn_num = int(last_user["turn_number"])
    user_query = str(last_user["content"])

    prior = df[df["turn_number"] < turn_num]
    history = [
        {"role": str(r["role"]), "content": r["content"], "turn_number": int(r["turn_number"])}
        for _, r in prior.iterrows()
    ]
    music_turns = [str(r["content"]).strip() for _, r in prior.iterrows() if r["role"] == "music"]
    return turn_num, user_query, history, music_turns


def parse_all_turns(item: dict[str, Any]) -> list[tuple[int, str, list[dict], list[str]]]:
    """Yield (turn_number, user_query, history, music_turns) for every user turn."""
    df = pd.DataFrame(item["conversations"]).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    out = []
    for _, ur in user_rows.iterrows():
        turn_num = int(ur["turn_number"])
        user_query = str(ur["content"])
        prior = df[df["turn_number"] < turn_num]
        history = [
            {"role": str(r["role"]), "content": r["content"], "turn_number": int(r["turn_number"])}
            for _, r in prior.iterrows()
        ]
        music_turns = [str(r["content"]).strip() for _, r in prior.iterrows() if r["role"] == "music"]
        out.append((turn_num, user_query, history, music_turns))
    return out


# ---------- A' (max-recent-5 cosine) ------------------------------------- #

class APrimeMaxRecent:
    """Score every track by max cosine similarity over the last K played tracks.

    Uses the qwen3 embedding cache (already L2-normalised by
    TrackSimilarityRetriever's build path). Returns top-N candidate ids.
    """

    def __init__(self, retriever: TrackSimilarityRetriever, recent_k: int = A_PRIME_RECENT_K):
        self.recent_k = int(recent_k)
        self.track_ids = retriever.track_ids
        self.vectors = retriever.vectors  # (N, D), already L2-normalised
        self.id_to_idx = retriever._id_to_idx  # noqa: SLF001 — reuse mapping

    def topn(self, played: list[str], topn: int) -> list[str]:
        idxs = [self.id_to_idx.get(str(tid)) for tid in played[-self.recent_k:]]
        idxs = [i for i in idxs if i is not None]
        if not idxs:
            return []
        anchors = self.vectors[idxs]              # (k, D)
        sims = self.vectors @ anchors.T           # (N, k)
        scores = sims.max(axis=1)                 # (N,)
        # Exclude the played tracks themselves to avoid trivial self-match in the pool.
        exclude = set(idxs)
        # Argpartition for top (topn + |exclude|) then sort.
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


# ---------- Weighted RRF fusion ------------------------------------------ #

def weighted_rrf(sources: dict[str, list[str]], weights: dict[str, float], topk: int, k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for name, ranked in sources.items():
        w = weights.get(name, 0.0)
        if w == 0.0 or not ranked:
            continue
        for rank, tid in enumerate(ranked, start=1):
            scores[tid] = scores.get(tid, 0.0) + w / (k + rank)
    return sorted(scores, key=scores.__getitem__, reverse=True)[:topk]


# ---------- Per-session feature builder (matches build_pool_features) ----- #

def _ensure_meta_maps(track_ids: list[str], metadata: dict, maps: dict[str, dict]) -> None:
    """Lazily populate per-track metadata maps with the same keys as the cached
    payload used during Powell fit.
    """
    track_artist = maps["track_artist"]
    track_tags = maps["track_tags"]
    track_title_toks = maps["track_title_toks"]
    track_artist_toks = maps["track_artist_toks"]
    track_meta_toks = maps["track_meta_toks"]
    for tid in track_ids:
        if tid in track_artist:
            continue
        meta = metadata.get(str(tid), {})
        artist = str(meta.get("artist_name", "")).lower().strip()
        raw_tags = meta.get("tag_list") or []
        tags = (
            {str(t).lower().strip() for t in raw_tags if str(t).strip()}
            if isinstance(raw_tags, list) else set()
        )
        title = str(meta.get("track_name", ""))
        album = str(meta.get("album_name", ""))
        meta_parts = [title, str(meta.get("artist_name", "")), album]
        if isinstance(raw_tags, list):
            meta_parts.extend(str(t) for t in raw_tags[:12])
        track_artist[tid] = artist
        track_tags[tid] = tags
        track_title_toks[tid] = tokens(title)
        track_artist_toks[tid] = tokens(meta.get("artist_name", ""))
        track_meta_toks[tid] = tokens(" ".join(meta_parts))


def build_row_features(
    pool: list[str],
    user_messages: list[str],
    music_turns: list[str],
    maps: dict[str, dict],
) -> np.ndarray:
    """Return a (POOL_K, F) feature matrix matching scripts.r3_confirm_400_deterministic
    semantics. Empty/short pools are zero-padded so X has fixed shape.
    """
    track_artist = maps["track_artist"]
    track_tags = maps["track_tags"]
    track_title_toks = maps["track_title_toks"]
    track_artist_toks = maps["track_artist_toks"]
    track_meta_toks = maps["track_meta_toks"]

    F = len(FEATURE_NAMES)
    X = np.zeros((POOL_K, F), dtype=np.float64)
    if not pool:
        return X

    now_tokens = tokens(user_messages[-1]) if user_messages else set()
    all_user_tokens = tokens(" ".join(user_messages)) if user_messages else set()
    played_set = set(music_turns)
    last_artist = track_artist.get(music_turns[-1], "") if music_turns else ""
    last_tags = track_tags.get(music_turns[-1], set()) if music_turns else set()
    prior = [
        (1.0 / (j + 1), track_artist.get(tid, ""), track_tags.get(tid, set()))
        for j, tid in enumerate(reversed(music_turns))
    ]

    for rank, tid in enumerate(pool[:POOL_K], start=1):
        cand_artist = track_artist.get(tid, "")
        cand_tags = track_tags.get(tid, set())
        row = X[rank - 1]
        row[0] = 1.0 / rank
        row[1] = 1.0 if cand_artist and cand_artist == last_artist else 0.0
        if cand_tags or last_tags:
            row[2] = len(cand_tags & last_tags) / len(cand_tags | last_tags)
        row[3] = float(len(track_artist_toks.get(tid, set()) & now_tokens))
        row[4] = float(len(track_title_toks.get(tid, set()) & now_tokens))
        row[5] = float(len(track_meta_toks.get(tid, set()) & all_user_tokens))
        row[6] = 1.0 if tid in played_set else 0.0
        rec = 0.0
        for w_dec, p_artist, p_tags in prior:
            artist_match = 1.0 if cand_artist and cand_artist == p_artist else 0.0
            if cand_tags or p_tags:
                tag_match = len(cand_tags & p_tags) / len(cand_tags | p_tags)
            else:
                tag_match = 0.0
            rec += w_dec * (artist_match + tag_match)
        row[7] = rec
    return X


def postrank_topk(pool: list[str], features: np.ndarray, weights: np.ndarray, k: int = TOP_K) -> list[str]:
    if not pool:
        return []
    n = min(len(pool), POOL_K)
    scores = features[:n] @ weights
    order = np.argsort(-scores, kind="stable")
    return [pool[i] for i in order[:k]]


# ---------- Response generation (Haiku, optional) ------------------------- #

def build_session_memory_for_response(
    history: list[dict],
    user_query: str,
    item_db,
) -> list[dict]:
    """Convert raw conversation rows into the format the response_generation
    prompt expects. role=music turns are translated to assistant metadata
    strings (mirrors mcrs.crs_system.chat behaviour for blind sessions).
    """
    memory: list[dict] = []
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        if role == "music":
            try:
                content = item_db.id_to_metadata(str(content).strip())
            except KeyError:
                content = f"track_id: {content}"
            role = "assistant"
        elif role == "system":
            role = "assistant"
        memory.append({"role": role, "content": str(content)})
    memory.append({"role": "user", "content": user_query})
    return memory


# ---------- Main --------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    require_cache = bool(os.environ.get("MCRS_REQUIRE_LLM_CACHE"))
    if args.skip_response_generation and not require_cache:
        # Belt-and-suspenders fence: any LLM call inside retrieval/ranking
        # would raise. Set it ourselves when the user runs smoke without it.
        os.environ["MCRS_REQUIRE_LLM_CACHE"] = "1"
        require_cache = True

    log.info("=" * 70)
    log.info("Deterministic R3 blind inference driver — cfg0209 fusion (v2)")
    log.info("Fusion config: RRF k=%d, A' depth=%d, weights (A,B,C,D)=(%g,%g,%g,%g)",
             RRF_K, A_PRIME_K,
             SOURCE_WEIGHTS["A"], SOURCE_WEIGHTS["B"], SOURCE_WEIGHTS["C"], SOURCE_WEIGHTS["D"])
    log.info("A' variant: %s (recent_k=%d, top=%d)", A_PRIME_VARIANT, A_PRIME_RECENT_K, A_PRIME_K)
    log.info(
        "RRF weights: A=%g  B=%g  C=%g  D=%g  (RRF_k=%d, pool_k=%d)",
        SOURCE_WEIGHTS["A"], SOURCE_WEIGHTS["B"], SOURCE_WEIGHTS["C"], SOURCE_WEIGHTS["D"],
        RRF_K, POOL_K,
    )
    log.info("BM25 depth: %d  Track-neighbors depth: %d", BM25_K, TRACK_NEIGHBOR_K)
    log.info("Track-sim cache: %s", TRACK_SIM_DIR)
    log.info("Powell fit cache: %s", POWELL_FIT_PKL)
    log.info("MCRS_REQUIRE_LLM_CACHE=%s", os.environ.get("MCRS_REQUIRE_LLM_CACHE", "0"))
    log.info("skip_response_generation=%s", args.skip_response_generation)
    log.info("=" * 70)

    # ----- Powell fit ----- #
    weights, fit_info = fit_powell_weights()
    log.info("Powell fitted weights:")
    for name, val in zip(FEATURE_NAMES, weights):
        log.info("  %-36s % .6f", name, val)
    log.info("Powell train nDCG@20 (full data): %.4f", fit_info["train_ndcg20"])

    # ----- Build retrievers ----- #
    log.info("Loading track metadata catalog (HF arrow cache)")
    metadata = load_track_metadata()

    log.info("Loading BM25 cache")
    bm25 = CachedBM25()

    log.info("Loading track similarity retriever (qwen3 0.6b)")
    track_sim = TrackSimilarityRetriever(cache_dir=str(REPO_ROOT / "cache"))
    a_prime = APrimeMaxRecent(track_sim, recent_k=A_PRIME_RECENT_K)

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
                "session_id": sid,
                "user_id": uid,
                "turn_number": turn_num,
                "user_query": user_query,
                "history": history,
                "music_turns": music_turns,
            })
        else:
            for turn_num, user_query, history, music_turns in parse_all_turns(item):
                rows.append({
                    "session_id": sid,
                    "user_id": uid,
                    "turn_number": turn_num,
                    "user_query": user_query,
                    "history": history,
                    "music_turns": music_turns,
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
        log.info("Resuming from %s: %d completed turns", out_json, len(completed))

    pending = []
    for r in rows:
        key = (
            str(r["session_id"]),
            None if r["user_id"] is None else str(r["user_id"]),
            int(r["turn_number"]),
        )
        if key in completed:
            continue
        pending.append(r)

    if completed:
        log.info("Skipping %d completed turns; %d pending", len(completed), len(pending))

    # ----- Maps for postrank features (lazy fill) ----- #
    maps = {
        "track_artist": {},
        "track_tags": {},
        "track_title_toks": {},
        "track_artist_toks": {},
        "track_meta_toks": {},
    }

    # ----- Retrieval + RRF + postrank ----- #
    t_retrieve = time.time()
    queries_b: list[str] = []
    queries_c: list[str] = []
    for r in pending:
        q_b = " ".join(query_parts(r["history"], r["user_query"], metadata, "last_music_meta"))
        q_c = " ".join(query_parts(r["history"], r["user_query"], metadata, "full"))
        queries_b.append(q_b or r["user_query"])
        queries_c.append(q_c or r["user_query"])

    log.info("BM25 retrieval (B=last_music_meta@%d, C=full@%d)", BM25_K, BM25_K)
    src_b = bm25.retrieve_batch(queries_b, topk=BM25_K) if pending else []
    src_c = bm25.retrieve_batch(queries_c, topk=BM25_K) if pending else []

    log.info("Building per-session A' (max recent %d) + D (track neighbors @ %d)",
             A_PRIME_RECENT_K, TRACK_NEIGHBOR_K)
    src_a: list[list[str]] = []
    src_d: list[list[str]] = []
    for r in pending:
        played = r["music_turns"]
        src_a.append(a_prime.topn(played, topn=A_PRIME_K) if played else [])
        anchor = played[-1] if played else None
        src_d.append(track_sim.track_id_to_neighbors(anchor, topk=TRACK_NEIGHBOR_K) if anchor else [])

    log.info("Retrieval+sim build done in %.2fs", time.time() - t_retrieve)

    log.info("Fusing weighted-RRF and applying postrank for %d rows", len(pending))
    pending_outputs: list[dict[str, Any]] = []
    fallback_zero_pool = 0
    for i, r in enumerate(tqdm(pending, desc="rank", disable=not pending)):
        sources = {"A": src_a[i], "B": src_b[i], "C": src_c[i], "D": src_d[i]}
        pool = weighted_rrf(sources, SOURCE_WEIGHTS, topk=POOL_K, k=RRF_K)
        if not pool:
            # Last-ditch fallback: top BM25 from query alone (preserves submission validity)
            log.warning("Empty pool for session %s/turn%d — falling back to bm25(user_query)",
                        r["session_id"], r["turn_number"])
            pool = bm25.retrieve(r["user_query"], topk=POOL_K)
            fallback_zero_pool += 1

        # Build features and pad pool to exactly POOL_K with deterministic filler
        # (we draw extra unique tracks from BM25 if pool < POOL_K, mostly to avoid
        # ever emitting fewer than 20 tracks). Filler order is deterministic.
        if len(pool) < POOL_K:
            seen = set(pool)
            for tid in src_b[i] + src_c[i]:
                if tid not in seen:
                    pool.append(tid)
                    seen.add(tid)
                    if len(pool) >= POOL_K:
                        break

        # Ensure metadata maps cover every candidate for this row + played history.
        _ensure_meta_maps(pool + r["music_turns"], metadata, maps)

        user_messages = [str(h["content"]) for h in r["history"] if h["role"] == "user"] + [r["user_query"]]
        feats = build_row_features(pool, user_messages, r["music_turns"], maps)
        top20 = postrank_topk(pool, feats, weights, k=TOP_K)

        # Top-20 must be exactly 20 unique track ids; pad from pool/bm25 if needed.
        if len(top20) < TOP_K:
            seen = set(top20)
            for tid in pool + src_b[i] + src_c[i]:
                if tid not in seen:
                    top20.append(tid)
                    seen.add(tid)
                    if len(top20) >= TOP_K:
                        break
        top20 = top20[:TOP_K]

        pending_outputs.append({
            "session_id": r["session_id"],
            "user_id": r["user_id"],
            "turn_number": r["turn_number"],
            "predicted_track_ids": top20,
            "predicted_response": "",
            "_pool_for_response": pool,  # internal, stripped before save
        })

    # ----- Response generation (Haiku) ----- #
    if not args.skip_response_generation and pending_outputs:
        if not os.environ.get("ANTHROPIC_RECSYS_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_RECSYS_API_KEY is not set — required for Haiku response generation. "
                "Set it or pass --skip_response_generation."
            )
        # Lazy import — avoids touching the LLM module under MCRS_REQUIRE_LLM_CACHE smoke runs.
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

        log.info("Generating responses for %d turns (Haiku, sequential)", len(pending_outputs))
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
        log.info("Skipping response generation (skip_response_generation=%s)", args.skip_response_generation)

    # ----- Persist (atomic, with optional batching) ----- #
    for out in pending_outputs:
        out.pop("_pool_for_response", None)
    inference_results.extend(pending_outputs)
    _atomic_write_json(str(out_json), inference_results)
    log.info("Wrote %d results → %s", len(inference_results), out_json)

    # ----- Submission zip ----- #
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(str(out_json), "prediction.json")
    log.info("Submission zip: %s", out_zip)

    # ----- Metadata sidecar ----- #
    runtime_meta = {
        "output_tid": output_tid,
        "blind_dataset": blind_dataset,
        "sample_size": args.sample_size,
        "last_turn_only": args.last_turn_only,
        "skip_response_generation": args.skip_response_generation,
        "n_results": len(inference_results),
        "n_pending_this_run": len(pending),
        "fallback_zero_pool": fallback_zero_pool,
        "a_prime_variant": A_PRIME_VARIANT,
        "a_prime_recent_k": A_PRIME_RECENT_K,
        "a_prime_topn": A_PRIME_K,
        "bm25_k": BM25_K,
        "track_neighbor_k": TRACK_NEIGHBOR_K,
        "pool_k": POOL_K,
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "source_weights": SOURCE_WEIGHTS,
        "powell_fit": fit_info,
        "track_sim_cache": TRACK_SIM_DIR,
        "no_sonnet_rerank": True,
        "no_candidate_pool_reuse": True,
        "mcrs_require_llm_cache": os.environ.get("MCRS_REQUIRE_LLM_CACHE", "0"),
    }
    _atomic_write_json(str(out_meta), runtime_meta)
    log.info("Metadata: %s", out_meta)
    log.info("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_tid", type=str, required=True)
    parser.add_argument("--blind_dataset", type=str,
                        default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Truncate dataset to first N sessions (smoke).")
    parser.add_argument("--last_turn_only", action="store_true",
                        help="Only predict the final user turn per session (Codabench scoring).")
    parser.add_argument("--skip_response_generation", action="store_true",
                        help="Skip Haiku response generation (smoke / retrieval-only).")
    parser.add_argument("--save_intermediate", action="store_true", default=True)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing artifact JSON if present.")
    args = parser.parse_args()
    main(args)
