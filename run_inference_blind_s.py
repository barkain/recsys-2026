"""ABCDF+S blind-set inference driver: cfg0209 + CF-BPR + Source S.

Extends F1 driver with Source S (sequence model, epoch 8 checkpoint).
  1. A' = max(cosine_sim, qwen3) over last 5 played → top 200
  2. B  = last_music_meta BM25 @ 500
  3. C  = full_history BM25 @ 500
  4. D  = track_neighbors (qwen3) @ 200
  5. F  = CF-BPR max_recent_5 @ 200
  6. S  = sequence model predicted embedding @ 200  ← NEW
  7. Weighted-RRF fuse (A=1, B=1, C=1, D=0.5, F=1.0, S=1.0, k=20) → top-50
  8. 8-feature Powell postrank → top-20
  9. Optional Haiku response generation
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
import torch
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
from offline_retrieval_sweep import CachedBM25, load_track_metadata, query_parts
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from mcrs.retrieval_modules.seq_model import SequenceRecommender
from scripts.tune_postrank_v23 import FEATURE_NAMES
from run_inference_blind_f1 import CFBPRMaxRecent, load_cfbpr_index, CFBPR_DEPTH, CFBPR_RECENT_K

# Source S constants
SEQ_MODEL_PATH = REPO_ROOT / "cache" / "seq_model" / "runs" / "20260428_200502_fdddfe" / "epoch_8.pt"
UTT_NPY = REPO_ROOT / "cache" / "seq_model" / "utt_embeddings.npy"
UTT_INDEX_PATH = REPO_ROOT / "cache" / "seq_model" / "utt_embedding_index.json"
S_DEPTH = 200

SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "S": 1.0}

log = logging.getLogger(__name__)


class SourceSRetriever:
    """Wraps the sequence model for inference."""

    def __init__(self, model_path, catalog_ids, catalog_matrix, utt_embs, utt_index, device="cpu"):
        self.device = torch.device(device)
        self.catalog_ids = catalog_ids
        self.catalog_matrix = torch.from_numpy(catalog_matrix).float()
        self.utt_embs = utt_embs
        self.utt_index = utt_index
        self.track_emb_dict = {tid: catalog_matrix[i] for i, tid in enumerate(catalog_ids)}
        self.catalog_id_to_idx = {tid: i for i, tid in enumerate(catalog_ids)}

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        config = checkpoint.get("config", {})
        self.model = SequenceRecommender(
            track_emb_dim=config.get("track_emb_dim", 1024),
            utt_emb_dim=config.get("utt_emb_dim", 384),
            d_model=config.get("d_model", 256),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 4),
            output_dim=config.get("output_dim", 1024),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _get_utt(self, session_id, turn):
        key = f"{session_id}:{turn}"
        idx = self.utt_index.get(key)
        if idx is not None and idx < len(self.utt_embs):
            return self.utt_embs[idx]
        return np.zeros(384, dtype=np.float32)

    @torch.no_grad()
    def topn(self, session_id, music_turns_with_turns, current_turn, topn=200):
        """Return top-N track_ids. music_turns_with_turns = [(turn_num, track_id), ...]"""
        if not music_turns_with_turns:
            return []

        T_hist = len(music_turns_with_turns)
        T = T_hist + 1
        te = np.zeros((T_hist, 1024), dtype=np.float32)
        ue = np.zeros((T, 384), dtype=np.float32)
        ac = np.ones(T_hist, dtype=np.int64)
        ti = np.zeros(T, dtype=np.int64)

        for j, (turn, tid) in enumerate(music_turns_with_turns):
            emb = self.track_emb_dict.get(tid)
            if emb is not None:
                te[j] = emb
            ue[j] = self._get_utt(session_id, turn)
            ti[j] = min(turn - 1, 7)

        ue[T_hist] = self._get_utt(session_id, current_turn)
        ti[T_hist] = min(current_turn - 1, 7)

        target = self.model(
            torch.from_numpy(te).unsqueeze(0).to(self.device),
            torch.from_numpy(ue).unsqueeze(0).to(self.device),
            torch.from_numpy(ac).unsqueeze(0).to(self.device),
            torch.from_numpy(ti).unsqueeze(0).to(self.device),
            torch.tensor([T], device=self.device),
        )

        scores = (target @ self.catalog_matrix.to(self.device).T).squeeze(0).cpu().numpy()

        played_set = {tid for _, tid in music_turns_with_turns}
        for tid in played_set:
            idx = self.catalog_id_to_idx.get(tid)
            if idx is not None:
                scores[idx] = -float("inf")

        topk_idx = np.argpartition(-scores, min(topn, len(scores) - 1))[:topn]
        topk_idx = topk_idx[np.argsort(-scores[topk_idx])]
        return [self.catalog_ids[i] for i in topk_idx]


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    require_cache = bool(os.environ.get("MCRS_REQUIRE_LLM_CACHE"))
    if args.skip_response_generation and not require_cache:
        os.environ["MCRS_REQUIRE_LLM_CACHE"] = "1"

    log.info("=" * 70)
    log.info("ABCDF+S blind inference driver")
    log.info("Sources: A'(qwen3) + B(bm25) + C(bm25) + D(qwen3) + F(cf-bpr) + S(seq model)")
    log.info("Weights: %s  RRF_k=%d  pool_k=%d", SOURCE_WEIGHTS, RRF_K, POOL_K)
    log.info("=" * 70)

    # Powell fit
    weights, fit_info = fit_powell_weights()
    log.info("Powell train nDCG@20: %.4f", fit_info["train_ndcg20"])

    # Retrievers
    log.info("Loading retrievers...")
    metadata = load_track_metadata()
    bm25 = CachedBM25()
    track_sim = TrackSimilarityRetriever(cache_dir=str(REPO_ROOT / "cache"))
    a_prime = APrimeMaxRecent(track_sim, recent_k=A_PRIME_RECENT_K)

    cf_ids, cf_vecs, cf_idx = load_cfbpr_index()
    cfbpr = CFBPRMaxRecent(cf_ids, cf_vecs, cf_idx, recent_k=CFBPR_RECENT_K)
    log.info("CF-BPR: %d tracks", len(cf_ids))

    # Source S
    log.info("Loading Source S (seq model epoch 8)...")
    catalog_ids = json.load(open(REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "track_ids.json"))
    catalog_matrix = np.load(REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "vectors.npy")
    utt_embs = np.load(UTT_NPY)
    utt_index = json.load(open(UTT_INDEX_PATH))
    source_s = SourceSRetriever(str(SEQ_MODEL_PATH), catalog_ids, catalog_matrix, utt_embs, utt_index)
    log.info("Source S loaded: %d catalog tracks", len(catalog_ids))

    # Blind dataset
    blind_dataset = args.blind_dataset
    log.info("Loading blind dataset: %s", blind_dataset)
    db = load_dataset(blind_dataset, split="test")
    log.info("Blind dataset: %d sessions", len(db))

    if args.sample_size:
        db = db.select(range(min(args.sample_size, len(db))))
        log.info("Selected first %d sessions (smoke)", len(db))

    # Build rows
    rows = []
    for item in db:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        if args.last_turn_only:
            turn_num, user_query, history, music_turns = parse_last_turn(item)
            rows.append({"session_id": sid, "user_id": uid, "turn_number": turn_num,
                         "user_query": user_query, "history": history, "music_turns": music_turns})
        else:
            for turn_num, user_query, history, music_turns in parse_all_turns(item):
                rows.append({"session_id": sid, "user_id": uid, "turn_number": turn_num,
                             "user_query": user_query, "history": history, "music_turns": music_turns})

    log.info("Total turns to predict: %d", len(rows))

    output_tid = args.output_tid
    out_dir = REPO_ROOT / "exp" / "inference" / "blind_a"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{output_tid}.json"
    out_zip = out_dir / f"{output_tid}_submission.zip"
    out_meta = out_dir / f"{output_tid}_metadata.json"

    # Resume support
    inference_results = []
    completed = set()
    if args.resume and out_json.exists():
        with open(out_json) as f:
            inference_results = json.load(f)
        completed = {_result_key(r) for r in inference_results}
        log.info("Resuming: %d completed", len(completed))

    pending = [r for r in rows if (str(r["session_id"]),
               None if r["user_id"] is None else str(r["user_id"]),
               int(r["turn_number"])) not in completed]

    maps = {"track_artist": {}, "track_tags": {}, "track_title_toks": {},
            "track_artist_toks": {}, "track_meta_toks": {}}

    # Retrieval
    t_retrieve = time.time()
    queries_b, queries_c = [], []
    for r in pending:
        q_b = " ".join(query_parts(r["history"], r["user_query"], metadata, "last_music_meta"))
        q_c = " ".join(query_parts(r["history"], r["user_query"], metadata, "full"))
        queries_b.append(q_b or r["user_query"])
        queries_c.append(q_c or r["user_query"])

    log.info("BM25 retrieval...")
    src_b = bm25.retrieve_batch(queries_b, topk=BM25_K) if pending else []
    src_c = bm25.retrieve_batch(queries_c, topk=BM25_K) if pending else []

    log.info("Building A' + D + F + S for %d rows", len(pending))
    src_a, src_d, src_f, src_s = [], [], [], []
    n_s_nonempty = 0
    for r in pending:
        played = r["music_turns"]
        src_a.append(a_prime.topn(played, topn=A_PRIME_K) if played else [])
        anchor = played[-1] if played else None
        src_d.append(track_sim.track_id_to_neighbors(anchor, topk=TRACK_NEIGHBOR_K) if anchor else [])
        src_f.append(cfbpr.topn(played, topn=CFBPR_DEPTH) if played else [])

        # Source S: need (turn_number, track_id) pairs
        history_music = [(int(h["turn_number"]), str(h["content"]).strip())
                         for h in r["history"] if h["role"] == "music"]
        s_result = source_s.topn(r["session_id"], history_music, r["turn_number"], topn=S_DEPTH) if history_music else []
        src_s.append(s_result)
        if s_result:
            n_s_nonempty += 1

    log.info("Retrieval done in %.2fs (S nonempty: %d/%d)", time.time() - t_retrieve, n_s_nonempty, len(pending))

    # Fusion + postrank
    log.info("Fusing ABCDFS + postrank")
    pending_outputs = []
    fallback_zero_pool = 0

    for i, r in enumerate(tqdm(pending, desc="rank", disable=not pending)):
        sources = {"A": src_a[i], "B": src_b[i], "C": src_c[i],
                    "D": src_d[i], "F": src_f[i], "S": src_s[i]}
        pool = weighted_rrf(sources, SOURCE_WEIGHTS, topk=POOL_K, k=RRF_K)

        if not pool:
            log.warning("Empty pool for %s/turn%d", r["session_id"], r["turn_number"])
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

        assert len(top20) == TOP_K and len(set(top20)) == TOP_K

        pending_outputs.append({
            "session_id": r["session_id"], "user_id": r["user_id"],
            "turn_number": r["turn_number"],
            "predicted_track_ids": top20,
            "predicted_response": "",
            "_pool_for_response": pool,
        })

    log.info("S nonempty: %d, Fallback: %d", n_s_nonempty, fallback_zero_pool)

    # Response generation
    if not args.skip_response_generation and pending_outputs:
        if not os.environ.get("ANTHROPIC_RECSYS_API_KEY"):
            raise EnvironmentError("ANTHROPIC_RECSYS_API_KEY not set")
        from mcrs.db_item.music_catalog import MusicCatalogDB
        from mcrs.lm_modules.claude import ClaudeModule
        item_db = MusicCatalogDB(dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                                  split_types=["all_tracks"])
        prompts_dir = REPO_ROOT / "mcrs" / "system_prompts"
        sys_prompt = ((prompts_dir / "roleplay.txt").read_text(encoding="utf-8") + "\n" +
                      (prompts_dir / "response_generation.txt").read_text(encoding="utf-8"))
        haiku = ClaudeModule(model="claude-haiku-4-5-20251001")
        log.info("Generating responses for %d turns", len(pending_outputs))
        for r, out in zip(pending, tqdm(pending_outputs, desc="response")):
            top_id = out["predicted_track_ids"][0]
            try:
                top_item = item_db.id_to_metadata(top_id)
            except KeyError:
                top_item = f"track_id: {top_id}"
            session_memory = build_session_memory_for_response(r["history"], r["user_query"], item_db)
            response = haiku.response_generation(sys_prompt, session_memory, top_item)
            out["predicted_response"] = response or ""
    else:
        log.info("Skipping response generation")

    # Save
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
        "driver": "run_inference_blind_s.py",
        "config": "cfg0209 + CF-BPR + Source S (seq model epoch 8)",
        "source_weights": SOURCE_WEIGHTS,
        "seq_model_checkpoint": str(SEQ_MODEL_PATH),
        "s_depth": S_DEPTH,
        "s_nonempty": n_s_nonempty,
        "n_results": len(inference_results),
        "fallback_zero_pool": fallback_zero_pool,
        "local_cv5_estimate": "0.1589 (dev400 quick sanity, BCD+F+S)",
    }
    _atomic_write_json(str(out_meta), runtime_meta)
    log.info("Metadata: %s", out_meta)
    log.info("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_tid", type=str, required=True)
    parser.add_argument("--blind_dataset", type=str, default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--last_turn_only", action="store_true")
    parser.add_argument("--skip_response_generation", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    main(args)
