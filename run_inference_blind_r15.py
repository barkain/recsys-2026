"""R15 blind-set inference: ABCDF+ALS + Q/R14/G weak sources → LightGBM rerank.

Sources: A'(qwen3) + B(BM25) + C(BM25) + D(qwen3) + F(CF-BPR) + ALS
       + Q(Qwen3 query→track) + R14(expanded BM25) + G(LLM generative)
Fusion: weighted-RRF (base=1.0, weak=0.25, k=20) → top-300
Ranker: 37-feature LightGBM LambdaRank → top-20
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

import bm25s
import lightgbm as lgb
import numpy as np
from collections import Counter
from datasets import Dataset, DownloadConfig, concatenate_datasets, load_dataset
from scipy import sparse
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from run_inference_blind_r3_det import (
    APrimeMaxRecent,
    A_PRIME_K,
    A_PRIME_RECENT_K,
    BM25_K,
    RRF_K,
    TOP_K,
    TRACK_NEIGHBOR_K,
    _atomic_write_json,
    _ensure_meta_maps,
    _result_key,
    build_session_memory_for_response,
    parse_all_turns,
    parse_last_turn,
    weighted_rrf,
)
from run_inference_blind_f1 import CFBPRMaxRecent, load_cfbpr_index
from offline_retrieval_sweep import CachedBM25, load_track_metadata
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever
from scripts.tune_postrank_v23 import tokens
from scripts.expS2_lambdarank import FEATURE_NAMES_LR
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.expR15_weak_source_fusion import (
    FEATURE_NAMES_V3, expand_query,
)

from implicit.als import AlternatingLeastSquares

POOL_K = 300
WEAK_W = 0.25
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}
WEAK_WEIGHTS = {"Q": WEAK_W, "R14": WEAK_W, "G": WEAK_W}
ALL_WEIGHTS = {**SOURCE_WEIGHTS, **WEAK_WEIGHTS}
CFBPR_DEPTH = 200
CFBPR_RECENT_K = 5
R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"

log = logging.getLogger(__name__)


def build_als_model():
    """Train ALS on full training data, return factors + mappings."""
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        download_config=DownloadConfig(local_files_only=True),
    )
    train = ds["train"]
    track_set = set()
    session_tracks = []
    for item in train:
        tracks = []
        for c in item["conversations"]:
            if c["role"] == "music":
                tid = str(c["content"]).strip()
                tracks.append(tid)
                track_set.add(tid)
        session_tracks.append(tracks)

    track_ids = sorted(track_set)
    track_to_idx = {t: i for i, t in enumerate(track_ids)}

    rows, cols, vals = [], [], []
    for si, tracks in enumerate(session_tracks):
        for tid in tracks:
            rows.append(si)
            cols.append(track_to_idx[tid])
            vals.append(1.0)
    matrix = sparse.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(session_tracks), len(track_ids)),
        dtype=np.float32,
    )

    model = AlternatingLeastSquares(
        factors=128, alpha=100, regularization=0.05,
        iterations=20, random_state=42, use_gpu=False,
    )
    model.fit(matrix)
    factors = model.item_factors
    if hasattr(factors, "to_numpy"):
        factors = factors.to_numpy()
    elif not isinstance(factors, np.ndarray):
        factors = np.array(factors)
    return factors, track_ids, track_to_idx


def als_retrieve(played, track_to_idx, factors, track_ids, topk=200, decay=0.8):
    """Retrieve top-k tracks using ALS item factors."""
    anchors = [track_to_idx[t] for t in played if t in track_to_idx]
    if not anchors:
        return [], None
    n = len(anchors)
    w = np.array([decay ** (n - 1 - j) for j in range(n)], dtype=np.float32)
    w /= w.sum()
    vec = np.zeros(factors.shape[1], dtype=np.float32)
    for j, idx in enumerate(anchors):
        vec += w[j] * factors[idx]
    scores = factors @ vec
    played_set = set(anchors)
    for idx in played_set:
        scores[idx] = -np.inf
    top_idx = np.argpartition(-scores, min(topk, len(scores) - 1))[:topk]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [track_ids[i] for i in top_idx], vec


def build_r14_index():
    """Build BM25 index for R14 expanded retrieval."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    splits = []
    for split in ["all_tracks", "test_tracks"]:
        matches = sorted(hf_cache.glob(
            f"talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            f"talk_play_data-challenge-track-metadata-{split}.arrow"))
        if matches:
            splits.append(Dataset.from_file(str(matches[-1])))
    combined = concatenate_datasets(splits)
    cols = combined.to_dict()
    track_ids = [str(tid) for tid in cols["track_id"]]
    meta = {track_ids[i]: {k: cols[k][i] for k in cols} for i in range(len(track_ids))}

    corpus_texts = []
    for tid in track_ids:
        m = meta[tid]
        name = " ".join(m.get("track_name", []) if isinstance(m.get("track_name"), list)
                        else [str(m.get("track_name", ""))])
        artist = " ".join(m.get("artist_name", []) if isinstance(m.get("artist_name"), list)
                          else [str(m.get("artist_name", ""))])
        album = " ".join(m.get("album_name", []) if isinstance(m.get("album_name"), list)
                         else [str(m.get("album_name", ""))])
        tags = m.get("tag_list", [])
        tags_str = " ".join(str(t) for t in tags) if isinstance(tags, list) else str(tags)
        corpus_texts.append(f"{name} {artist} {album} {tags_str}")

    corpus_tokens = bm25s.tokenize(corpus_texts)
    model = bm25s.BM25()
    model.index(corpus_tokens)
    return model, track_ids


def r14_retrieve_batch(r14_model, r14_track_ids, rows, maps, topk=300):
    """R14 kitchen_sink query construction + retrieval."""
    ta = maps["track_artist"]
    tt = maps["track_tags"]
    queries = []
    for r in rows:
        q = expand_query(r["user_query"])
        played = r["music_turns"]
        parts = [q]
        if played:
            last_tags = tt.get(played[-1], set())
            if last_tags:
                parts.append(" ".join(last_tags))
        artists = set()
        for tid in played:
            a = ta.get(tid, "")
            if isinstance(a, list):
                artists.update(a)
            elif a:
                artists.add(a)
        if artists:
            parts.append(" ".join(artists))
        queries.append(" ".join(parts))

    toks = bm25s.tokenize([q.lower() for q in queries])
    results, _ = r14_model.retrieve(toks, k=topk)
    out = []
    for i in range(len(rows)):
        out.append([r14_track_ids[int(idx)] for idx in results[i] if int(idx) >= 0])
    return out


def embed_blind_queries(queries):
    """Embed queries with Qwen3 in a subprocess to avoid implicit conflicts."""
    import subprocess, tempfile
    tmp_dir = Path(tempfile.mkdtemp())
    queries_path = tmp_dir / "queries.json"
    emb_path = tmp_dir / "embeddings.npy"

    with open(queries_path, "w") as f:
        json.dump(queries, f)

    script = f'''
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json, numpy as np, torch
from sentence_transformers import SentenceTransformer

with open("{queries_path}") as f:
    queries = json.load(f)
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True,
                            model_kwargs={{"torch_dtype": torch.float16}}, device=device)
emb = model.encode(queries, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
np.save("{emb_path}", emb.astype(np.float32))
print(f"Embedded {{len(queries)}} queries on {{device}}", flush=True)
'''
    result = subprocess.run(["uv", "run", "python", "-c", script],
                            capture_output=True, text=True, timeout=600)
    log.info("Q embed subprocess: %s", result.stdout.strip())
    if result.returncode != 0:
        log.error("Q embed stderr: %s", result.stderr[:500])
        raise RuntimeError(f"Q embedding failed (code {result.returncode})")
    emb = np.load(emb_path)
    queries_path.unlink()
    emb_path.unlink()
    tmp_dir.rmdir()
    return emb


def q_retrieve(query_embs, track_vecs, track_ids_q, played_lists, topk=300):
    """Retrieve tracks by query embedding similarity."""
    out = []
    for i in range(len(query_embs)):
        scores = track_vecs @ query_embs[i]
        played_set = set(played_lists[i])
        for idx, tid in enumerate(track_ids_q):
            if tid in played_set:
                scores[idx] = -np.inf
        top_idx = np.argpartition(-scores, min(topk, len(scores) - 1))[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        out.append([track_ids_q[j] for j in top_idx])
    return out


def train_lambdarank(als_factors, als_track_ids, als_track_to_idx, track_pop):
    """Train LambdaRank on R12 payload with R15 37-feature set."""
    log.info("Loading R12 payload for LambdaRank training...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)

    log.info("Building ALS source for training data...")
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        tracks, vec = als_retrieve(played, als_track_to_idx, als_factors, als_track_ids)
        als_source.append(tracks)
        als_vecs.append(vec)

    # Build Q source from cached embeddings
    log.info("Loading cached Q embeddings for training...")
    emb_context = np.load(REPO_ROOT / "cache" / "r13_query_emb" / "emb_context.npy")
    track_ids_q = json.load(open(REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "track_ids.json"))
    track_vecs = np.load(REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "vectors.npy")
    played_lists_train = [c["music_turns"] for c in cases]
    q_source = q_retrieve(emb_context, track_vecs, track_ids_q, played_lists_train, topk=300)
    del emb_context, track_vecs

    # Build R14 source
    log.info("Building R14 source for training...")
    r14_model, r14_tids = build_r14_index()
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    train_rows = [{"user_query": c["user_query"], "music_turns": c["music_turns"],
                   "history": c["history"]} for c in cases]
    maps_train = {"track_artist": ta, "track_tags": tt}
    r14_source = r14_retrieve_batch(r14_model, r14_tids, train_rows, maps_train, topk=300)
    del r14_model

    # G source from payload
    g_source = payload["src_g"]

    n_feat = len(FEATURE_NAMES_V3)
    log.info("Building %d-feature matrix (pool_k=%d)...", n_feat, POOL_K)
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        base_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
        }
        all_lists = dict(base_lists)
        all_lists["Q"] = q_source[i]
        all_lists["R14"] = r14_source[i]
        all_lists["G"] = g_source[i] or []
        pool = weighted_rrf(all_lists, ALL_WEIGHTS, topk=POOL_K, k=RRF_K)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank_base = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                         for sn, sl in base_lists.items()}
        q_rank = {tid: r + 1 for r, tid in enumerate(q_source[i][:300])}
        r14_rank = {tid: r + 1 for r, tid in enumerate(r14_source[i][:300])}
        g_rank = {tid: r + 1 for r, tid in enumerate((g_source[i] or [])[:300])}
        base_set = set()
        for sl in base_lists.values():
            base_set.update(sl)

        user_msgs = ([str(r["content"]) for r in c["history"] if r["role"] == "user"]
                     + [c["user_query"]])
        played = c["music_turns"]
        n_hist = len(played)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        played_set = set(played)
        l_artist = ta.get(played[-1], "") if played else ""
        l_tags = tt.get(played[-1], set()) if played else set()
        prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]
        sv = als_vecs[i]
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)

        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[i, rank - 1]
            row[0] = 1.0 / rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags:
                row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(tat.get(tid, set()) & now_tok))
            row[4] = float(len(ttl.get(tid, set()) & now_tok))
            row[5] = float(len(tmt.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank_base[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank_base[sname] else 0.0
            row[20] = sum(1 for sn in base_lists if tid in src_rank_base[sn])
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            pop = track_pop.get(tid, 0)
            row[23] = pop / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            # R15 weak source features
            row[27] = 1.0 / q_rank[tid] if tid in q_rank else 0.0
            row[28] = 1.0 / r14_rank[tid] if tid in r14_rank else 0.0
            row[29] = 1.0 / g_rank[tid] if tid in g_rank else 0.0
            row[30] = 1.0 if tid in q_rank else 0.0
            row[31] = 1.0 if tid in r14_rank else 0.0
            row[32] = 1.0 if tid in g_rank else 0.0
            row[33] = (tid in q_rank) + (tid in r14_rank) + (tid in g_rank)
            row[34] = 1.0 / np.log2(pop + 2)
            content_only = tid not in base_set
            row[35] = 1.0 if content_only else 0.0
            row[36] = row[35] * row[34]

    pool_hit = float(np.mean(gt_idx >= 0))
    log.info("Training pool_hit@%d = %.4f", POOL_K, pool_hit)

    X_flat = X.reshape(-1, n_feat)
    labels = np.zeros(n * POOL_K, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels[i * POOL_K + gt_idx[i]] = 1.0
    group_sizes = np.array([int(sizes[i]) for i in range(n)], dtype=np.int32)

    train_flat = []
    for i in range(n):
        for k in range(int(sizes[i])):
            train_flat.append(i * POOL_K + k)
    X_train = X_flat[train_flat]
    y_train = labels[train_flat]

    dtrain = lgb.Dataset(X_train, y_train, group=group_sizes,
                         feature_name=FEATURE_NAMES_V3, free_raw_data=False)

    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [20],
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbose": -1,
        "random_state": 42,
        "force_col_wise": True,
    }

    log.info("Training LambdaRank (n=%d, %d features)...", len(X_train), n_feat)
    model = lgb.train(lgb_params, dtrain, num_boost_round=300)
    log.info("LambdaRank training complete (%d trees)", model.num_trees())
    return model


def build_row_features_r15(pool, user_messages, played, base_lists, als_vec,
                           als_factors, als_track_to_idx, maps, track_pop,
                           max_pop, q_list, r14_list, g_list):
    """Build R15 37-feature vector for each candidate in pool."""
    n_feat = len(FEATURE_NAMES_V3)
    feats = np.zeros((len(pool), n_feat), dtype=np.float64)

    ta = maps["track_artist"]
    tt = maps["track_tags"]
    ttl = maps["track_title_toks"]
    tat = maps["track_artist_toks"]
    tmt = maps["track_meta_toks"]

    now_tok = tokens(user_messages[-1]) if user_messages else set()
    all_tok = tokens(" ".join(user_messages))
    played_set = set(played)
    n_hist = len(played)
    l_artist = ta.get(played[-1], "") if played else ""
    l_tags = tt.get(played[-1], set()) if played else set()
    prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
             for j, t in enumerate(reversed(played))]

    src_rank_base = {sn: {tid: r + 1 for r, tid in enumerate(sl)}
                     for sn, sl in base_lists.items()}
    q_rank = {tid: r + 1 for r, tid in enumerate(q_list[:300])}
    r14_rank = {tid: r + 1 for r, tid in enumerate(r14_list[:300])}
    g_rank = {tid: r + 1 for r, tid in enumerate(g_list[:300])}
    base_set = set()
    for sl in base_lists.values():
        base_set.update(sl)

    pool_artists = [ta.get(tid, "") for tid in pool]
    artist_counts = Counter(a for a in pool_artists if a)

    for rank, tid in enumerate(pool, start=1):
        ca = ta.get(tid, "")
        ct = tt.get(tid, set())
        row = feats[rank - 1]
        row[0] = 1.0 / rank
        row[1] = 1.0 if ca and ca == l_artist else 0.0
        if ct or l_tags:
            row[2] = len(ct & l_tags) / len(ct | l_tags)
        row[3] = float(len(tat.get(tid, set()) & now_tok))
        row[4] = float(len(ttl.get(tid, set()) & now_tok))
        row[5] = float(len(tmt.get(tid, set()) & all_tok))
        row[6] = 1.0 if tid in played_set else 0.0
        rec = 0.0
        for wd, pa, pt in prior:
            am = 1.0 if ca and ca == pa else 0.0
            tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
            rec += wd * (am + tm)
        row[7] = rec
        for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
            sr = src_rank_base[sname].get(tid)
            row[8 + fi] = 1.0 / sr if sr else 0.0
        for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
            row[14 + fi] = 1.0 if tid in src_rank_base[sname] else 0.0
        row[20] = sum(1 for sn in base_lists if tid in src_rank_base[sn])
        if als_vec is not None:
            aidx = als_track_to_idx.get(tid)
            if aidx is not None:
                row[21] = float(np.dot(als_vec, als_factors[aidx]))
        row[22] = float(n_hist)
        pop = track_pop.get(tid, 0)
        row[23] = pop / max_pop
        row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
        row[25] = float(artist_counts.get(ca, 0)) if ca else 0
        row[26] = row[20]
        # R15 weak source features
        row[27] = 1.0 / q_rank[tid] if tid in q_rank else 0.0
        row[28] = 1.0 / r14_rank[tid] if tid in r14_rank else 0.0
        row[29] = 1.0 / g_rank[tid] if tid in g_rank else 0.0
        row[30] = 1.0 if tid in q_rank else 0.0
        row[31] = 1.0 if tid in r14_rank else 0.0
        row[32] = 1.0 if tid in g_rank else 0.0
        row[33] = (tid in q_rank) + (tid in r14_rank) + (tid in g_rank)
        row[34] = 1.0 / np.log2(pop + 2)
        content_only = tid not in base_set
        row[35] = 1.0 if content_only else 0.0
        row[36] = row[35] * row[34]

    return feats


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Don't set MCRS_REQUIRE_LLM_CACHE here — G source needs API calls.
    # We'll set it later, just before response generation.

    log.info("=" * 70)
    log.info("R15 blind inference: ABCDF+ALS+Q+R14+G → LightGBM rerank")
    log.info("Sources: A'+B+C+D+F+ALS (w=1) + Q+R14+G (w=%.2f)", WEAK_W)
    log.info("Pool_k=%d  Features=%d  Ranker=LightGBM LambdaRank R15", POOL_K, len(FEATURE_NAMES_V3))
    log.info("=" * 70)

    # ----- Train ALS ----- #
    log.info("Training ALS model...")
    als_factors, als_track_ids, als_track_to_idx = build_als_model()
    log.info("ALS: %d tracks, %d factors", len(als_track_ids), als_factors.shape[1])

    # ----- Build popularity stats ----- #
    log.info("Building popularity stats...")
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1
    log.info("Popularity: %d tracks", len(track_pop))

    # ----- Train LambdaRank ----- #
    lr_model = train_lambdarank(als_factors, als_track_ids, als_track_to_idx, track_pop)

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

    # ----- Load blind dataset ----- #
    log.info("Loading blind dataset: %s", args.blind_dataset)
    db = load_dataset(args.blind_dataset, split="test")
    log.info("Blind dataset: %d sessions", len(db))

    if args.sample_size:
        db = db.select(range(min(args.sample_size, len(db))))
        log.info("Selected first %d sessions (smoke)", len(db))

    # ----- Build per-row context ----- #
    rows: list[dict[str, Any]] = []
    for item in db:
        sid = str(item["session_id"])
        uid = item.get("user_id")
        if not args.all_turns:
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

    # ----- Maps for features ----- #
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

    log.info("Building A' + D + F + ALS for %d rows", len(pending))
    src_a: list[list[str]] = []
    src_d: list[list[str]] = []
    src_f: list[list[str]] = []
    src_als: list[list[str]] = []
    als_vecs: list[np.ndarray | None] = []
    for r in pending:
        played = r["music_turns"]
        src_a.append(a_prime.topn(played, topn=A_PRIME_K) if played else [])
        anchor = played[-1] if played else None
        src_d.append(track_sim.track_id_to_neighbors(anchor, topk=TRACK_NEIGHBOR_K) if anchor else [])
        src_f.append(cfbpr.topn(played, topn=CFBPR_DEPTH) if played else [])
        als_result, als_vec = als_retrieve(played, als_track_to_idx, als_factors, als_track_ids)
        src_als.append(als_result)
        als_vecs.append(als_vec)

    log.info("Base retrieval done in %.2fs", time.time() - t_retrieve)

    # ----- Weak sources: Q, R14, G ----- #
    t_weak = time.time()

    # Q source: embed blind queries with Qwen3 (context variant)
    log.info("Q source: embedding %d blind queries...", len(pending))
    blind_queries = []
    for r in pending:
        hist_user = [str(h["content"]) for h in r["history"] if h["role"] == "user"]
        if hist_user:
            blind_queries.append(hist_user[-1] + " " + r["user_query"])
        else:
            blind_queries.append(r["user_query"])
    q_embs = embed_blind_queries(blind_queries)
    track_ids_q = json.load(open(REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "track_ids.json"))
    track_vecs_q = np.load(REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "vectors.npy")
    played_lists = [r["music_turns"] for r in pending]
    src_q = q_retrieve(q_embs, track_vecs_q, track_ids_q, played_lists, topk=300)
    del q_embs, track_vecs_q
    log.info("Q source: done (%d rows)", len(src_q))

    # R14 source: expanded BM25
    log.info("R14 source: building index + retrieval...")
    r14_model, r14_tids = build_r14_index()
    src_r14 = r14_retrieve_batch(r14_model, r14_tids, pending, maps, topk=300)
    del r14_model
    log.info("R14 source: done (%d rows)", len(src_r14))

    # G source: generative retrieval
    log.info("G source: LLM generative retrieval...")
    from mcrs.retrieval_modules.generative import GenerativeRetriever
    gen_retriever = GenerativeRetriever(model="claude-haiku-4-5-20251001")
    src_g: list[list[str]] = []
    for r in pending:
        session_memory = r["history"]
        suggestions = gen_retriever.get_suggestions(session_memory, r["user_query"])
        gen_queries = gen_retriever.suggestions_to_queries(suggestions)
        g_tracks = []
        seen = set()
        for gq in gen_queries:
            for tid in bm25.retrieve(gq, topk=3):
                if tid not in seen:
                    g_tracks.append(tid)
                    seen.add(tid)
        src_g.append(g_tracks)
    log.info("G source: done (%d rows, avg %.1f tracks)", len(src_g),
             np.mean([len(g) for g in src_g]) if src_g else 0)

    log.info("Weak source retrieval done in %.2fs", time.time() - t_weak)

    # ----- Fusion + LambdaRank rerank ----- #
    log.info("Fusing ABCDF+ALS+Q+R14+G + LambdaRank for %d rows", len(pending))
    pending_outputs: list[dict[str, Any]] = []
    fallback_zero_pool = 0

    for i, r in enumerate(tqdm(pending, desc="rank", disable=not pending)):
        base_lists = {
            "A": src_a[i], "B": src_b[i], "C": src_c[i],
            "D": src_d[i], "F": src_f[i], "ALS": src_als[i],
        }
        all_lists = dict(base_lists)
        all_lists["Q"] = src_q[i]
        all_lists["R14"] = src_r14[i]
        all_lists["G"] = src_g[i]
        pool = weighted_rrf(all_lists, ALL_WEIGHTS, topk=POOL_K, k=RRF_K)

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

        feats = build_row_features_r15(
            pool, user_messages, r["music_turns"], base_lists,
            als_vecs[i], als_factors, als_track_to_idx, maps,
            track_pop, max_pop, src_q[i], src_r14[i], src_g[i])

        scores = lr_model.predict(feats)
        ranked_idx = np.argsort(-scores)
        top20 = [pool[j] for j in ranked_idx[:TOP_K]]

        if len(top20) < TOP_K:
            seen = set(top20)
            for tid in pool + src_b[i] + src_c[i]:
                if tid not in seen:
                    top20.append(tid)
                    seen.add(tid)
                    if len(top20) >= TOP_K:
                        break
        top20 = top20[:TOP_K]

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

    log.info("Fallback zero-pool: %d", fallback_zero_pool)

    # ----- Response assembly ----- #
    if args.skip_response_generation:
        os.environ["MCRS_REQUIRE_LLM_CACHE"] = "1"

    if args.hybrid_responses_from:
        log.info("Loading hybrid responses from %s", args.hybrid_responses_from)
        with open(args.hybrid_responses_from, encoding="utf-8") as f:
            prior_results = json.load(f)
        prior_by_sid = {r["session_id"]: r for r in prior_results}
        n_reused, n_generated = 0, 0
        for out in pending_outputs:
            sid = out["session_id"]
            top20_set = set(out["predicted_track_ids"])
            prior = prior_by_sid.get(sid)
            if prior and prior["predicted_track_ids"][0] in top20_set:
                resp = prior["predicted_response"].lstrip(",").lstrip()
                if resp.strip():
                    out["predicted_response"] = resp
                    n_reused += 1
                    continue
            out["predicted_response"] = ""
            n_generated += 1
        log.info("Hybrid responses: %d reused, %d need generation", n_reused, n_generated)

        if n_generated > 0 and not args.skip_response_generation:
            if not os.environ.get("ANTHROPIC_RECSYS_API_KEY"):
                raise EnvironmentError(
                    f"{n_generated} rows need new responses but ANTHROPIC_RECSYS_API_KEY not set."
                )
            from mcrs.db_item.music_catalog import MusicCatalogDB
            from mcrs.lm_modules.claude import ClaudeModule

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
            for r, out in zip(pending, pending_outputs):
                if out["predicted_response"]:
                    continue
                top_id = out["predicted_track_ids"][0]
                try:
                    top_item = item_db.id_to_metadata(top_id)
                except KeyError:
                    top_item = f"track_id: {top_id}"
                session_memory = build_session_memory_for_response(
                    r["history"], r["user_query"], item_db)
                response = haiku.response_generation(sys_prompt, session_memory, top_item)
                out["predicted_response"] = (response or "").lstrip(",").lstrip()
            log.info("Generated %d new responses", n_generated)
        elif n_generated > 0:
            log.warning("%d rows have empty responses (no API key, skip_response_generation)", n_generated)

    elif not args.skip_response_generation and pending_outputs:
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
            session_memory = build_session_memory_for_response(
                r["history"], r["user_query"], item_db)
            response = haiku.response_generation(sys_prompt, session_memory, top_item)
            out["predicted_response"] = (response or "").lstrip(",").lstrip()
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
        "driver": "run_inference_blind_r15.py",
        "config": "ABCDF+ALS+Q+R14+G → LightGBM LambdaRank R15",
        "blind_dataset": args.blind_dataset,
        "sample_size": args.sample_size,
        "last_turn_only": not args.all_turns,
        "skip_response_generation": args.skip_response_generation,
        "n_results": len(inference_results),
        "n_pending_this_run": len(pending),
        "fallback_zero_pool": fallback_zero_pool,
        "source_weights": ALL_WEIGHTS,
        "pool_k": POOL_K,
        "weak_w": WEAK_W,
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "n_features": len(FEATURE_NAMES_V3),
        "feature_names": FEATURE_NAMES_V3,
        "ranker": "LightGBM LambdaRank R15 (300 rounds, 31 leaves, 37 features)",
        "local_cv5_estimate": "0.2129 (v3+weak_w0.25_p300, expR15)",
        "local_last_turn_estimate": "0.2315",
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
    parser.add_argument("--all_turns", action="store_true",
                        help="Predict all turns (default: last turn only)")
    parser.add_argument("--skip_response_generation", action="store_true")
    parser.add_argument("--hybrid_responses_from", type=str, default=None,
                        help="Path to prior artifact JSON for hybrid response reuse")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
