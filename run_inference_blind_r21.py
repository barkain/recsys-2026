"""R21 blind-set inference: ABCDF+ALS+R21 → LightGBM rerank.

Sources: A'(qwen3) + B(BM25) + C(BM25) + D(qwen3) + F(CF-BPR) + ALS + R21(supervised BGE)
Fusion: weighted-RRF (base=1.0, R21=1.0, k=20) → top-300
Ranker: 29-feature LightGBM LambdaRank → top-20
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

import lightgbm as lgb
import numpy as np
from datasets import DownloadConfig, load_dataset
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

from implicit.als import AlternatingLeastSquares

POOL_K = 300
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
FEATURE_NAMES_R21 = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
R21_CACHE = REPO_ROOT / "cache" / "r21_production"
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


def train_lambdarank(als_factors, als_track_ids, als_track_to_idx, track_pop):
    """Train LambdaRank on R12 payload with V2 + R21 features."""
    log.info("Loading R12 payload for LambdaRank training...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)

    log.info("Building ALS source for training data...")
    als_source = []
    als_vecs = []
    for c in cases:
        played = c["music_turns"]
        tracks, vec = als_retrieve(played, als_track_to_idx, als_factors, als_track_ids)
        als_source.append(tracks)
        als_vecs.append(vec)

    # Load OUT-OF-FOLD R21 retrieval lists (honest, not in-sample)
    oof_path = R21_CACHE / "dev_r21_oof_lists.json"
    if oof_path.exists():
        log.info("Loading out-of-fold R21 retrieval lists...")
        with open(oof_path) as f:
            r21_source = json.load(f)
    else:
        log.info("No out-of-fold R21 lists found, using empty lists (R21 features=0 for training)")
        r21_source = [[] for _ in range(n)]
    n_nonempty = sum(1 for x in r21_source if x)
    log.info("R21 training lists: %d/%d non-empty (out-of-fold)", n_nonempty, n)

    n_feat = len(FEATURE_NAMES_R21)
    log.info("Building %d-feature matrix (pool_k=%d)...", n_feat, POOL_K)
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]

    max_pop = max(track_pop.values()) if track_pop else 1
    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i],
        }
        pool = weighted_rrf(src_lists, SOURCE_WEIGHTS, topk=POOL_K, k=RRF_K)
        sizes[i] = len(pool)
        if c["gt"] in pool:
            gt_idx[i] = pool.index(c["gt"])

        src_rank = {}
        for sname, slist in src_lists.items():
            src_rank[sname] = {tid: rank + 1 for rank, tid in enumerate(slist)}

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
        from collections import Counter
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}

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
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
            row[20] = sum(1 for sname in src_lists if tid in src_rank[sname])
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            # V2 features
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            # R21 features
            row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            row[28] = 1.0 if tid in r21_rank_map else 0.0

    pool_hit = float(np.mean(gt_idx >= 0))
    log.info("Training pool_hit@%d = %.4f", POOL_K, pool_hit)

    # Flatten for LightGBM
    X_flat = X.reshape(-1, n_feat)
    labels = np.zeros(n * POOL_K, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels[i * POOL_K + gt_idx[i]] = 1.0
    group_sizes = np.array([int(sizes[i]) for i in range(n)], dtype=np.int32)

    # Build flat arrays matching group sizes
    train_flat = []
    for i in range(n):
        for k in range(int(sizes[i])):
            train_flat.append(i * POOL_K + k)
    X_train = X_flat[train_flat]
    y_train = labels[train_flat]

    dtrain = lgb.Dataset(X_train, y_train, group=group_sizes,
                         feature_name=FEATURE_NAMES_R21, free_raw_data=False)

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


def build_row_features_lr(pool, user_messages, played, src_lists, als_vec,
                          als_factors, als_track_to_idx, maps, track_pop,
                          max_pop, r21_list=None):
    """Build R21 29-feature vector for each candidate in pool."""
    n_feat = len(FEATURE_NAMES_R21)
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

    src_rank = {}
    for sname, slist in src_lists.items():
        src_rank[sname] = {tid: rank + 1 for rank, tid in enumerate(slist)}

    pool_artists = [ta.get(tid, "") for tid in pool]
    from collections import Counter
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
            sr = src_rank[sname].get(tid)
            row[8 + fi] = 1.0 / sr if sr else 0.0
        for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
            row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
        row[20] = sum(1 for sname in src_lists if tid in src_rank[sname])
        if als_vec is not None:
            aidx = als_track_to_idx.get(tid)
            if aidx is not None:
                row[21] = float(np.dot(als_vec, als_factors[aidx]))
        row[22] = float(n_hist)
        # V2 features
        row[23] = track_pop.get(tid, 0) / max_pop
        row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
        row[25] = float(artist_counts.get(ca, 0)) if ca else 0
        row[26] = row[20]
        # R21 features
        r21_rank_map = {tid: r + 1 for r, tid in enumerate((r21_list or [])[:300])}
        row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
        row[28] = 1.0 if tid in r21_rank_map else 0.0

    return feats


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.skip_response_generation:
        os.environ.setdefault("MCRS_REQUIRE_LLM_CACHE", "1")

    log.info("=" * 70)
    log.info("R21 blind inference: ABCDF+ALS+R21 → LightGBM rerank")
    log.info("Sources: A'+B+C+D+F+ALS+R21(supervised BGE)")
    log.info("Pool_k=%d  Features=%d  Ranker=LightGBM LambdaRank R21", POOL_K, len(FEATURE_NAMES_R21))
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

    # ----- R21 supervised retrieval (subprocess to avoid implicit/torch conflict) ----- #
    t_r21 = time.time()
    log.info("R21: Encoding blind queries in subprocess...")

    import subprocess as _sp
    import tempfile as _tmp

    # Save queries to temp file
    _tmpdir = Path(_tmp.mkdtemp())
    _queries_file = _tmpdir / "queries.json"

    from scripts.train_r21_production import build_query_text as r21_query_text
    blind_queries_r21 = [r21_query_text(r["history"], r["user_query"]) for r in pending]
    blind_played = [r["music_turns"] for r in pending]
    with open(_queries_file, "w") as f:
        json.dump({"queries": blind_queries_r21, "played": blind_played}, f)

    _r21_script = f'''
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json, numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

cache = Path("{R21_CACHE}")
model = SentenceTransformer(str(cache / "model"), device="cpu")
track_embs = np.load(cache / "track_embeddings.npy")
with open(cache / "track_ids.json") as f:
    track_ids = json.load(f)

with open("{_queries_file}") as f:
    data = json.load(f)
queries = data["queries"]
played_lists = data["played"]

embs = model.encode(queries, batch_size=64, normalize_embeddings=True).astype(np.float32)
print(f"Encoded {{len(queries)}} queries", flush=True)

results = []
for i in range(len(queries)):
    scores = track_embs @ embs[i]
    played_set = set(played_lists[i])
    for idx, tid in enumerate(track_ids):
        if tid in played_set:
            scores[idx] = -np.inf
    top_idx = np.argpartition(-scores, 300)[:300]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    results.append([track_ids[j] for j in top_idx])

out = Path("{_tmpdir}") / "r21_results.json"
with open(out, "w") as f:
    json.dump(results, f)
print(f"Saved {{len(results)}} retrieval lists", flush=True)
'''
    _result = _sp.run(["uv", "run", "python", "-c", _r21_script],
                       capture_output=True, text=True, timeout=600)
    log.info("R21 subprocess: %s", _result.stdout.strip())
    if _result.returncode != 0:
        log.error("R21 subprocess stderr: %s", _result.stderr[:500])
        raise RuntimeError(f"R21 encoding failed (code {_result.returncode})")

    with open(_tmpdir / "r21_results.json") as f:
        src_r21 = json.load(f)
    _queries_file.unlink()
    (_tmpdir / "r21_results.json").unlink()
    _tmpdir.rmdir()

    log.info("R21: %d retrieval lists, done in %.2fs", len(src_r21), time.time() - t_r21)

    # ----- Fusion + LambdaRank rerank ----- #
    log.info("Fusing ABCDF+ALS+R21 + LambdaRank for %d rows", len(pending))
    pending_outputs: list[dict[str, Any]] = []
    fallback_zero_pool = 0

    for i, r in enumerate(tqdm(pending, desc="rank", disable=not pending)):
        src_lists = {
            "A": src_a[i], "B": src_b[i], "C": src_c[i],
            "D": src_d[i], "F": src_f[i], "ALS": src_als[i],
            "R21": src_r21[i],
        }
        pool = weighted_rrf(src_lists, SOURCE_WEIGHTS, topk=POOL_K, k=RRF_K)

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

        feats = build_row_features_lr(
            pool, user_messages, r["music_turns"], src_lists,
            als_vecs[i], als_factors, als_track_to_idx, maps,
            track_pop, max_pop, r21_list=src_r21[i])

        # LambdaRank scoring
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
        "driver": "run_inference_blind_r21.py",
        "config": "ABCDF+ALS+R21 → LightGBM LambdaRank",
        "blind_dataset": args.blind_dataset,
        "sample_size": args.sample_size,
        "last_turn_only": not args.all_turns,
        "skip_response_generation": args.skip_response_generation,
        "n_results": len(inference_results),
        "n_pending_this_run": len(pending),
        "fallback_zero_pool": fallback_zero_pool,
        "source_weights": SOURCE_WEIGHTS,
        "pool_k": POOL_K,
        "top_k": TOP_K,
        "rrf_k": RRF_K,
        "n_features": len(FEATURE_NAMES_R21),
        "feature_names": FEATURE_NAMES_R21,
        "ranker": "LightGBM LambdaRank R21 (300 rounds, 31 leaves, 29 features)",
        "local_cv5_estimate": "0.2152 (v3+r21_w1.0_p300, 2-fold subset)",
        "r21_model": "BGE-base fine-tuned on competition (conversation, GT) pairs",
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
