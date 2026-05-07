#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301,S603,S607
"""R26 blind submission: Q2+Q3 intent retrieval + LambdaRank.

Steps:
1. Extract intents for 80 blind cases (Haiku, cached)
2. Build Q2 BM25 + Q3 dense retrieval for blind
3. Train LambdaRank on dev with Q2+Q3 features
4. Build blind pools with Q2+Q3, rerank with LambdaRank
5. Assemble hybrid responses from R27b where possible
6. Validate and package submission
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als
from scripts.expS2_lambdarank_grouped import als_session_vector
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R21_MODEL = REPO / "cache" / "r21_production" / "model"
R21_TRACK_IDS = REPO / "cache" / "r21_production" / "track_ids.json"
R21_TRACK_EMBS = REPO / "cache" / "r21_production" / "track_embeddings.npy"
Q3_DEV_LISTS = REPO / "cache" / "r26" / "q3_dense_results.json"
INTENTS_DEV = REPO / "cache" / "r26" / "intents_dev.json"
INTENTS_BLIND = REPO / "cache" / "r26" / "intents_blind_a.json"
BLIND_OUT = REPO / "exp" / "inference" / "blind_a"

RRF_K = 20
POOL_K = 300
SOURCE_WEIGHTS = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
R26_WEIGHTS = {**SOURCE_WEIGHTS, "Q3": 0.5, "Q2": 0.25}

FEATURE_NAMES_R21 = FEATURE_NAMES_V2 + ["r21_rank_inv", "r21_presence"]
FEATURE_NAMES_Q2Q3 = FEATURE_NAMES_R21 + ["q3_rank_inv", "q3_presence", "q2_rank_inv", "q2_presence"]

EXTRACT_MODEL = "claude-haiku-4-5-20251001"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


# ---------------------------------------------------------------------------
# Step 1: Extract blind intents
# ---------------------------------------------------------------------------

def extract_blind_intents():
    """Extract intents for 80 blind cases."""
    from datasets import DownloadConfig, load_dataset

    if INTENTS_BLIND.exists():
        with open(INTENTS_BLIND) as f:
            existing = json.load(f)
        if len(existing) >= 80:
            print(f"  Blind intents already cached ({len(existing)} cases)")
            return existing

    print(f"{ts()} Extracting intents for blind cases...")
    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                      download_config=DownloadConfig(local_files_only=True))

    from scripts.expR26_intent_retrieval import build_conversation_text, extract_single_intent

    results = []
    for item in db:
        convs = sorted(item["conversations"], key=lambda x: x["turn_number"])
        last_user = ""
        for msg in reversed(convs):
            if msg["role"] == "user":
                last_user = msg["content"]
                break

        case = {
            "session_id": str(item["session_id"]),
            "turn_number": max(c["turn_number"] for c in convs),
            "user_query": last_user,
            "history": [{"role": c["role"], "content": c["content"]}
                       for c in convs[:-1] if c["role"] in ("user", "assistant", "music")],
        }
        conv_text = build_conversation_text(case)
        intent = extract_single_intent(conv_text)
        if intent is None:
            intent = extract_single_intent(conv_text)
        results.append({
            "session_id": case["session_id"],
            "turn_number": case["turn_number"],
            "user_query": last_user,
            "intent": intent,
        })

    INTENTS_BLIND.parent.mkdir(parents=True, exist_ok=True)
    with open(INTENTS_BLIND, "w") as f:
        json.dump(results, f, indent=1)
    valid = sum(1 for r in results if r.get("intent"))
    print(f"  {valid}/80 valid blind intents")
    return results


# ---------------------------------------------------------------------------
# Step 2: Build blind retrieval lists (Q2 BM25 + Q3 dense)
# ---------------------------------------------------------------------------

def build_blind_q2_lists(blind_intents):
    """Build Q2 artist-boost BM25 lists for blind."""
    from mcrs.retrieval_modules.bm25 import BM25Retriever

    queries = []
    for r in blind_intents:
        intent = r.get("intent")
        if intent:
            parts = []
            for artist in intent.get("positive_artists", []):
                parts.extend([artist] * 3)
            for genre in intent.get("genres", []):
                parts.extend([genre] * 2)
            parts.extend(intent.get("moods", []))
            if intent.get("era"):
                parts.append(intent["era"])
            if intent.get("context"):
                parts.append(intent["context"])
            parts.extend(intent.get("similarity_anchors", []))
            queries.append(" ".join(parts) if parts else intent.get("summary", r["user_query"]))
        else:
            queries.append(r["user_query"])

    bm25 = BM25Retriever(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
        corpus_types=["track_name", "artist_name", "album_name", "tag_list"],
        cache_dir=str(REPO / "cache"),
    )
    return bm25.batch_text_to_item_retrieval(queries, topk=300)


def build_blind_q3_lists(blind_intents):
    """Build Q3 intent dense lists for blind using R21 model."""
    cache_path = REPO / "cache" / "r26" / "q3_blind_results.json"
    if cache_path.exists():
        print("  Q3 blind lists cached")
        with open(cache_path) as f:
            return json.load(f)

    flat_queries = []
    case_indices = []
    for i, r in enumerate(blind_intents):
        intent = r.get("intent")
        if intent:
            variants = intent.get("query_variants", [])
            if not variants:
                variants = [intent.get("summary", r["user_query"])]
        else:
            variants = [r["user_query"]]
        for v in variants:
            flat_queries.append(v)
            case_indices.append(i)

    queries_path = REPO / "cache" / "r26" / "q3_blind_queries.json"
    with open(queries_path, "w") as f:
        json.dump({"queries": flat_queries, "case_indices": case_indices}, f)

    print(f"{ts()}   Encoding {len(flat_queries)} blind query variants...")
    script = f'''
import json, numpy as np, time
from pathlib import Path
from sentence_transformers import SentenceTransformer

REPO = Path("{REPO}")
with open(REPO / "cache/r26/q3_blind_queries.json") as f:
    data = json.load(f)
queries = data["queries"]
case_indices = data["case_indices"]

track_ids = json.loads((REPO / "cache/r21_production/track_ids.json").read_text())
track_embs = np.load(REPO / "cache/r21_production/track_embeddings.npy")

print("Loading R21 model...", flush=True)
model = SentenceTransformer(str(REPO / "cache/r21_production/model"))

print(f"Encoding {{len(queries)}} queries...", flush=True)
query_embs = model.encode(queries, batch_size=64, normalize_embeddings=True, show_progress_bar=True)
query_embs = query_embs.astype(np.float32)

n_cases = max(case_indices) + 1
topk = 300
results = [[] for _ in range(n_cases)]

for qi in range(len(queries)):
    ci = case_indices[qi]
    scores = track_embs @ query_embs[qi]
    top_idx = np.argpartition(-scores, topk)[:topk]
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    for j in top_idx:
        results[ci].append((track_ids[j], float(scores[j])))

merged = []
for ci in range(n_cases):
    track_scores = {{}}
    for tid, sc in results[ci]:
        if tid not in track_scores or sc > track_scores[tid]:
            track_scores[tid] = sc
    ranked = sorted(track_scores, key=track_scores.__getitem__, reverse=True)[:topk]
    merged.append(ranked)

out_path = REPO / "cache/r26/q3_blind_results.json"
with open(out_path, "w") as f:
    json.dump(merged, f)
print(f"Saved {{len(merged)}} blind Q3 lists", flush=True)
'''
    result = subprocess.run(
        ["uv", "run", "python", "-c", script],
        capture_output=True, text=True, timeout=600,
        cwd=str(REPO),
    )
    print(result.stdout, flush=True)
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[:2000]}", flush=True)
        raise RuntimeError("Q3 blind dense retrieval failed")

    with open(cache_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 3: Train LambdaRank with Q2+Q3 features on dev
# ---------------------------------------------------------------------------

def train_lambdarank_r26(als_factors, als_track_ids, als_track_to_idx, track_pop,
                          q3_dev, q2_dev):
    """Train LambdaRank on full dev data with Q2+Q3 features."""
    print(f"{ts()} Training LambdaRank with Q2+Q3 features on full dev...")

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)

    als_source = []
    als_vecs = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
            als_vecs.append(sv)
        else:
            als_source.append([])
            als_vecs.append(None)

    with open(R21_OOF) as f:
        r21_source = json.load(f)

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    ttl = payload["track_title_toks"]
    tat = payload["track_artist_toks"]
    tmt = payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    n_feat = len(FEATURE_NAMES_Q2Q3)
    X = np.zeros((n, POOL_K, n_feat), dtype=np.float64)
    gt_idx = np.full(n, -1, dtype=np.int64)
    sizes = np.zeros(n, dtype=np.int64)

    for i, c in enumerate(cases):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R21": r21_source[i], "Q3": q3_dev[i], "Q2": q2_dev[i],
        }
        pool = weighted_rrf(src_lists, R26_WEIGHTS, topk=POOL_K, k=RRF_K)
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
        prior_list = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(played))]
        sv = als_vecs[i]
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_source[i][:300])}
        q3_rank_map = {tid: r + 1 for r, tid in enumerate(q3_dev[i][:300])}
        q2_rank_map = {tid: r + 1 for r, tid in enumerate(q2_dev[i][:300])}

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
            for wd, pa, pt in prior_list:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
            row[20] = sum(1 for sname in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sname, {}))
            if sv is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(sv, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            row[28] = 1.0 if tid in r21_rank_map else 0.0
            row[29] = 1.0 / q3_rank_map[tid] if tid in q3_rank_map else 0.0
            row[30] = 1.0 if tid in q3_rank_map else 0.0
            row[31] = 1.0 / q2_rank_map[tid] if tid in q2_rank_map else 0.0
            row[32] = 1.0 if tid in q2_rank_map else 0.0

    pool_hit = float(np.mean(gt_idx >= 0))
    print(f"  Dev pool_hit@{POOL_K}: {pool_hit:.4f}")

    X_flat = X.reshape(-1, n_feat)
    labels = np.zeros(n * POOL_K, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels[i * POOL_K + gt_idx[i]] = 1.0
    group_sizes = [int(sizes[i]) for i in range(n)]

    ds = lgb.Dataset(X_flat[:sum(group_sizes)], label=labels[:sum(group_sizes)],
                     group=group_sizes, feature_name=list(FEATURE_NAMES_Q2Q3))

    params = {
        "objective": "lambdarank", "metric": "ndcg",
        "eval_at": [20], "num_leaves": 31, "learning_rate": 0.05,
        "min_data_in_leaf": 10, "verbose": -1, "seed": 0,
    }
    model = lgb.train(params, ds, num_boost_round=300)
    print(f"  LambdaRank trained ({model.num_trees()} trees)")
    return model, payload


# ---------------------------------------------------------------------------
# Step 4: Blind inference
# ---------------------------------------------------------------------------

def blind_inference(lr_model, blind_intents, q2_blind, q3_blind,
                    als_factors, als_track_ids, als_track_to_idx, track_pop, dev_payload):
    """Run LambdaRank inference on blind cases."""
    from datasets import DownloadConfig, load_dataset
    from run_inference_blind_r21 import (
        als_retrieve, parse_last_turn,
    )
    from run_inference_blind_f1 import CFBPRMaxRecent, load_cfbpr_index
    from offline_retrieval_sweep import CachedBM25
    from run_inference_blind_r3_det import APrimeMaxRecent
    from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever

    print(f"{ts()} Loading blind infrastructure...")
    bm25 = CachedBM25()
    track_sim = TrackSimilarityRetriever(cache_dir=str(REPO / "cache"))
    a_prime = APrimeMaxRecent(track_sim, recent_k=3)
    cf_ids, cf_vecs, cf_idx = load_cfbpr_index()
    cfbpr = CFBPRMaxRecent(cf_ids, cf_vecs, cf_idx, recent_k=3)

    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                      download_config=DownloadConfig(local_files_only=True))

    ta = dev_payload["track_artist"]
    tt = dev_payload["track_tags"]
    ttl = dev_payload["track_title_toks"]
    tat = dev_payload["track_artist_toks"]
    tmt = dev_payload["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1

    print(f"{ts()} Building blind R21 retrieval lists...")
    r21_model_obj = None
    r21_track_ids_list = json.loads(Path(R21_TRACK_IDS).read_text())
    r21_track_embs_arr = np.load(R21_TRACK_EMBS)

    from sentence_transformers import SentenceTransformer
    r21_model_obj = SentenceTransformer(str(R21_MODEL))

    n_feat = len(FEATURE_NAMES_Q2Q3)
    results = []

    for idx, item in enumerate(db):
        sid = str(item["session_id"])
        turn_num, user_query, history, music_turns = parse_last_turn(item)

        # Build query for R21
        user_parts = [str(h["content"]) for h in history if h["role"] == "user"]
        user_parts.append(user_query)
        r21_query = " ".join(user_parts[-3:])

        # R21 dense retrieval
        q_emb = r21_model_obj.encode([r21_query], normalize_embeddings=True).astype(np.float32)[0]
        r21_scores = r21_track_embs_arr @ q_emb
        played_set = set(music_turns)
        for ti, tid in enumerate(r21_track_ids_list):
            if tid in played_set:
                r21_scores[ti] = -np.inf
        r21_top = np.argpartition(-r21_scores, 300)[:300]
        r21_top = r21_top[np.argsort(-r21_scores[r21_top])]
        r21_list = [r21_track_ids_list[j] for j in r21_top]

        src_a = a_prime.topn(music_turns, topn=100) if music_turns else []
        src_b = bm25.retrieve(user_query, topk=100)
        src_c = bm25.retrieve(user_query, topk=100)
        anchor = music_turns[-1] if music_turns else None
        src_d = track_sim.track_id_to_neighbors(anchor, topk=100) if anchor else []
        src_f = cfbpr.topn(music_turns, topn=100) if music_turns else []
        als_tracks, als_vec = als_retrieve(music_turns, als_track_to_idx, als_factors, als_track_ids)

        src_lists = {
            "A": src_a, "B": src_b, "C": src_c, "D": src_d,
            "F": src_f, "ALS": als_tracks, "R21": r21_list,
            "Q3": q3_blind[idx], "Q2": q2_blind[idx],
        }
        pool = weighted_rrf(src_lists, R26_WEIGHTS, topk=POOL_K, k=RRF_K)

        src_rank = {}
        for sname, slist in src_lists.items():
            src_rank[sname] = {tid: r + 1 for r, tid in enumerate(slist)}

        user_msgs = user_parts
        n_hist = len(music_turns)
        now_tok = tokens(user_msgs[-1]) if user_msgs else set()
        all_tok = tokens(" ".join(user_msgs))
        l_artist = ta.get(music_turns[-1], "") if music_turns else ""
        l_tags = tt.get(music_turns[-1], set()) if music_turns else set()
        prior_list = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
                 for j, t in enumerate(reversed(music_turns))]
        pool_artists = [ta.get(tid, "") for tid in pool[:POOL_K]]
        artist_counts = Counter(a for a in pool_artists if a)
        r21_rank_map = {tid: r + 1 for r, tid in enumerate(r21_list[:300])}
        q3_rank_map = {tid: r + 1 for r, tid in enumerate(q3_blind[idx][:300])}
        q2_rank_map = {tid: r + 1 for r, tid in enumerate(q2_blind[idx][:300])}

        X = np.zeros((POOL_K, n_feat), dtype=np.float64)
        for rank, tid in enumerate(pool[:POOL_K], start=1):
            ca = ta.get(tid, "")
            ct = tt.get(tid, set())
            row = X[rank - 1]
            row[0] = 1.0 / rank
            row[1] = 1.0 if ca and ca == l_artist else 0.0
            if ct or l_tags:
                row[2] = len(ct & l_tags) / len(ct | l_tags)
            row[3] = float(len(tat.get(tid, set()) & now_tok))
            row[4] = float(len(ttl.get(tid, set()) & now_tok))
            row[5] = float(len(tmt.get(tid, set()) & all_tok))
            row[6] = 1.0 if tid in played_set else 0.0
            rec = 0.0
            for wd, pa, pt in prior_list:
                am = 1.0 if ca and ca == pa else 0.0
                tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
                rec += wd * (am + tm)
            row[7] = rec
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                sr = src_rank[sname].get(tid)
                row[8 + fi] = 1.0 / sr if sr else 0.0
            for fi, sname in enumerate(["A", "B", "C", "D", "F", "ALS"]):
                row[14 + fi] = 1.0 if tid in src_rank[sname] else 0.0
            row[20] = sum(1 for sname in ["A", "B", "C", "D", "F", "ALS"] if tid in src_rank.get(sname, {}))
            if als_vec is not None:
                aidx = als_track_to_idx.get(tid)
                if aidx is not None:
                    row[21] = float(np.dot(als_vec, als_factors[aidx]))
            row[22] = float(n_hist)
            row[23] = track_pop.get(tid, 0) / max_pop
            row[24] = artist_counts.get(ca, 0) / max(len(pool), 1) if ca else 0
            row[25] = float(artist_counts.get(ca, 0)) if ca else 0
            row[26] = row[20]
            row[27] = 1.0 / r21_rank_map[tid] if tid in r21_rank_map else 0.0
            row[28] = 1.0 if tid in r21_rank_map else 0.0
            row[29] = 1.0 / q3_rank_map[tid] if tid in q3_rank_map else 0.0
            row[30] = 1.0 if tid in q3_rank_map else 0.0
            row[31] = 1.0 / q2_rank_map[tid] if tid in q2_rank_map else 0.0
            row[32] = 1.0 if tid in q2_rank_map else 0.0

        scores = lr_model.predict(X[:len(pool)])
        ranked_idx = np.argsort(-scores)
        top20 = [pool[j] for j in ranked_idx[:20]]

        results.append({
            "session_id": sid,
            "turn_number": turn_num,
            "predicted_track_ids": top20,
            "predicted_response": "",
        })
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/80 blind rows processed", flush=True)

    print(f"  {len(results)} blind predictions built")
    return results


# ---------------------------------------------------------------------------
# Step 5: Hybrid responses + validation
# ---------------------------------------------------------------------------

def assemble_responses(results):
    """Assemble hybrid responses from R27b where top-1 overlaps."""
    r27b_path = BLIND_OUT / "r27b_agent_audit_submission.json"
    r25_path = BLIND_OUT / "r25_lexdiv_v2.json"
    r21_path = BLIND_OUT / "lr_r21_v1_hybrid.json"

    # Try R27b first (best composite), then R25, then R21
    source_paths = [(r27b_path, "R27b"), (r25_path, "R25"), (r21_path, "R21")]
    prev_by_sid: dict[str, dict] = {}
    for path, name in source_paths:
        if path.exists():
            with open(path) as f:
                for r in json.load(f):
                    sid = r["session_id"]
                    if sid not in prev_by_sid:
                        prev_by_sid[sid] = {"response": r["predicted_response"],
                                           "tracks": r["predicted_track_ids"],
                                           "source": name}

    reused = 0
    need_gen = []
    for r in results:
        sid = r["session_id"]
        prev = prev_by_sid.get(sid)
        if prev and prev["tracks"][0] in set(r["predicted_track_ids"]):
            r["predicted_response"] = prev["response"]
            reused += 1
        else:
            need_gen.append(r)

    print(f"  Responses: {reused} reused, {len(need_gen)} need generation")

    if need_gen:
        from mcrs.db_item.music_catalog import MusicCatalogDB
        from mcrs.lm_modules.claude import ClaudeModule
        from run_inference_blind_r3_det import build_session_memory_for_response, parse_last_turn
        from datasets import DownloadConfig, load_dataset

        item_db = MusicCatalogDB(dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                                  split_types=["all_tracks"])
        prompts_dir = REPO / "mcrs" / "system_prompts"
        sys_prompt = (prompts_dir / "roleplay.txt").read_text() + "\n" + (prompts_dir / "response_generation.txt").read_text()
        haiku = ClaudeModule(model="claude-haiku-4-5-20251001")
        db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                          download_config=DownloadConfig(local_files_only=True))
        blind_by_sid = {str(item["session_id"]): item for item in db}

        for r in need_gen:
            item = blind_by_sid[r["session_id"]]
            turn_num, user_query, history, music_turns = parse_last_turn(item)
            top_id = r["predicted_track_ids"][0]
            try:
                top_item = item_db.id_to_metadata(top_id)
            except KeyError:
                top_item = f"track_id: {top_id}"
            session_memory = build_session_memory_for_response(history, user_query, item_db)
            response = haiku.response_generation(sys_prompt, session_memory, top_item)
            r["predicted_response"] = (response or "").lstrip(",").lstrip()

    return results


def validate_and_package(results):
    """Validate and create submission zip."""
    if len(results) != 80:
        raise ValueError(f"Expected 80 rows, got {len(results)}")

    sids = set()
    for r in results:
        sid = r["session_id"]
        if len(r["predicted_track_ids"]) != 20:
            raise ValueError(f"Row {sid}: {len(r['predicted_track_ids'])} tracks")
        if len(set(r["predicted_track_ids"])) != 20:
            raise ValueError(f"Row {sid}: duplicate tracks")
        if not r["predicted_response"].strip():
            raise ValueError(f"Row {sid}: empty response")
        if r["predicted_response"].startswith(","):
            raise ValueError(f"Row {sid}: leading comma")
        sids.add(sid)
    if len(sids) != 80:
        raise ValueError(f"Expected 80 unique sessions, got {len(sids)}")

    out_json = BLIND_OUT / "r26_q2q3_submission.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    out_zip = BLIND_OUT / "r26_q2q3_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")

    print(f"\n  SUBMISSION: {out_zip}")
    print("  80 rows, 20 tracks each, all unique, no empty responses")
    return out_zip


# ---------------------------------------------------------------------------
# Comparison with previous submissions
# ---------------------------------------------------------------------------

def compare_with_previous(results):
    """Compare R26 tracks with R21/R27b."""
    prev_paths = [
        (BLIND_OUT / "lr_r21_v1_hybrid.json", "R21"),
        (BLIND_OUT / "r27b_agent_audit_submission.json", "R27b"),
    ]
    for path, name in prev_paths:
        if not path.exists():
            continue
        with open(path) as f:
            prev = {r["session_id"]: r for r in json.load(f)}

        top1_changed = 0
        top20_overlaps = []
        for r in results:
            p = prev.get(r["session_id"])
            if not p:
                continue
            if r["predicted_track_ids"][0] != p["predicted_track_ids"][0]:
                top1_changed += 1
            overlap = len(set(r["predicted_track_ids"]) & set(p["predicted_track_ids"]))
            top20_overlaps.append(overlap)

        avg_overlap = np.mean(top20_overlaps) if top20_overlaps else 0
        print(f"\n  vs {name}:")
        print(f"    top-1 changed: {top1_changed}/80")
        print(f"    avg top-20 overlap: {avg_overlap:.1f}/20")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"{ts()} R26 Blind Submission Pipeline (Q2+Q3)")
    print("=" * 70)

    # Step 1: Extract blind intents
    print(f"\n{ts()} Step 1: Blind intent extraction")
    blind_intents = extract_blind_intents()

    # Step 2: Build blind Q2+Q3 lists
    print(f"\n{ts()} Step 2: Blind retrieval lists")
    print("  Building Q2 BM25...")
    q2_blind = build_blind_q2_lists(blind_intents)
    print(f"  Q2: {len(q2_blind)} lists")
    print("  Building Q3 dense...")
    q3_blind = build_blind_q3_lists(blind_intents)
    print(f"  Q3: {len(q3_blind)} lists")

    # Load dev Q2+Q3 for training
    print("  Loading dev Q3 lists...")
    with open(Q3_DEV_LISTS) as f:
        q3_dev = json.load(f)

    print("  Building dev Q2 lists...")
    with open(R12_CACHE, "rb") as f:
        dev_cases = pickle.load(f)["cases"]
    with open(INTENTS_DEV) as f:
        dev_intents = json.load(f)
    dev_intent_map = {(r["session_id"], r["turn_number"]): r.get("intent") for r in dev_intents}
    q2_dev = build_dev_q2(dev_cases, dev_intent_map)

    # Step 3: Train LambdaRank
    print(f"\n{ts()} Step 3: ALS + LambdaRank training")
    als_factors, als_track_ids, als_track_to_idx = build_als()
    track_pop = build_popularity_stats()
    lr_model, dev_payload = train_lambdarank_r26(
        als_factors, als_track_ids, als_track_to_idx, track_pop, q3_dev, q2_dev)

    # Step 4: Blind inference
    print(f"\n{ts()} Step 4: Blind inference (80 rows)")
    results = blind_inference(lr_model, blind_intents, q2_blind, q3_blind,
                              als_factors, als_track_ids, als_track_to_idx,
                              track_pop, dev_payload)

    # Step 5: Hybrid responses
    print(f"\n{ts()} Step 5: Response assembly")
    results = assemble_responses(results)

    # Step 6: Validate + compare
    print(f"\n{ts()} Step 6: Validation")
    compare_with_previous(results)
    out_zip = validate_and_package(results)

    print(f"\n{ts()} DONE: {out_zip}")


def build_dev_q2(cases, intent_map):
    """Build Q2 lists for dev using cached intents."""
    from mcrs.retrieval_modules.bm25 import BM25Retriever

    queries = []
    for case in cases:
        intent = intent_map.get((case["session_id"], case["turn_number"]))
        if intent:
            parts = []
            for artist in intent.get("positive_artists", []):
                parts.extend([artist] * 3)
            for genre in intent.get("genres", []):
                parts.extend([genre] * 2)
            parts.extend(intent.get("moods", []))
            if intent.get("era"):
                parts.append(intent["era"])
            if intent.get("context"):
                parts.append(intent["context"])
            parts.extend(intent.get("similarity_anchors", []))
            queries.append(" ".join(parts) if parts else intent.get("summary", case["user_query"]))
        else:
            queries.append(case["user_query"])

    bm25 = BM25Retriever(
        dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split_types=["all_tracks"],
        corpus_types=["track_name", "artist_name", "album_name", "tag_list"],
        cache_dir=str(REPO / "cache"),
    )
    return bm25.batch_text_to_item_retrieval(queries, topk=300)


if __name__ == "__main__":
    main()
