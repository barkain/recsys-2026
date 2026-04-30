#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R15: Controlled fusion of complementary weak sources Q + R14 + G.

Stage 2: LambdaRank evaluation with extended features.
Union diagnostic showed 158 unique unreachable recoveries (gate: ≥150 PASS).

Pool strategies:
  1. V3 baseline (ABCDF+ALS, pool_k=200)
  2. V3 + Q/R14/G via RRF (pool_k=200)
  3. V3 + Q/R14/G via RRF (pool_k=300)
  4. Reserved-tail: V3 top-200 + dedupe top-50 from Q/R14/G
  5. Two-stage: V3 top-200 + Q/R14/G candidates appearing in ≥2 weak sources
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import bm25s
import lightgbm as lgb
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank import build_als, FEATURE_NAMES_LR
from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds
from scripts.expS2_lr_v2 import FEATURE_NAMES_V2, build_popularity_stats
from scripts.tune_postrank_v23 import tokens

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20

FEATURE_NAMES_V3 = FEATURE_NAMES_V2 + [
    "rank_Q",          # reciprocal rank in Q source
    "rank_R14",        # reciprocal rank in R14 source
    "rank_G",          # reciprocal rank in G source
    "in_Q",            # present in Q
    "in_R14",          # present in R14
    "in_G",            # present in G
    "weak_source_count",    # how many weak sources have this
    "is_cold",              # log(1 + pop) inverted
    "content_only",         # in Q/R14/G but not ABCDF+ALS
    "cold_content_match",   # content_only * is_cold
]


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


MOOD_TERMS = {
    "chill": "chillout lounge ambient relaxing mellow",
    "relaxing": "chillout lounge ambient mellow calm",
    "calm": "ambient mellow soft acoustic peaceful",
    "upbeat": "energetic dance party uplifting happy",
    "energetic": "dance party power upbeat workout",
    "sad": "melancholy emotional ballad heartbreak",
    "happy": "upbeat feel-good joyful positive cheerful",
    "romantic": "love ballad smooth soft intimate",
    "dark": "gothic noir brooding intense atmospheric",
    "intense": "heavy aggressive powerful epic driving",
    "dreamy": "ethereal ambient shoegaze atmospheric",
    "funky": "funk groove bass danceable soul",
    "workout": "energetic power intense driving motivational",
    "party": "dance upbeat energetic club electronic",
}
GENRE_TERMS = {
    "rock": "rock alternative indie guitar",
    "pop": "pop mainstream catchy",
    "hip hop": "hip-hop rap hiphop", "hip-hop": "hip-hop rap hiphop",
    "rap": "hip-hop rap hiphop",
    "jazz": "jazz swing bebop smooth",
    "electronic": "electronic edm synth dance",
    "r&b": "rnb r&b soul rhythm", "soul": "soul rnb motown rhythm",
    "country": "country americana folk",
    "folk": "folk acoustic singer-songwriter",
    "metal": "metal heavy thrash",
    "indie": "indie alternative lo-fi",
    "grunge": "grunge alternative 90s rock",
}


def expand_query(q):
    ql = q.lower()
    exps = []
    for term, exp in MOOD_TERMS.items():
        if term in ql:
            exps.append(exp)
    for term, exp in GENRE_TERMS.items():
        if term in ql:
            exps.append(exp)
    return q + " " + " ".join(exps) if exps else q


def build_r14_source(cases, ta, tt, r14_meta, r14_track_ids):
    """Build R14 base__q_kitchen_sink retrieval."""
    corpus_texts = []
    for tid in r14_track_ids:
        m = r14_meta[tid]
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

    queries = []
    for c in cases:
        q = expand_query(c["user_query"])
        played = c["music_turns"]
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
    results, _ = model.retrieve(toks, k=300)
    out = []
    for i in range(len(cases)):
        out.append([r14_track_ids[int(idx)] for idx in results[i] if int(idx) >= 0])
    return out


def build_q_source(cases, emb_context, track_ids_q, track_vecs):
    """Build Q_context retrieval from cached embeddings."""
    out = []
    for i in range(len(cases)):
        scores = track_vecs @ emb_context[i]
        played_set = set(cases[i]["music_turns"])
        for idx, tid in enumerate(track_ids_q):
            if tid in played_set:
                scores[idx] = -np.inf
        top_idx = np.argpartition(-scores, 300)[:300]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        out.append([track_ids_q[j] for j in top_idx])
    return out


def build_pool_strategy(strategy, i, payload, als_source, q_source, r14_source,
                        g_source, base_weights, weak_weights, pool_k):
    """Build candidate pool for a given strategy."""
    base_lists = {
        "A": payload["src_a"][i], "B": payload["src_b"][i],
        "C": payload["src_c"][i], "D": payload["src_d"][i],
        "F": payload["src_f"][i], "ALS": als_source[i],
    }

    if strategy == "v3_baseline":
        pool = weighted_rrf(base_lists, base_weights, topk=pool_k, k=RRF_K)
        return pool

    elif strategy == "v3_plus_weak_rrf":
        all_lists = dict(base_lists)
        all_lists["Q"] = q_source[i]
        all_lists["R14"] = r14_source[i]
        all_lists["G"] = g_source[i]
        all_weights = dict(base_weights)
        all_weights.update(weak_weights)
        pool = weighted_rrf(all_lists, all_weights, topk=pool_k, k=RRF_K)
        return pool

    elif strategy == "reserved_tail":
        v3_pool = weighted_rrf(base_lists, base_weights, topk=200, k=RRF_K)
        v3_set = set(v3_pool)
        weak_candidates = []
        for src in [q_source[i], r14_source[i], g_source[i]]:
            for tid in src[:200]:
                if tid not in v3_set and tid not in {t for t in weak_candidates}:
                    weak_candidates.append(tid)
                if len(weak_candidates) >= 50:
                    break
            if len(weak_candidates) >= 50:
                break
        pool = list(v3_pool) + weak_candidates[:50]
        return pool

    elif strategy == "two_stage":
        v3_pool = weighted_rrf(base_lists, base_weights, topk=200, k=RRF_K)
        v3_set = set(v3_pool)
        q_set = set(q_source[i][:200])
        r14_set = set(r14_source[i][:200])
        g_set = set((g_source[i] or [])[:200])
        extra = []
        seen = set(v3_set)
        for tid in (q_set | r14_set | g_set) - v3_set:
            count = (tid in q_set) + (tid in r14_set) + (tid in g_set)
            if count >= 2 and tid not in seen:
                extra.append(tid)
                seen.add(tid)
        pool = list(v3_pool) + extra[:100]
        return pool

    raise ValueError(f"Unknown strategy: {strategy}")


def build_features_v3(pool, case, payload_i, src_rank_base, als_vecs_i,
                      als_factors, als_track_to_idx, track_pop, max_pop,
                      q_source_i, r14_source_i, g_source_i, ta, tt, n_feat):
    """Build V3 feature vector for one case."""
    user_msgs = ([str(r["content"]) for r in case["history"] if r["role"] == "user"]
                 + [case["user_query"]])
    played = case["music_turns"]
    n_hist = len(played)
    now_tok = tokens(user_msgs[-1]) if user_msgs else set()
    all_tok = tokens(" ".join(user_msgs))
    played_set = set(played)
    l_artist = ta.get(played[-1], "") if played else ""
    l_tags = tt.get(played[-1], set()) if played else set()
    prior = [(1.0 / (j + 1), ta.get(t, ""), tt.get(t, set()))
             for j, t in enumerate(reversed(played))]
    sv = als_vecs_i

    ttl = payload_i["ttl"]
    tat = payload_i["tat"]
    tmt = payload_i["tmt"]

    pool_artists = [ta.get(tid, "") for tid in pool]
    artist_counts = Counter(a for a in pool_artists if a)

    q_rank = {tid: r + 1 for r, tid in enumerate(q_source_i[:300])}
    r14_rank = {tid: r + 1 for r, tid in enumerate(r14_source_i[:300])}
    g_rank = {tid: r + 1 for r, tid in enumerate((g_source_i or [])[:300])}

    base_set = set()
    for sname in ["A", "B", "C", "D", "F", "ALS"]:
        base_set.update(src_rank_base[sname].keys())

    X = np.zeros((len(pool), n_feat), dtype=np.float64)
    for rank, tid in enumerate(pool, start=1):
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
        row[20] = sum(1 for sname in src_rank_base if tid in src_rank_base[sname])
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

        # V3 weak source features
        row[27] = 1.0 / q_rank[tid] if tid in q_rank else 0.0
        row[28] = 1.0 / r14_rank[tid] if tid in r14_rank else 0.0
        row[29] = 1.0 / g_rank[tid] if tid in g_rank else 0.0
        row[30] = 1.0 if tid in q_rank else 0.0
        row[31] = 1.0 if tid in r14_rank else 0.0
        row[32] = 1.0 if tid in g_rank else 0.0
        row[33] = (tid in q_rank) + (tid in r14_rank) + (tid in g_rank)
        row[34] = 1.0 / np.log2(pop + 2) if pop >= 0 else 1.0  # coldness
        content_only = tid not in base_set
        row[35] = 1.0 if content_only else 0.0
        row[36] = row[35] * row[34]  # cold_content_match

    return X


def run_cv(X_all, gt_idx, sizes, cases, sessions, n, pool_k_max, n_feat, seeds=(0, 1, 2)):
    """Run grouped-session CV5 with LambdaRank."""
    lgb_params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [20],
        "num_leaves": 31, "learning_rate": 0.05, "min_child_samples": 20,
        "subsample": 0.8, "colsample_bytree": 0.8, "verbose": -1,
        "random_state": 42, "force_col_wise": True,
    }

    X_flat = X_all.reshape(-1, n_feat)
    labels = np.zeros(n * pool_k_max, dtype=np.float32)
    for i in range(n):
        if gt_idx[i] >= 0:
            labels[i * pool_k_max + gt_idx[i]] = 1.0

    cv5_seeds, lt_seeds = [], []
    slice_results = defaultdict(lambda: defaultdict(list))

    for seed in seeds:
        folds = grouped_session_folds(sessions, seed)
        fold_ndcgs, fold_lt = [], []

        for fold in folds:
            held = set(fold.tolist())
            train_cases = [j for j in range(n) if j not in held]
            val_cases = fold.tolist()
            train_flat = [j * pool_k_max + k for j in train_cases for k in range(int(sizes[j]))]
            val_flat = [j * pool_k_max + k for j in val_cases for k in range(int(sizes[j]))]
            g_train = np.array([int(sizes[j]) for j in train_cases], dtype=np.int32)
            g_val = np.array([int(sizes[j]) for j in val_cases], dtype=np.int32)
            dtrain = lgb.Dataset(X_flat[train_flat], labels[train_flat],
                                 group=g_train, feature_name=FEATURE_NAMES_V3,
                                 free_raw_data=False)
            dval = lgb.Dataset(X_flat[val_flat], labels[val_flat],
                               group=g_val, reference=dtrain, free_raw_data=False)
            model = lgb.train(lgb_params, dtrain, num_boost_round=300,
                              valid_sets=[dval],
                              callbacks=[lgb.early_stopping(30, verbose=False)])
            val_scores = model.predict(X_flat[val_flat])
            offset = 0
            case_ndcgs = []
            for j in val_cases:
                sz = int(sizes[j])
                if sz == 0:
                    case_ndcgs.append(0.0)
                    continue
                sc = val_scores[offset:offset + sz]
                gt = gt_idx[j]
                if gt >= 0:
                    gt_score = sc[gt]
                    rank0 = int(np.sum(sc > gt_score) +
                                np.sum((sc == gt_score) & (np.arange(sz) < gt)))
                    ndcg = 1.0 / np.log2(rank0 + 2) if rank0 < 20 else 0.0
                else:
                    ndcg = 0.0
                case_ndcgs.append(ndcg)

                # Sliced metrics
                played = cases[j]["music_turns"]
                n_hist = len(played)
                is_last_turn = n_hist == 7
                if is_last_turn:
                    slice_results["last_turn"][seed].append(ndcg)
                if n_hist == 0:
                    slice_results["hist_0"][seed].append(ndcg)

                gt_tid = cases[j]["gt"]
                gt_pop = track_pop_global.get(gt_tid, 0)
                if gt_pop == 0:
                    slice_results["pop_0"][seed].append(ndcg)

                if played:
                    la = ta_global.get(played[-1], "")
                    ga = ta_global.get(gt_tid, "")
                    if isinstance(la, list): la = la[0] if la else ""
                    if isinstance(ga, list): ga = ga[0] if ga else ""
                    if ga and la:
                        if ga == la:
                            slice_results["same_artist"][seed].append(ndcg)
                        else:
                            slice_results["diff_artist"][seed].append(ndcg)

                offset += sz
            fold_ndcgs.append(float(np.mean(case_ndcgs)))
        cv5_seeds.append(float(np.mean(fold_ndcgs)))
        lt_vals = slice_results["last_turn"].get(seed, [])
        if lt_vals:
            lt_seeds.append(float(np.mean(lt_vals)))

    cv5 = float(np.mean(cv5_seeds))
    lt_ndcg = float(np.mean(lt_seeds)) if lt_seeds else 0.0

    slices = {}
    for sl_name, seed_data in slice_results.items():
        all_vals = []
        for sv in seed_data.values():
            all_vals.extend(sv)
        slices[sl_name] = float(np.mean(all_vals)) if all_vals else 0.0

    return cv5, lt_ndcg, slices


# Globals for slice computation inside run_cv
track_pop_global = {}
ta_global = {}


def main():
    global track_pop_global, ta_global
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1
    track_pop_global = track_pop
    ta_global = ta

    # Build Q source
    print(f"{ts()} Loading Q_context...", flush=True)
    emb_context = np.load(REPO_ROOT / "cache" / "r13_query_emb" / "emb_context.npy")
    track_ids_q = json.load(open(REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "track_ids.json"))
    track_vecs = np.load(REPO_ROOT / "cache" / "track_sim" / "metadata-qwen3_embedding_0.6b" / "vectors.npy")
    print(f"{ts()} Building Q source (8000 queries)...", flush=True)
    q_source = build_q_source(cases, emb_context, track_ids_q, track_vecs)
    del emb_context, track_vecs

    # Build R14 source
    print(f"{ts()} Building R14 source...", flush=True)
    from datasets import Dataset, concatenate_datasets
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
    r14_track_ids = [str(tid) for tid in cols["track_id"]]
    r14_meta = {r14_track_ids[i]: {k: cols[k][i] for k in cols} for i in range(len(r14_track_ids))}
    r14_source = build_r14_source(cases, ta, tt, r14_meta, r14_track_ids)
    del r14_meta, combined, cols

    # G source
    g_source = payload["src_g"]

    # Build ALS
    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source, als_vecs = [], []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        als_vecs.append(sv)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    base_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}

    # Pre-compute per-case data lookups
    print(f"{ts()} Pre-computing per-case data...", flush=True)
    case_data = []
    for i in range(n):
        src_rank_base = {}
        for sname, key in [("A", "src_a"), ("B", "src_b"), ("C", "src_c"),
                           ("D", "src_d"), ("F", "src_f")]:
            src_rank_base[sname] = {tid: r + 1 for r, tid in enumerate(payload[key][i])}
        src_rank_base["ALS"] = {tid: r + 1 for r, tid in enumerate(als_source[i])}
        case_data.append({
            "src_rank_base": src_rank_base,
            "ttl": payload["track_title_toks"],
            "tat": payload["track_artist_toks"],
            "tmt": payload["track_meta_toks"],
        })

    n_feat = len(FEATURE_NAMES_V3)

    # Configs to test
    configs = [
        {"name": "v3_baseline_200", "strategy": "v3_baseline", "pool_k": 200,
         "weak_weights": {}},
        {"name": "v3+weak_w0.1_p200", "strategy": "v3_plus_weak_rrf", "pool_k": 200,
         "weak_weights": {"Q": 0.1, "R14": 0.1, "G": 0.1}},
        {"name": "v3+weak_w0.25_p200", "strategy": "v3_plus_weak_rrf", "pool_k": 200,
         "weak_weights": {"Q": 0.25, "R14": 0.25, "G": 0.25}},
        {"name": "v3+weak_w0.5_p200", "strategy": "v3_plus_weak_rrf", "pool_k": 200,
         "weak_weights": {"Q": 0.5, "R14": 0.5, "G": 0.5}},
        {"name": "v3+weak_w0.25_p300", "strategy": "v3_plus_weak_rrf", "pool_k": 300,
         "weak_weights": {"Q": 0.25, "R14": 0.25, "G": 0.25}},
        {"name": "reserved_tail_250", "strategy": "reserved_tail", "pool_k": 250,
         "weak_weights": {}},
        {"name": "two_stage_multi", "strategy": "two_stage", "pool_k": 300,
         "weak_weights": {}},
    ]

    results = {}

    for cfg in configs:
        config_name = cfg["name"]
        strategy = cfg["strategy"]
        pool_k = cfg["pool_k"]
        weak_w = cfg["weak_weights"]

        print(f"\n{ts()} === {config_name} ===", flush=True)

        # Build pools and features
        X_all = np.zeros((n, pool_k, n_feat), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)

        for i in range(n):
            pool = build_pool_strategy(
                strategy, i, payload, als_source, q_source, r14_source,
                g_source, base_weights, weak_w, pool_k)
            pool = pool[:pool_k]
            sizes[i] = len(pool)
            if cases[i]["gt"] in pool:
                gt_idx[i] = pool.index(cases[i]["gt"])

            X_case = build_features_v3(
                pool, cases[i], case_data[i], case_data[i]["src_rank_base"],
                als_vecs[i], als_factors, als_track_to_idx, track_pop, max_pop,
                q_source[i], r14_source[i], g_source[i], ta, tt, n_feat)
            X_all[i, :len(pool)] = X_case

        pool_hit = float(np.mean(gt_idx >= 0))
        print(f"  pool_hit@{pool_k}: {pool_hit:.4f}", flush=True)

        # Run CV
        cv5, lt_ndcg, slices = run_cv(X_all, gt_idx, sizes, cases, sessions, n,
                                       pool_k, n_feat)

        print(f"  CV5={cv5:.4f}  last_turn={lt_ndcg:.4f}")
        print(f"  Slices: same_artist={slices.get('same_artist',0):.4f}  "
              f"diff_artist={slices.get('diff_artist',0):.4f}  "
              f"hist_0={slices.get('hist_0',0):.4f}  "
              f"pop_0={slices.get('pop_0',0):.4f}")

        results[config_name] = {
            "pool_hit": pool_hit, "cv5": cv5, "last_turn": lt_ndcg,
            "slices": slices, "pool_k": pool_k, "strategy": strategy,
        }

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"{ts()} R15 Stage 2 complete. Elapsed: {elapsed:.1f}s")

    baseline_lt = results["v3_baseline_200"]["last_turn"]
    baseline_sa = results["v3_baseline_200"]["slices"].get("same_artist", 0)
    print(f"\nBaseline V3: CV5={results['v3_baseline_200']['cv5']:.4f}  "
          f"last_turn={baseline_lt:.4f}")

    print(f"\nGate evaluation (last_turn +0.005, same_artist degradation <0.003):")
    for name, r in results.items():
        if name == "v3_baseline_200":
            continue
        lt_delta = r["last_turn"] - baseline_lt
        sa_delta = r["slices"].get("same_artist", 0) - baseline_sa
        gate_lt = lt_delta >= 0.005
        gate_sa = sa_delta > -0.003
        gate_improve = (r["slices"].get("pop_0", 0) > results["v3_baseline_200"]["slices"].get("pop_0", 0) or
                        r["slices"].get("diff_artist", 0) > results["v3_baseline_200"]["slices"].get("diff_artist", 0))
        all_pass = gate_lt and gate_sa and gate_improve
        print(f"  {name}: lt={r['last_turn']:.4f} (Δ={lt_delta:+.4f} {'PASS' if gate_lt else 'FAIL'})  "
              f"sa_Δ={sa_delta:+.4f} ({'PASS' if gate_sa else 'FAIL'})  "
              f"slice_improve={'PASS' if gate_improve else 'FAIL'}  "
              f"→ {'PASS' if all_pass else 'FAIL'}")

    out_path = REPO_ROOT / "exp" / "eval" / "expR15_weak_source_fusion.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "elapsed_s": elapsed,
                   "feature_names": FEATURE_NAMES_V3}, f, indent=2, default=str)
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
