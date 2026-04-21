"""Offline devset evaluator for Echo's BM25+CF-BPR system.

Computes nDCG@20 and LexDiv locally against devset ground truth.
Use this BEFORE submitting to Codabench to validate changes.

Usage:
    python3 eval_devset.py
"""
import json
import math
import os
import argparse
import pandas as pd
from collections import defaultdict
from datasets import load_dataset
from omegaconf import OmegaConf

from mcrs.retrieval_modules.bm25 import BM25Retriever
from mcrs.retrieval_modules.cf_bpr import CFBPRRetriever


# ── Retrieval helpers (same as run_inference_blind_bm25cf.py) ────────────────

def rrf_fuse(bm25_results, cf_results, k=60, bm25_weight=0.5, cf_weight=0.5, topk=20):
    scores = {}
    for rank, (tid, _) in enumerate(bm25_results):
        scores[tid] = scores.get(tid, 0) + bm25_weight / (k + rank + 1)
    for rank, (tid, _) in enumerate(cf_results):
        scores[tid] = scores.get(tid, 0) + cf_weight / (k + rank + 1)
    return [tid for tid, _ in sorted(scores.items(), key=lambda x: -x[1])][:topk]


def build_bm25_query(history, user_query, metadata_dict=None):
    """Enriched BM25 query — mirrors run_inference_blind_bm25cf.py exactly.

    1. Confirmed track metadata (role=music turns): artist + tags
    2. Recency-weighted user turns: last prior turn 2×, current query 3×
    """
    parts = []
    if metadata_dict is not None:
        for msg in history:
            if msg["role"] == "music":
                track_id = msg["content"].strip()
                if track_id in metadata_dict:
                    meta = metadata_dict[track_id]
                    artist = meta.get("artist_name", "")
                    if artist:
                        parts.append(artist)
                    tags = meta.get("tag_list", [])
                    if isinstance(tags, list):
                        parts.extend(str(t) for t in tags[:5])
    user_turns = [m["content"] for m in history if m["role"] == "user"]
    n = len(user_turns)
    for i, turn in enumerate(user_turns):
        parts.append(turn)
        if i == n - 1:
            parts.append(turn)
    parts.extend([user_query, user_query, user_query])
    return " ".join(parts)


def get_last_user_turn(conversations):
    """Return (turn_number, user_query, history) for the last user turn.

    Preserves role=music in history for the query builder.
    """
    df = pd.DataFrame(conversations).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    row = user_rows.iloc[-1]
    turn_num = int(row["turn_number"])
    query = row["content"]
    history = []
    for _, h in df[df["turn_number"] < turn_num].iterrows():
        history.append({"role": h["role"], "content": h["content"]})
    return turn_num, query, history


def get_ground_truth(conversations, turn_number):
    """Return the ground-truth track_id for the music turn at turn_number."""
    df = pd.DataFrame(conversations)
    music_rows = df[(df["role"] == "music") & (df["turn_number"] == turn_number)]
    if music_rows.empty:
        return None
    return music_rows.iloc[0]["content"]


def generate_response(track_ids, metadata_dict):
    """Clean template response (no noisy tags)."""
    track_info = []
    seen_artists = set()
    for tid in track_ids[:5]:
        meta = metadata_dict.get(tid)
        if not meta:
            continue
        track = meta.get("track_name", "")
        artist = meta.get("artist_name", "")
        if isinstance(track, list):
            track = track[0] if track else ""
        if isinstance(artist, list):
            artist = ", ".join(str(a) for a in artist) if artist else ""
        if not track or not artist or artist.lower() in seen_artists:
            continue
        track_info.append((track, artist))
        seen_artists.add(artist.lower())

    if not track_info:
        return "Here are some tracks I think you'll enjoy based on our conversation!"

    parts = [f'"{t}" by {a}' for t, a in track_info]
    recs = ", ".join(parts[:-1]) + f", and {parts[-1]}" if len(parts) > 1 else parts[0]
    extra = len(track_ids) - len(track_info)
    return (
        f"Here are my top picks for you: {recs} — "
        f"plus {extra} more tracks I think you'll love. "
        f"Let me know if you'd like something more specific!"
    )


# ── Metrics ──────────────────────────────────────────────────────────────────

def ndcg_at_k(predicted, relevant_set, k=20):
    """Compute nDCG@k. relevant_set is a set of relevant track IDs."""
    dcg = 0.0
    for i, tid in enumerate(predicted[:k]):
        if tid in relevant_set:
            dcg += 1.0 / math.log2(i + 2)
    # Ideal: relevant item at rank 1
    idcg = 1.0 / math.log2(2) if relevant_set else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def lexical_diversity(responses):
    """Mean pairwise Jaccard distance on unigrams — proxy for LexDiv."""
    tokenized = [set(r.lower().split()) for r in responses if r]
    if len(tokenized) < 2:
        return 0.0
    total, count = 0.0, 0
    # Sample 200 pairs for speed
    import random
    random.seed(42)
    pairs = [(i, j) for i in range(len(tokenized)) for j in range(i + 1, len(tokenized))]
    sample = random.sample(pairs, min(200, len(pairs)))
    for i, j in sample:
        a, b = tokenized[i], tokenized[j]
        union = a | b
        if union:
            total += 1 - len(a & b) / len(union)
            count += 1
    return total / count if count else 0.0


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    config = OmegaConf.load(f"config/{args.tid}.yaml")

    print("Loading BM25...")
    bm25 = BM25Retriever(
        dataset_name=config.item_db_name,
        split_types=list(config.track_split_types),
        corpus_types=list(config.corpus_types),
        cache_dir=config.cache_dir,
    )

    cf_weight = getattr(config, "cf_weight", 0.5)
    cf = None
    if cf_weight > 0.0:
        print("Loading CF-BPR...")
        cf = CFBPRRetriever(
            track_embed_dataset="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            user_embed_dataset="talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
            track_split_types=list(config.track_split_types),
            cache_dir=config.cache_dir,
        )
    else:
        print("CF weight=0.0 — skipping CF-BPR load.")

    devset_name = getattr(config, "devset_name", "talkpl-ai/TalkPlayData-Challenge-Devset")
    print(f"Loading devset: {devset_name}")
    db = load_dataset(devset_name, split="test")

    rrf_k = getattr(config, "rrf_k", 60)
    bm25_weight = getattr(config, "bm25_weight", 0.5)
    candidate_k = getattr(config, "candidate_k", 50)

    ndcg_scores = []
    responses = []
    skipped = 0

    for item in db:
        user_id = item["user_id"]
        df = pd.DataFrame(item["conversations"]).sort_values("turn_number")
        user_rows = df[df["role"] == "user"]
        cf_ranked = cf.retrieve_for_user(user_id, topk=candidate_k) if cf is not None else []

        for _, row in user_rows.iterrows():
            turn_num = int(row["turn_number"])
            user_query = row["content"]
            gt = get_ground_truth(item["conversations"], turn_num)
            if gt is None:
                skipped += 1
                continue

            # Preserve role=music so query builder uses confirmed track signals
            history = []
            for _, h in df[df["turn_number"] < turn_num].iterrows():
                history.append({"role": h["role"], "content": h["content"]})

            bm25_query = build_bm25_query(history, user_query, metadata_dict=bm25.metadata_dict)
            bm25_ranked = bm25.scored_retrieval(bm25_query, topk=candidate_k)
            fused = rrf_fuse(bm25_ranked, cf_ranked, k=rrf_k,
                             bm25_weight=bm25_weight, cf_weight=cf_weight, topk=20)

            score = ndcg_at_k(fused, {gt})
            ndcg_scores.append(score)
            responses.append(generate_response(fused, bm25.metadata_dict))

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
    lex_div = lexical_diversity(responses)

    print(f"\n{'='*50}")
    print(f"Sessions evaluated : {len(ndcg_scores)} (skipped {skipped} — no GT)")
    print(f"nDCG@20            : {mean_ndcg:.4f}")
    print(f"LexDiv (proxy)     : {lex_div:.4f}")
    print(f"{'='*50}\n")

    os.makedirs("exp/eval", exist_ok=True)
    out = f"exp/eval/{args.tid}_devset.json"
    with open(out, "w") as f:
        json.dump({"ndcg20": mean_ndcg, "lexdiv": lex_div, "n": len(ndcg_scores)}, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", default="echo_bm25_cf_blind_a")
    args = parser.parse_args()
    main(args)
