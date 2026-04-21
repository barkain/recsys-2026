"""BM25 + CF-BPR hybrid blind-set inference — no model inference needed.

Fuses BM25 (conversation text → track) with CF-BPR (user taste → track) via RRF.
"""
import os
import json
import zipfile
import argparse
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset

from mcrs.retrieval_modules.bm25 import BM25Retriever


def rrf_fuse(
    bm25_results: list[tuple[str, float]],
    cf_results: list[tuple[str, float]],
    k: int = 60,
    bm25_weight: float = 0.5,
    cf_weight: float = 0.5,
    topk: int = 20,
) -> list[str]:
    scores: dict[str, float] = {}
    for rank, (tid, _) in enumerate(bm25_results):
        scores[tid] = scores.get(tid, 0) + bm25_weight / (k + rank + 1)
    for rank, (tid, _) in enumerate(cf_results):
        scores[tid] = scores.get(tid, 0) + cf_weight / (k + rank + 1)
    return [tid for tid, _ in sorted(scores.items(), key=lambda x: -x[1])][:topk]



def last_turn(conversations: list[dict]) -> tuple[int, str, list[dict]]:
    """Return (turn_number, user_query, prior_history_dicts) for the LAST user turn only.

    The Codabench scorer expects exactly one prediction per session (80 sessions = 80 entries).
    Preserves role=music so query builder can use confirmed track IDs.
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


def build_bm25_query(history: list[dict], user_query: str, metadata_dict: dict | None = None) -> str:
    """Build enriched BM25 query from conversation history.

    Improvements over naive concatenation:
    1. Confirmed tracks (role=music turns): add their artist + tags as direct relevance signal
    2. Recency weighting: repeat the last user turn 2x and current query 3x
    3. All prior user turns included for full context
    """
    parts = []

    # 1. Confirmed track metadata — strongest positive signal
    if metadata_dict is not None:
        for msg in history:
            if msg["role"] == "music":
                track_id = msg["content"].strip()
                if track_id in metadata_dict:
                    meta = metadata_dict[track_id]
                    artist = meta.get("artist_name", "")
                    if isinstance(artist, list):
                        artist = " ".join(str(a) for a in artist)
                    if artist:
                        parts.append(str(artist))
                    tags = meta.get("tag_list", [])
                    if isinstance(tags, list):
                        parts.extend(str(t) for t in tags[:5])

    # 2. Prior user turns — recency weighted
    user_turns = [m["content"] for m in history if m["role"] == "user"]
    n = len(user_turns)
    for i, turn in enumerate(user_turns):
        parts.append(turn)
        if i == n - 1:          # last prior turn: repeat once more
            parts.append(turn)

    # 3. Current query — highest weight (3x)
    parts.extend([user_query, user_query, user_query])

    return " ".join(parts)


def build_session_memory(history: list[dict], user_query: str, metadata_dict: dict) -> list[dict]:
    """Build conversation memory for the LLM reranker.

    Translates role=music track IDs into readable "{track} by {artist}" strings
    so the LLM understands what music the user was offered/confirmed.
    """
    memory = []
    for msg in history:
        if msg["role"] == "music":
            track_id = msg["content"].strip()
            meta = metadata_dict.get(track_id, {})
            track_name = meta.get("track_name", track_id)
            if isinstance(track_name, list):
                track_name = track_name[0] if track_name else track_id
            artist = meta.get("artist_name", "")
            if isinstance(artist, list):
                artist = ", ".join(str(a) for a in artist)
            readable = f"{track_name} by {artist}" if artist else str(track_name)
            memory.append({"role": "assistant", "content": f"[Played: {readable}]"})
        else:
            memory.append({"role": msg["role"], "content": msg["content"]})
    memory.append({"role": "user", "content": user_query})
    return memory


def main(args):
    print("Clearing local BM25 cache...")
    os.system("rm -rf cache")

    config = OmegaConf.load(f"config/{args.tid}.yaml")

    print("Loading BM25 retriever...")
    bm25 = BM25Retriever(
        dataset_name=config.item_db_name,
        split_types=list(config.track_split_types),
        corpus_types=list(config.corpus_types),
        cache_dir=config.cache_dir,
    )

    rrf_k = getattr(config, "rrf_k", 60)
    bm25_weight = getattr(config, "bm25_weight", 0.5)
    cf_weight = getattr(config, "cf_weight", 0.5)
    candidate_k = getattr(config, "candidate_k", 50)
    use_reranker = getattr(config, "use_reranker", False)
    reranker_model = getattr(config, "lm_type", "claude-haiku-4-5-20251001")

    cf = None
    if cf_weight > 0.0:
        print("Loading CF-BPR retriever...")
        from mcrs.retrieval_modules.cf_bpr import CFBPRRetriever
        cf = CFBPRRetriever(
            track_embed_dataset="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            user_embed_dataset="talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
            track_split_types=list(config.track_split_types),
            cache_dir=config.cache_dir,
        )
    else:
        print("CF weight=0.0 — skipping CF-BPR retriever load.")

    reranker = None
    if use_reranker:
        from mcrs.reranking_modules.llm_reranker import LLMListwiseReranker
        print(f"LLM reranker enabled — model: {reranker_model}")
        reranker = LLMListwiseReranker(model=reranker_model, topk=20, window_size=candidate_k)

    blind_dataset_name = getattr(config, "test_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Blind-A")
    print(f"Loading blind dataset: {blind_dataset_name}")
    db = load_dataset(blind_dataset_name, split="test")

    # Collect all sessions first for batch reranking (parallel API calls)
    all_sessions = []
    for item in tqdm(db, desc="Retrieving candidates"):
        user_id = item["user_id"]
        session_id = item["session_id"]
        cf_ranked = cf.retrieve_for_user(user_id, topk=candidate_k) if cf is not None else []

        turn_num, user_query, history = last_turn(item["conversations"])
        bm25_query = build_bm25_query(history, user_query, metadata_dict=bm25.metadata_dict)
        bm25_ranked = bm25.scored_retrieval(bm25_query, topk=candidate_k)
        # Fuse to candidate_k (not 20) so reranker has full candidate pool
        rerank_topk = candidate_k if reranker else 20
        fused = rrf_fuse(bm25_ranked, cf_ranked, k=rrf_k,
                         bm25_weight=bm25_weight, cf_weight=cf_weight, topk=rerank_topk)
        session_memory = build_session_memory(history, user_query, bm25.metadata_dict)
        all_sessions.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": turn_num,
            "candidates": fused,
            "session_memory": session_memory,
        })

    # Rerank in parallel batch (one LLM call per session, 8 threads)
    if reranker:
        print(f"Reranking {len(all_sessions)} sessions with LLM (parallel)...")
        batch_candidates = [s["candidates"] for s in all_sessions]
        batch_memory = [s["session_memory"] for s in all_sessions]
        reranked_batch = reranker.batch_rerank(batch_candidates, batch_memory, bm25.metadata_dict)
        for session, reranked in zip(all_sessions, reranked_batch):
            session["candidates"] = reranked

    results = []
    for session in all_sessions:
        results.append({
            "session_id": session["session_id"],
            "user_id": session["user_id"],
            "turn_number": session["turn_number"],
            "predicted_track_ids": session["candidates"][:20],
            "predicted_response": "",
        })

    print(f"Total predictions: {len(results)}")
    os.makedirs("exp/inference/blind_a", exist_ok=True)
    out_json = f"exp/inference/blind_a/{args.tid}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved to {out_json}")

    out_zip = f"exp/inference/blind_a/{args.tid}_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")
    print(f"Submission zip: {out_zip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", type=str, default="echo_bm25_cf_blind_a")
    args = parser.parse_args()
    main(args)
