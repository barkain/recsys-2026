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
from mcrs.retrieval_modules.cf_bpr import CFBPRRetriever


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
    """
    df = pd.DataFrame(conversations).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    row = user_rows.iloc[-1]
    turn_num = int(row["turn_number"])
    query = row["content"]
    history = []
    for _, h in df[df["turn_number"] < turn_num].iterrows():
        role = h["role"] if h["role"] != "music" else "assistant"
        history.append({"role": role, "content": h["content"]})
    return turn_num, query, history


def build_bm25_query(history: list[dict], user_query: str) -> str:
    parts = [m["content"] for m in history if m["role"] == "user"]
    parts.append(user_query)
    return " ".join(parts)


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

    print("Loading CF-BPR retriever...")
    cf = CFBPRRetriever(
        track_embed_dataset="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
        user_embed_dataset="talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
        track_split_types=list(config.track_split_types),
        cache_dir=config.cache_dir,
    )

    rrf_k = getattr(config, "rrf_k", 60)
    bm25_weight = getattr(config, "bm25_weight", 0.5)
    cf_weight = getattr(config, "cf_weight", 0.5)
    candidate_k = getattr(config, "candidate_k", 50)

    blind_dataset_name = getattr(config, "test_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Blind-A")
    print(f"Loading blind dataset: {blind_dataset_name}")
    db = load_dataset(blind_dataset_name, split="test")

    results = []
    for item in tqdm(db, desc="Sessions"):
        user_id = item["user_id"]
        session_id = item["session_id"]
        cf_ranked = cf.retrieve_for_user(user_id, topk=candidate_k)

        turn_num, user_query, history = last_turn(item["conversations"])
        bm25_query = build_bm25_query(history, user_query)
        bm25_ranked = bm25.scored_retrieval(bm25_query, topk=candidate_k)
        fused = rrf_fuse(bm25_ranked, cf_ranked, k=rrf_k,
                         bm25_weight=bm25_weight, cf_weight=cf_weight, topk=20)
        results.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": turn_num,
            "predicted_track_ids": fused,
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
