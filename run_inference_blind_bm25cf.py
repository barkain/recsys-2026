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


def generate_response(user_query: str, track_ids: list[str], metadata_dict: dict) -> str:
    """Generate a natural-language recommendation response from retrieved tracks.

    Focuses on the music (genres/tags) rather than echoing the raw user query,
    which avoids incoherent phrases when the user query is conversational/confirmatory.
    """
    track_info = []
    all_tags: list[str] = []
    seen_artists: set[str] = set()

    for tid in track_ids[:5]:
        meta = metadata_dict.get(tid)
        if not meta:
            continue
        track = meta.get("track_name", "")
        artist = meta.get("artist_name", "")
        tags = meta.get("tag_list", [])
        if isinstance(track, list):
            track = track[0] if track else ""
        if isinstance(artist, list):
            artist = ", ".join(str(a) for a in artist) if artist else ""
        if not track or not artist:
            continue
        if isinstance(tags, list):
            all_tags.extend(str(t) for t in tags[:3])
        artist_key = artist.lower()
        if artist_key not in seen_artists:
            track_info.append((track, artist))
            seen_artists.add(artist_key)

    if not track_info:
        return "Here are some tracks I think you'll enjoy based on our conversation!"

    # Format track list
    track_parts = [f'"{t}" by {a}' for t, a in track_info]
    if len(track_parts) == 1:
        recs = track_parts[0]
    elif len(track_parts) == 2:
        recs = f"{track_parts[0]} and {track_parts[1]}"
    else:
        recs = ", ".join(track_parts[:-1]) + f", and {track_parts[-1]}"

    extra = len(track_ids) - len(track_info)
    return (
        f"Here are my top picks for you: {recs} — "
        f"plus {extra} more tracks I think you'll love. "
        f"Let me know if you'd like something more specific!"
    )


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

    cf = None
    if cf_weight > 0.0:
        print("Loading CF-BPR retriever...")
        cf = CFBPRRetriever(
            track_embed_dataset="talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
            user_embed_dataset="talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
            track_split_types=list(config.track_split_types),
            cache_dir=config.cache_dir,
        )
    else:
        print("CF weight=0.0 — skipping CF-BPR retriever load.")

    blind_dataset_name = getattr(config, "test_dataset_name", "talkpl-ai/TalkPlayData-Challenge-Blind-A")
    print(f"Loading blind dataset: {blind_dataset_name}")
    db = load_dataset(blind_dataset_name, split="test")

    results = []
    for item in tqdm(db, desc="Sessions"):
        user_id = item["user_id"]
        session_id = item["session_id"]
        cf_ranked = cf.retrieve_for_user(user_id, topk=candidate_k) if cf is not None else []

        turn_num, user_query, history = last_turn(item["conversations"])
        bm25_query = build_bm25_query(history, user_query)
        bm25_ranked = bm25.scored_retrieval(bm25_query, topk=candidate_k)
        fused = rrf_fuse(bm25_ranked, cf_ranked, k=rrf_k,
                         bm25_weight=bm25_weight, cf_weight=cf_weight, topk=20)
        response = generate_response(user_query, fused, bm25.metadata_dict)
        results.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": turn_num,
            "predicted_track_ids": fused,
            "predicted_response": response,
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
