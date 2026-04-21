"""Blind set inference script for Echo's Music CRS — RecSys 2026.

Runs inference on TalkPlayData-Challenge-Blind-A and produces prediction.json
in the Codabench submission format, then zips as submission.zip.
"""
import os
import json
import zipfile
import argparse
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset
from mcrs import load_crs_system


def chat_history_parser(conversations, crs):
    """Parse the LAST user turn from a blind-set conversation.

    The Codabench scorer expects exactly one prediction per session (80 sessions = 80 entries).
    Returns a single (turn_number, chat_history, user_query) tuple.
    """
    df = pd.DataFrame(conversations)
    user_rows = df[df["role"] == "user"].sort_values("turn_number")
    row = user_rows.iloc[-1]
    turn_number = int(row["turn_number"])
    user_query = row["content"]

    history_rows = df[df["turn_number"] < turn_number].sort_values("turn_number")
    chat_history = []
    for _, h in history_rows.iterrows():
        role = h["role"]
        content = h["content"]
        if role == "music":
            role = "assistant"
            content = crs.item_db.id_to_metadata(content)
        elif role == "system":
            role = "assistant"
        chat_history.append({"role": role, "content": content})

    return turn_number, chat_history, user_query


def main(args):
    print("Clearing cache to prevent memory issues...")
    os.system("rm -rf cache")

    config = OmegaConf.load(f"config/{args.tid}.yaml")
    config_dict = OmegaConf.to_container(config, resolve=True)

    crs = load_crs_system(**config_dict)

    blind_dataset_name = args.blind_dataset or "talkpl-ai/TalkPlayData-Challenge-Blind-A"
    print(f"Loading blind dataset: {blind_dataset_name}")
    db = load_dataset(blind_dataset_name, split="test")

    batch_data, metadata = [], []
    for item in db:
        user_id = item["user_id"]
        session_id = item["session_id"]
        turn_number, chat_history, user_query = chat_history_parser(item["conversations"], crs)
        batch_data.append({
            "user_query": user_query,
            "user_id": user_id,
            "session_memory": chat_history,
        })
        metadata.append({
            "session_id": session_id,
            "user_id": user_id,
            "turn_number": turn_number,
        })

    print(f"Total sessions to predict: {len(batch_data)}")

    inference_results = []
    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Inference"):
        batch = batch_data[i:i + args.batch_size]
        batch_meta = metadata[i:i + args.batch_size]
        results = crs.batch_chat(batch)
        for j, result in enumerate(results):
            inference_results.append({
                "session_id": batch_meta[j]["session_id"],
                "user_id": batch_meta[j]["user_id"],
                "turn_number": batch_meta[j]["turn_number"],
                "predicted_track_ids": result["retrieval_items"],
                "predicted_response": result.get("response", ""),
            })

    os.makedirs("exp/inference/blind_a", exist_ok=True)
    out_json = f"exp/inference/blind_a/{args.tid}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(inference_results)} results to {out_json}")

    # Package as submission.zip
    out_zip = f"exp/inference/blind_a/{args.tid}_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")
    print(f"Submission zip: {out_zip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", type=str, default="echo_bm25_enriched_devset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--blind_dataset", type=str, default="talkpl-ai/TalkPlayData-Challenge-Blind-A")
    args = parser.parse_args()
    main(args)
