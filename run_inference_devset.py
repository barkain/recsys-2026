"""Batch inference script for Echo's Music CRS — RecSys 2026."""
import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import load_dataset
from mcrs import load_crs_system


def chat_history_parser(conversations, crs, target_turn_number):
    """Parse conversation history up to target_turn_number.

    Returns:
        (chat_history, user_query)
    """
    df = pd.DataFrame(conversations)
    df_history = df[df["turn_number"] < target_turn_number]
    chat_history = []
    for row in df_history.to_dict(orient="records"):
        role = row["role"]
        content = row["content"]
        if role == "music":
            role = "assistant"
            content = crs.item_db.id_to_metadata(content)
        chat_history.append({"role": role, "content": content})
    user_query = df[df["turn_number"] == target_turn_number].iloc[0]["content"]
    return chat_history, user_query


def main(args):
    print("Clearing cache to prevent memory issues...")
    os.system("rm -rf cache")

    config = OmegaConf.load(f"config/{args.tid}.yaml")
    config_dict = OmegaConf.to_container(config, resolve=True)

    crs = load_crs_system(**config_dict)

    db = load_dataset(config.test_dataset_name, split="test")

    batch_data, metadata = [], []
    for item in db:
        user_id = item["user_id"]
        session_id = item["session_id"]
        for turn in range(1, 9):
            chat_history, user_query = chat_history_parser(item["conversations"], crs, turn)
            batch_data.append({
                "user_query": user_query,
                "user_id": user_id,
                "session_memory": chat_history,
            })
            metadata.append({
                "session_id": session_id,
                "user_id": user_id,
                "turn_number": turn,
            })

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
                "predicted_response": result["response"],
            })

    os.makedirs("exp/inference/devset", exist_ok=True)
    out_path = f"exp/inference/devset/{args.tid}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(inference_results)} results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", type=str, default="echo_hybrid_claude_devset")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)
