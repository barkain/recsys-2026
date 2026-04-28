"""Batch inference script for Echo's Music CRS — RecSys 2026."""
# ruff: noqa: T201, S311, S605, S607
import os
import json
import argparse
import random
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from datasets import Dataset, DownloadConfig, load_dataset
from mcrs import load_crs_system
from eval_inference import cached_test_arrow_path


def _result_key(result: dict) -> tuple[str, str | None, int]:
    user_id = result.get("user_id")
    return str(result["session_id"]), None if user_id is None else str(user_id), int(result["turn_number"])


def _atomic_write_json(path: str, data) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _load_case_filter(path: str) -> tuple[set[str], set[tuple[str, str]]]:
    session_ids = set()
    case_ids = set()
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                case_ids.add((parts[0], parts[1]))
            else:
                session_ids.add(line)
    return session_ids, case_ids


def _select_by_case_filter(db, session_ids: set[str], case_ids: set[tuple[str, str]]):
    indices = [
        i for i, item in enumerate(db)
        if str(item["session_id"]) in session_ids
        or (str(item["session_id"]), str(item.get("user_id", ""))) in case_ids
    ]
    return db.select(indices)


def _sample_sessions(db, sample_size: int, seed: int):
    n = min(sample_size, len(db))
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(db)), n))
    return db.select(indices)


def _write_session_ids(path: str, db) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in db:
            f.write(f"{item['session_id']}\t{item.get('user_id', '')}\n")


def _load_test_dataset(dataset_name: str):
    arrow_path = cached_test_arrow_path()
    if dataset_name == "talkpl-ai/TalkPlayData-Challenge-Dataset" and arrow_path:
        return Dataset.from_file(arrow_path)
    return load_dataset(
        dataset_name,
        split="test",
        download_config=DownloadConfig(local_files_only=True),
    )


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
    if not os.environ.get("ANTHROPIC_RECSYS_API_KEY"):
        raise EnvironmentError("ANTHROPIC_RECSYS_API_KEY is not set — cannot run LLM inference")

    # Cache cleared only when explicitly requested — BM25 index re-build takes minutes
    if os.environ.get("CLEAR_CACHE"):
        print("Clearing cache (CLEAR_CACHE=1)...")
        os.system("rm -rf cache")

    config = OmegaConf.load(f"config/{args.tid}.yaml")
    config_dict = OmegaConf.to_container(config, resolve=True)
    config_dict["generate_responses"] = not args.skip_response_generation
    if args.fail_fast_llm:
        config_dict["fail_fast_llm"] = True
    # CLI override for the persisted rerank pool width. Config-file key
    # save_rerank_pool_k is also honored — CLI wins.
    if args.save_rerank_pool_k is not None:
        config_dict["save_rerank_pool_k"] = args.save_rerank_pool_k

    print(f"Loading CRS config: {args.tid}", flush=True)
    crs = load_crs_system(**config_dict)
    print("CRS loaded", flush=True)

    print(f"Loading test dataset: {config.test_dataset_name}", flush=True)
    db = _load_test_dataset(config.test_dataset_name)
    print(f"Test dataset loaded: {len(db)} sessions", flush=True)

    if args.session_ids:
        session_ids, case_ids = _load_case_filter(args.session_ids)
        db = _select_by_case_filter(db, session_ids, case_ids)
        print(f"Selected {len(db)} sessions from {args.session_ids}")

    if args.sample_size:
        db = _sample_sessions(db, args.sample_size, args.sample_seed)
        print(f"Selected deterministic sample: {len(db)} sessions (seed={args.sample_seed})")

    # Legacy convenience limit. Prefer --sample_size for unbiased small evals.
    if args.max_sessions:
        db = db.select(range(min(args.max_sessions, len(db))))
        print(f"Selected first {len(db)} sessions via --max_sessions")

    if args.write_session_ids:
        _write_session_ids(args.write_session_ids, db)
        print(f"Wrote session IDs to {args.write_session_ids}")

    # Determine which turns to run inference on.
    # --last_turn_only: only run turn 8 (nDCG@20 is measured on the last turn).
    # This reduces LLM calls 8x for quick offline measurements.
    turns_to_run = [8] if args.last_turn_only else list(range(1, 9))

    batch_data, metadata = [], []
    for item in db:
        user_id = item["user_id"]
        session_id = item["session_id"]
        for turn in turns_to_run:
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
    print(f"Total turns to run: {len(batch_data)}", flush=True)

    os.makedirs("exp/inference/devset", exist_ok=True)
    output_tid = args.output_tid or args.tid
    out_path = f"exp/inference/devset/{output_tid}.json"

    inference_results = []
    completed: set[tuple[str, str | None, int]] = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            inference_results = json.load(f)
        completed = {_result_key(r) for r in inference_results}
        print(f"Resuming from {out_path}: {len(completed)} completed turns")

    if completed:
        filtered_batch_data, filtered_metadata = [], []
        for item, meta in zip(batch_data, metadata):
            user_id = meta.get("user_id")
            key = (
                str(meta["session_id"]),
                None if user_id is None else str(user_id),
                int(meta["turn_number"]),
            )
            if key not in completed:
                filtered_batch_data.append(item)
                filtered_metadata.append(meta)
        skipped = len(batch_data) - len(filtered_batch_data)
        print(f"Skipping {skipped} already-completed turns; {len(filtered_batch_data)} remaining")
        batch_data, metadata = filtered_batch_data, filtered_metadata

    for i in tqdm(range(0, len(batch_data), args.batch_size), desc="Inference"):
        batch = batch_data[i:i + args.batch_size]
        batch_meta = metadata[i:i + args.batch_size]
        results = crs.batch_chat(batch)
        for j, result in enumerate(results):
            # top-20 is unchanged in shape & content — persisted identically.
            # candidate_pool_track_ids is additive: the wider pool (up to
            # save_rerank_pool_k, default 50) composed of the LLM-reranked
            # prefix + remaining rerank-input-window candidates in retrieval
            # order (deduped). The provenance block documents truth about
            # which ranks are LLM-ranked vs. upstream retrieval order.
            row = {
                "session_id": batch_meta[j]["session_id"],
                "user_id": batch_meta[j]["user_id"],
                "turn_number": batch_meta[j]["turn_number"],
                "predicted_track_ids": result["retrieval_items"],
                "predicted_response": result["response"],
            }
            pool = result.get("candidate_pool_items")
            if pool is not None:
                row["candidate_pool_track_ids"] = pool
            prov = result.get("candidate_pool_provenance")
            if prov is not None:
                row["candidate_pool_provenance"] = prov
            inference_results.append(row)
        # Incremental save after each batch so a late-stage crash doesn't wipe the run.
        _atomic_write_json(out_path, inference_results)

    print(f"Saved {len(inference_results)} results to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", type=str, default="echo_hybrid_claude_devset")
    parser.add_argument("--output_tid", type=str, default=None,
                        help="Write output under this ID instead of --tid (useful for samples)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_sessions", type=int, default=None,
                        help="Limit to the first N sessions (legacy; prefer --sample_size)")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="Run a deterministic random sample of N sessions")
    parser.add_argument("--sample_seed", type=int, default=42,
                        help="Seed for --sample_size")
    parser.add_argument("--session_ids", type=str, default=None,
                        help="Path to newline-delimited session IDs or tab-delimited session_id/user_id cases")
    parser.add_argument("--write_session_ids", type=str, default=None,
                        help="Write selected tab-delimited session_id/user_id case IDs to this file")
    parser.add_argument("--last_turn_only", action="store_true",
                        help="Only run inference on turn 8 (mirrors Codabench nDCG@20 scoring)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip turns already present in exp/inference/devset/{tid}.json")
    parser.add_argument("--skip_response_generation", action="store_true",
                        help="Do retrieval/reranking only and write empty responses (saves one LLM call per turn)")
    parser.add_argument("--fail_fast_llm", action="store_true",
                        help="Abort if QR/reranker LLM calls fail instead of falling back silently")
    parser.add_argument("--save_rerank_pool_k", type=int, default=None,
                        help="Persist candidate_pool_track_ids of length up to K alongside top-20 predicted_track_ids. Default: config value or 50.")
    args = parser.parse_args()
    main(args)
