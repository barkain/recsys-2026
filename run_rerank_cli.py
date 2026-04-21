"""Rerank BM25 candidates using the Claude CLI subscription (no API key needed).

Workflow:
  1. Load BM25 from existing cache (no rebuild) → get top-50 candidates per session
  2. For each session: format conversation + candidates as text
  3. Call `claude -p --no-session-persistence` in 8 parallel threads
  4. Merge reranked tracks with v6 responses → final submission zip

Usage:
    cd recsys-work
    HF_DATASETS_CACHE=/workspace/group/hf_cache \\
    HF_HOME=/workspace/group/hf_home \\
    USER=echo LOGNAME=echo \\
    uv run python3 run_rerank_cli.py
"""
import json
import os
import re
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd
from datasets import load_dataset
from omegaconf import OmegaConf

from mcrs.retrieval_modules.bm25 import BM25Retriever


TID = "echo_bm25_cf_blind_a"
CANDIDATE_K = 50
N_WORKERS = 8
CLAUDE_MODEL = "haiku"   # claude-haiku-4-5-20251001 alias


# ── Query / history helpers ──────────────────────────────────────────────────

def last_turn(conversations):
    df = pd.DataFrame(conversations).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    row = user_rows.iloc[-1]
    turn_num = int(row["turn_number"])
    query = row["content"]
    history = []
    for _, h in df[df["turn_number"] < turn_num].iterrows():
        history.append({"role": h["role"], "content": h["content"]})
    return turn_num, query, history


def build_bm25_query(history, user_query, metadata_dict=None):
    parts = []
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
    user_turns = [m["content"] for m in history if m["role"] == "user"]
    n = len(user_turns)
    for i, turn in enumerate(user_turns):
        parts.append(turn)
        if i == n - 1:
            parts.append(turn)
    parts.extend([user_query, user_query, user_query])
    return " ".join(parts)


def _resolve_artist(meta):
    artist = meta.get("artist_name", "")
    if isinstance(artist, list):
        artist = ", ".join(str(a) for a in artist)
    return str(artist) or "Unknown Artist"


def _resolve_track(meta, track_id):
    name = meta.get("track_name", track_id)
    if isinstance(name, list):
        name = name[0] if name else track_id
    return str(name)


# ── Conversation + candidates formatting ─────────────────────────────────────

def format_conversation(history, user_query, metadata_dict):
    lines = []
    for msg in history:
        if msg["role"] == "music":
            tid = msg["content"].strip()
            meta = metadata_dict.get(tid, {})
            name = _resolve_track(meta, tid)
            artist = _resolve_artist(meta)
            lines.append(f"[Track played: {name} by {artist}]")
        elif msg["role"] == "user":
            lines.append(f"User: {msg['content']}")
        elif msg["role"] in ("assistant", "system"):
            pass  # skip bot turns to keep prompt compact
    lines.append(f"User (latest): {user_query}")
    return "\n".join(lines) if lines else user_query


def format_candidates(candidates, metadata_dict):
    lines = []
    for i, tid in enumerate(candidates, 1):
        meta = metadata_dict.get(tid, {})
        name = _resolve_track(meta, tid)
        artist = _resolve_artist(meta)
        tags = meta.get("tag_list", [])
        if isinstance(tags, list):
            tags_str = ", ".join(str(t) for t in tags[:6])
        else:
            tags_str = ""
        lines.append(f"{i}. [{tid}] {name} — {artist} | tags: {tags_str}")
    return "\n".join(lines)


# ── Claude CLI reranker ──────────────────────────────────────────────────────

def claude_rerank(session_idx, session_id, candidates, conv_text, cands_text):
    topk = 20
    prompt = (
        f"Music recommendation task. Rank the {topk} best matching tracks for this user.\n\n"
        f"Conversation:\n{conv_text}\n\n"
        f"Candidate tracks (ID in brackets):\n{cands_text}\n\n"
        f"Return ONLY a JSON array of exactly {topk} track_id strings from the candidates above, "
        f"best match first. No explanation, no markdown, no extra text. Example: "
        f'["id1","id2",...]\n'
    )
    try:
        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence", "--model", CLAUDE_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=90,
        )
        text = result.stdout.strip()
        # Extract JSON array
        match = re.search(r"\[.*?\]", text, re.DOTALL)
        if match:
            ids = json.loads(match.group())
            valid = set(candidates)
            reranked = [str(i) for i in ids if str(i) in valid]
            if reranked:
                remaining = [c for c in candidates if c not in set(reranked)]
                return session_idx, session_id, (reranked + remaining)[:topk]
    except Exception as e:
        print(f"  [WARN] Session {session_id} rerank failed: {e}")
    # Fallback: original BM25 order
    return session_idx, session_id, candidates[:topk]


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = OmegaConf.load(f"config/{TID}.yaml")

    print("Loading BM25 retriever (from cache if available)...")
    bm25 = BM25Retriever(
        dataset_name=config.item_db_name,
        split_types=list(config.track_split_types),
        corpus_types=list(config.corpus_types),
        cache_dir=config.cache_dir,
    )

    print("Loading blind dataset from HF cache...")
    db = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Blind-A",
        split="test",
    )

    # Stage 1: BM25 retrieval — get top-50 candidates per session
    print(f"\nStage 1: BM25 retrieval ({CANDIDATE_K} candidates per session)...")
    sessions = []
    for item in tqdm(db, desc="Retrieving"):
        turn_num, user_query, history = last_turn(item["conversations"])
        bm25_query = build_bm25_query(history, user_query, bm25.metadata_dict)
        bm25_ranked = bm25.scored_retrieval(bm25_query, topk=CANDIDATE_K)
        candidates = [tid for tid, _ in bm25_ranked]

        conv_text = format_conversation(history, user_query, bm25.metadata_dict)
        cands_text = format_candidates(candidates, bm25.metadata_dict)
        sessions.append({
            "session_id": item["session_id"],
            "user_id": item["user_id"],
            "turn_number": turn_num,
            "candidates": candidates,
            "conv_text": conv_text,
            "cands_text": cands_text,
        })

    print(f"\nStage 2: LLM reranking ({len(sessions)} sessions, {N_WORKERS} parallel threads)...")
    results = [None] * len(sessions)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(
                claude_rerank,
                i,
                s["session_id"],
                s["candidates"],
                s["conv_text"],
                s["cands_text"],
            ): i
            for i, s in enumerate(sessions)
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Reranking"):
            idx, sid, ranked = f.result()
            results[idx] = (sid, ranked)

    # Stage 3: Load v6 responses and merge
    v6_path = "exp/inference/blind_a/echo_bm25_responses_v6.json"
    print(f"\nStage 3: Merging with v6 responses ({v6_path})...")
    with open(v6_path) as f:
        v6_data = {s["session_id"]: s["predicted_response"] for s in json.load(f)}

    predictions = []
    for i, (sid, ranked) in enumerate(results):
        sess = sessions[i]
        predictions.append({
            "session_id": sid,
            "user_id": sess["user_id"],
            "turn_number": sess["turn_number"],
            "predicted_track_ids": ranked,
            "predicted_response": v6_data.get(sid, ""),
        })

    # Save
    os.makedirs("exp/inference/blind_a", exist_ok=True)
    out_json = "exp/inference/blind_a/echo_bm25_llm_reranked.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions: {out_json}")

    out_zip = "/workspace/group/echo_v8_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")
    print(f"Submission zip: {out_zip}")
    print(f"\nTotal: {len(predictions)} sessions predicted.")


if __name__ == "__main__":
    main()
