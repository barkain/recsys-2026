"""v9 inference: BM25 retrieval → Claude rerank + personalized response generation.

Key improvement over v8:
  v8 reranked tracks with Claude but kept stale v6 responses → LLM judge penalized mismatch.
  v9 combines reranking + response generation in ONE Claude call per session.
  The response explicitly references the top recommended tracks → judge sees coherent output.

Workflow:
  1. BM25 retrieval → top-50 candidates per session
  2. Claude Haiku: rerank to top-20 AND write personalized response (single call)
  3. Output submission zip

Usage:
    cd recsys-work
    HF_DATASETS_CACHE=/workspace/group/hf_cache \\
    HF_HOME=/workspace/group/hf_home \\
    USER=echo LOGNAME=echo \\
    .venv/bin/python3.14 run_v9_cli.py [--tid echo_bm25_cf_blind_a] [--out /workspace/group/echo_v9_submission.zip]
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


CANDIDATE_K = 50
N_WORKERS = 8
CLAUDE_MODEL = "haiku"   # claude-haiku-4-5-20251001


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


def format_conversation(history, user_query, metadata_dict):
    lines = []
    for msg in history:
        if msg["role"] == "music":
            tid = msg["content"].strip()
            meta = metadata_dict.get(tid, {})
            name = _resolve_track(meta, tid)
            artist = _resolve_artist(meta)
            lines.append(f"[Previously played: {name} by {artist}]")
        elif msg["role"] == "user":
            lines.append(f"User: {msg['content']}")
        # skip assistant/system turns to keep prompt compact
    lines.append(f"User: {user_query}")
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
        year = meta.get("release_year", "")
        year_str = f" ({year})" if year else ""
        lines.append(f"{i}. [{tid}] {name}{year_str} — {artist} | {tags_str}")
    return "\n".join(lines)


# ── Combined rerank + response generation ────────────────────────────────────

COMBINED_PROMPT_TEMPLATE = """\
You are a music recommendation assistant. Complete two tasks for this user.

**Conversation:**
{conv_text}

**Candidate tracks to choose from:**
{cands_text}

**Task 1:** Select and rank the 20 best matching tracks from the candidates above, best first.

**Task 2:** Write a personalized response (2-4 sentences) that:
- Directly addresses what the user asked for in their last message
- Mentions 2-3 specific track names and briefly explains why they fit their taste
- Shows understanding of their preferences from the conversation
- Ends with a warm invitation to refine further

Return ONLY valid JSON, no markdown, no extra text:
{{
  "ranked_tracks": ["track_id_1", "track_id_2", ...],
  "response": "Your personalized response here..."
}}

The ranked_tracks array must contain exactly 20 track IDs from the candidates above.\
"""


_FALLBACK_RESPONSE = "Here are some tracks I think you'll enjoy based on our conversation!"


def claude_rerank_and_respond(session_idx, session_id, candidates, conv_text, cands_text):
    topk = 20
    prompt = COMBINED_PROMPT_TEMPLATE.format(conv_text=conv_text, cands_text=cands_text)
    try:
        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence", "--model", CLAUDE_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  [WARN] Session {session_id} CLI error (rc={result.returncode}): "
                  f"{result.stderr[:300]}")
        else:
            text = result.stdout.strip()
            # Extract JSON object (handle possible surrounding text)
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                ranked = [str(i) for i in data.get("ranked_tracks", []) if str(i) in set(candidates)]
                response = str(data.get("response", "")).strip()
                if ranked and response:
                    remaining = [c for c in candidates if c not in set(ranked)]
                    return session_idx, session_id, (ranked + remaining)[:topk], response, False
    except Exception as e:
        print(f"  [WARN] Session {session_id} failed: {e} | stderr: {getattr(e, 'stderr', '')[:200]}")

    # Fallback: BM25 order + template response
    return session_idx, session_id, candidates[:topk], _FALLBACK_RESPONSE, True


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    import shutil
    if not shutil.which("claude"):
        raise RuntimeError(
            "claude CLI not found in PATH. The v9 inference requires `claude -p` "
            "for subscription-auth inference."
        )

    config = OmegaConf.load(f"config/{args.tid}.yaml")

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

    # Stage 1: BM25 retrieval
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

    # Stage 2: LLM rerank + response generation (combined)
    print(f"\nStage 2: Claude rerank + response ({len(sessions)} sessions, {N_WORKERS} threads)...")
    results = [None] * len(sessions)
    fallback_count = 0

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(
                claude_rerank_and_respond,
                i,
                s["session_id"],
                s["candidates"],
                s["conv_text"],
                s["cands_text"],
            ): i
            for i, s in enumerate(sessions)
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Reranking+Response"):
            try:
                idx, sid, ranked, response, is_fallback = f.result()
            except Exception as e:
                orig_idx = futures[f]
                print(f"  [ERROR] Thread for session index {orig_idx} crashed: {e}")
                s = sessions[orig_idx]
                idx, sid, ranked, response, is_fallback = (
                    orig_idx, s["session_id"], s["candidates"][:20], _FALLBACK_RESPONSE, True
                )
            results[idx] = (sid, ranked, response)
            if is_fallback:
                fallback_count += 1

    print(f"Fallbacks (BM25 order + template): {fallback_count}/{len(sessions)}")

    # Stage 3: Build predictions — guard against any None slots
    none_slots = [i for i, r in enumerate(results) if r is None]
    if none_slots:
        print(f"  [WARN] {len(none_slots)} result slots still None — applying fallback")
        for i in none_slots:
            s = sessions[i]
            results[i] = (s["session_id"], s["candidates"][:20], _FALLBACK_RESPONSE)

    predictions = []
    for i, (sid, ranked, response) in enumerate(results):
        sess = sessions[i]
        predictions.append({
            "session_id": sid,
            "user_id": sess["user_id"],
            "turn_number": sess["turn_number"],
            "predicted_track_ids": ranked,
            "predicted_response": response,
        })

    # Save
    os.makedirs("exp/inference/blind_a", exist_ok=True)
    out_json = "exp/inference/blind_a/echo_v9_reranked_with_response.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved predictions: {out_json}")

    out_zip = args.out
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")
    print(f"Submission zip: {out_zip}")
    print(f"\nDone. {len(predictions)} sessions.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tid", default="echo_bm25_cf_blind_a",
                        help="Config TID (looks up config/{tid}.yaml)")
    parser.add_argument("--out", default="/workspace/group/echo_v9_submission.zip",
                        help="Output zip path")
    args = parser.parse_args()
    main(args)
