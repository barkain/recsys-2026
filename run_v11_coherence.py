"""v11: Coherence-chain-of-thought response patch for 3.5-4.0 proxy sessions.

Strategy (per Codi's recommendation):
  - Tracks are already right (P=4.0) — don't touch retrieval
  - The problem is Explanation Quality (E=3.0): responses name tracks but
    don't articulate WHY they form a coherent set for THIS user
  - Fix: chain-of-thought prompt that forces the model to identify the
    musical thread connecting the tracks BEFORE writing the response

Target: 26 sessions with proxy avg in (3.0, 4.0] — untouched in v10,
        P=4.0 but E=3.0 dragging the average.

Usage:
    cd recsys-work
    HF_DATASETS_CACHE=/workspace/group/hf_cache \\
    uv run python3 run_v11_coherence.py \\
        --scores /workspace/group/v9_judge_scores.json \\
        --base /workspace/group/echo_v10_submission.zip \\
        --out /workspace/group/echo_v11_submission.zip \\
        --min-threshold 3.0 --max-threshold 4.0
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from datasets import load_dataset

N_WORKERS = 8
CLAUDE_MODEL = "haiku"

# Chain-of-thought coherence prompt — forces thread identification before writing
COHERENCE_PROMPT = """\
You are a music recommendation assistant. Your task is to write a SHORT, warm, personalized recommendation message (3-5 sentences).

**Conversation history (what the user said):**
{conversation}

**Recommended tracks (in order):**
{tracks}

**Instructions — follow in order:**

Step 1 — Musical thread (internal reasoning, do NOT include in output):
Identify the specific musical thread connecting these tracks for this user. Be precise: tempo arc, mood journey, production era, vocal style, genre relationship, or emotional progression. Every track should connect back to it.

Step 2 — Write the response:
- Open by naming the thread you found (e.g. "These tracks share a..." or "What ties these together is...")
- Connect at least 3 specific tracks back to the thread by name
- Reference something the user actually said (quote or closely paraphrase their words)
- 3-5 sentences max. No bullet points. Sound like a knowledgeable friend.
- Do NOT use generic phrases like "based on your preferences" or "I think you'll enjoy" without a specific reason

Return ONLY the response text (Step 2). Do not include Step 1 reasoning, JSON, or any other text.
"""


def load_blind_sessions(hf_cache):
    if hf_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_cache
    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test")
    return {item["session_id"]: item for item in db}


def load_metadata(hf_cache):
    if hf_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_cache
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    combined = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    return {item["track_id"]: item for item in combined}


def format_conversation(session):
    convs = sorted(session["conversations"], key=lambda x: x["turn_number"])
    lines = []
    for msg in convs:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f'User: "{content}"')
        elif role == "music":
            lines.append(f"[Track played: {content}]")
    return "\n".join(lines) if lines else "(no prior conversation)"


def format_tracks(track_ids, metadata, n=10):
    lines = []
    for i, tid in enumerate(track_ids[:n], 1):
        meta = metadata.get(tid, {})
        name = meta.get("track_name", tid)
        artist = meta.get("artist_name", "Unknown")
        tags = meta.get("tag_list", [])
        if isinstance(name, list):
            name = name[0] if name else tid
        if isinstance(artist, list):
            artist = ", ".join(str(a) for a in artist)
        if isinstance(tags, list):
            tags_str = ", ".join(str(t) for t in tags[:5])
        else:
            tags_str = ""
        lines.append(f"{i}. \"{name}\" by {artist}" + (f" [{tags_str}]" if tags_str else ""))
    return "\n".join(lines)


def regenerate_response(session_id, conversation_text, tracks_text):
    prompt = COHERENCE_PROMPT.format(
        conversation=conversation_text,
        tracks=tracks_text,
    )
    try:
        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence", "--model", CLAUDE_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if result.returncode != 0:
            print(f"  [WARN] {session_id[:8]}: CLI error: {result.stderr[:100]}")
            return None
        text = result.stdout.strip()
        if text.startswith("{") or text.startswith("["):
            print(f"  [WARN] {session_id[:8]}: JSON leaked into response")
            return None
        if len(text) < 50:
            print(f"  [WARN] {session_id[:8]}: Response too short ({len(text)} chars)")
            return None
        return text
    except Exception as ex:
        print(f"  [WARN] {session_id[:8]}: {ex}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="/workspace/group/v9_judge_scores.json")
    parser.add_argument("--base", default="/workspace/group/echo_v10_submission.zip",
                        help="Base submission (v10) — tracks kept, responses patched")
    parser.add_argument("--out", default="/workspace/group/echo_v11_submission.zip")
    parser.add_argument("--min-threshold", type=float, default=3.0,
                        help="Patch sessions with proxy avg > this (exclusive)")
    parser.add_argument("--max-threshold", type=float, default=4.0,
                        help="Patch sessions with proxy avg <= this (inclusive)")
    parser.add_argument("--hf-cache", default="/workspace/group/hf_cache")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not shutil.which("claude"):
        raise RuntimeError("claude CLI not found in PATH")

    # 1. Load proxy scores → identify target sessions
    with open(args.scores) as f:
        scores_data = json.load(f)

    target_sessions = []
    for sid, v in scores_data["sessions"].items():
        if v["personalization"] is None:
            avg = 0.0
            p, e, r = 0, 0, "scoring failed"
        else:
            p, e = v["personalization"], v["explanation"]
            r = v.get("reasoning", "")
            avg = (p + e) / 2
        if args.min_threshold < avg <= args.max_threshold:
            target_sessions.append((sid, avg, p, e, r))
    target_sessions.sort(key=lambda x: x[1])

    print(f"Sessions to patch (avg in ({args.min_threshold}, {args.max_threshold}]): {len(target_sessions)}")
    for sid, avg, p, e, r in target_sessions:
        print(f"  {sid[:8]}: avg={avg:.2f} P={p:.1f} E={e:.1f} | {r[:80]}")

    if args.dry_run:
        print("\n[dry-run] Stopping here.")
        return

    # 2. Load base submission (v10)
    with zipfile.ZipFile(args.base) as zf:
        with zf.open("prediction.json") as f:
            predictions = json.load(f)
    pred_by_sid = {p["session_id"]: p for p in predictions}

    # 3. Load blind sessions + metadata
    if args.hf_cache:
        os.environ["HF_DATASETS_CACHE"] = args.hf_cache
    print("\nLoading blind sessions and metadata...")
    sessions_db = load_blind_sessions(args.hf_cache)
    metadata = load_metadata(args.hf_cache)
    print(f"  {len(sessions_db)} sessions, {len(metadata)} tracks loaded")

    # 4. Build tasks
    tasks = []
    for sid, avg, p, e, r in target_sessions:
        pred = pred_by_sid.get(sid)
        session = sessions_db.get(sid)
        if pred is None or session is None:
            print(f"  [WARN] {sid[:8]}: not found in base or blind-A, skipping")
            continue
        conv_text = format_conversation(session)
        tracks_text = format_tracks(pred["predicted_track_ids"], metadata)
        tasks.append((sid, conv_text, tracks_text, avg))

    # 5. Regenerate in parallel
    print(f"\nRegenerating {len(tasks)} responses ({N_WORKERS} workers, coherence-CoT prompt)...")
    improved = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(regenerate_response, sid, conv, tracks): sid
            for sid, conv, tracks, _ in tasks
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Regenerating"):
            sid = futures[f]
            try:
                response = f.result()
            except Exception as ex:
                response = None
                print(f"  [WARN] {sid[:8]}: thread error: {ex}")
            if response:
                improved[sid] = response

    print(f"\nSuccessfully regenerated: {len(improved)}/{len(tasks)}")
    failures = [sid for sid, *_ in tasks if sid not in improved]
    if failures:
        print(f"  Failed (keeping v10 response): {[s[:8] for s in failures]}")

    # 6. Patch
    patched = 0
    for pred in predictions:
        sid = pred["session_id"]
        if sid in improved:
            pred["predicted_response"] = improved[sid]
            patched += 1

    print(f"Patched {patched} sessions")

    # 7. Validate
    dup_sessions = [p["session_id"] for p in predictions
                    if len(p["predicted_track_ids"]) != len(set(p["predicted_track_ids"]))]
    if dup_sessions:
        raise ValueError(f"Duplicate track IDs in {len(dup_sessions)} sessions: {dup_sessions}")
    short_sessions = [(p["session_id"][:8], len(p["predicted_track_ids"])) for p in predictions
                      if len(p["predicted_track_ids"]) < 20]
    if short_sessions:
        print(f"[WARN] Short track lists: {short_sessions}")
    print(f"Validation passed: {len(predictions)} sessions, all unique track IDs.")

    # 8. Save
    out_json = args.out.replace(".zip", ".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")
    print(f"Submission zip: {args.out}")
    print(f"Total: {len(predictions)} sessions ({patched} with new coherence responses)")


if __name__ == "__main__":
    main()
