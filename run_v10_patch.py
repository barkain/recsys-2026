"""v10: Targeted response regeneration for worst-scoring sessions.

Strategy:
  1. Load v9_judge_scores.json → identify worst sessions (avg ≤ threshold)
  2. Load v9 submission → get track IDs for those sessions
  3. Load blind-A dataset → get conversation context
  4. Load track metadata → translate IDs to readable track info
  5. Regenerate responses with an explanation-focused prompt that:
     - Quotes specific user preferences verbatim
     - Connects each recommended track to what the user said
     - Avoids generic filler
  6. Patch v9 submission with improved responses → save as v10

Usage:
    cd recsys-work
    HF_DATASETS_CACHE=/workspace/group/hf_cache \\
    uv run python3 run_v10_patch.py \\
        --scores /workspace/group/v9_judge_scores.json \\
        --base /workspace/group/echo_v9_submission.zip \\
        --out /workspace/group/echo_v10_submission.zip \\
        --threshold 3.0
"""
import argparse
import json
import os
import re
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from datasets import load_dataset

N_WORKERS = 8
CLAUDE_MODEL = "haiku"

# Stronger prompt: forces explicit track-to-preference connections
EXPLANATION_PROMPT = """\
You are a music recommendation assistant. Write a SHORT, warm, personalized recommendation message (3-5 sentences max).

**Conversation history (what the user said):**
{conversation}

**Tracks you are recommending (top 5):**
{tracks}

Rules:
- QUOTE or closely paraphrase what the user specifically said (e.g. "Since you love [their exact words]...")
- For EACH of the top 3 tracks, briefly explain WHY it matches — connect a specific track attribute (genre, mood, artist style, instruments) to something the user mentioned
- Do NOT use generic phrases like "based on your preferences" or "I think you'll enjoy" without explaining WHY
- Sound like a knowledgeable friend, not a robot
- No lists — write flowing sentences
- DO NOT include any JSON. Return ONLY plain text (the recommendation message).
"""


def load_blind_sessions(hf_cache):
    """Load blind-A sessions for conversation context."""
    if hf_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_cache
    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test")
    sessions = {}
    for item in db:
        sessions[item["session_id"]] = item
    return sessions


def load_metadata(hf_cache):
    """Load track metadata."""
    if hf_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_cache
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    combined = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    return {item["track_id"]: item for item in combined}


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


def format_conversation(session):
    """Format conversation history for the prompt."""
    convs = sorted(session["conversations"], key=lambda x: x["turn_number"])
    lines = []
    for msg in convs:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f'User: "{content}"')
        elif role == "music":
            lines.append(f"[Track played: {content}]")
        # skip assistant turns — keep prompt focused on user voice
    return "\n".join(lines) if lines else "(no prior conversation)"


def format_tracks_for_response(track_ids, metadata, n=5):
    """Format top tracks with rich metadata for the response prompt."""
    lines = []
    for i, tid in enumerate(track_ids[:n], 1):
        meta = metadata.get(tid, {})
        name = _resolve_track(meta, tid)
        artist = _resolve_artist(meta)
        tags = meta.get("tag_list", [])
        if isinstance(tags, list):
            tags_str = ", ".join(str(t) for t in tags[:6])
        else:
            tags_str = ""
        lines.append(f"{i}. \"{name}\" by {artist}" + (f" [{tags_str}]" if tags_str else ""))
    return "\n".join(lines)


def regenerate_response(session_id, conversation_text, tracks_text):
    """Call Claude to generate an improved explanation-focused response."""
    prompt = EXPLANATION_PROMPT.format(
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
        # Reject if it looks like JSON leaked through
        if text.startswith("{") or text.startswith("["):
            print(f"  [WARN] {session_id[:8]}: Got JSON instead of text response")
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
    parser.add_argument("--scores", default="/workspace/group/v9_judge_scores.json",
                        help="Per-session proxy judge scores JSON")
    parser.add_argument("--base", default="/workspace/group/echo_v9_submission.zip",
                        help="Base submission zip to patch (tracks kept, responses patched)")
    parser.add_argument("--out", default="/workspace/group/echo_v10_submission.zip",
                        help="Output submission zip")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Regenerate sessions with proxy avg <= this threshold")
    parser.add_argument("--hf-cache", default="/workspace/group/hf_cache")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show which sessions would be patched, don't call Claude")
    args = parser.parse_args()

    import shutil
    if not shutil.which("claude"):
        raise RuntimeError("claude CLI not found in PATH")

    # 1. Load proxy scores → identify target sessions
    with open(args.scores) as f:
        scores_data = json.load(f)
    sessions_scores = scores_data["sessions"]

    target_sessions = []
    for sid, v in sessions_scores.items():
        if v["personalization"] is None:
            # Scoring failed entirely — treat as worst possible score (0) so
            # these sessions are always included in the regeneration target list.
            avg = 0.0
            p, e, r = 0, 0, "scoring failed"
        else:
            p = v["personalization"]
            e = v["explanation"]
            r = v.get("reasoning", "")
            avg = (p + e) / 2
        if avg <= args.threshold:
            target_sessions.append((sid, avg, p, e, r))
    target_sessions.sort(key=lambda x: x[1])

    print(f"Sessions to regenerate (avg ≤ {args.threshold}): {len(target_sessions)}")
    for sid, avg, p, e, r in target_sessions:
        print(f"  {sid[:8]}: avg={avg:.2f} P={p:.1f} E={e:.1f} | {r[:80]}")

    if args.dry_run:
        print("\n[dry-run] Stopping here.")
        return

    # 2. Load base submission
    with zipfile.ZipFile(args.base) as zf:
        with zf.open("prediction.json") as f:
            predictions = json.load(f)
    pred_by_sid = {p["session_id"]: p for p in predictions}

    # 3. Load blind sessions + metadata
    os.environ["HF_DATASETS_CACHE"] = args.hf_cache
    print("\nLoading blind sessions and metadata...")
    sessions_db = load_blind_sessions(args.hf_cache)
    metadata = load_metadata(args.hf_cache)
    print(f"  {len(sessions_db)} sessions, {len(metadata)} tracks loaded")

    # 4. Build tasks for target sessions
    tasks = []
    for sid, avg, p, e, r in target_sessions:
        pred = pred_by_sid.get(sid)
        session = sessions_db.get(sid)
        if pred is None or session is None:
            print(f"  [WARN] {sid[:8]}: not found in base or blind-A, skipping")
            continue
        conv_text = format_conversation(session)
        tracks_text = format_tracks_for_response(pred["predicted_track_ids"], metadata)
        tasks.append((sid, conv_text, tracks_text, avg))

    # 5. Regenerate in parallel
    print(f"\nRegenerating {len(tasks)} responses ({N_WORKERS} workers)...")
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
    failures = [sid for sid, _, _, _ in tasks if sid not in improved]
    if failures:
        print(f"  Failed (keeping original): {[s[:8] for s in failures]}")

    # 6. Patch predictions
    patched = 0
    for pred in predictions:
        sid = pred["session_id"]
        if sid in improved:
            pred["predicted_response"] = improved[sid]
            patched += 1

    print(f"Patched {patched} sessions in submission")

    # 7. Validate — no duplicate track IDs
    dup_sessions = [
        p["session_id"] for p in predictions
        if len(p["predicted_track_ids"]) != len(set(p["predicted_track_ids"]))
    ]
    if dup_sessions:
        raise ValueError(f"Duplicate track IDs in {len(dup_sessions)} sessions: {dup_sessions}")

    # 8. Save
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    out_json = args.out.replace(".zip", ".json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"Saved patched JSON: {out_json}")

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")
    print(f"Submission zip: {args.out}")
    print(f"\nTotal: {len(predictions)} sessions ({patched} with improved responses)")


if __name__ == "__main__":
    main()
