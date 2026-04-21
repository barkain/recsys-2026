"""Local proxy for the Codabench LLM-as-a-Judge scorer.

The real judge is Gemini-based and evaluates:
  - Personalization (1-5): Does the response show understanding of the user's
    specific preferences expressed throughout the conversation?
  - Explanation Quality (1-5): Does the response explain WHY these tracks are
    recommended, connecting track attributes to user preferences?

We mimic this with Claude, scoring each session on the same rubric.
Use this to:
  1. Score all sessions in a submission → identify worst performers
  2. Iterate on responses offline before burning submission slots
  3. Calibrate against known blind scores: v6=3.55, v7=3.65, v8=3.25, v9=3.80

Usage:
    # Score a submission zip
    .venv/bin/python3.14 score_judge_proxy.py --zip /workspace/group/echo_v9_submission.zip

    # Score and dump per-session breakdown
    .venv/bin/python3.14 score_judge_proxy.py --zip /workspace/group/echo_v9_submission.zip --verbose

    # Compare two submissions
    .venv/bin/python3.14 score_judge_proxy.py --zip /workspace/group/echo_v9_submission.zip \\
                                               --compare /workspace/group/echo_v7_submission.zip
"""
import argparse
import json
import os
import re
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import pandas as pd
from datasets import load_dataset

N_WORKERS = 8
CLAUDE_MODEL = "haiku"

# Known blind scores for calibration
KNOWN_SCORES = {
    "v6": 3.55,
    "v7": 3.65,
    "v8": 3.25,
    "v9": 3.80,
}

JUDGE_PROMPT = """\
You are a music recommendation evaluator. Score this recommendation response on two criteria.

**Conversation history:**
{conversation}

**System response:**
{response}

**Recommended tracks (top 5 shown):**
{tracks}

Score on these two dimensions (each 1-5):

**Personalization (1-5):**
- 5: Response clearly demonstrates deep understanding of THIS user's specific preferences from the conversation. References specific things they said, adapts tone, shows it "gets" the user.
- 4: Good personalization, references conversation context meaningfully.
- 3: Some personalization but generic in parts.
- 2: Minimal personalization, mostly generic recommendation language.
- 1: Completely generic, could be copy-pasted to any user.

**Explanation Quality (1-5):**
- 5: Clearly explains WHY specific tracks match the user's stated preferences. Connects track attributes (genre, mood, artist style) to what user asked for.
- 4: Good explanations for most tracks recommended.
- 3: Some explanation but vague or incomplete.
- 2: Minimal explanation, just names tracks.
- 1: No explanation at all.

Return ONLY valid JSON:
{{"personalization": <1-5>, "explanation": <1-5>, "reasoning": "<one sentence>"}}
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
    """Load track metadata for formatting track names."""
    if hf_cache:
        os.environ["HF_DATASETS_CACHE"] = hf_cache
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    combined = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    return {item["track_id"]: item for item in combined}


def load_submission(zip_path):
    """Load predictions from a submission zip."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("prediction.json") as f:
            return json.load(f)


def format_conversation(session):
    """Format conversation history for the judge prompt."""
    convs = sorted(session["conversations"], key=lambda x: x["turn_number"])
    lines = []
    for msg in convs:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "music":
            lines.append(f"[Track played: {content}]")
        elif role == "assistant":
            lines.append(f"Assistant: {content[:100]}...")
    return "\n".join(lines) if lines else "(no history)"


def format_tracks(track_ids, metadata):
    """Format top-5 track names for the judge prompt."""
    lines = []
    for tid in track_ids[:5]:
        meta = metadata.get(tid, {})
        name = meta.get("track_name", tid)
        artist = meta.get("artist_name", "Unknown")
        tags = meta.get("tag_list", [])
        if isinstance(name, list):
            name = name[0] if name else tid
        if isinstance(artist, list):
            artist = ", ".join(str(a) for a in artist)
        if isinstance(tags, list):
            tags = ", ".join(str(t) for t in tags[:4])
        lines.append(f"- {name} by {artist} | {tags}")
    return "\n".join(lines)


def score_session(session_id, conversation_text, response, tracks_text):
    """Call Claude to score a single session. Returns (personalization, explanation, reasoning)."""
    prompt = JUDGE_PROMPT.format(
        conversation=conversation_text,
        response=response,
        tracks=tracks_text,
    )
    try:
        result = subprocess.run(
            ["claude", "-p", "--no-session-persistence", "--model", CLAUDE_MODEL],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return None, None, f"CLI error: {result.stderr[:100]}"
        text = result.stdout.strip()
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            p = float(data.get("personalization", 0))
            e = float(data.get("explanation", 0))
            r = str(data.get("reasoning", ""))
            if 1 <= p <= 5 and 1 <= e <= 5:
                return p, e, r
    except Exception as ex:
        return None, None, str(ex)
    return None, None, "parse error"


def score_submission(zip_path, sessions_db, metadata, verbose=False):
    """Score all sessions in a submission. Returns list of per-session results."""
    predictions = load_submission(zip_path)
    print(f"\nScoring {len(predictions)} sessions from {zip_path}...")

    tasks = []
    for pred in predictions:
        sid = pred["session_id"]
        session = sessions_db.get(sid)
        if session is None:
            continue
        conv_text = format_conversation(session)
        tracks_text = format_tracks(pred["predicted_track_ids"], metadata)
        response = pred.get("predicted_response", "")
        tasks.append((sid, conv_text, response, tracks_text))

    results = {}
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(score_session, sid, conv, resp, tracks): sid
            for sid, conv, resp, tracks in tasks
        }
        for f in tqdm(as_completed(futures), total=len(futures), desc="Scoring"):
            sid = futures[f]
            try:
                p, e, r = f.result()
            except Exception as ex:
                p, e, r = None, None, str(ex)
            results[sid] = {"personalization": p, "explanation": e, "reasoning": r}

    # Compute averages
    valid = [(v["personalization"], v["explanation"])
             for v in results.values()
             if v["personalization"] is not None]
    if not valid:
        print("No valid scores!")
        return results, None

    avg_p = sum(v[0] for v in valid) / len(valid)
    avg_e = sum(v[1] for v in valid) / len(valid)
    avg_combined = (avg_p + avg_e) / 2
    failed = len(results) - len(valid)

    print(f"\n{'='*50}")
    print(f"Proxy Judge Results")
    print(f"  Personalization avg : {avg_p:.3f}")
    print(f"  Explanation avg     : {avg_e:.3f}")
    print(f"  Combined avg        : {avg_combined:.3f}")
    print(f"  Scored              : {len(valid)}/{len(results)} (failed: {failed})")
    print(f"{'='*50}")
    print(f"\nCalibration reference (blind scores):")
    for ver, score in KNOWN_SCORES.items():
        print(f"  {ver}: {score}")

    if verbose:
        print(f"\n{'─'*60}")
        print(f"{'Session':40} {'P':>4} {'E':>4} {'Avg':>5}  Reasoning")
        print(f"{'─'*60}")
        # Sort by combined score ascending (worst first)
        sorted_results = sorted(
            [(sid, v) for sid, v in results.items() if v["personalization"] is not None],
            key=lambda x: (x[1]["personalization"] + x[1]["explanation"]) / 2
        )
        for sid, v in sorted_results:
            avg = (v["personalization"] + v["explanation"]) / 2
            print(f"{sid[:36]:36}  {v['personalization']:4.1f} {v['explanation']:4.1f} {avg:5.2f}  {v['reasoning'][:60]}")

    return results, avg_combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Submission zip to score")
    parser.add_argument("--compare", help="Optional second zip to compare against")
    parser.add_argument("--verbose", action="store_true", help="Show per-session breakdown")
    parser.add_argument("--out", help="Save per-session scores to JSON file")
    parser.add_argument("--hf-cache", default="/workspace/group/hf_cache")
    args = parser.parse_args()

    import shutil
    if not shutil.which("claude"):
        raise RuntimeError("claude CLI not found in PATH")

    os.environ["HF_DATASETS_CACHE"] = args.hf_cache

    print("Loading blind sessions and metadata...")
    sessions_db = load_blind_sessions(args.hf_cache)
    metadata = load_metadata(args.hf_cache)
    print(f"  {len(sessions_db)} sessions, {len(metadata)} tracks loaded")

    results_a, avg_a = score_submission(args.zip, sessions_db, metadata, verbose=args.verbose)

    if args.compare:
        print(f"\n{'='*50}")
        print(f"Comparing against: {args.compare}")
        results_b, avg_b = score_submission(args.compare, sessions_db, metadata, verbose=False)
        if avg_a is not None and avg_b is not None:
            delta = avg_a - avg_b
            print(f"\nDelta (primary vs compare): {delta:+.3f}")

    if args.out and results_a:
        with open(args.out, "w") as f:
            json.dump({
                "zip": args.zip,
                "avg_combined": avg_a,
                "sessions": results_a,
            }, f, indent=2)
        print(f"\nPer-session scores saved to {args.out}")


if __name__ == "__main__":
    main()
