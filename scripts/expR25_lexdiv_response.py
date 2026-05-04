#!/usr/bin/env python3
# ruff: noqa: T201
"""R25: Response-only LexDiv improvement.

Regenerates responses for the R21 submission with a diversity-aware prompt.
Track IDs are frozen — only the predicted_response field changes.

Usage:
    # Generate new responses (costs ~80 Haiku calls)
    uv run python3 scripts/expR25_lexdiv_response.py --generate

    # Validate artifact matches R21 track IDs
    uv run python3 scripts/expR25_lexdiv_response.py --validate

    # Compute LexDiv proxy comparison
    uv run python3 scripts/expR25_lexdiv_response.py --compare

    # All steps
    uv run python3 scripts/expR25_lexdiv_response.py --generate --validate --compare
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R21_ZIP = REPO / "exp" / "inference" / "blind_a" / "lr_r21_v1_hybrid_submission.zip"
OUT_DIR = REPO / "exp" / "inference" / "blind_a"
OUT_TAG = "r25_lexdiv_v2"

DIVERSE_SYSTEM_PROMPT = """\
You are an expert music recommendation assistant with deep knowledge of genres, \
artists, moods, and music history.

RULES — follow every one:
1. FIRST SENTENCE must address the user directly — reference what THEY asked for, \
their mood, their request, or what they said. Use "you" in sentence one.
2. SECOND SENTENCE names the track (title and artist) and connects ONE specific \
musical attribute (tempo, production technique, instrumentation, vocal style, era) \
to the user's request.
3. DO NOT end with a question. No "are you looking for...?", "would you prefer...?", \
"do you want...?", "or would you...?", "what do you think?". End with a statement.
4. DO NOT use these filler phrases: "something with", "that same", "got that", \
"you're after", "fans of", "you might enjoy", "or are you", "would you prefer", \
"or would you"
5. Keep it 2-3 sentences. Be concise and vivid.
6. Use music-specific vocabulary. Each response must feel distinct from every other.
"""

STYLE_VARIANTS = [
    "After addressing the user, highlight a specific sonic texture of the track.",
    "After addressing the user, connect the track to a cultural moment or era.",
    "After addressing the user, describe the physical feeling the track creates.",
    "After addressing the user, contrast this track with something they already heard.",
    "After addressing the user, name a specific production technique in the track.",
    "After addressing the user, describe the emotional arc the track builds.",
    "After addressing the user, place the track in its genre lineage.",
    "After addressing the user, highlight the vocal delivery or instrumental tone.",
]

BANNED_WORDS = {
    "suggest", "recommendation", "enjoy", "vibe", "vibing", "vibes",
    "fans", "lovers", "absolutely", "certainly", "definitely",
    "perfect", "awesome", "great choice", "good pick",
}


def load_r21_predictions():
    import zipfile
    with zipfile.ZipFile(R21_ZIP) as zf:
        with zf.open("prediction.json") as f:
            return json.load(f)


def load_blind_sessions():
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Blind-A",
        split="test",
        download_config=DownloadConfig(local_files_only=True),
    )
    return {str(item["session_id"]): item for item in ds}


def load_track_metadata():
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        download_config=DownloadConfig(local_files_only=True),
    )
    combined = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    return {str(item["track_id"]): item for item in combined}


def format_conversation(session_item) -> str:
    convs = sorted(session_item["conversations"], key=lambda x: x["turn_number"])
    lines = []
    for msg in convs:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "music":
            lines.append(f"[Track played: {content}]")
        elif role == "assistant":
            lines.append(f"Assistant: {content[:150]}...")
    return "\n".join(lines) if lines else "(no history)"


def format_track_info(track_id: str, metadata: dict) -> str:
    meta = metadata.get(track_id, {})
    name = meta.get("track_name", track_id)
    artist = meta.get("artist_name", "Unknown")
    tags = meta.get("tag_list", [])
    if isinstance(name, list):
        name = name[0] if name else track_id
    if isinstance(artist, list):
        artist = ", ".join(str(a) for a in artist)
    tag_str = ""
    if isinstance(tags, list) and tags:
        tag_str = " | Tags: " + ", ".join(str(t) for t in tags[:6])
    return f'"{name}" by {artist}{tag_str}'


def generate_response(
    sys_prompt: str,
    conversation: str,
    track_info: str,
    user_query: str,
    style_hint: str,
    idx: int,
) -> str:
    from mcrs.utils import call_llm_api

    user_msg = (
        f"Conversation so far:\n{conversation}\n\n"
        f"User's latest message: {user_query}\n\n"
        f"You are recommending: {track_info}\n\n"
        f"Style instruction: {style_hint}\n\n"
        "Write your response (2-4 sentences, flowing prose, vivid and specific):"
    )

    result = call_llm_api(
        sys_prompt,
        [{"role": "user", "content": user_msg}],
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
    )
    return (result or "").strip().lstrip(",").strip()


def lexical_diversity(responses: list[str], n: int = 2) -> float:
    """Distinct-n: unique n-grams / total n-grams across all responses.
    Matches the competition's metrics_diversity.py exactly."""
    unique_ngrams: set[tuple[str, ...]] = set()
    total_ngrams = 0
    for r in responses:
        if not r:
            continue
        tokens = r.lower().split()
        for i in range(len(tokens) - n + 1):
            ng = tuple(tokens[i : i + n])
            unique_ngrams.add(ng)
            total_ngrams += 1
    return len(unique_ngrams) / total_ngrams if total_ngrams else 0.0


def validate_artifact(new_preds: list[dict], r21_preds: list[dict]) -> dict:
    issues = []

    if len(new_preds) != 80:
        issues.append(f"Row count: {len(new_preds)} (expected 80)")

    if len(new_preds) != len(r21_preds):
        issues.append(f"Row count mismatch: {len(new_preds)} vs R21's {len(r21_preds)}")
        return {"valid": False, "issues": issues}

    for i, (new, r21) in enumerate(zip(new_preds, r21_preds)):
        if new["session_id"] != r21["session_id"]:
            issues.append(f"Row {i}: session_id mismatch")
        if new["turn_number"] != r21["turn_number"]:
            issues.append(f"Row {i}: turn_number mismatch")
        if new["predicted_track_ids"] != r21["predicted_track_ids"]:
            issues.append(f"Row {i}: track IDs differ!")

        tids = new["predicted_track_ids"]
        if len(tids) != 20:
            issues.append(f"Row {i}: {len(tids)} track IDs (expected 20)")
        if len(set(tids)) != len(tids):
            issues.append(f"Row {i}: duplicate track IDs")

        resp = new.get("predicted_response", "")
        if not resp.strip():
            issues.append(f"Row {i}: empty response")
        if resp.startswith(","):
            issues.append(f"Row {i}: comma prefix")
        if any(phrase in resp.lower() for phrase in ["i apologize", "i'm sorry", "i cannot", "unfortunately"]):
            issues.append(f"Row {i}: apology/uncertainty boilerplate")
        if len(resp) < 50:
            issues.append(f"Row {i}: suspiciously short response ({len(resp)} chars)")
        if len(resp) > 1500:
            issues.append(f"Row {i}: overly long response ({len(resp)} chars)")

    return {"valid": len(issues) == 0, "issues": issues}


def do_generate():
    print("Loading R21 predictions...")
    r21_preds = load_r21_predictions()
    print(f"  {len(r21_preds)} rows")

    print("Loading blind sessions...")
    sessions = load_blind_sessions()
    print(f"  {len(sessions)} sessions")

    print("Loading track metadata...")
    metadata = load_track_metadata()
    print(f"  {len(metadata)} tracks")

    # Disable LLM cache for fresh responses
    os.environ["MCRS_LLM_CACHE_DISABLE"] = "1"

    new_preds = []
    t0 = time.time()
    for i, pred in enumerate(r21_preds):
        sid = pred["session_id"]
        session = sessions.get(sid)
        if session is None:
            print(f"  WARNING: session {sid} not found in blind dataset")
            new_preds.append(pred)
            continue

        top1_id = pred["predicted_track_ids"][0]
        track_info = format_track_info(top1_id, metadata)
        conversation = format_conversation(session)

        convs = sorted(session["conversations"], key=lambda x: x["turn_number"])
        user_query = ""
        for msg in reversed(convs):
            if msg["role"] == "user":
                user_query = msg["content"]
                break

        style_hint = STYLE_VARIANTS[i % len(STYLE_VARIANTS)]

        resp = generate_response(
            DIVERSE_SYSTEM_PROMPT,
            conversation,
            track_info,
            user_query,
            style_hint,
            i,
        )

        new_pred = {
            "session_id": pred["session_id"],
            "user_id": pred["user_id"],
            "turn_number": pred["turn_number"],
            "predicted_track_ids": pred["predicted_track_ids"],
            "predicted_response": resp,
        }
        new_preds.append(new_pred)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (80 - i - 1) / rate if rate > 0 else 0
        print(f"  [{i+1:2d}/80] {elapsed:.0f}s  ETA {eta:.0f}s  len={len(resp):3d}  {resp[:80]}...")

    out_json = OUT_DIR / f"{OUT_TAG}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(new_preds, f, indent=2)
    print(f"\nSaved: {out_json}")

    out_zip = OUT_DIR / f"{OUT_TAG}_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", json.dumps(new_preds, indent=2))
    print(f"Saved: {out_zip}")

    return new_preds


def do_validate():
    r21_preds = load_r21_predictions()
    out_json = OUT_DIR / f"{OUT_TAG}.json"
    if not out_json.exists():
        print(f"ERROR: {out_json} not found. Run --generate first.")
        return False

    with open(out_json) as f:
        new_preds = json.load(f)

    result = validate_artifact(new_preds, r21_preds)
    if result["valid"]:
        print("VALIDATION PASSED: all checks OK")
    else:
        print(f"VALIDATION FAILED: {len(result['issues'])} issues")
        for issue in result["issues"][:20]:
            print(f"  - {issue}")
    return result["valid"]


def do_compare():
    r21_preds = load_r21_predictions()
    out_json = OUT_DIR / f"{OUT_TAG}.json"
    if not out_json.exists():
        print(f"ERROR: {out_json} not found. Run --generate first.")
        return

    with open(out_json) as f:
        new_preds = json.load(f)

    r21_responses = [p["predicted_response"] for p in r21_preds]
    new_responses = [p["predicted_response"] for p in new_preds]

    ld_r21 = lexical_diversity(r21_responses)
    ld_new = lexical_diversity(new_responses)

    print(f"\n{'='*50}")
    print("LexDiv Proxy Comparison")
    print(f"{'='*50}")
    print(f"  R21 (baseline) : {ld_r21:.4f}")
    print(f"  R25 (new)      : {ld_new:.4f}")
    print(f"  Delta          : {ld_new - ld_r21:+.4f}")
    print()

    # Per-response stats
    r21_lens = [len(r) for r in r21_responses]
    new_lens = [len(r) for r in new_responses]
    print(f"  R21 response lengths: min={min(r21_lens)} max={max(r21_lens)} mean={sum(r21_lens)/len(r21_lens):.0f}")
    print(f"  R25 response lengths: min={min(new_lens)} max={max(new_lens)} mean={sum(new_lens)/len(new_lens):.0f}")

    # Vocabulary overlap analysis
    r21_vocab = set()
    new_vocab = set()
    for r in r21_responses:
        r21_vocab.update(r.lower().split())
    for r in new_responses:
        new_vocab.update(r.lower().split())
    shared = r21_vocab & new_vocab
    print(f"\n  R21 unique words: {len(r21_vocab)}")
    print(f"  R25 unique words: {len(new_vocab)}")
    print(f"  Shared words    : {len(shared)}")
    print(f"  R25 new words   : {len(new_vocab - r21_vocab)}")

    # Check banned word usage
    banned_count = 0
    for r in new_responses:
        r_lower = r.lower()
        for bw in BANNED_WORDS:
            if bw in r_lower:
                banned_count += 1
                break
    print(f"\n  R25 responses with banned words: {banned_count}/80")

    # Opening word diversity
    from collections import Counter
    r21_openers = Counter()
    new_openers = Counter()
    for r in r21_responses:
        w = r.strip().split()[0] if r.strip() else ""
        r21_openers[w] += 1
    for r in new_responses:
        w = r.strip().split()[0] if r.strip() else ""
        new_openers[w] += 1
    print(f"\n  R21 unique openers: {len(r21_openers)}")
    print(f"  R25 unique openers: {len(new_openers)}")


def main():
    parser = argparse.ArgumentParser(description="R25: LexDiv response improvement")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    if not any([args.generate, args.validate, args.compare]):
        parser.print_help()
        return

    if args.generate:
        do_generate()
    if args.validate:
        do_validate()
    if args.compare:
        do_compare()


if __name__ == "__main__":
    main()
