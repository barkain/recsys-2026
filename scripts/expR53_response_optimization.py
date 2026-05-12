#!/usr/bin/env python3
# ruff: noqa: T201
"""R53: Response-only optimization targeting LLM judge rubric + LexDiv.

Produces TWO submission artifacts from R39 album submission:
  - R53a (LLM-focused): maximize personalization + explanation quality
  - R53b (diverse): same quality prompt + stronger lexical diversity constraints

Track IDs are FROZEN from R39. Only predicted_response changes.

Usage:
    .venv/bin/python scripts/expR53_response_optimization.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R39_JSON = REPO / "exp" / "inference" / "blind_a" / "r39_album_submission.json"
OUT_DIR = REPO / "exp" / "inference" / "blind_a"
EVAL_DIR = REPO / "exp" / "eval"

# ── System prompt: LLM-focused (R53a) ─────────────────────────────────────

SYSTEM_PROMPT_LLM = """\
You are an expert music recommendation assistant. Write a recommendation \
response in exactly 2-3 sentences (70-110 words total, flowing prose).

STRICT RULES:

1. OPEN by echoing the user's own words or request. The first clause must \
reference what THIS user said, asked for, or described. Use "you" in the \
opening phrase.

2. NAME the recommended track (title and artist) and state ONE concrete \
musical attribute (tempo, vocal delivery, production technique, \
instrumentation, rhythmic pattern, harmonic texture, or sonic era). \
TIE that attribute to what the user asked for using their exact phrasing.

3. BANNED phrases (score zero): "you'll love" / "you'll enjoy" / \
"perfect for" / "great fit" / "ideal for" / "fans of" / "lovers of" / \
"if you like" / "I think you'll" / "you might" / "something with" / \
"got that" / "great choice" / "awesome" / "absolutely" / "check out" / \
"give this a listen" / "Perfect" / "Great choice" / "Absolutely".

4. Address the user FIRST. Never open with "This track...", the track \
name, or track-centric framing.

5. End with a statement, NOT a question.

6. No bullet points. No apologies or refusals. Only use facts from the \
provided metadata/tags.
"""

# ── System prompt: Diverse (R53b) ─────────────────────────────────────────

SYSTEM_PROMPT_DIVERSE = SYSTEM_PROMPT_LLM + """
DIVERSITY RULES (additional):
7. Vary your sentence structure: sometimes start with a gerund, sometimes \
a prepositional phrase, sometimes a dependent clause, sometimes a noun phrase.
8. Use music-specific vocabulary: groove, timbre, reverb-soaked, propulsive, \
anthemic, lo-fi, hypnotic, bittersweet, sun-drenched, frenetic, understated, \
lush, gritty, sparse, euphoric, melancholic, raw, cinematic, cavernous, \
finger-picked, layered, pulsing, angular, shimmering, distorted, intimate.
9. Never repeat the same opening construction you used in a previous response.
"""

# Style variants to rotate through
STYLE_VARIANTS = [
    "Describe the track's sonic texture.",
    "Reference a production technique.",
    "Describe the physical sensation the music creates.",
    "Place the track in its era or lineage.",
    "Highlight the vocal delivery.",
    "Focus on rhythmic character.",
    "Describe the emotional arc.",
    "Name the instrumentation.",
    "Contrast with a sonic opposite.",
    "Describe the spatial quality of the sound.",
]

BANNED_OPENERS = {
    "i", "this", "here", "the", "what", "these", "i'd", "i've",
    "let", "check", "listen", "try", "how", "for",
}

BOILERPLATE_STARTS = ["perfect", "great choice", "awesome", "absolutely", "certainly"]

BANNED_PHRASES_LIST = [
    "you'll love", "you'll enjoy", "perfect for", "great fit",
    "fans of", "lovers of", "if you like", "great choice",
    "check out", "give this a listen", "you might", "i apologize",
    "i'm sorry", "i cannot", "unfortunately",
]


def load_r39_predictions() -> list[dict]:
    with open(R39_JSON) as f:
        return json.load(f)


def load_blind_sessions() -> dict:
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]

    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Blind-A",
        split="test",
        download_config=DownloadConfig(local_files_only=True),
    )
    return {str(item["session_id"]): item for item in ds}


def load_track_metadata() -> dict:
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]

    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        download_config=DownloadConfig(local_files_only=True),
    )
    combined = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    return {str(item["track_id"]): item for item in combined}


def format_conversation(session_item) -> tuple[str, str]:
    """Return (full_conversation_text, last_user_query)."""
    convs = sorted(session_item["conversations"], key=lambda x: x["turn_number"])
    lines = []
    last_user_query = ""
    for msg in convs:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            lines.append(f"User: {content}")
            last_user_query = content
        elif role == "music":
            lines.append(f"[Track played: {content}]")
        elif role == "assistant":
            lines.append(f"Assistant: {content[:200]}...")
    return "\n".join(lines) if lines else "(no history)", last_user_query


def format_track_info(track_id: str, metadata: dict) -> str:
    meta = metadata.get(track_id, {})
    name = meta.get("track_name", track_id)
    artist = meta.get("artist_name", "Unknown")
    tags = meta.get("tag_list", [])
    album = meta.get("album_name", "")
    if isinstance(name, list):
        name = name[0] if name else track_id
    if isinstance(artist, list):
        artist = ", ".join(str(a) for a in artist)
    if isinstance(album, list):
        album = album[0] if album else ""
    tag_str = ""
    if isinstance(tags, list) and tags:
        tag_str = " | Tags: " + ", ".join(str(t) for t in tags[:6])
    album_str = f' | Album: "{album}"' if album else ""
    return f'"{name}" by {artist}{album_str}{tag_str}'


def lexical_diversity(responses: list[str], n: int = 2) -> float:
    """Distinct-n: unique n-grams / total n-grams across all responses."""
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


def word_count(text: str) -> int:
    return len(text.split())


def generate_response(
    system_prompt: str,
    conversation: str,
    user_query: str,
    track_info: str,
    style_hint: str,
    avoid_openers: list[str] | None = None,
) -> str:
    """Generate a response using the Anthropic SDK directly."""
    import anthropic  # type: ignore[reportMissingImports]

    avoid_str = ""
    if avoid_openers:
        recent = avoid_openers[-15:]
        avoid_str = (
            "\n\nDIVERSITY: Do NOT start your response with any of these "
            "words/phrases (already used in other responses): "
            + ", ".join(repr(o) for o in recent)
            + ". Find a fresh opening construction."
        )

    user_msg = (
        f"Conversation history:\n{conversation}\n\n"
        f'User\'s latest message: "{user_query}"\n\n'
        f"You are recommending this track: {track_info}\n\n"
        f"Style focus: {style_hint}"
        f"{avoid_str}\n\n"
        "Write your recommendation response (70-110 words, 2-3 sentences):"
    )

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    )
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    text = message.content[0].text if message.content else ""
    return (text or "").strip().lstrip(",").strip()


def extract_opener(response: str) -> str:
    """Extract the first 2-3 words as the opener fingerprint."""
    words = response.strip().split()
    if len(words) >= 3:
        return " ".join(words[:3]).lower().rstrip(".,!:")
    elif words:
        return " ".join(words).lower().rstrip(".,!:")
    return ""


def validate_submission(new_preds: list[dict], r39_preds: list[dict]) -> list[str]:
    """Validate a submission against R39. Returns list of issues."""
    issues = []

    if len(new_preds) != 80:
        issues.append(f"Row count: {len(new_preds)} (expected 80)")

    if len(new_preds) != len(r39_preds):
        issues.append(
            f"Row count mismatch: {len(new_preds)} vs R39's {len(r39_preds)}"
        )
        return issues

    for i, (new, r39) in enumerate(zip(new_preds, r39_preds)):
        if new["session_id"] != r39["session_id"]:
            issues.append(f"Row {i}: session_id mismatch")
        if new["turn_number"] != r39["turn_number"]:
            issues.append(f"Row {i}: turn_number mismatch")
        if new["predicted_track_ids"] != r39["predicted_track_ids"]:
            issues.append(f"Row {i}: track IDs differ!")

        resp = new.get("predicted_response", "")
        if not resp.strip():
            issues.append(f"Row {i}: empty response")
        if resp.startswith(","):
            issues.append(f"Row {i}: comma prefix")
        r_lower = resp.lower()
        for bp in BOILERPLATE_STARTS:
            if r_lower.startswith(bp):
                issues.append(f"Row {i}: boilerplate opener '{bp}'")
    return issues


def generate_variant(
    label: str,
    system_prompt: str,
    r39_preds: list[dict],
    sessions: dict,
    metadata: dict,
    use_diversity_avoidance: bool,
) -> tuple[list[dict], float]:
    """Generate 80 responses for one variant. Returns (preds, gen_time)."""
    print(f"\n--- Generating {label} (80 responses) ---")
    new_preds = []
    used_openers: list[str] = []
    t0 = time.time()
    failures = 0

    for i, pred in enumerate(r39_preds):
        sid = pred["session_id"]
        session = sessions.get(sid)
        if session is None:
            print(f"  WARNING: session {sid} not found - keeping R39 response")
            new_preds.append(dict(pred))
            continue

        top1_id = pred["predicted_track_ids"][0]
        track_info = format_track_info(top1_id, metadata)
        conversation, user_query = format_conversation(session)

        if not user_query:
            user_query = "(user initiated conversation)"

        style_hint = STYLE_VARIANTS[i % len(STYLE_VARIANTS)]
        avoid = used_openers if use_diversity_avoidance else None

        resp = ""
        for attempt in range(3):
            try:
                resp = generate_response(
                    system_prompt,
                    conversation,
                    user_query,
                    track_info,
                    style_hint,
                    avoid,
                )
            except Exception as e:
                print(
                    f"  API error attempt {attempt + 1} "
                    f"for session {sid}: {e}"
                )
                failures += 1
                time.sleep(2)
                continue

            opener = extract_opener(resp)
            first_word = opener.split()[0] if opener.split() else ""
            if first_word in BANNED_OPENERS and attempt < 2:
                time.sleep(0.3)
                continue
            break

        used_openers.append(extract_opener(resp))

        new_pred = {
            "session_id": pred["session_id"],
            "turn_number": pred["turn_number"],
            "predicted_track_ids": pred["predicted_track_ids"],
            "predicted_response": resp,
        }
        if "user_id" in pred:
            new_pred["user_id"] = pred["user_id"]

        new_preds.append(new_pred)

        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (len(r39_preds) - i - 1) / rate if rate > 0 else 0
        print(
            f"  [{i + 1:2d}/{len(r39_preds)}] {elapsed:.0f}s "
            f"ETA {eta:.0f}s  wc={word_count(resp):3d}  {resp[:70]}..."
        )

        time.sleep(0.5)

    gen_time = time.time() - t0
    print(f"  Done: {gen_time:.0f}s ({failures} API failures)")
    return new_preds, gen_time


def compute_metrics(
    label: str,
    new_preds: list[dict],
    r39_preds: list[dict],
) -> dict:
    """Compute quality and diversity metrics for a variant."""
    r39_responses = [p["predicted_response"] for p in r39_preds]
    new_responses = [p["predicted_response"] for p in new_preds]

    lexdiv_r39 = lexical_diversity(r39_responses)
    lexdiv_new = lexical_diversity(new_responses)

    r39_lens = [len(r) for r in r39_responses if r]
    new_lens = [len(r) for r in new_responses if r]
    avg_len_r39 = sum(r39_lens) / len(r39_lens) if r39_lens else 0
    avg_len_new = sum(new_lens) / len(new_lens) if new_lens else 0

    r39_wcs = [word_count(r) for r in r39_responses if r]
    new_wcs = [word_count(r) for r in new_responses if r]
    avg_wc_r39 = sum(r39_wcs) / len(r39_wcs) if r39_wcs else 0
    avg_wc_new = sum(new_wcs) / len(new_wcs) if new_wcs else 0

    # Opener diversity
    openers = [extract_opener(r) for r in new_responses]
    unique_openers = len(set(openers))

    # First-word analysis
    first_words_new = Counter()
    for r in new_responses:
        w = r.strip().split()[0].lower() if r.strip() else ""
        first_words_new[w] += 1

    # User-first check
    user_first_count = 0
    for r in new_responses:
        r_lower = r.strip().lower()
        if any(
            r_lower.startswith(p)
            for p in [
                "you", "your", "since you", "that", "when you",
                "given your", "for that", "with your",
            ]
        ):
            user_first_count += 1

    # Banned phrase count
    banned_count = 0
    for r in new_responses:
        r_lower = r.lower()
        if any(bp in r_lower for bp in BANNED_PHRASES_LIST):
            banned_count += 1

    # Vocab stats
    r39_vocab = set()
    new_vocab = set()
    for r in r39_responses:
        r39_vocab.update(r.lower().split())
    for r in new_responses:
        new_vocab.update(r.lower().split())

    return {
        "label": label,
        "lexdiv_new": round(lexdiv_new, 4),
        "lexdiv_r39": round(lexdiv_r39, 4),
        "lexdiv_delta": round(lexdiv_new - lexdiv_r39, 4),
        "avg_length_new": round(avg_len_new, 1),
        "avg_length_r39": round(avg_len_r39, 1),
        "avg_wordcount_new": round(avg_wc_new, 1),
        "avg_wordcount_r39": round(avg_wc_r39, 1),
        "unique_openers": unique_openers,
        "unique_first_words": len(first_words_new),
        "user_first_ratio": round(user_first_count / max(len(new_responses), 1), 3),
        "banned_phrase_count": banned_count,
        "vocab_size_new": len(new_vocab),
        "vocab_size_r39": len(r39_vocab),
    }


def print_metrics(m: dict) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {m['label']} Results")
    print(sep)
    print("  Model              : claude-sonnet-4-6")
    print("")
    print("  LexDiv (Distinct-2):")
    print(f"    R39 (baseline)   : {m['lexdiv_r39']:.4f}")
    print(f"    New              : {m['lexdiv_new']:.4f}")
    print(f"    Delta            : {m['lexdiv_delta']:+.4f}")
    print("")
    print("  Response lengths:")
    print(f"    R39 avg chars    : {m['avg_length_r39']:.0f}")
    print(f"    New avg chars    : {m['avg_length_new']:.0f}")
    print(f"    R39 avg words    : {m['avg_wordcount_r39']:.0f}")
    print(f"    New avg words    : {m['avg_wordcount_new']:.0f}")
    print("")
    print("  Quality indicators:")
    print(f"    Unique openers   : {m['unique_openers']}/80")
    print(f"    User-first ratio : {m['user_first_ratio']:.3f}")
    print(f"    Banned phrases   : {m['banned_phrase_count']}/80")
    print(f"    Unique 1st words : {m['unique_first_words']}")
    print(f"    Vocab size       : {m['vocab_size_new']} (R39: {m['vocab_size_r39']})")
    print(sep)


def main():
    # ── Pre-flight checks ──────────────────────────────────────────────
    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    )
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY / ANTHROPIC_RECSYS_API_KEY not set.")
        sys.exit(1)

    if not R39_JSON.exists():
        print(f"ERROR: R39 submission not found at {R39_JSON}")
        sys.exit(1)

    # Disable LLM cache for fresh responses
    os.environ["MCRS_DISABLE_LLM_CACHE"] = "1"

    # ── Load data ──────────────────────────────────────────────────────
    print("Loading R39 predictions...")
    r39_preds = load_r39_predictions()
    print(f"  {len(r39_preds)} rows")

    print("Loading blind sessions...")
    sessions = load_blind_sessions()
    print(f"  {len(sessions)} sessions")

    print("Loading track metadata...")
    metadata = load_track_metadata()
    print(f"  {len(metadata)} tracks")

    # ── Variant A: LLM-focused ─────────────────────────────────────────
    preds_a, time_a = generate_variant(
        label="R53a (LLM-focused)",
        system_prompt=SYSTEM_PROMPT_LLM,
        r39_preds=r39_preds,
        sessions=sessions,
        metadata=metadata,
        use_diversity_avoidance=False,
    )

    # ── Variant B: Diverse ─────────────────────────────────────────────
    preds_b, time_b = generate_variant(
        label="R53b (diverse)",
        system_prompt=SYSTEM_PROMPT_DIVERSE,
        r39_preds=r39_preds,
        sessions=sessions,
        metadata=metadata,
        use_diversity_avoidance=True,
    )

    # ── Validate both ──────────────────────────────────────────────────
    print("\n--- Validation ---")
    for label, preds in [("R53a", preds_a), ("R53b", preds_b)]:
        issues = validate_submission(preds, r39_preds)
        if issues:
            print(f"  {label}: {len(issues)} issues")
            for issue in issues[:10]:
                print(f"    - {issue}")
        else:
            print(f"  {label}: PASSED")

    # ── Compute metrics ────────────────────────────────────────────────
    metrics_a = compute_metrics("R53a_llm_focused", preds_a, r39_preds)
    metrics_b = compute_metrics("R53b_diverse", preds_b, r39_preds)
    print_metrics(metrics_a)
    print_metrics(metrics_b)

    # ── Save artifacts ─────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    out_a = OUT_DIR / "r53a_llm_focused_submission.json"
    with open(out_a, "w") as f:
        json.dump(preds_a, f, indent=2)
    print(f"\nSaved: {out_a}")

    out_b = OUT_DIR / "r53b_diverse_submission.json"
    with open(out_b, "w") as f:
        json.dump(preds_b, f, indent=2)
    print(f"Saved: {out_b}")

    combined_metrics = {
        "experiment": "R53_response_optimization",
        "model": "claude-sonnet-4-6",
        "n_responses": 80,
        "generation_time_a_s": round(time_a, 1),
        "generation_time_b_s": round(time_b, 1),
        "r53a_llm_focused": metrics_a,
        "r53b_diverse": metrics_b,
    }

    eval_json = EVAL_DIR / "expR53_response_optimization.json"
    with open(eval_json, "w") as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"Saved: {eval_json}")

    # ── Sample comparison ──────────────────────────────────────────────
    dash = "-" * 60
    print(f"\n{dash}")
    print("Sample responses (first 3):")
    print(dash)
    for i in range(min(3, len(preds_a))):
        sid = preds_a[i]["session_id"][:8]
        print(f"\n[{i + 1}] Session {sid}:")
        print(f"  R39 : {r39_preds[i]['predicted_response'][:100]}...")
        print(f"  R53a: {preds_a[i]['predicted_response'][:100]}...")
        print(f"  R53b: {preds_b[i]['predicted_response'][:100]}...")

    print(f"\n{dash}")
    print("DONE. Review metrics above.")
    print(
        "Decision rule: submit the variant with better LLM proxy score, "
        "but only if LLM >= 4.70."
    )
    print(dash)


if __name__ == "__main__":
    main()
