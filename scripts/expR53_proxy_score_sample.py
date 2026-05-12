# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301,S311
"""R53 proxy scorer — compare R39, R53a, R53b on 14-session sample.

Uses Anthropic Python SDK with the same rubric as score_judge_proxy.py
(claude-sonnet-4-6 as judge). Saves results to exp/eval/expR53_proxy_score_sample.json.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime

import anthropic
from datasets import load_dataset
from datasets.download import DownloadConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JUDGE_MODEL = "claude-sonnet-4-6"
N_SAMPLE = 14
DELAY_BETWEEN_CALLS = 1.0  # seconds

JUDGE_PROMPT = """\
You are replicating the scoring behavior of a Gemini-based LLM judge for a music recommendation challenge.

The judge evaluates how well the recommendation response serves THIS specific user based on THEIR conversation.

**Conversation history (what this user said):**
{conversation}

**System recommendation response:**
{response}

**Recommended tracks (shown to user):**
{tracks}

---

Score on exactly these two dimensions (each 1-5, half-points allowed):

**PERSONALIZATION (1-5)** — Does the response talk TO this user, not just ABOUT music?
- 5: Opens by directly quoting or closely paraphrasing the user's own words. Every sentence addresses this specific user's situation. No sentence could be copy-pasted to a different conversation.
- 4: Clearly references what this user said. Feels written for them specifically. Minor generic phrases OK.
- 3: Some user references but mixed with generic filler ("fans of X", "you might enjoy"). Feels partly templated.
- 2: Mostly generic. Vague nod to conversation context but nothing specific.
- 1: Completely generic. Could be sent to any user unchanged.

CRITICAL: If the response opens with track-centric framing ("What ties these together...", "These tracks share...", "I've selected tracks that...", "All of these songs...") BEFORE addressing the user — cap Personalization at 3, even if the rest is good.

**EXPLANATION QUALITY (1-5)** — Does the response name specific track attributes tied to what the user asked for?
- 5: For each recommended track, names ONE specific concrete attribute (tempo, vocal style, production era, mood, instrumentation) AND connects it to something the user explicitly said. e.g. "X has the breathy vocal style you asked for."
- 4: Most tracks get specific attribute connections. One or two vague.
- 3: Some attribute names but connections to user preferences are weak or missing.
- 2: Vague adjectives only ("mellow", "energetic") with no connection to user's words.
- 1: No explanation. Just names the tracks.

CRITICAL: Generic phrases like "you'll love this", "perfect for your taste", "fans of X will enjoy" do NOT count as explanations — they must name a specific musical attribute.

Return ONLY valid JSON, no other text:
{{"personalization": <1-5>, "explanation": <1-5>, "reasoning": "<one sentence explaining the main strength or weakness>"}}
"""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_blind_sessions() -> dict:
    """Load blind-A sessions for conversation context."""
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Blind-A",
        download_config=DownloadConfig(local_files_only=True),
        split="test",
    )
    sessions: dict = {}
    for item in ds:
        sessions[item["session_id"]] = item
    return sessions


def load_metadata() -> dict:
    """Load track metadata."""
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        download_config=DownloadConfig(local_files_only=True),
    )
    # Use the split that has the most tracks
    combined = None
    for split_name in ["all_tracks", "train"]:
        if split_name in ds:
            combined = ds[split_name]
            break
    if combined is None:
        combined = ds[list(ds.keys())[0]]
    return {item["track_id"]: item for item in combined}


def load_submission(path: str) -> dict:
    """Load a JSON submission file."""
    with open(path) as f:
        preds = json.load(f)
    # Index by session_id
    return {p["session_id"]: p for p in preds}


# ---------------------------------------------------------------------------
# Formatting (same logic as score_judge_proxy.py)
# ---------------------------------------------------------------------------
def format_conversation(session: dict) -> str:
    """Format conversation history for the judge prompt."""
    convs = sorted(session["conversations"], key=lambda x: x["turn_number"])
    lines: list[str] = []
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


def format_tracks(track_ids: list, metadata: dict) -> str:
    """Format top-5 track names for the judge prompt."""
    lines: list[str] = []
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


# ---------------------------------------------------------------------------
# Session selection
# ---------------------------------------------------------------------------
def select_sample_sessions(
    sessions: dict, sub_r39: dict, sub_r53a: dict
) -> list[str]:
    """Select 14 sessions covering short/medium/long + divergent."""
    # Categorize by conversation length
    short: list[str] = []
    medium: list[str] = []
    long_: list[str] = []
    for sid, sess in sessions.items():
        # Count user turns specifically
        user_turns = sum(1 for c in sess["conversations"] if c["role"] == "user")
        if user_turns <= 2:
            short.append(sid)
        elif user_turns <= 5:
            medium.append(sid)
        else:
            long_.append(sid)

    print(f"  Short (1-2 user turns): {len(short)}")
    print(f"  Medium (3-5 user turns): {len(medium)}")
    print(f"  Long (6+ user turns): {len(long_)}")

    # Pick 4 from each bucket (deterministic, sorted by session_id)
    short.sort()
    medium.sort()
    long_.sort()
    selected = short[:4] + medium[:4] + long_[:4]

    # Find 2 sessions where R53a response differs most from R39
    # Use simple word-level Jaccard distance
    already = set(selected)
    diffs: list[tuple[str, float]] = []
    for sid in sessions:
        if sid in already:
            continue
        r39_resp = sub_r39.get(sid, {}).get("predicted_response", "")
        r53a_resp = sub_r53a.get(sid, {}).get("predicted_response", "")
        if not r39_resp or not r53a_resp:
            continue
        words_39 = set(r39_resp.lower().split())
        words_53a = set(r53a_resp.lower().split())
        if not words_39 and not words_53a:
            continue
        jaccard = len(words_39 & words_53a) / max(len(words_39 | words_53a), 1)
        diffs.append((sid, 1.0 - jaccard))  # higher = more different

    diffs.sort(key=lambda x: -x[1])
    for sid, dist in diffs[:2]:
        selected.append(sid)
        print(f"  Most divergent: {sid} (word-Jaccard distance={dist:.3f})")

    print(f"  Total selected: {len(selected)}")
    return selected


# ---------------------------------------------------------------------------
# Scoring via Anthropic SDK
# ---------------------------------------------------------------------------
def score_session_sdk(
    client: anthropic.Anthropic,
    conversation_text: str,
    response: str,
    tracks_text: str,
) -> tuple[float | None, float | None, str]:
    """Score a single session using the Anthropic Python SDK.

    Returns (personalization, explanation, reasoning) or (None, None, error_msg).
    """
    prompt = JUDGE_PROMPT.format(
        conversation=conversation_text,
        response=response,
        tracks=tracks_text,
    )
    try:
        msg = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            p = float(data.get("personalization", 0))
            e = float(data.get("explanation", 0))
            r = str(data.get("reasoning", ""))
            if 1 <= p <= 5 and 1 <= e <= 5:
                return p, e, r
        return None, None, f"parse error: {text[:100]}"
    except Exception as ex:
        return None, None, str(ex)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] R53 proxy score sample — starting")

    # Set API key
    api_key = os.environ.get("ANTHROPIC_RECSYS_API_KEY") or os.environ.get(
        "ANTHROPIC_API_KEY"
    )
    if not api_key:
        raise RuntimeError(
            "No Anthropic API key found in ANTHROPIC_RECSYS_API_KEY or ANTHROPIC_API_KEY"
        )
    client = anthropic.Anthropic(api_key=api_key)

    # Load data
    print("Loading blind sessions and metadata...")
    sessions = load_blind_sessions()
    metadata = load_metadata()
    print(f"  {len(sessions)} sessions, {len(metadata)} tracks")

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sub_r39 = load_submission(
        os.path.join(base, "exp/inference/blind_a/r39_album_submission.json")
    )
    sub_r53a = load_submission(
        os.path.join(base, "exp/inference/blind_a/r53a_llm_focused_submission.json")
    )
    sub_r53b = load_submission(
        os.path.join(base, "exp/inference/blind_a/r53b_diverse_submission.json")
    )
    print(
        f"  Loaded submissions: R39={len(sub_r39)}, R53a={len(sub_r53a)}, R53b={len(sub_r53b)}"
    )

    # Select sample
    print("\nSelecting sample sessions...")
    sample_sids = select_sample_sessions(sessions, sub_r39, sub_r53a)

    # Score all sessions x variants
    variants = {"R39": sub_r39, "R53a": sub_r53a, "R53b": sub_r53b}
    total_calls = len(sample_sids) * len(variants)
    print(
        f"\nScoring {len(sample_sids)} sessions x {len(variants)} variants = {total_calls} API calls"
    )
    print(f"Using model: {JUDGE_MODEL}")

    results: dict = {}  # sid -> {variant -> {p, e, reasoning}}
    call_count = 0
    for sid in sample_sids:
        sess = sessions[sid]
        conv_text = format_conversation(sess)
        n_user_turns = sum(
            1 for c in sess["conversations"] if c["role"] == "user"
        )

        results[sid] = {"n_user_turns": n_user_turns}
        for vname, sub in variants.items():
            pred = sub.get(sid)
            if pred is None:
                results[sid][vname] = {
                    "personalization": None,
                    "explanation": None,
                    "reasoning": "missing from submission",
                }
                continue

            response = pred.get("predicted_response", "")
            track_ids = pred.get("predicted_track_ids", [])
            tracks_text = format_tracks(track_ids, metadata)

            call_count += 1
            ts_now = datetime.now().strftime("%H:%M:%S")
            print(
                f"  [{ts_now}] ({call_count}/{total_calls}) {vname} / {sid[:8]}...",
                end="",
                flush=True,
            )

            p, e, reasoning = score_session_sdk(
                client, conv_text, response, tracks_text
            )

            if p is not None and e is not None:
                avg = (p + e) / 2
                print(f" P={p:.1f} E={e:.1f} avg={avg:.2f}")
            else:
                print(f" FAILED: {reasoning[:60]}")

            results[sid][vname] = {
                "personalization": p,
                "explanation": e,
                "reasoning": reasoning,
            }

            time.sleep(DELAY_BETWEEN_CALLS)

    # ---------------------------------------------------------------------------
    # Aggregate
    # ---------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")

    # Per-session table
    print(
        f"\n{'Session':>10} {'Turns':>5}  {'R39':>8}  {'R53a':>8}  {'R53b':>8}  {'Best':>5}"
    )
    print("-" * 55)

    per_session_list: list[dict] = []
    for sid in sample_sids:
        r = results[sid]
        ut = r["n_user_turns"]
        avgs: dict[str, float | None] = {}
        for vname in ["R39", "R53a", "R53b"]:
            v = r[vname]
            if v["personalization"] is not None and v["explanation"] is not None:
                avgs[vname] = (v["personalization"] + v["explanation"]) / 2
            else:
                avgs[vname] = None

        valid_avgs = {vn: a for vn, a in avgs.items() if a is not None}
        best = max(valid_avgs, key=lambda vn: valid_avgs[vn]) if valid_avgs else "?"

        row: dict = {
            "session_id": sid,
            "n_user_turns": ut,
        }
        for vname in ["R39", "R53a", "R53b"]:
            v = r[vname]
            row[f"{vname}_personalization"] = v["personalization"]
            row[f"{vname}_explanation"] = v["explanation"]
            row[f"{vname}_avg"] = avgs.get(vname)
            row[f"{vname}_reasoning"] = v["reasoning"]

        per_session_list.append(row)

        def _fmt(x: float | None) -> str:
            return f"{x:.2f}" if x is not None else "FAIL"

        print(
            f"{sid[:10]:>10} {ut:>5}  "
            f"{_fmt(avgs.get('R39')):>8}  "
            f"{_fmt(avgs.get('R53a')):>8}  "
            f"{_fmt(avgs.get('R53b')):>8}  "
            f"{best:>5}"
        )

    # Averages
    agg: dict[str, float | None | int] = {}
    for vname in ["R39", "R53a", "R53b"]:
        vals = [
            r[f"{vname}_avg"]
            for r in per_session_list
            if r[f"{vname}_avg"] is not None
        ]
        agg[f"avg_{vname}"] = sum(vals) / len(vals) if vals else None
        agg[f"n_valid_{vname}"] = len(vals)

    print(f"\n{'='*50}")
    for vname in ["R39", "R53a", "R53b"]:
        avg_val = agg[f"avg_{vname}"]
        n_val = agg[f"n_valid_{vname}"]
        avg_str = f"{avg_val:.3f}" if avg_val is not None else "N/A"
        print(f"  avg_{vname}: {avg_str}  (n={n_val})")

    # Win/tie/loss
    comparisons: dict = {}
    for comp_name, comp_variant in [
        ("R53a_vs_R39", "R53a"),
        ("R53b_vs_R39", "R53b"),
    ]:
        win, tie, loss = 0, 0, 0
        for r in per_session_list:
            r39_avg = r.get("R39_avg")
            comp_avg = r.get(f"{comp_variant}_avg")
            if r39_avg is None or comp_avg is None:
                continue
            diff = comp_avg - r39_avg
            if diff > 0.1:
                win += 1
            elif diff < -0.1:
                loss += 1
            else:
                tie += 1
        comparisons[comp_name] = {"win": win, "tie": tie, "loss": loss}
        print(f"  {comp_name}: win={win} tie={tie} loss={loss}")

    # Check for quality failures (any session with avg < 2.0)
    failures: list[str] = []
    for r in per_session_list:
        for vname in ["R53a", "R53b"]:
            avg_val = r.get(f"{vname}_avg")
            if avg_val is not None and avg_val < 2.0:
                failures.append(
                    f"{vname}/{r['session_id'][:8]}: avg={avg_val:.2f}"
                    f" — {r[f'{vname}_reasoning'][:80]}"
                )

    if failures:
        print("\n  QUALITY FAILURES:")
        for failure_msg in failures:
            print(f"    {failure_msg}")

    # Decision
    avg_r39 = agg["avg_R39"]
    avg_r53a = agg["avg_R53a"]
    avg_r53b = agg["avg_R53b"]

    lift_a: float = 0.0
    lift_b: float = 0.0
    if isinstance(avg_r53a, float) and isinstance(avg_r39, float):
        lift_a = avg_r53a - avg_r39
    if isinstance(avg_r53b, float) and isinstance(avg_r39, float):
        lift_b = avg_r53b - avg_r39

    has_failures = len(failures) > 0

    if lift_a >= 0.05 and not has_failures:
        decision = "submit R53a"
    elif lift_b >= 0.05 and not has_failures:
        decision = "submit R53b"
    elif lift_a < 0.03 and lift_b < 0.03:
        decision = "do not submit"
    elif lift_a >= 0.03 and not has_failures:
        decision = "submit R53a (marginal lift)"
    elif lift_b >= 0.03 and not has_failures:
        decision = "submit R53b (marginal lift)"
    else:
        decision = "do not submit"

    print(f"\n  Lift R53a vs R39: {lift_a:+.3f}")
    print(f"  Lift R53b vs R39: {lift_b:+.3f}")
    print(f"  DECISION: {decision}")

    # Save output
    output = {
        "n_sessions_scored": len(sample_sids),
        "judge_model": JUDGE_MODEL,
        "timestamp": datetime.now().isoformat(),
        "per_session": per_session_list,
        "aggregates": {
            "avg_R39": agg["avg_R39"],
            "avg_R53a": agg["avg_R53a"],
            "avg_R53b": agg["avg_R53b"],
            "lift_R53a_vs_R39": lift_a,
            "lift_R53b_vs_R39": lift_b,
            "R53a_vs_R39": comparisons["R53a_vs_R39"],
            "R53b_vs_R39": comparisons["R53b_vs_R39"],
            "quality_failures": failures,
        },
        "decision": decision,
    }

    out_path = os.path.join(base, "exp/eval/expR53_proxy_score_sample.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
