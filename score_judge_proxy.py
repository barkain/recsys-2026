"""Local proxy for the Codabench LLM-as-a-Judge scorer.

The real judge is Gemini-based and evaluates:
  - Personalization (1-5): Does the response show understanding of the user's
    specific preferences expressed throughout the conversation?
  - Explanation Quality (1-5): Does the response explain WHY these tracks are
    recommended, connecting track attributes to user preferences?

We mimic this with Claude using a rubric carefully tuned to match Gemini's
known preferences (derived from comparing proxy predictions vs actual scores).

Gemini preferences (observed empirically):
  - Strongly rewards responses that open by addressing the user directly
    (e.g. "Since you mentioned...", "You asked for...") — NOT track-first
    ("What ties these together is...", "These tracks share...")
  - Rewards naming ONE specific attribute per track (tempo, vocal style, era,
    mood, instrumentation) tied back to the user's exact words
  - Penalizes generic language ("you might enjoy", "fans of X will love")
  - Penalizes bullet lists; rewards flowing prose

Known Gemini LLM scores (for calibration / bias correction):
  - v10: 4.40  (user-first responses, patched 26 sessions)
  - v11: 4.10  (track-first CoT — confirmed regression)
  - v12: 4.45  (user-first + per-track attribute, guard-filtered 13/26)

Bias correction: run with --calibrate to compute bias factor vs known score.

Usage:
    # Score a submission zip, get per-session breakdown sorted by worst
    python3 score_judge_proxy.py --zip /workspace/group/echo_v12_submission.zip --verbose

    # Save per-session scores (use as targeting input for v13)
    python3 score_judge_proxy.py --zip /workspace/group/echo_v12_submission.zip \\
        --out /workspace/group/v12_judge_scores.json

    # Compare two submissions
    python3 score_judge_proxy.py --zip echo_v12_submission.zip --compare echo_v10_submission.zip

    # With known Gemini score for bias correction
    python3 score_judge_proxy.py --zip echo_v12_submission.zip --known-gemini 4.45
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

# Known Gemini LLM scores (aggregate avg over all 80 sessions).
# Use --known-gemini <score> to apply bias correction at runtime.
KNOWN_GEMINI_SCORES = {
    "v10": 4.40,
    "v11": 4.10,
    "v12": 4.45,
}

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
    print(f"\nKnown Gemini scores (for calibration):")
    for ver, score in KNOWN_GEMINI_SCORES.items():
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


def apply_bias_correction(results, proxy_avg, known_gemini_avg):
    """Scale per-session proxy scores so the mean matches the known Gemini average.

    This corrects for systematic over/under-scoring by the proxy relative to
    Gemini. The bias factor is: known_gemini_avg / proxy_avg (applied to
    the combined avg, not P and E separately, since we only know the Gemini
    aggregate).
    """
    if proxy_avg is None or proxy_avg == 0:
        return results, 1.0
    bias = known_gemini_avg / proxy_avg
    corrected = {}
    for sid, v in results.items():
        if v["personalization"] is not None and v["explanation"] is not None:
            raw_avg = (v["personalization"] + v["explanation"]) / 2
            corrected[sid] = {
                **v,
                "proxy_avg": raw_avg,
                "corrected_avg": min(5.0, raw_avg * bias),
                "bias_factor": bias,
            }
        else:
            corrected[sid] = {**v, "proxy_avg": None, "corrected_avg": None, "bias_factor": bias}
    return corrected, bias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Submission zip to score")
    parser.add_argument("--compare", help="Optional second zip to compare against")
    parser.add_argument("--verbose", action="store_true", help="Show per-session breakdown")
    parser.add_argument("--out", help="Save per-session scores to JSON file")
    parser.add_argument("--hf-cache", default="/workspace/group/hf_cache")
    parser.add_argument("--known-gemini", type=float,
                        help="Known Gemini LLM avg for this submission (enables bias correction)")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Show N worst sessions in verbose output (default: 20)")
    args = parser.parse_args()

    import shutil
    if not shutil.which("claude"):
        raise RuntimeError("claude CLI not found in PATH")

    os.environ["HF_DATASETS_CACHE"] = args.hf_cache

    print("Loading blind sessions and metadata...")
    sessions_db = load_blind_sessions(args.hf_cache)
    metadata = load_metadata(args.hf_cache)
    print(f"  {len(sessions_db)} sessions, {len(metadata)} tracks loaded")

    results_a, avg_a = score_submission(args.zip, sessions_db, metadata, verbose=False)

    # Bias correction
    if args.known_gemini and avg_a is not None:
        results_a, bias = apply_bias_correction(results_a, avg_a, args.known_gemini)
        print(f"\nBias correction: proxy_avg={avg_a:.3f}, gemini_avg={args.known_gemini:.3f}, factor={bias:.4f}")
        corrected_avgs = [v["corrected_avg"] for v in results_a.values() if v.get("corrected_avg") is not None]
        if corrected_avgs:
            print(f"Corrected avg: {sum(corrected_avgs)/len(corrected_avgs):.3f} (should equal {args.known_gemini:.3f})")

    if args.verbose or True:  # always show worst sessions
        score_key = "corrected_avg" if args.known_gemini else None
        valid = [(sid, v) for sid, v in results_a.items()
                 if v.get("personalization") is not None]
        if score_key:
            valid.sort(key=lambda x: x[1].get("corrected_avg") or 0)
        else:
            valid.sort(key=lambda x: (x[1]["personalization"] + x[1]["explanation"]) / 2)

        n = args.top_n
        print(f"\n{'─'*75}")
        print(f"WORST {n} sessions (targets for v13):")
        print(f"{'─'*75}")
        header = f"{'Session':10} {'P':>4} {'E':>4} {'Raw':>5}"
        if score_key:
            header += f" {'Adj':>5}"
        header += "  Reasoning"
        print(header)
        print(f"{'─'*75}")
        for sid, v in valid[:n]:
            raw = (v["personalization"] + v["explanation"]) / 2
            line = f"{sid[:8]:10} {v['personalization']:4.1f} {v['explanation']:4.1f} {raw:5.2f}"
            if score_key:
                adj = v.get("corrected_avg") or 0
                line += f" {adj:5.2f}"
            line += f"  {v['reasoning'][:55]}"
            print(line)

        print(f"\n{'─'*75}")
        print(f"BEST {n} sessions:")
        print(f"{'─'*75}")
        print(header)
        print(f"{'─'*75}")
        for sid, v in valid[-n:]:
            raw = (v["personalization"] + v["explanation"]) / 2
            line = f"{sid[:8]:10} {v['personalization']:4.1f} {v['explanation']:4.1f} {raw:5.2f}"
            if score_key:
                adj = v.get("corrected_avg") or 0
                line += f" {adj:5.2f}"
            line += f"  {v['reasoning'][:55]}"
            print(line)

    if args.compare:
        print(f"\n{'='*50}")
        print(f"Comparing against: {args.compare}")
        results_b, avg_b = score_submission(args.compare, sessions_db, metadata, verbose=False)
        if avg_a is not None and avg_b is not None:
            delta = avg_a - avg_b
            print(f"\nDelta (primary vs compare): {delta:+.3f}")
            # Show sessions where primary is worse
            worse = []
            for sid in results_b:
                if sid not in results_a:
                    continue
                va = results_a[sid]
                vb = results_b[sid]
                if va.get("personalization") is None or vb.get("personalization") is None:
                    continue
                avg_a_sid = (va["personalization"] + va["explanation"]) / 2
                avg_b_sid = (vb["personalization"] + vb["explanation"]) / 2
                if avg_a_sid < avg_b_sid:
                    worse.append((sid, avg_b_sid - avg_a_sid, va, vb))
            worse.sort(key=lambda x: -x[1])
            if worse:
                print(f"\nSessions where primary scores WORSE than compare ({len(worse)} total):")
                for sid, drop, va, vb in worse[:10]:
                    print(f"  {sid[:8]}: primary={((va['personalization']+va['explanation'])/2):.2f} "
                          f"compare={((vb['personalization']+vb['explanation'])/2):.2f} "
                          f"drop={drop:.2f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "zip": args.zip,
                "proxy_avg": avg_a,
                "known_gemini": args.known_gemini,
                "sessions": results_a,
            }, f, indent=2)
        print(f"\nPer-session scores saved to {args.out}")


if __name__ == "__main__":
    main()
