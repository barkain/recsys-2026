#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Lever-1 keystone: a calibrated Gemini-2.5-Pro PER-ASPECT local judge.

The competition's data-generation judge is Gemini-2.5-Pro (NOT Flash, NOT Claude),
scoring each response on SEPARATE aspects (Personalization, Explanation-Quality), each
with its own rubric, and given the recommended track's full metadata (TalkPlayData-2
paper). Our prior "proxy judge unreliable" failures used Claude + holistic scoring — the
wrong model and wrong granularity. This rebuilds the right instrument so we get an
OFFLINE GATE and can iterate the 30%-weight judge lever without burning blind slots.

GATE / VALIDATION (run this BEFORE trusting it): score our four already-officially-scored
Blind-A submissions whose true LLM-judge averages we know —
  R74 -> 4.85 ; R77 -> 4.90 ; R84c -> 4.90 ; R106 A-clean -> 4.90
If this judge reproduces R74 < {R77, R84c, R106} and the 4.90 plateau, it is a usable
local gate. If it cannot, the judge is locally un-modelable -> stop all judge work
(a clean, high-value negative either way).

Requires: GEMINI_API_KEY (Google AI Studio).  pip install google-genai
Usage:
  GEMINI_API_KEY=... python scripts/judge_gemini_pro.py --submission <zip> --blind-cases <hf|json>
  GEMINI_API_KEY=... python scripts/judge_gemini_pro.py --validate   # scores R74/R77/R84c/R106
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = "gemini-2.5-pro"

# Per-aspect rubrics (1-5). Grounded in the known judge dimensions + reference_llm_judge_rubric;
# REFINE against TalkPlayData-2 paper Table 7 once validation runs. The two aspects are scored
# in SEPARATE calls (per-aspect granularity is the key fix vs the failed holistic proxy).
RUBRICS = {
    "personalization": (
        "You score ONLY how well the assistant's response is PERSONALIZED to THIS user. "
        "Reward responses that VISIBLY use the user's stated preferences, their listening "
        "history / previously played tracks, and their profile (age group, country, gender) "
        "to justify the recommendation. PENALIZE generic responses, and responses that merely "
        "acknowledge user data exists without operationalizing it. 5 = recommendation is "
        "explicitly, specifically tied to this user's expressed taste/history/profile; "
        "1 = generic, could be sent to anyone."),
    "explanation_quality": (
        "You score ONLY the EXPLANATION QUALITY of the assistant's response: is it clear, "
        "fluent, confident, and does it GROUND the recommendation in concrete, correct facts "
        "about the recommended track (artist, album, year, genre, instrumentation, mood) with "
        "a coherent causal justification for why it fits the conversation? PENALIZE vagueness, "
        "hedging, factual errors, and filler. 5 = vivid, accurate, well-justified critic-quality "
        "explanation; 1 = vague or unjustified."),
}


def track_meta(tid, meta):
    m = meta.get(tid, {})
    def g(k):
        v = m.get(k, [])
        return v[0] if isinstance(v, list) and v else (v if isinstance(v, str) else "")
    tags = m.get("tag_list", [])
    tags = ", ".join(str(t) for t in (tags[:12] if isinstance(tags, list) else []))
    return (f"track_name: {g('track_name')}\nartist: {g('artist_name')}\nalbum: {g('album_name')}\n"
            f"release_date: {g('release_date')}\ntags: {tags}")


def conv_text(case):
    out = []
    for h in case.get("history", case.get("conversations", [])):
        role, c = h.get("role"), str(h.get("content", ""))
        if role in ("user", "assistant") and c:
            out.append(f"{role}: {c}")
    prof = case.get("user_profile") or ""
    demo = case.get("_demo") or ""
    return (f"User profile/demographics: {demo} {prof}\n" if (prof or demo) else "") + "\n".join(out[-8:])


def score_one(client, aspect, case, response, tid, meta):
    prompt = (
        f"{RUBRICS[aspect]}\n\n"
        f"=== Conversation so far ===\n{conv_text(case)}\n\n"
        f"=== Recommended track (full metadata) ===\n{track_meta(tid, meta)}\n\n"
        f"=== Assistant's response to score ===\n{response}\n\n"
        f"Output ONLY an integer 1-5 for the {aspect} aspect. No other text.")
    r = client.models.generate_content(model=MODEL, contents=prompt)
    m = re.search(r"[1-5]", r.text or "")
    return int(m.group()) if m else None


def judge_submission(client, rows, cases_by_key, meta, limit=None):
    per = {"personalization": [], "explanation_quality": []}
    rows = rows[:limit] if limit else rows
    for i, r in enumerate(rows):
        key = (r["session_id"], r.get("turn_number"))
        case = cases_by_key.get(key, {})
        resp = r.get("predicted_response", "")
        tid = (r.get("predicted_track_ids") or [None])[0]
        for asp in per:
            s = score_one(client, asp, case, resp, tid, meta)
            if s is not None:
                per[asp].append(s)
        if (i + 1) % 20 == 0:
            print(f"  scored {i+1}/{len(rows)}", flush=True)
    avg = {a: (sum(v) / len(v) if v else 0.0) for a, v in per.items()}
    overall = sum(avg.values()) / len(avg)
    return overall, avg


def load_cases():
    """Blind-A cases keyed by (session_id, turn_number), with profile/demographics."""
    from datasets import load_dataset
    b = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A")
    ds = b[list(b.keys())[0]]
    by_key = {}
    for r in ds:
        demo = f"age_group={r.get('user_profile','')}"  # Blind-A ships user_profile; demographics via User-Metadata if needed
        for tn in range(1, 9):
            by_key[(r["session_id"], tn)] = {"history": r["conversations"], "user_profile": r.get("user_profile", ""), "_demo": demo}
    return by_key


def load_rows(path):
    p = Path(path)
    if p.suffix == ".zip":
        return json.loads(zipfile.ZipFile(p).read("prediction.json"))
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in open(p)]
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set. Get one at https://aistudio.google.com/apikey and "
                 "`export GEMINI_API_KEY=...` (pip install google-genai). This is the hard "
                 "dependency for Lever 1.")
    try:
        from google import genai
    except ImportError:
        sys.exit("pip install google-genai")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    meta = json.load(open(REPO / "cache/metadata/track_metadata_all_tracks.json"))
    cases = load_cases()

    if args.validate:
        # the known-score boundary: must reproduce R74(4.85) < R77/R84c/R106(4.90)
        VAL = {
            "R74 (true 4.85)": "exp/inference/blind_a/r74_lexdiv_rows_final.jsonl",
            "R77 (true 4.90)": "exp/inference/blind_a/r77_ceiling_rows_final.jsonl",
            "R84c (true 4.90)": "exp/inference/blind_a/r84c_selective_submission.zip",
            "R106 (true 4.90)": "exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip",
        }
        print(f"VALIDATION — local Gemini-2.5-Pro per-aspect judge vs known LLM scores:")
        for name, path in VAL.items():
            if not (REPO / path).exists():
                print(f"  {name}: MISSING {path}"); continue
            ov, avg = judge_submission(client, load_rows(REPO / path), cases, meta, args.limit or 80)
            print(f"  {name}: local={ov:.3f}  (pers={avg['personalization']:.2f} expl={avg['explanation_quality']:.2f})")
        print("\nGATE: PASS iff R74 < {R77,R84c,R106} and the three 4.90s cluster. "
              "PASS -> usable offline gate for Levers 2-3. FAIL -> judge un-modelable, stop judge work.")
    elif args.submission:
        ov, avg = judge_submission(client, load_rows(args.submission), cases, meta, args.limit)
        print(f"{args.submission}: local judge={ov:.3f} (pers={avg['personalization']:.2f} expl={avg['explanation_quality']:.2f})")


if __name__ == "__main__":
    main()
