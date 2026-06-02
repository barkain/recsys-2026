#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Lever-1 (sharper instrument): PAIRWISE Gemini-2.5-Pro per-aspect judge.

The absolute per-aspect judge (judge_gemini_pro.py) saturates at the top and cannot
resolve our 0.05 inter-submission band (R74 4.85 vs R106 4.90). Pairwise A-vs-B
preference is the standard fix: far more sensitive to small differences. This validates
the instrument on the ONE externally-known ordered pair we have — R74 (true 4.85) <
R106 (true 4.90) — with position-bias counterbalancing (each case asked in BOTH orders;
a 'win' counts only if the preference is consistent across both orders, else tie).

VALIDATION GATE: on the 39 cases where R74/R106 responses differ, pairwise must prefer
R106 by a clear majority of consistent (order-invariant) decisions. The 41 identical-
response cases are a null check (should be ~all tie). PASS -> pairwise is the usable
offline gate for Lever 2 (personalization-as-content). FAIL -> judge saturated for our
style; 30% lever is offline-ungated -> hold R106, pivot to Blind-B readiness.

  GEMINI_API_KEY=$(cat ~/.gemini_key_recsys) python scripts/judge_gemini_pairwise.py --validate
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = "gemini-2.5-pro"

ASPECTS = {
    "personalization": (
        "Judge which assistant response is better PERSONALIZED to THIS user: which one "
        "more visibly and specifically uses the user's stated preferences, listening "
        "history, and profile to justify its recommendation (penalize generic responses "
        "and responses that merely acknowledge user data without operationalizing it)."),
    "explanation_quality": (
        "Judge which assistant response has higher EXPLANATION QUALITY: clearer, more "
        "fluent and confident, and better grounds its recommendation in concrete correct "
        "facts about its recommended track with a coherent causal justification "
        "(penalize vagueness, hedging, factual errors, filler)."),
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
    for h in case.get("history", []):
        role, c = h.get("role"), str(h.get("content", ""))
        if role in ("user", "assistant") and c:
            out.append(f"{role}: {c}")
    prof = case.get("user_profile") or ""
    return (f"User profile: {prof}\n" if prof else "") + "\n".join(out[-8:])


def _gen(client, prompt, attempts=5, backoff=20):
    """generate_content with 429-backoff so the soft daily-cap boundary is ridden out."""
    for i in range(attempts):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt)
        except Exception as e:
            if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and i < attempts - 1:
                time.sleep(backoff); continue
            raise


def ask_both(client, case, ra, ta, rb, tb, meta):
    """ONE call judging BOTH aspects (quota-frugal). Returns {aspect: 'A'|'B'|'TIE'}."""
    prompt = (
        "You are comparing two assistant responses (A and B) in a music-recommendation "
        "conversation. Judge TWO aspects INDEPENDENTLY:\n"
        f"1. PERSONALIZATION — {ASPECTS['personalization']}\n"
        f"2. EXPLANATION — {ASPECTS['explanation_quality']}\n\n"
        f"=== Conversation so far ===\n{conv_text(case)}\n\n"
        f"=== Response A (recommends this track) ===\n{track_meta(ta, meta)}\nA: {ra}\n\n"
        f"=== Response B (recommends this track) ===\n{track_meta(tb, meta)}\nB: {rb}\n\n"
        "Output EXACTLY two lines, each value one of A, B, or TIE:\n"
        "PERSONALIZATION: <A|B|TIE>\nEXPLANATION: <A|B|TIE>")
    t = (_gen(client, prompt).text or "").upper()
    def pick(label):
        m = re.search(label + r"\s*:\s*(A|B|TIE)", t)
        return m.group(1) if m else "TIE"
    return {"personalization": pick("PERSONALIZATION"), "explanation_quality": pick("EXPLANATION")}


def counterbalanced_both(client, case, rx, tx, ry, ty, meta):
    """Ask both orders, both aspects. Per aspect: win only if order-consistent. X is the
    first response argument. Returns {aspect: 'X'|'Y'|'TIE'}. Costs 2 calls total."""
    d1 = ask_both(client, case, rx, tx, ry, ty, meta)   # X=A, Y=B
    d2 = ask_both(client, case, ry, ty, rx, tx, meta)   # Y=A, X=B
    out = {}
    for asp in ASPECTS:
        v1 = {"A": "X", "B": "Y", "TIE": "TIE"}[d1[asp]]
        v2 = {"A": "Y", "B": "X", "TIE": "TIE"}[d2[asp]]
        out[asp] = v1 if (v1 == v2 and v1 != "TIE") else "TIE"
    return out


def load_rows(path):
    p = Path(path)
    if p.suffix == ".zip":
        return json.loads(zipfile.ZipFile(p).read("prediction.json"))
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in open(p)]
    return json.load(open(p))


def load_cases():
    from datasets import load_dataset
    b = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A")
    ds = b[list(b.keys())[0]]
    return {r["session_id"]: {"history": r["conversations"], "user_profile": r.get("user_profile", "")} for r in ds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", default="exp/inference/blind_a/r74_lexdiv_rows_final.jsonl")
    ap.add_argument("--B", default="exp/inference/blind_a/r106_lexdiv_Aclean_submission.zip")
    ap.add_argument("--name-A", default="R74(4.85)")
    ap.add_argument("--name-B", default="R106(4.90)")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--only-differing", action="store_true", help="skip identical-response cases")
    ap.add_argument("--ckpt", default=".scratch/pairwise_ckpt.jsonl", help="resume/checkpoint file")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY not set")
    from google import genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    meta = json.load(open(REPO / "cache/metadata/track_metadata_all_tracks.json"))
    cases = load_cases()
    A = {(r["session_id"], r.get("turn_number")): r for r in load_rows(REPO / args.A)}
    B = {(r["session_id"], r.get("turn_number")): r for r in load_rows(REPO / args.B)}
    keys = sorted(set(A) & set(B))
    if args.limit:
        keys = keys[:args.limit]

    # resume: each decision persisted as a line keyed by session|turn|aspect (survives 429 crashes)
    ckpt = REPO / args.ckpt
    done = {}
    if ckpt.exists():
        for line in open(ckpt):
            d = json.loads(line)
            done[(d["sid"], d["tn"], d["aspect"])] = d
    ck = open(ckpt, "a")

    def tallies():
        tally = {a: {"A": 0, "B": 0, "TIE": 0} for a in ASPECTS}
        null = {a: {"A": 0, "B": 0, "TIE": 0} for a in ASPECTS}
        diff_keys = set()
        for (sid, tn, asp), d in done.items():
            (null if d["identical"] else tally)[asp][d["lab"]] += 1
            if not d["identical"]:
                diff_keys.add((sid, tn))
        return tally, null, len(diff_keys)

    interrupted = None
    try:
        for i, k in enumerate(keys):
            ra = A[k].get("predicted_response", ""); rb = B[k].get("predicted_response", "")
            ta = (A[k].get("predicted_track_ids") or [None])[0]
            tb = (B[k].get("predicted_track_ids") or [None])[0]
            identical = (ra == rb) and (ta == tb)
            if args.only_differing and identical:
                continue
            case = cases.get(k[0], {})
            if all((k[0], k[1], asp) in done for asp in ASPECTS):
                continue  # both aspects already scored in a prior run
            verdicts = counterbalanced_both(client, case, ra, ta, rb, tb, meta)  # 2 calls, both aspects
            for asp in ASPECTS:
                lab = {"X": "A", "Y": "B", "TIE": "TIE"}[verdicts[asp]]
                rec = {"sid": k[0], "tn": k[1], "aspect": asp, "lab": lab, "identical": identical}
                done[(k[0], k[1], asp)] = rec
                ck.write(json.dumps(rec) + "\n"); ck.flush()
            if (i + 1) % 20 == 0:
                _, _, nd = tallies()
                print(f"  {i+1}/{len(keys)} (differing scored: {nd})", flush=True)
    except Exception as e:
        interrupted = str(e)[:200]
        print(f"\n[INTERRUPTED: {interrupted}] — printing partial tally from {len(done)} decisions", flush=True)
    finally:
        ck.close()

    tally, null, n_diff = tallies()
    print(f"\nPAIRWISE {args.name_A} (A) vs {args.name_B} (B) — order-counterbalanced, consistent-only wins")
    print(f"DIFFERING-RESPONSE cases (n={n_diff}):")
    for asp in ASPECTS:
        t = tally[asp]
        print(f"  {asp:20s}  A={t['A']:2d}  B={t['B']:2d}  TIE={t['TIE']:2d}   "
              f"(B-A net = {t['B']-t['A']:+d})")
    nn = sum(null['personalization'].values())
    if nn:
        print(f"NULL CHECK — identical-response cases (n={nn}, expect ~all TIE):")
        for asp in ASPECTS:
            t = null[asp]
            print(f"  {asp:20s}  A={t['A']:2d}  B={t['B']:2d}  TIE={t['TIE']:2d}")
    print(f"\nGATE: PASS iff B ({args.name_B}) wins differing cases by a clear net margin "
          f"on >=1 aspect AND null cases ~all TIE. PASS -> pairwise is the offline gate for "
          f"Lever 2. FAIL -> judge saturated for our style; 30% lever offline-ungated.")


if __name__ == "__main__":
    main()
