#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54b: response-aligned submission.

Takes R54 Phase 3 exploratory submission (already on the leaderboard:
nDCG 0.4925, composite 0.6050, LLM 4.65). Track IDs are kept bitwise
identical. Only responses change:
  - For rows where R54 top-1 == R39 top-1: keep the R39 response (already aligned).
  - For rows where R54 top-1 differs from R39 top-1: regenerate response with
    Haiku, prompted directly on the new R54 top-1 track (no trailing
    questions, references the specific recommendation).

Goal: restore LLM judge score from 4.65 toward 4.70+ without any nDCG risk.

Hard requirements:
  - Bitwise-identical predicted_track_ids per row vs R54 exploratory.
  - LexDiv >= 0.80 across 80 responses.
  - No empty responses, no boilerplate (caught by the response-quality checks).

Output:
  exp/inference/blind_a/r54b_aligned_submission.{json,zip}
  exp/inference/blind_a/r54b_aligned_metadata.json
"""
from __future__ import annotations

import json
import os
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R54_EXPLORATORY = REPO / "exp" / "inference" / "blind_a" / "r54_phase3_exploratory_submission.json"
R39_SUBMISSION = REPO / "exp" / "inference" / "blind_a" / "r39_album_submission.json"
BLIND_OUT = REPO / "exp" / "inference" / "blind_a"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


BANNED_PATTERNS = [
    "i'm sorry",
    "i apologize",
    "i don't have",
    "i cannot",
    "as an ai",
    "let me know if",
    "would you like",
    "is there anything",
    "do you want",
]


def has_banned(resp: str) -> str | None:
    low = resp.lower()
    for p in BANNED_PATTERNS:
        if p in low:
            return p
    return None


def parse_last_turn_local(item):
    import pandas as pd
    df = pd.DataFrame(item["conversations"]).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    last_user = user_rows.iloc[-1]
    turn_num = int(last_user["turn_number"])
    user_query = str(last_user["content"])
    prior = df[df["turn_number"] < turn_num]
    history = [{"role": str(r["role"]), "content": r["content"],
                "turn_number": int(r["turn_number"])} for _, r in prior.iterrows()]
    return turn_num, user_query, history


def main():
    print(f"{ts()} R54b: response-aligned submission")
    print("=" * 70)

    if not R54_EXPLORATORY.exists():
        print(f"ERROR: missing {R54_EXPLORATORY}")
        sys.exit(1)
    if not R39_SUBMISSION.exists():
        print(f"ERROR: missing {R39_SUBMISSION}")
        sys.exit(1)

    with open(R54_EXPLORATORY) as f:
        r54_results = json.load(f)
    with open(R39_SUBMISSION) as f:
        r39_results = json.load(f)

    r39_by_sid = {r["session_id"]: r for r in r39_results}
    r54_by_sid = {r["session_id"]: r for r in r54_results}
    assert len(r39_by_sid) == 80 and len(r54_by_sid) == 80
    assert set(r39_by_sid) == set(r54_by_sid), "Session ID mismatch"

    # Classify each session: kept (top-1 matches R39) vs changed (regenerate)
    kept = []
    to_regenerate = []
    for sid, r54 in r54_by_sid.items():
        r39 = r39_by_sid[sid]
        if r54["predicted_track_ids"][0] == r39["predicted_track_ids"][0]:
            kept.append(sid)
        else:
            to_regenerate.append(sid)
    print(f"  Sessions where R54 top-1 == R39 top-1: {len(kept)} (keep R39 response)")
    print(f"  Sessions where R54 top-1 != R39 top-1: {len(to_regenerate)} (regenerate)")

    # Build base results from R54 (bitwise identical track IDs)
    results = []
    for r54 in r54_results:
        sid = r54["session_id"]
        new_r = {
            "session_id": sid,
            "turn_number": r54["turn_number"],
            "predicted_track_ids": list(r54["predicted_track_ids"]),  # COPY, unchanged
            "predicted_response": "",  # filled below
        }
        results.append(new_r)
    results_by_sid = {r["session_id"]: r for r in results}

    # Stage 1: keep R39 responses where R54 top-1 == R39 top-1
    for sid in kept:
        results_by_sid[sid]["predicted_response"] = r39_by_sid[sid]["predicted_response"]

    # Stage 2: regenerate responses for changed top-1
    if to_regenerate:
        print(f"\n{ts()} Generating {len(to_regenerate)} new responses with Haiku")
        from datasets import DownloadConfig, load_dataset
        from mcrs.db_item.music_catalog import MusicCatalogDB
        from mcrs.lm_modules.claude import ClaudeModule
        from run_inference_blind_r3_det import build_session_memory_for_response

        item_db = MusicCatalogDB(dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                                  split_types=["all_tracks"])
        prompts_dir = REPO / "mcrs" / "system_prompts"
        sys_prompt = (prompts_dir / "roleplay.txt").read_text() + "\n" + \
                     (prompts_dir / "response_generation.txt").read_text()
        # Append guardrails to the system prompt for these regenerations
        sys_prompt += (
            "\n\nAdditional instructions for this response:\n"
            "- The response MUST reference the specific recommended track directly.\n"
            "- Do NOT end with a question or ask the user what they want next.\n"
            "- Do NOT use boilerplate like 'is there anything else', 'would you like', "
            "'let me know if', 'I'm sorry', 'as an AI'.\n"
            "- Keep response concise (1-3 sentences).\n"
        )
        haiku = ClaudeModule(model="claude-haiku-4-5-20251001")

        db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                          download_config=DownloadConfig(local_files_only=True))
        blind_by_sid = {str(item["session_id"]): item for item in db}

        for i, sid in enumerate(to_regenerate):
            item = blind_by_sid[sid]
            _, user_query_r, history_r = parse_last_turn_local(item)
            top_id = results_by_sid[sid]["predicted_track_ids"][0]
            try:
                top_item = item_db.id_to_metadata(top_id)
            except KeyError:
                top_item = f"track_id: {top_id}"
            session_memory = build_session_memory_for_response(history_r, user_query_r, item_db)
            # Retry once if banned phrase appears
            response = None
            for attempt in range(2):
                resp = haiku.response_generation(sys_prompt, session_memory, top_item)
                resp = (resp or "").lstrip(",").lstrip()
                if not resp:
                    continue
                bp = has_banned(resp)
                if bp is None:
                    response = resp
                    break
                print(f"    [{i+1}/{len(to_regenerate)}] sid={sid[:8]} attempt {attempt+1}: "
                      f"banned phrase '{bp}', retrying", flush=True)
            if response is None:
                # If retries failed, use the last generation anyway and warn
                response = resp or "Try this track."
                print(f"    WARN sid={sid[:8]}: using unfiltered response (fallback)",
                      flush=True)
            results_by_sid[sid]["predicted_response"] = response
            if (i + 1) % 5 == 0:
                print(f"    {i+1}/{len(to_regenerate)} regenerated", flush=True)

    # Verification: track IDs bitwise identical to R54 exploratory
    print(f"\n{ts()} Verifying bitwise-identical track IDs vs R54 exploratory")
    for r in results:
        sid = r["session_id"]
        r54 = r54_by_sid[sid]
        if r["predicted_track_ids"] != r54["predicted_track_ids"]:
            raise RuntimeError(f"Track ID mismatch at sid={sid}")
    print(f"  PASS: all 80 rows have bitwise-identical track IDs")

    # Validation
    print(f"\n{ts()} Validation")
    # Need catalog for valid track ID check
    from scripts.expR54_phase3_blind_submission import load_track_albums
    track_album = load_track_albums()
    valid_catalog = set(track_album.keys())

    sids = set()
    banned_hits = []
    for r in results:
        sid = r["session_id"]
        if len(r["predicted_track_ids"]) != 20:
            raise ValueError(f"Row {sid}: {len(r['predicted_track_ids'])} tracks")
        if len(set(r["predicted_track_ids"])) != 20:
            raise ValueError(f"Row {sid}: duplicate tracks")
        if not r["predicted_response"].strip():
            raise ValueError(f"Row {sid}: empty response")
        if r["predicted_response"].startswith(","):
            raise ValueError(f"Row {sid}: leading comma")
        invalid = [t for t in r["predicted_track_ids"] if t not in valid_catalog]
        if invalid:
            raise ValueError(f"Row {sid}: invalid track IDs: {invalid[:3]}")
        bp = has_banned(r["predicted_response"])
        if bp:
            banned_hits.append((sid, bp, r["predicted_response"][:80]))
        sids.add(sid)
    if len(results) != 80 or len(sids) != 80:
        raise ValueError(f"Expected 80 unique rows, got {len(results)}/{len(sids)}")
    print(f"  rows=80  unique=80  empty=0  invalid_ids=0")
    if banned_hits:
        print(f"  WARN {len(banned_hits)} responses still contain banned phrases:")
        for sid, bp, snippet in banned_hits[:5]:
            print(f"    sid={sid[:8]}  phrase={bp!r}  resp={snippet!r}")

    # LexDiv check (Distinct-2 across all 80 responses)
    all_bigrams = []
    for r in results:
        toks = r["predicted_response"].lower().split()
        all_bigrams.extend(zip(toks, toks[1:]))
    lexdiv = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    print(f"  LexDiv (Distinct-2): {lexdiv:.4f}")
    if lexdiv < 0.80:
        print(f"  FAIL: LexDiv {lexdiv:.4f} < 0.80 — do NOT submit")
        sys.exit(1)
    print(f"  PASS: LexDiv {lexdiv:.4f} >= 0.80")

    # Diff vs R54 exploratory (responses)
    response_changed = sum(
        1 for r in results
        if r["predicted_response"] != r54_by_sid[r["session_id"]]["predicted_response"]
    )
    print(f"  Responses changed vs R54 exploratory: {response_changed}/80")

    # Save
    report = {
        "submission_label": "R54b_response_aligned",
        "track_ids_source": "r54_phase3_exploratory_submission.json (bitwise identical)",
        "responses": {
            "kept_R39": len(kept),
            "regenerated": len(to_regenerate),
            "changed_vs_r54_exploratory": int(response_changed),
        },
        "validation": {
            "track_ids_bitwise_identical": True,
            "rows": 80, "duplicates": 0, "empty": 0, "invalid_ids": 0,
            "lexdiv_distinct2": float(lexdiv),
            "banned_phrase_remaining": len(banned_hits),
        },
        "context": {
            "r54_exploratory_scores": {
                "ndcg20": 0.4925, "composite": 0.6050,
                "lexdiv": 0.8198, "llm_judge": 4.65,
            },
            "r39_scores": {
                "ndcg20": 0.4798, "composite": 0.6024,
                "lexdiv": 0.8198, "llm_judge": 4.70,
            },
            "rationale": "Bitwise-identical R54 tracks (preserve nDCG 0.4925). "
                         "Regenerate only the 27 responses where top-1 changed. "
                         "Target: restore LLM 4.65 -> 4.70+.",
        },
        "created_at": datetime.now().isoformat(),
    }

    out_json = BLIND_OUT / "r54b_aligned_submission.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    out_meta = BLIND_OUT / "r54b_aligned_metadata.json"
    with open(out_meta, "w") as f:
        json.dump(report, f, indent=2)
    out_zip = BLIND_OUT / "r54b_aligned_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")

    print(f"\n  SUBMISSION: {out_zip}")
    print(f"  METADATA:   {out_meta}")
    print(f"  80 rows.  kept={len(kept)} R39 responses, regenerated={len(to_regenerate)}.")
    print(f"  LexDiv: {lexdiv:.4f}")
    print(f"\n{ts()} DONE")


if __name__ == "__main__":
    main()
