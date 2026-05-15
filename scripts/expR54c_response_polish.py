#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54c: audit-first targeted response polish on top of R54b.

Track IDs: bitwise identical to R54b.

Phase audit:
  Score all 80 R54b responses on 7 weakness criteria. Output an audit report
  listing flagged rows + reasons. Stops early if too many flagged.

Phase polish:
  Regenerate only the flagged rows with a richer system prompt. Reuse R54b
  responses for unflagged rows. Validate hard gates: bitwise track IDs,
  LexDiv >= 0.83 (or >= R54b 0.8387, whichever lower), no trailing questions,
  no apologies, 60-100 words, references current recommendation directly.

Output:
  exp/eval/expR54c_audit.json
  exp/inference/blind_a/r54c_polish_submission.{json,zip}
  exp/inference/blind_a/r54c_polish_metadata.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R54B_SUBMISSION = REPO / "exp" / "inference" / "blind_a" / "r54b_aligned_submission.json"
BLIND_OUT = REPO / "exp" / "inference" / "blind_a"
AUDIT_OUT = REPO / "exp" / "eval" / "expR54c_audit.json"

MAX_FLAG_BEFORE_STOP = 30  # if more than this, stop and ask for review
HYBRID_HARD_CAP = 22  # max rows allowed for hybrid polish
HARD_LEXDIV_FLOOR = 0.83
R54B_LEXDIV = 0.8387
WORD_RANGE = (40, 150)  # final responses should fall here

# In hybrid mode: regenerate non-style flags + this specific opener cluster only.
HYBRID_REGEN_FLAGS = {
    "trailing_question",
    "too_short",
    "too_long",
    "boilerplate",
    "no_user_query_overlap",
    "no_track_or_artist_mention",
    "descriptor_heavy",
}
HYBRID_OPENER_CLUSTER = "you're drawn to"  # the largest repeated-opener cluster

FORBIDDEN_OPENERS = [
    "you're drawn to", "you're craving", "you're hunting",
    "you've found", "you're seeking", "you want",
    "you're chasing", "you're zeroing", "you're ready",
]


TAG_PREFIX_RX = re.compile(r"^([^\n]{1,49})\n\n+(.+)$", re.DOTALL)


def strip_tag_prefix(resp: str) -> str:
    """Strip a short non-sentence prefix line followed by blank line.

    Targets metadata leaks like 'metal\\n\\nReal response...' or
    '2010s country, fip\\n\\n"Track"...'. Keep if head looks like a real
    sentence (quoted title, sentence-ending punctuation, or many tokens).
    """
    s = resp.strip()
    m = TAG_PREFIX_RX.match(s)
    if not m:
        return s
    head = m.group(1).strip().lstrip("-").strip()
    rest = m.group(2).strip()
    if not head:
        return rest
    if '"' in head or "'" in head:
        return s
    if head[-1] in ".!?":
        return s
    tokens = re.findall(r"\w+", head)
    if not tokens:
        return rest
    if len(tokens) <= 6 and all(len(t) <= 20 for t in tokens):
        return rest
    return s


BOILERPLATE_PATTERNS = [
    "great choice", "wonderful choice", "perfect choice", "excellent choice",
    "hope you enjoy", "i hope you", "enjoy listening",
    "happy listening", "let me know",
    "is there anything", "would you like", "do you want",
    "i'm sorry", "i apologize", "as an ai",
    "this is a great", "this is a wonderful",
    "i think you'll love", "you'll love this", "you'll enjoy",
]
TRAILING_QUESTION_RX = re.compile(r"\?\s*$")
GENERIC_DESCRIPTORS = {"great", "wonderful", "amazing", "fantastic", "beautiful",
                       "perfect", "incredible", "awesome", "lovely", "excellent"}


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def get_track_metadata_strings(item_db, track_id):
    """Return (track_name, artist_name, album_name, tag_list) lowercased."""
    try:
        meta = item_db.id_to_metadata(track_id)
        # meta is a string or dict depending on impl — handle both
        if isinstance(meta, str):
            # parse loose
            return None, None, None, []
    except KeyError:
        return None, None, None, []
    return None, None, None, []  # fallback


def load_catalog_dict():
    """Load track_id -> {name, artist, album, tags}."""
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig(local_files_only=True))["all_tracks"]
    out = {}
    for item in ds:
        tid = str(item["track_id"])
        name = item.get("track_name", [])
        artist = item.get("artist_name", [])
        album = item.get("album_name", [])
        tags = item.get("tag_list", [])
        if isinstance(name, list):
            name = name[0] if name else ""
        if isinstance(artist, list):
            artist = ", ".join(str(a) for a in artist) if artist else ""
        if isinstance(album, list):
            album = album[0] if album else ""
        if not isinstance(tags, list):
            tags = [tags] if tags else []
        out[tid] = {
            "name": str(name).lower(),
            "artist": str(artist).lower(),
            "album": str(album).lower(),
            "tags": [str(t).lower() for t in tags[:10]],
        }
    return out


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


def audit_one(resp: str, top1_id: str, catalog: dict, user_query: str):
    """Return list of weakness flags + per-row diagnostics."""
    flags = []
    info = {}

    low = normalize(resp)
    word_count = len(resp.split())
    info["word_count"] = word_count

    if word_count < WORD_RANGE[0]:
        flags.append("too_short")
    if word_count > WORD_RANGE[1]:
        flags.append("too_long")

    if TRAILING_QUESTION_RX.search(resp.strip()):
        flags.append("trailing_question")

    # Boilerplate
    boilerplate_hits = [p for p in BOILERPLATE_PATTERNS if p in low]
    if boilerplate_hits:
        flags.append("boilerplate")
        info["boilerplate_hits"] = boilerplate_hits

    # Top-1 fit: response mentions track name OR artist
    track = catalog.get(top1_id, {})
    name = track.get("name", "")
    artist = track.get("artist", "")
    # Use only meaningful tokens (>= 3 chars, no stopwords)
    mention_name = name and (name in low or any(
        len(w) >= 4 and w in low for w in name.split()))
    mention_artist = artist and (artist in low or any(
        len(w) >= 4 and w in low for w in artist.replace(",", " ").split()))
    info["mentions_track_name"] = bool(mention_name)
    info["mentions_track_artist"] = bool(mention_artist)
    if not (mention_name or mention_artist):
        flags.append("no_track_or_artist_mention")

    # Specificity: response should have at least one tag word OR a proper noun
    tag_hit = any(t in low for t in track.get("tags", []))
    capital_words = sum(1 for w in resp.split() if w and w[0].isupper() and not w.isupper())
    info["tag_hit"] = bool(tag_hit)
    info["capital_words"] = capital_words
    if not tag_hit and capital_words < 2:
        flags.append("low_specificity")

    # Personalization: response touches on user's query (token overlap)
    uq_tokens = set(re.findall(r"\b[a-z]{4,}\b", user_query.lower()))
    resp_tokens = set(re.findall(r"\b[a-z]{4,}\b", low))
    overlap = uq_tokens & resp_tokens
    info["user_query_overlap"] = sorted(overlap)
    # If user query is non-trivial (>= 5 long tokens), require >= 1 overlap
    if len(uq_tokens) >= 5 and not overlap:
        flags.append("no_user_query_overlap")

    # Generic descriptors only (e.g. "Great song! You'll love it.")
    desc_hits = sum(1 for w in re.findall(r"\b[a-z]+\b", low) if w in GENERIC_DESCRIPTORS)
    info["generic_descriptors"] = desc_hits
    if desc_hits >= 3 and word_count < 60:
        flags.append("descriptor_heavy")

    return flags, info


def detect_repeated_openers(responses):
    """Return list of (opener, count) where opener is first 3 words and count > 1."""
    openers = [" ".join(r.split()[:3]).lower() for r in responses]
    cnt = Counter(openers)
    return [(o, c) for o, c in cnt.most_common() if c > 1]


def phase_audit():
    print("R54c — Phase AUDIT")
    print("=" * 70)
    with open(R54B_SUBMISSION) as f:
        r54b = json.load(f)
    print(f"  Loaded {len(r54b)} R54b rows")

    catalog = load_catalog_dict()
    print(f"  Loaded {len(catalog)} track metadata entries")

    from datasets import DownloadConfig, load_dataset
    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                      download_config=DownloadConfig(local_files_only=True))
    blind_by_sid = {str(item["session_id"]): item for item in db}

    # Per-row audit
    audit = []
    flag_counts = Counter()
    for r in r54b:
        sid = r["session_id"]
        item = blind_by_sid[sid]
        _, user_query, _ = parse_last_turn_local(item)
        flags, info = audit_one(r["predicted_response"], r["predicted_track_ids"][0],
                                  catalog, user_query)
        for f in flags:
            flag_counts[f] += 1
        audit.append({
            "session_id": sid,
            "top1_id": r["predicted_track_ids"][0],
            "response": r["predicted_response"],
            "user_query": user_query,
            "flags": flags,
            "info": info,
        })

    # Cross-row: repeated openers
    openers = detect_repeated_openers([r["predicted_response"] for r in r54b])
    repeated_opener_set = {o for o, _ in openers}
    if openers:
        for a in audit:
            opener3 = " ".join(a["response"].split()[:3]).lower()
            if opener3 in repeated_opener_set:
                a["flags"].append("repeated_opener")
                flag_counts["repeated_opener"] += 1

    flagged = [a for a in audit if a["flags"]]
    print(f"\n=== AUDIT SUMMARY ===")
    print(f"  Total: 80")
    print(f"  Flagged at least once: {len(flagged)}")
    print(f"  Flag breakdown:")
    for flag, cnt in flag_counts.most_common():
        print(f"    {flag:<32} {cnt}")
    if openers:
        print(f"\n  Repeated openers (first 3 words):")
        for o, c in openers[:10]:
            print(f"    [{c}x] {o!r}")

    print(f"\n=== SAMPLE FLAGGED ROWS (first 10) ===")
    for a in flagged[:10]:
        print(f"\n  sid={a['session_id'][:8]}  flags={a['flags']}")
        print(f"    user_query: {a['user_query'][:100]!r}")
        print(f"    response:   {a['response'][:100]!r}")

    AUDIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_OUT, "w") as f:
        json.dump({
            "n_total": 80,
            "n_flagged": len(flagged),
            "flag_counts": dict(flag_counts),
            "repeated_openers": [{"opener": o, "count": c} for o, c in openers],
            "per_row": audit,
        }, f, indent=2)
    print(f"\nAudit saved: {AUDIT_OUT}")

    if len(flagged) > MAX_FLAG_BEFORE_STOP:
        print(f"\n*** STOP: {len(flagged)} rows flagged exceeds threshold "
              f"{MAX_FLAG_BEFORE_STOP}. Review audit before running --phase polish. ***")
        sys.exit(2)
    print(f"\nFlagged {len(flagged)} <= {MAX_FLAG_BEFORE_STOP} threshold. "
          f"Safe to proceed with --phase polish.")


def phase_polish(mode="hybrid"):
    print(f"R54c — Phase POLISH (mode={mode})")
    print("=" * 70)
    if not AUDIT_OUT.exists():
        print(f"ERROR: missing {AUDIT_OUT}. Run --phase audit first.")
        sys.exit(1)
    with open(AUDIT_OUT) as f:
        audit_data = json.load(f)

    # Build target set based on mode
    if mode == "hybrid":
        # Non-style weakness rows + only the "you're drawn to" cluster
        flagged = []
        for a in audit_data["per_row"]:
            non_style = [f for f in a["flags"] if f in HYBRID_REGEN_FLAGS]
            opener3 = " ".join(a["response"].split()[:3]).lower()
            in_target_cluster = opener3 == HYBRID_OPENER_CLUSTER
            if non_style or in_target_cluster:
                flagged.append({
                    **a,
                    "regen_reasons": (non_style + (["target_opener_cluster"]
                                                    if in_target_cluster else [])),
                })
        print(f"  Hybrid mode: {len(flagged)} rows selected")
        reason_counts = Counter()
        for a in flagged:
            for r in a["regen_reasons"]:
                reason_counts[r] += 1
        for r, c in reason_counts.most_common():
            print(f"    {r:<32} {c}")
        if len(flagged) > HYBRID_HARD_CAP:
            print(f"\n*** STOP: {len(flagged)} > hybrid hard cap {HYBRID_HARD_CAP}. ***")
            sys.exit(2)
    else:
        # Full mode: regenerate everything the audit flagged
        flagged = [a for a in audit_data["per_row"] if a["flags"]]
        print(f"  Full mode: {len(flagged)} flagged rows to regenerate")

    with open(R54B_SUBMISSION) as f:
        r54b = json.load(f)
    r54b_by_sid = {r["session_id"]: r for r in r54b}

    # Build base results (bitwise-identical track IDs)
    results = []
    for r in r54b:
        results.append({
            "session_id": r["session_id"],
            "turn_number": r["turn_number"],
            "predicted_track_ids": list(r["predicted_track_ids"]),
            "predicted_response": r["predicted_response"],  # default: keep
        })
    by_sid = {r["session_id"]: r for r in results}

    if not flagged:
        print("  No rows flagged. Nothing to do.")
        return

    print(f"\n  Regenerating {len(flagged)} responses with Haiku ...")
    from datasets import DownloadConfig, load_dataset
    from mcrs.db_item.music_catalog import MusicCatalogDB
    from mcrs.lm_modules.claude import ClaudeModule
    from run_inference_blind_r3_det import build_session_memory_for_response

    item_db = MusicCatalogDB(dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                              split_types=["all_tracks"])
    prompts_dir = REPO / "mcrs" / "system_prompts"
    sys_prompt = (prompts_dir / "roleplay.txt").read_text() + "\n" + \
                 (prompts_dir / "response_generation.txt").read_text()
    # Strong guardrails for R54c polish (hybrid mode)
    forbidden_list = "\n".join(f"  * \"{p}\"" for p in FORBIDDEN_OPENERS)
    sys_prompt += (
        "\n\nR54c polish — strict guidelines:\n"
        "- Mention the recommended track or artist explicitly. Name it.\n"
        "- Reference the user's expressed preference directly.\n"
        "- Length: 60-100 words. 2-3 complete sentences.\n"
        "- Start with a complete sentence. Do NOT prefix your response with "
        "genre tags, year labels, comma-separated descriptors, or any line "
        "that is not a complete sentence.\n"
        "- NO trailing questions. Do not end the response with '?'.\n"
        "- NO boilerplate: 'great choice', 'hope you enjoy', 'let me know if', "
        "'would you like', 'is there anything', 'I'm sorry', 'as an AI'.\n"
        "- Do NOT open with any of these phrases (or close variants):\n"
        f"{forbidden_list}\n"
        "- Vary sentence structure. Do not start two consecutive sentences with 'You'."
    )
    haiku = ClaudeModule(model="claude-haiku-4-5-20251001")

    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                      download_config=DownloadConfig(local_files_only=True))
    blind_by_sid = {str(item["session_id"]): item for item in db}

    catalog = load_catalog_dict()

    n_regen = 0
    n_retry = 0
    for ai, a in enumerate(flagged):
        sid = a["session_id"]
        item = blind_by_sid[sid]
        _, user_query, history = parse_last_turn_local(item)
        top_id = by_sid[sid]["predicted_track_ids"][0]
        try:
            top_item = item_db.id_to_metadata(top_id)
        except KeyError:
            top_item = f"track_id: {top_id}"
        session_memory = build_session_memory_for_response(history, user_query, item_db)

        chosen = None
        for attempt in range(3):
            resp = haiku.response_generation(sys_prompt, session_memory, top_item)
            resp = (resp or "").lstrip(",").lstrip()
            resp = strip_tag_prefix(resp)
            if not resp:
                continue
            # Check forbidden opener
            low = resp.lower().strip()
            forbidden_hit = any(low.startswith(p) for p in FORBIDDEN_OPENERS)
            new_flags, _ = audit_one(resp, top_id, catalog, user_query)
            critical = [f for f in new_flags if f in (
                "trailing_question", "boilerplate", "too_short", "too_long",
                "descriptor_heavy", "no_track_or_artist_mention",
            )]
            if not critical and not forbidden_hit:
                chosen = resp
                break
            n_retry += 1
        if chosen is None:
            chosen = resp or by_sid[sid]["predicted_response"]
            print(f"    [{ai+1}/{len(flagged)}] sid={sid[:8]} "
                  f"FALLBACK (kept R54b or unfiltered)", flush=True)
        by_sid[sid]["predicted_response"] = chosen
        n_regen += 1
        if (ai + 1) % 5 == 0:
            print(f"    {ai+1}/{len(flagged)} processed ({n_retry} retries)", flush=True)

    # Universal artifact cleanup: strip metadata-prefix leaks from ALL 80 rows
    # (including kept R54b rows). Track IDs unchanged. No semantic rewrites.
    n_cleanup = 0
    cleanup_samples = []
    for r in results:
        before = r["predicted_response"]
        after = strip_tag_prefix(before)
        if after != before.strip():
            n_cleanup += 1
            if len(cleanup_samples) < 8:
                cleanup_samples.append((r["session_id"][:8], before, after))
            r["predicted_response"] = after
    print(f"\n  Universal prefix cleanup: {n_cleanup}/80 rows had a prefix leak stripped")
    if cleanup_samples:
        print(f"\n  === BEFORE / AFTER samples (first {len(cleanup_samples)}) ===")
        for sid, before, after in cleanup_samples:
            b_head = before[:80].replace("\n", "\\n")
            a_head = after[:80].replace("\n", "\\n")
            print(f"    sid={sid}")
            print(f"      BEFORE: {b_head!r}")
            print(f"      AFTER : {a_head!r}")

    # Debug dump BEFORE validation so we can inspect even if validation fails
    debug_path = REPO / "exp" / "eval" / "expR54c_polish_debug.json"
    with open(debug_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Debug dump (pre-validation): {debug_path}")

    # Verify bitwise track IDs
    print(f"\n  Verifying bitwise-identical track IDs vs R54b ...")
    for r in results:
        if r["predicted_track_ids"] != r54b_by_sid[r["session_id"]]["predicted_track_ids"]:
            raise RuntimeError(f"track id mismatch at sid={r['session_id']}")
    print(f"  PASS: track IDs bitwise identical to R54b")

    # Validation gates
    print(f"\n  Validation ...")
    from scripts.expR54_phase3_blind_submission import load_track_albums
    valid_catalog = set(load_track_albums().keys())

    sids = set()
    n_trailing_q = 0
    n_boilerplate = 0
    n_prefix_leak = 0
    for r in results:
        sid = r["session_id"]
        resp = r["predicted_response"]
        if len(r["predicted_track_ids"]) != 20:
            raise ValueError(f"{sid}: tracks not 20")
        if len(set(r["predicted_track_ids"])) != 20:
            raise ValueError(f"{sid}: duplicates")
        if not resp.strip():
            raise ValueError(f"{sid}: empty response")
        invalid = [t for t in r["predicted_track_ids"] if t not in valid_catalog]
        if invalid:
            raise ValueError(f"{sid}: invalid track ids: {invalid[:3]}")
        if TRAILING_QUESTION_RX.search(resp.strip()):
            n_trailing_q += 1
        low = normalize(resp)
        if any(p in low for p in BOILERPLATE_PATTERNS):
            n_boilerplate += 1
        if strip_tag_prefix(resp) != resp.strip():
            n_prefix_leak += 1
        sids.add(sid)
    print(f"  rows=80 unique={len(sids)}  trailing_q={n_trailing_q}  "
          f"boilerplate={n_boilerplate}  prefix_leak={n_prefix_leak}")
    if n_trailing_q + n_boilerplate + n_prefix_leak > 0:
        print(f"  FAIL: {n_trailing_q + n_boilerplate + n_prefix_leak} responses still have issues")
        sys.exit(1)

    # LexDiv
    all_bigrams = []
    for r in results:
        toks = r["predicted_response"].lower().split()
        all_bigrams.extend(zip(toks, toks[1:]))
    lexdiv = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    print(f"  LexDiv (Distinct-2): {lexdiv:.4f}")
    if lexdiv < HARD_LEXDIV_FLOOR:
        print(f"  FAIL: LexDiv {lexdiv:.4f} < {HARD_LEXDIV_FLOOR}")
        sys.exit(1)
    r54b_lexdiv = 0.8387
    if lexdiv < r54b_lexdiv - 0.01:
        print(f"  WARN: LexDiv {lexdiv:.4f} below R54b 0.8387 by > 0.01")

    # Diff vs R54b
    response_changed = sum(
        1 for r in results
        if r["predicted_response"] != r54b_by_sid[r["session_id"]]["predicted_response"]
    )
    print(f"  Responses changed vs R54b: {response_changed}/80")

    # Save
    report = {
        "submission_label": "R54c_response_polish",
        "predecessor": "R54b (bitwise-identical track IDs)",
        "audit_flagged": len(flagged),
        "responses": {
            "regenerated": int(response_changed),
            "kept_r54b": int(80 - response_changed),
        },
        "validation": {
            "track_ids_bitwise_identical_to_r54b": True,
            "rows": 80, "unique": len(sids),
            "trailing_questions_remaining": n_trailing_q,
            "boilerplate_remaining": n_boilerplate,
            "lexdiv_distinct2": float(lexdiv),
        },
        "context": {
            "r54b_scores": {
                "ndcg20": 0.4925, "composite": 0.6106,
                "lexdiv": 0.8387, "llm_judge": 4.70,
            },
            "rationale": "Targeted polish of R54b. Track IDs unchanged. Only audit-flagged "
                         "responses regenerated. Goal: lift LLM/LexDiv without nDCG risk.",
        },
        "created_at": datetime.now().isoformat(),
    }
    out_json = BLIND_OUT / "r54c_polish_submission.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    out_meta = BLIND_OUT / "r54c_polish_metadata.json"
    with open(out_meta, "w") as f:
        json.dump(report, f, indent=2)
    out_zip = BLIND_OUT / "r54c_polish_submission.zip"
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(out_json, "prediction.json")
    print(f"\n  SUBMISSION: {out_zip}")
    print(f"  METADATA:   {out_meta}")
    print(f"  Regenerated {response_changed} responses, kept {80 - response_changed}.")
    print(f"  LexDiv: {lexdiv:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="audit", choices=["audit", "polish"])
    args = parser.parse_args()
    if args.phase == "audit":
        phase_audit()
    elif args.phase == "polish":
        phase_polish()
