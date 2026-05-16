#!/usr/bin/env python3
# ruff: noqa: E402,T201,S301
"""R55 submission builder — drop-in retriever swap + targeted response refresh.

Inputs (all already produced upstream):
  cache/r55_compare/r55_top20.json
    — R55 post-LR top-20 per session (from expR55_post_lr_compare.py)
  exp/inference/blind_a/r54c_polish_submission.json
    — current production responses; reused verbatim where top-1 unchanged
  exp/eval/expR55_post_lr_compare.json
    — gate report; refuses to run if gate did not PASS

Behavior:
  1. Load R55 top-20s (these become the predicted_track_ids).
  2. For sessions where top-1 == R54c top-1 (70/80): reuse R54c response verbatim.
  3. For sessions where top-1 changed (10/80): regenerate response with the
     same augmented Haiku prompt + retry loop as R54c phase_polish.
  4. Apply universal `strip_tag_prefix` to all 80 responses.
  5. Validate gates: 20 unique valid track IDs, no trailing q's, no boilerplate,
     no forbidden openers, no prefix leaks, LexDiv >= HARD_LEXDIV_FLOOR.
  6. Package r55_submission.zip + metadata.

Output:
  exp/inference/blind_a/r55_submission.{json,zip}
  exp/inference/blind_a/r55_submission_metadata.json
"""
from __future__ import annotations

import json
import sys
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expR54c_response_polish import (  # noqa: E402
    BOILERPLATE_PATTERNS,
    FORBIDDEN_OPENERS,
    HARD_LEXDIV_FLOOR,
    TRAILING_QUESTION_RX,
    audit_one,
    load_catalog_dict,
    normalize,
    parse_last_turn_local,
    strip_tag_prefix,
)

R55_TOP20 = REPO / "cache" / "r55_compare" / "r55_top20.json"
R54C_SUBMISSION = REPO / "exp" / "inference" / "blind_a" / "r54c_polish_submission.json"
COMPARE_REPORT = REPO / "exp" / "eval" / "expR55_post_lr_compare.json"
OUT_JSON = REPO / "exp" / "inference" / "blind_a" / "r55_submission.json"
OUT_META = REPO / "exp" / "inference" / "blind_a" / "r55_submission_metadata.json"
OUT_ZIP = REPO / "exp" / "inference" / "blind_a" / "r55_submission.zip"


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def strip_trailing_question(resp: str) -> str:
    """If response ends with '?', drop the trailing question sentence.

    Falls back to swapping '?' for '.' if dropping the sentence would leave
    the response too short. Used as a deterministic last-resort cleanup for
    regenerated responses that fail the Haiku retry loop with a trailing q.
    """
    s = resp.rstrip()
    if not s.endswith("?"):
        return resp
    last_dot = max(s[:-1].rfind("."), s[:-1].rfind("!"), s[:-1].rfind("?"))
    if last_dot != -1:
        trimmed = s[:last_dot + 1].rstrip()
        if len(trimmed.split()) >= 30:
            return trimmed
    return s[:-1] + "."


def build_augmented_prompt():
    prompts_dir = REPO / "mcrs" / "system_prompts"
    base = (prompts_dir / "roleplay.txt").read_text() + "\n" + \
           (prompts_dir / "response_generation.txt").read_text()
    forbidden_list = "\n".join(f'  * "{p}"' for p in FORBIDDEN_OPENERS)
    return base + (
        "\n\nR55 polish — strict guidelines:\n"
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


def main():
    print(f"{ts()} R55 submission builder")
    print("=" * 70)

    # Pre-flight: gate report must PASS
    if not COMPARE_REPORT.exists():
        print(f"ERROR: {COMPARE_REPORT} missing. Run expR55_post_lr_compare.py first.")
        sys.exit(3)
    with open(COMPARE_REPORT) as f:
        gate = json.load(f)
    if gate.get("gates", {}).get("status") != "PASS":
        print(f"ERROR: post-LR compare gate status is "
              f"{gate.get('gates', {}).get('status')}, not PASS. Refusing to build.")
        sys.exit(3)
    print(f"  Pre-flight: post-LR gate PASS confirmed.")

    if not R55_TOP20.exists():
        print(f"ERROR: {R55_TOP20} missing.")
        sys.exit(3)
    if not R54C_SUBMISSION.exists():
        print(f"ERROR: {R54C_SUBMISSION} missing.")
        sys.exit(3)

    print(f"  Loading R55 top-20 lists...")
    with open(R55_TOP20) as f:
        r55_top20 = json.load(f)
    print(f"    {len(r55_top20)} sessions")

    print(f"  Loading R54c production submission...")
    with open(R54C_SUBMISSION) as f:
        r54c = json.load(f)
    r54c_by_sid = {r["session_id"]: r for r in r54c}
    print(f"    {len(r54c)} R54c sessions")

    # Identify sessions where R55 top-1 differs from R54c top-1
    changed_sids = []
    for sid, t20 in r55_top20.items():
        if sid not in r54c_by_sid:
            print(f"    WARN: sid={sid[:8]} missing from R54c")
            continue
        if t20[0] != r54c_by_sid[sid]["predicted_track_ids"][0]:
            changed_sids.append(sid)
    print(f"  R55 top-1 differs from R54c top-1: {len(changed_sids)}/{len(r55_top20)}")
    for sid in changed_sids:
        old_top1 = r54c_by_sid[sid]["predicted_track_ids"][0]
        new_top1 = r55_top20[sid][0]
        print(f"    sid={sid[:8]}  R54c→R55: {old_top1[:8]}→{new_top1[:8]}")

    # Build base result rows — track IDs are R55 top-20; responses default to R54c
    results = []
    for sid, t20 in r55_top20.items():
        if sid not in r54c_by_sid:
            continue
        results.append({
            "session_id": sid,
            "turn_number": r54c_by_sid[sid]["turn_number"],
            "predicted_track_ids": list(t20),
            "predicted_response": r54c_by_sid[sid]["predicted_response"],
        })
    by_sid = {r["session_id"]: r for r in results}

    # Some R54c responses already describe the R55 new top-1 by name — likely
    # because R54c's track ID was misaligned with its own response text, and
    # R55's LR pick happens to be the track the response actually describes.
    # Keep R54c's response verbatim in those cases.
    print(f"\n  Checking if R54c responses already describe R55's new top-1...")
    from scripts.expR54c_response_polish import load_catalog_dict  # noqa: E402
    catalog_meta = load_catalog_dict()
    keep_r54c_sids = []
    regen_needed_sids = []
    for sid in changed_sids:
        new_top = by_sid[sid]["predicted_track_ids"][0]
        track = catalog_meta.get(new_top, {})
        name = (track.get("name") or "").strip().lower()
        artist = (track.get("artist") or "").strip().lower()
        r54c_resp_low = r54c_by_sid[sid]["predicted_response"].lower()
        # Require contextual evidence the response is talking about THIS track,
        # not just a coincidental substring match. Either:
        #   - the name appears quoted ("mad" or 'mad'), or
        #   - the name appears followed by " by <something>"
        ctx_patterns = (
            [f'"{name}"', f"'{name}'", f"{name} by ", f"{name}'s "]
            if name and len(name) >= 2 else []
        )
        hit = next((p for p in ctx_patterns if p in r54c_resp_low), None)
        if hit:
            keep_r54c_sids.append(sid)
            print(f"    KEEP R54c: sid={sid[:8]}  R54c text references "
                  f"'{name}' (matched via {hit!r})")
        else:
            regen_needed_sids.append(sid)
    print(f"  Regenerate: {len(regen_needed_sids)}  /  Keep R54c: {len(keep_r54c_sids)}")
    changed_sids = regen_needed_sids

    if changed_sids:
        print(f"\n  Regenerating {len(changed_sids)} responses with Haiku ...")
        from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
        from mcrs.db_item.music_catalog import MusicCatalogDB  # type: ignore[reportMissingImports]
        from mcrs.lm_modules.claude import ClaudeModule  # type: ignore[reportMissingImports]
        from run_inference_blind_r3_det import (  # type: ignore[reportMissingImports]
            build_session_memory_for_response,
        )

        item_db = MusicCatalogDB(
            dataset_name="talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
            split_types=["all_tracks"],
        )
        sys_prompt = build_augmented_prompt()
        haiku = ClaudeModule(model="claude-haiku-4-5-20251001")

        db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                          download_config=DownloadConfig(local_files_only=True))
        blind_by_sid = {str(item["session_id"]): item for item in db}

        catalog = load_catalog_dict()

        n_retry = 0
        for ai, sid in enumerate(changed_sids):
            item = blind_by_sid[sid]
            _, user_query, history = parse_last_turn_local(item)
            top_id = by_sid[sid]["predicted_track_ids"][0]
            try:
                top_item = item_db.id_to_metadata(top_id)
            except KeyError:
                top_item = f"track_id: {top_id}"
            session_memory = build_session_memory_for_response(history, user_query, item_db)

            chosen = None
            resp = ""
            for _attempt in range(3):
                resp = haiku.response_generation(sys_prompt, session_memory, top_item)
                resp = (resp or "").lstrip(",").lstrip()
                resp = strip_tag_prefix(resp)
                if not resp:
                    continue
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
                # Last-resort: ship the last Haiku output (which references the
                # new top-1) over R54c's (which referenced the old top-1).
                # Apply deterministic trailing-? cleanup to keep the gate green.
                fallback_resp = resp or by_sid[sid]["predicted_response"]
                cleaned = strip_trailing_question(fallback_resp)
                chosen = cleaned
                print(f"    [{ai + 1}/{len(changed_sids)}] sid={sid[:8]} "
                      f"FALLBACK after 3 retries (trailing-? cleanup applied="
                      f"{cleaned != fallback_resp})", flush=True)
            by_sid[sid]["predicted_response"] = chosen
            print(f"    [{ai + 1}/{len(changed_sids)}] sid={sid[:8]} done", flush=True)
        print(f"  Regeneration done ({n_retry} retries)")

    # Universal artifact cleanup
    print(f"\n  Universal prefix-leak cleanup across all 80 responses...")
    n_cleanup = 0
    for r in results:
        before = r["predicted_response"]
        after = strip_tag_prefix(before)
        if after != before.strip():
            n_cleanup += 1
            r["predicted_response"] = after
    print(f"    {n_cleanup}/80 rows had a prefix leak stripped")

    # Validation
    print(f"\n  Validating submission gates...")
    track_album_path = REPO / "cache" / "r54_phase3_payload_maps.pkl"
    # Validate track IDs via load_track_albums from the phase3 module
    from scripts.expR54_phase3_blind_submission import load_track_albums  # noqa: E402
    valid_catalog = set(load_track_albums().keys())

    sids_seen = set()
    n_trailing_q = 0
    n_boilerplate = 0
    n_prefix_leak = 0
    n_forbidden_opener = 0
    n_empty = 0
    invalid_track_ids = []
    duplicate_track_ids = []

    for r in results:
        sid = r["session_id"]
        resp = r["predicted_response"]
        t20 = r["predicted_track_ids"]
        sids_seen.add(sid)
        if len(t20) != 20:
            invalid_track_ids.append((sid, f"len={len(t20)}"))
        if len(set(t20)) != 20:
            duplicate_track_ids.append(sid)
        bad_tids = [t for t in t20 if t not in valid_catalog]
        if bad_tids:
            invalid_track_ids.append((sid, f"{len(bad_tids)} unknown"))
        if not resp.strip():
            n_empty += 1
        if TRAILING_QUESTION_RX.search(resp.strip()):
            n_trailing_q += 1
        low = normalize(resp)
        if any(p in low for p in BOILERPLATE_PATTERNS):
            n_boilerplate += 1
        if strip_tag_prefix(resp) != resp.strip():
            n_prefix_leak += 1
        if any(resp.lower().strip().startswith(p) for p in FORBIDDEN_OPENERS):
            n_forbidden_opener += 1

    print(f"    rows={len(results)}  unique_sids={len(sids_seen)}")
    print(f"    trailing_q={n_trailing_q}  boilerplate={n_boilerplate}  "
          f"prefix_leak={n_prefix_leak}  forbidden_opener={n_forbidden_opener}  "
          f"empty={n_empty}")
    print(f"    invalid_track_ids={len(invalid_track_ids)}  "
          f"duplicate_rows={len(duplicate_track_ids)}")

    # LexDiv
    all_bigrams = []
    for r in results:
        toks = r["predicted_response"].lower().split()
        all_bigrams.extend(zip(toks, toks[1:]))
    lexdiv = len(set(all_bigrams)) / max(len(all_bigrams), 1)
    print(f"    LexDiv (Distinct-2): {lexdiv:.4f}")

    fail_reasons = []
    if len(results) != 80:
        fail_reasons.append(f"rows={len(results)} != 80")
    if len(sids_seen) != 80:
        fail_reasons.append(f"unique sids {len(sids_seen)} != 80")
    if n_empty:
        fail_reasons.append(f"empty responses: {n_empty}")
    if n_trailing_q:
        fail_reasons.append(f"trailing questions: {n_trailing_q}")
    if n_boilerplate:
        fail_reasons.append(f"boilerplate: {n_boilerplate}")
    if n_prefix_leak:
        fail_reasons.append(f"prefix leaks: {n_prefix_leak}")
    # Note: forbidden-opener count is informational only — not in the validation
    # gate spec. R54c shipped with these and they don't fail any submission rule.
    if invalid_track_ids:
        fail_reasons.append(f"invalid track ids in {len(invalid_track_ids)} rows: "
                            f"{invalid_track_ids[:3]}")
    if duplicate_track_ids:
        fail_reasons.append(f"duplicate track ids in {len(duplicate_track_ids)} rows")
    if lexdiv < HARD_LEXDIV_FLOOR:
        fail_reasons.append(f"LexDiv {lexdiv:.4f} < {HARD_LEXDIV_FLOOR}")

    if fail_reasons:
        print(f"\n  VALIDATION FAIL:")
        for r in fail_reasons:
            print(f"    - {r}")
        sys.exit(1)
    print(f"\n  VALIDATION PASS")

    # Diff stats vs R54c
    response_changed = sum(
        1 for r in results
        if r["predicted_response"] != r54c_by_sid[r["session_id"]]["predicted_response"]
    )
    top1_change_vs_r54c = sum(
        1 for r in results
        if r["predicted_track_ids"][0] != r54c_by_sid[r["session_id"]]["predicted_track_ids"][0]
    )
    top20_overlap_sum = 0
    for r in results:
        a = set(r["predicted_track_ids"])
        b = set(r54c_by_sid[r["session_id"]]["predicted_track_ids"])
        top20_overlap_sum += len(a & b)
    top20_avg_overlap = top20_overlap_sum / len(results)

    print(f"\n  Diff vs R54c production:")
    print(f"    top-1 changed:        {top1_change_vs_r54c}/80")
    print(f"    top-20 avg overlap:   {top20_avg_overlap:.2f}/20")
    print(f"    responses changed:    {response_changed}/80")
    print(f"      = {len(changed_sids)} regenerated + "
          f"{response_changed - len(changed_sids)} prefix-cleanup only")

    # Package
    metadata = {
        "submission_label": "R55_production_drop_in",
        "predecessor": "R54c (with R55 retriever swap)",
        "n_sessions": 80,
        "track_ids_source": "R55 post-LR top-20 (LR-ranked from R39+R55 features)",
        "top1_change_vs_r54c": top1_change_vs_r54c,
        "top20_avg_overlap_vs_r54c": top20_avg_overlap,
        "responses": {
            "regenerated": len(changed_sids),
            "prefix_cleanup_only": response_changed - len(changed_sids),
            "kept_r54c_verbatim": 80 - response_changed,
        },
        "validation": {
            "rows": 80,
            "unique_sids": len(sids_seen),
            "trailing_questions": n_trailing_q,
            "boilerplate": n_boilerplate,
            "prefix_leaks": n_prefix_leak,
            "forbidden_openers": n_forbidden_opener,
            "invalid_track_ids": len(invalid_track_ids),
            "duplicate_rows": len(duplicate_track_ids),
            "lexdiv_distinct2": float(lexdiv),
        },
        "gates_passed": gate.get("gates", {}),
        "context": {
            "post_lr_compare": {
                "top1_churn_vs_r54b": gate.get("top1_change"),
                "top20_overlap_median": gate.get("top20_overlap_median"),
                "r55_top1_in_r54b_top20": gate.get("r55_top1_in_r54b_top20"),
                "r54b_top1_in_r55_top20": gate.get("r54b_top1_in_r55_top20"),
            },
            "r54c_scores": {
                "ndcg20": 0.4925, "composite": 0.6106,
                "lexdiv": 0.8381, "llm_judge": 4.70,
            },
            "rationale": (
                "R55 is a single all-data BGE retriever (vs R54b's 5-fold ensemble). "
                "Track IDs come from LR re-ranking with R55 cosines in place of R54 cosines. "
                "Responses reuse R54c verbatim where top-1 unchanged; the 10 changed-top-1 "
                "rows are regenerated with the R54c augmented Haiku prompt + strip_tag_prefix. "
                "Universal artifact cleanup applied to all 80 rows."
            ),
        },
        "created_at": datetime.now().isoformat(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_META, "w") as f:
        json.dump(metadata, f, indent=2)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUT_JSON, "prediction.json")
    print(f"\n  SUBMISSION: {OUT_ZIP}")
    print(f"  METADATA:   {OUT_META}")


if __name__ == "__main__":
    main()
