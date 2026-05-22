#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R73 concise-direct polish: targeted regen of R63c-repair to push LLM 4.85→4.90+.

R63c-repair holds production at composite 0.6224 (LLM 4.85, LexDiv 0.8438).
Per-dimension leaderboard analysis shows LLM headroom (semintelligence at
4.95) is the highest-ROI dimension while retrieval is saturated (R72).

R73 strategy:
- Tracks bitwise identical to R63c-repair (zero churn).
- Audit R63c-repair rows for verbose/hedgy outliers (long openers, hedge
  phrases, multiple parentheticals, high comma count).
- Select 12-18 weakest rows by style score.
- Regenerate ONLY those rows with semintelligence-style prompt: direct
  verdict opener, concrete reason, no hedging, 60-80 words.
- Rotate through 5 direct-style archetypes for LexDiv preservation.
- Hard LexDiv floor 0.840 (R63c-repair is 0.8438; loss tolerance ~0.004).

Past context:
- R64 (full concise-direct pass) failed at LexDiv 0.8294 < 0.830 floor.
  R73 is partial pass (15-18 of 80 rows), proportional LexDiv impact ≈ 0.003.
- R63b (25 rows regen), R63c (15 rows), R63c-repair (15 rows) all used
  targeted regen successfully.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.expR63_e1_response_only as r63
import scripts.expR63c_targeted_polish as r63c

OUT_DIR = REPO / "exp" / "inference" / "blind_a"
R63C_REPAIR_ROWS = OUT_DIR / "r63c_repair_rows_final.jsonl"
R63C_REPAIR_METADATA = OUT_DIR / "r63c_repair_polish_submission.metadata.json"

PERSISTED_ROWS = OUT_DIR / "r73_concise_direct_rows_persisted.jsonl"
FINAL_ROWS = OUT_DIR / "r73_concise_direct_rows_final.jsonl"
OUT_ZIP = OUT_DIR / "r73_concise_direct_submission.zip"
OUT_METADATA = OUT_DIR / "r73_concise_direct_submission.metadata.json"
OUT_DOC = REPO / "docs" / "r73_concise_direct_result.md"
AUDIT_JSON = REPO / "exp" / "eval" / "expR73_audit.json"

MODEL_ID = "claude-opus-4-7"
PROMPT_VERSION = (
    "R73 v1; targeted concise-direct regen of 15 weakest R63c-repair rows; "
    "semintelligence-style direct verdict opener with 5-archetype rotation"
)
MAX_TOKENS = 240
REGEN_WORD_TARGET = (60, 82)
LEXDIV_FLOOR = 0.840
LEXDIV_BORDERLINE_FLOOR = 0.838
DEFAULT_TARGET_ROWS = 15
MIN_REGEN_ROWS = 12
MAX_REGEN_ROWS = 18

# Hedge / verbose patterns that R73 targets for removal
HEDGE_PATTERNS = [
    r"\bcomes off\b",
    r"\bfits that\b",
    r"\bcarries that\b",
    r"\bcaptures the\b",
    r"\bcaptures that\b",
    r"\bleans into\b",
    r"\bmatches the\b",
    r"\byou're (looking|chasing|trying|hunting|describing|digging|after)\b",
    r"\byou described\b",
    r"\byou mentioned\b",
    r"\bsounds like (?:what )?you\b",
    r"\bperfect for\b",
    r"\bexactly what\b",
    r"\bmakes it a\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)

# 5 direct-style archetypes for LexDiv balance
ARCHITECTURES = [
    {
        "name": "verdict-first",
        "instruction": (
            "Open with a direct verdict statement: name the recommendation as "
            "[Track] by [Artist] inside a confident, evidence-light claim about "
            "why it is the right pick. No hedging, no 'you're looking for' phrasing."
        ),
        "example": '"On and On by Pete Rock & C.L. Smooth is the textbook 90s East Coast pick."',
    },
    {
        "name": "concrete-detail-lead",
        "instruction": (
            "Open with one concrete musical detail (instrument, production "
            "element, vocal quality, or production technique), then name the "
            "recommendation as [Track] by [Artist] in the same sentence."
        ),
        "example": '"A chopped horn loop and dusty boom-bap drums drive On and On by Pete Rock & C.L. Smooth."',
    },
    {
        "name": "album-anchor",
        "instruction": (
            "Open with the album or release context (year, label, or album "
            "name), then introduce the track and a concrete reason inside one tight sentence."
        ),
        "example": '"From the 1992 Mecca and the Soul Brother sessions, On and On by Pete Rock & C.L. Smooth is the textbook East Coast cut."',
    },
    {
        "name": "comparison-pivot",
        "instruction": (
            "Open with a tight comparison or contrast pivoting from a prior "
            "track, artist, or session preference, then state the new pick "
            "as [Track] by [Artist] with one concrete reason."
        ),
        "example": '"Where the previous picks leaned soul-sample, On and On by Pete Rock & C.L. Smooth pulls in tighter East Coast boom-bap."',
    },
    {
        "name": "single-claim",
        "instruction": (
            "Open with a single declarative claim that ties the recommendation "
            "to one specific session detail in one sentence; name the track and "
            "artist inside that sentence."
        ),
        "example": '"The 90s sample-loop hip-hop you cited is exactly the territory of On and On by Pete Rock & C.L. Smooth."',
    },
]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def load_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clean_stale_package() -> None:
    for path in (OUT_ZIP, OUT_METADATA):
        if path.exists():
            path.unlink()


def write_zip(rows: list[dict[str, Any]]) -> None:
    payload = json.dumps(rows, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)


def r73_style_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score each row by 'verbose / hedgy / LLM-low' signals."""
    out = []
    for idx, row in enumerate(rows):
        resp = row.get("predicted_response", "")
        wc = len(resp.split())
        chars = len(resp)
        hedge_count = len(HEDGE_RE.findall(resp))
        em_dash = resp.count("—") + resp.count("–")
        parens = resp.count("(")
        comma_count = resp.count(",")
        first_sentence = resp.split(".", 1)[0] if "." in resp else resp
        first_words = first_sentence.split()
        opener_uses_track = bool(re.match(r"^[A-Z][^,]*\bby\s+[A-Z]", first_sentence))
        opener_hedges = bool(HEDGE_RE.search(first_sentence))
        long_opener = len(first_words) > 30

        score = 0
        if chars > 500: score += 2
        if chars > 540: score += 1
        if wc > 95: score += 1
        if hedge_count >= 2: score += 2
        elif hedge_count == 1: score += 1
        if em_dash >= 2: score += 1
        if parens >= 1: score += 1
        if comma_count >= 6: score += 1
        if long_opener: score += 1
        if opener_hedges: score += 1
        if not opener_uses_track and len(first_words) > 15: score += 1

        out.append({
            "row_index": idx,
            "session_id": row["session_id"],
            "weakness_score": score,
            "chars": chars,
            "word_count": wc,
            "hedge_count": hedge_count,
            "em_dash": em_dash,
            "parens": parens,
            "comma_count": comma_count,
            "long_opener": long_opener,
            "opener_uses_track": opener_uses_track,
            "opener_hedges": opener_hedges,
            "before_response": resp,
        })
    return out


def select_weak_rows(audit: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    if target < MIN_REGEN_ROWS or target > MAX_REGEN_ROWS:
        raise ValueError(
            f"target must be between {MIN_REGEN_ROWS} and {MAX_REGEN_ROWS}: {target}"
        )
    ranked = sorted(
        audit,
        key=lambda a: (-int(a["weakness_score"]), -int(a["chars"])),
    )
    return ranked[:target]


def render_prompt(
    item: dict[str, Any],
    top_meta: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    row_index: int,
    current_response: str,
    architecture: dict[str, str],
) -> str:
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    album = top_meta.get("album_name") or "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"
    tags = r63.clean_tags(top_meta.get("tag_list") or [])

    return "\n".join([
        "Rewrite the following music recommender response in a more direct, "
        "concise style that an LLM judge would reward.",
        "",
        "Hard requirements:",
        f"- {REGEN_WORD_TARGET[0]}-{REGEN_WORD_TARGET[1]} words. STRICT.",
        f"- Name the recommendation as {title} by {artist} in the first sentence.",
        "- Lead with a confident verdict or concrete musical claim — NO hedging.",
        "- Name at least 2 concrete musical attributes (genre, era, instrument, "
        "production technique, lyrical theme).",
        "- Reference at least 1 specific session detail (prior track, prior "
        "artist, stated preference, artwork clue, or conversation context).",
        "- Final sentence should land cleanly, not trail off.",
        "",
        f"Use the {architecture['name']} sentence pattern for this row: "
        f"{architecture['instruction']}",
        f"(Reference only — DO NOT copy this example wording: {architecture['example']})",
        "",
        "Banned phrases / patterns (do not use ANY of these):",
        "- 'comes off', 'fits that', 'carries that', 'captures the', 'captures that'",
        "- 'leans into', 'matches the', 'sounds like what you'",
        "- 'you're looking for', 'you're chasing', 'you're trying', 'you're after'",
        "- 'you described', 'you mentioned', 'perfect for', 'exactly what'",
        "- 'makes it a', 'lands', 'delivers', 'right in', 'pairs well'",
        "- Boilerplate openers: 'If you', 'You might', 'Here's a', 'For fans of'",
        "- Vague filler: 'vibe', 'journey', 'soundscape', 'captures the essence'",
        "- Em-dashes or parentheticals — write in clean direct prose only.",
        "- Do not end with a question.",
        "- Do not output prompt labels, bullets, markdown, or quotes around the full text.",
        "",
        f"User profile: {r63.compact_profile(item.get('user_profile'))}",
        f"Conversation goal: {r63.compact_goal(item.get('conversation_goal'))}",
        "Conversation:",
        r63.conversation_lines(item, catalog),
        "",
        "Top recommendation metadata:",
        f"Track: {title}",
        f"Artist: {artist}",
        f"Album: {album}",
        f"Release date: {release}",
        f"Tags: {', '.join(tags) if tags else '(none)'}",
        "",
        "Current R63c-repair response being polished. Keep true specifics, but "
        "deliver them in tighter, more direct prose:",
        current_response,
    ])


def validate_regenerated_text(text: str, top_meta: dict[str, Any]) -> list[str]:
    failures = []
    wc = len(text.split())
    if wc < REGEN_WORD_TARGET[0] or wc > REGEN_WORD_TARGET[1]:
        failures.append(f"word_count {wc} outside [{REGEN_WORD_TARGET[0]}, {REGEN_WORD_TARGET[1]}]")
    # Banned phrase check
    banned = [
        "comes off", "fits that", "carries that", "captures the",
        "captures that", "leans into", "matches the",
        "you're looking", "you're chasing", "you're trying",
        "you're after", "you described", "you mentioned",
        "perfect for", "exactly what", "makes it a",
        "vibe", "journey", "soundscape",
    ]
    for phrase in banned:
        if re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE):
            failures.append(f"banned_phrase: {phrase!r}")
    # Em-dash / parenthetical check (we want clean prose)
    if "—" in text or "–" in text:
        failures.append("contains em-dash")
    if "(" in text or ")" in text:
        failures.append("contains parenthetical")
    if text.rstrip().endswith("?"):
        failures.append("ends with question mark")
    # Must contain track + artist
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    if title.lower() not in text.lower():
        failures.append(f"missing_track_name: {title!r}")
    if artist.lower() not in text.lower():
        failures.append(f"missing_artist_name: {artist!r}")
    return failures


def call_opus(
    client: Any,
    system: str,
    user_prompt: str,
    usage: r63.UsageTotals,
    model: str,
) -> str:
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    usage.add(getattr(message, "usage", None))
    if getattr(message, "stop_reason", None) != "end_turn":
        raise RuntimeError(
            f"Claude stop_reason={getattr(message, 'stop_reason', None)} for model={model}"
        )
    return r63.normalize_ws("".join(getattr(part, "text", "") for part in message.content))


def generate_one(
    client: Any,
    item: dict[str, Any],
    top_meta: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    row_index: int,
    current_response: str,
    architecture: dict[str, str],
    usage: r63.UsageTotals,
    model: str,
    max_retries: int = 2,
) -> tuple[str, list[str], int]:
    """Try up to max_retries+1 times; return (text, failures, attempts_used)."""
    system = (
        "You write concise, personalized music recommendation responses with a "
        "direct, confident style. Follow every constraint EXACTLY. Output only "
        "the response text, no preamble."
    )
    last_response = ""
    last_failures: list[str] = []
    for attempt in range(max_retries + 1):
        prompt = render_prompt(
            item,
            top_meta,
            catalog,
            row_index,
            current_response,
            architecture,
        )
        if attempt > 0:
            prompt += (
                f"\n\nPrior attempt rejected for: {', '.join(last_failures)}. "
                "Try again. Stricter on word count and banned phrases."
            )
        response = call_opus(client, system, prompt, usage, model=model)
        failures = validate_regenerated_text(response, top_meta)
        last_response = response
        last_failures = failures
        if not failures:
            return response, [], attempt + 1
    return last_response, last_failures, max_retries + 1


def assert_tracks_unchanged(
    new_rows: list[dict[str, Any]],
    r63c_repair_rows: list[dict[str, Any]],
) -> None:
    """R73 must keep tracks bitwise identical to R63c-repair."""
    assert len(new_rows) == len(r63c_repair_rows)
    for i, (new, old) in enumerate(zip(new_rows, r63c_repair_rows)):
        if new["session_id"] != old["session_id"]:
            raise RuntimeError(f"row {i}: session_id mismatch")
        if new["turn_number"] != old["turn_number"]:
            raise RuntimeError(f"row {i}: turn_number mismatch")
        if new["predicted_track_ids"] != old["predicted_track_ids"]:
            raise RuntimeError(f"row {i}: predicted_track_ids mismatch (NOT bitwise identical)")


def compute_lexdiv_distinct2(rows: list[dict[str, Any]]) -> float:
    """Distinct-2 bigram diversity across all responses."""
    bigrams = []
    for r in rows:
        text = r["predicted_response"]
        words = re.findall(r"\b\w+\b", text.lower())
        bigrams.extend(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    return len(set(bigrams)) / len(bigrams)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    args = parser.parse_args()

    if args.model != MODEL_ID:
        raise SystemExit(
            f"Refusing model override {args.model!r}; R73 requires {MODEL_ID!r}."
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY. Pause.")

    import anthropic

    clean_stale_package()
    print(f"{ts()} R73 concise-direct polish")
    print(f"{ts()} Loading R63c-repair rows + Blind-A sessions + catalog")
    r63c_repair_rows = r63c.read_jsonl(R63C_REPAIR_ROWS)
    if len(r63c_repair_rows) != 80:
        raise RuntimeError(f"Expected 80 R63c-repair rows, found {len(r63c_repair_rows)}")
    r54c_rows = r63.load_r54c_rows()
    blind_by_sid = r63.load_blind_by_sid()
    catalog = r63.load_catalog()

    print(f"{ts()} Auditing R63c-repair for verbose/hedgy outliers")
    audit = r73_style_audit(r63c_repair_rows)
    selected = select_weak_rows(audit, args.target_rows)
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps({
        "all_audit": audit,
        "selected_indices": [int(a["row_index"]) for a in selected],
    }, indent=2), encoding="utf-8")
    print(f"{ts()} Selected {len(selected)} rows for regen:")
    for n, item in enumerate(selected):
        print(f"  {n+1}/{len(selected)}  idx={item['row_index']:2d}  "
              f"score={item['weakness_score']}  chars={item['chars']}  "
              f"hedge={item['hedge_count']}  em_dash={item['em_dash']}  "
              f"parens={item['parens']}")

    print(f"{ts()} Baseline LexDiv (R63c-repair): "
          f"{compute_lexdiv_distinct2(r63c_repair_rows):.4f}")

    client = anthropic.Anthropic(api_key=api_key)
    usage = r63.UsageTotals()
    print(f"{ts()} Confirming Anthropic model availability: {args.model}")
    r63.confirm_model(client, usage, args.model)
    print(f"{ts()} Model check passed")

    final_rows = copy.deepcopy(r63c_repair_rows)
    r63c.write_jsonl(PERSISTED_ROWS, final_rows)

    attempts: list[dict[str, Any]] = []
    generation_failures: list[dict[str, Any]] = []
    accepted_indices: set[int] = set()
    started_at = datetime.now().isoformat()
    start = time.time()

    for n_done, sel in enumerate(selected, start=1):
        idx = int(sel["row_index"])
        base_row = r63c_repair_rows[idx]
        sid = base_row["session_id"]
        top_id = base_row["predicted_track_ids"][0]
        top_meta = catalog[top_id]
        architecture = ARCHITECTURES[(n_done - 1) % len(ARCHITECTURES)]
        response, failures, n_attempts = generate_one(
            client,
            blind_by_sid[sid],
            top_meta,
            catalog,
            idx,
            base_row["predicted_response"],
            architecture,
            usage,
            args.model,
        )
        accepted = not failures
        if accepted:
            final_rows[idx]["predicted_response"] = response
            accepted_indices.add(idx)
        else:
            final_rows[idx]["predicted_response"] = base_row["predicted_response"]
            generation_failures.append({
                "row_index": idx,
                "session_id": sid,
                "top1_track_id": top_id,
                "top1_track": top_meta.get("track_name", ""),
                "top1_artist": top_meta.get("artist_name", ""),
                "architecture": architecture["name"],
                "failures": failures,
                "last_generated_response": response,
                "kept_original_response": True,
            })
        attempts.append({
            "row_index": idx,
            "session_id": sid,
            "weakness_score": sel["weakness_score"],
            "attempts": n_attempts,
            "accepted": accepted,
            "word_count": len(final_rows[idx]["predicted_response"].split()),
            "top1_track": top_meta.get("track_name", ""),
            "top1_artist": top_meta.get("artist_name", ""),
            "architecture": architecture["name"],
        })
        r63c.write_jsonl(PERSISTED_ROWS, final_rows)
        print(f"{ts()} Row {n_done}/{len(selected)} idx={idx} accepted={accepted} "
              f"attempts={n_attempts} architecture={architecture['name']} "
              f"api_calls={usage.calls}", flush=True)

    elapsed = time.time() - start
    print(f"{ts()} Regen done in {elapsed:.0f}s, "
          f"accepted={len(accepted_indices)}/{len(selected)}")

    # Save final rows
    r63c.write_jsonl(FINAL_ROWS, final_rows)

    # Validate tracks unchanged
    print(f"{ts()} Validating tracks bitwise identical to R63c-repair")
    assert_tracks_unchanged(final_rows, r63c_repair_rows)
    print(f"  PASS: tracks bitwise identical")

    # Compute LexDiv
    final_lexdiv = compute_lexdiv_distinct2(final_rows)
    baseline_lexdiv = compute_lexdiv_distinct2(r63c_repair_rows)
    lexdiv_delta = final_lexdiv - baseline_lexdiv
    print(f"{ts()} LexDiv: R63c-repair={baseline_lexdiv:.4f}  "
          f"R73={final_lexdiv:.4f}  Δ={lexdiv_delta:+.4f}")

    if final_lexdiv >= LEXDIV_FLOOR:
        gate = "PASS"
    elif final_lexdiv >= LEXDIV_BORDERLINE_FLOOR:
        gate = "BORDERLINE"
    else:
        gate = "FAIL"
    print(f"{ts()} LexDiv gate (floor {LEXDIV_FLOOR}): {gate}")

    # Standard validation
    validation = r63.validate_submission(final_rows, r54c_rows, catalog)
    r63.abort_on_track_mismatch(validation)

    # Build summary
    summary = {
        "experiment": "R73 concise-direct polish",
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "started_at": started_at,
        "elapsed_s": elapsed,
        "target_rows": args.target_rows,
        "n_selected": len(selected),
        "n_accepted": len(accepted_indices),
        "selected_indices": [int(s["row_index"]) for s in selected],
        "accepted_indices": sorted(accepted_indices),
        "lexdiv": {
            "baseline": baseline_lexdiv,
            "r73": final_lexdiv,
            "delta": lexdiv_delta,
            "floor": LEXDIV_FLOOR,
            "borderline_floor": LEXDIV_BORDERLINE_FLOOR,
            "gate": gate,
        },
        "validation": validation,
        "attempts": attempts,
        "generation_failures": generation_failures,
        "usage": {
            "calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
    }
    OUT_METADATA.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{ts()} Saved {OUT_METADATA}")

    # Write zip if gate clears
    if gate in ("PASS", "BORDERLINE"):
        write_zip(final_rows)
        print(f"{ts()} Wrote {OUT_ZIP}")
        ship_status = "READY" if gate == "PASS" else "BORDERLINE_HOLD_FOR_USER"
    else:
        ship_status = "DO_NOT_SHIP_LEXDIV_FAIL"

    # Doc
    sample_lines = []
    for n, idx in enumerate(sorted(accepted_indices)[:5]):
        before = r63c_repair_rows[idx]["predicted_response"]
        after = final_rows[idx]["predicted_response"]
        sample_lines.append(f"### Row {idx}")
        sample_lines.append(f"**Before (R63c-repair, {len(before)} chars):** {before}")
        sample_lines.append("")
        sample_lines.append(f"**After (R73, {len(after)} chars):** {after}")
        sample_lines.append("")

    doc = "\n".join([
        "# R73 concise-direct polish",
        "",
        f"Model: `{args.model}`  Prompt: `{PROMPT_VERSION}`",
        f"Target rows: {args.target_rows}  Selected: {len(selected)}  Accepted: {len(accepted_indices)}",
        f"Elapsed: {elapsed:.0f}s  API calls: {usage.calls}",
        "",
        f"## LexDiv Gate: **{gate}**",
        "",
        f"- Baseline (R63c-repair): **{baseline_lexdiv:.4f}**",
        f"- R73 (after polish): **{final_lexdiv:.4f}**",
        f"- Δ = **{lexdiv_delta:+.4f}**",
        f"- Floor: {LEXDIV_FLOOR}, Borderline: {LEXDIV_BORDERLINE_FLOOR}",
        "",
        f"## Ship status: **{ship_status}**",
        "",
        "## Selected rows (top by weakness score)",
        "",
        "| Idx | Score | Chars | Hedge | Em-dash | Parens | Accepted |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for s in selected:
        doc += (
            f"\n| {s['row_index']} | {s['weakness_score']} | {s['chars']} | "
            f"{s['hedge_count']} | {s['em_dash']} | {s['parens']} | "
            f"{'✓' if s['row_index'] in accepted_indices else '✗'} |"
        )
    doc += "\n\n## Before / After Samples\n\n" + "\n".join(sample_lines)
    OUT_DOC.write_text(doc, encoding="utf-8")
    print(f"{ts()} Saved {OUT_DOC}")


if __name__ == "__main__":
    main()
