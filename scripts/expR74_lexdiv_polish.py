#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R74 LexDiv polish: bigram-repeat-density regen of R73 to push LexDiv 0.8536 → 0.86+.

R73 lifted LexDiv 0.8438 → 0.8536 (+0.0098) via 15-row concise-direct regen.
LLM judge held at 4.85 (ceiling). nDCG unchanged (tracks identical).

R74 picks a DIFFERENT 15 rows: those with the highest bigram-repeat-density
(content bigrams that appear in many other rows of the R73 corpus). These
rows contribute disproportionately to the corpus's bigram redundancy.

Strategy:
- Tracks bitwise identical to R73 (zero nDCG risk).
- Audit R73 by bigram-repeat-density (count of content bigrams shared with
  5+ other rows).
- Identify top overused content bigrams across corpus.
- Select top 15 rows by repeat-density (largely disjoint from R73's 15).
- Regenerate with R73's direct/concise style + explicit ban on the
  overused bigrams found in the corpus audit.
- Archetype rotation preserved.
- Hold LLM ≥ 4.85 (preserve R73's style ceiling) and tracks identical.

Expected outcome:
- LexDiv +0.005 to +0.015 → composite +0.001 to +0.002
- LLM held at 4.85
- Lands ~0.625-0.628, likely passing el_presidente at 0.63 to #4
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
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.expR63_e1_response_only as r63
import scripts.expR63c_targeted_polish as r63c

OUT_DIR = REPO / "exp" / "inference" / "blind_a"
R73_ROWS = OUT_DIR / "r73_concise_direct_rows_final.jsonl"

PERSISTED_ROWS = OUT_DIR / "r74_lexdiv_rows_persisted.jsonl"
FINAL_ROWS = OUT_DIR / "r74_lexdiv_rows_final.jsonl"
OUT_ZIP = OUT_DIR / "r74_lexdiv_submission.zip"
OUT_METADATA = OUT_DIR / "r74_lexdiv_submission.metadata.json"
OUT_DOC = REPO / "docs" / "r74_lexdiv_result.md"
AUDIT_JSON = REPO / "exp" / "eval" / "expR74_bigram_audit.json"

MODEL_ID = "claude-opus-4-7"
PROMPT_VERSION = (
    "R74 v1; targeted LexDiv regen of 15 highest-bigram-repeat-density rows "
    "from R73; direct/concise style preserved + explicit ban on top corpus bigrams"
)
MAX_TOKENS = 240
REGEN_WORD_TARGET = (60, 82)
DEFAULT_TARGET_ROWS = 15
MIN_REGEN_ROWS = 12
MAX_REGEN_ROWS = 20
LEXDIV_FLOOR_DELTA = -0.001  # R74 must not regress LexDiv

# Stopwords for content-bigram filter
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "of", "to",
    "with", "for", "is", "are", "by", "that", "this", "as", "s", "t",
    "it", "its", "be", "was", "been", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "can", "so", "if", "from", "after",
    "before", "into", "through", "between", "while", "since",
}

# Banned phrases inherited from R73 + new R74 additions from corpus audit
BANNED_PHRASES_R73 = [
    "comes off", "fits that", "carries that", "captures the",
    "captures that", "leans into", "matches the",
    "you're looking", "you're chasing", "you're trying",
    "you're after", "you described", "you mentioned",
    "perfect for", "exactly what", "makes it a",
    "vibe", "journey", "soundscape",
]
# Will be augmented with audit-derived overused bigrams at runtime
BANNED_FROM_AUDIT_LIMIT = 8


# Same 5 archetypes as R73 — proven direct/concise
ARCHITECTURES = [
    {
        "name": "verdict-first",
        "instruction": (
            "Open with a direct verdict statement: name the recommendation as "
            "[Track] by [Artist] inside a confident, evidence-light claim about "
            "why it is the right pick."
        ),
        "example": '"Track is the textbook 90s East Coast pick by Pete Rock & C.L. Smooth."',
    },
    {
        "name": "concrete-detail-lead",
        "instruction": (
            "Open with one concrete musical detail (instrument, production "
            "element, vocal quality), then name the recommendation as [Track] "
            "by [Artist] in the same sentence."
        ),
        "example": '"A chopped horn loop drives Track by Artist."',
    },
    {
        "name": "album-anchor",
        "instruction": (
            "Open with the album or release context (year, label, or album name) "
            "before introducing the track and a concrete reason."
        ),
        "example": '"From the 1992 sessions, Track by Artist..."',
    },
    {
        "name": "comparison-pivot",
        "instruction": (
            "Open with a tight comparison or contrast pivoting from a prior "
            "track, artist, or session preference, then state the new pick."
        ),
        "example": '"Where prior picks leaned X, Track by Artist..."',
    },
    {
        "name": "single-claim",
        "instruction": (
            "Open with a single declarative claim that ties the recommendation "
            "to one specific session detail in one sentence."
        ),
        "example": '"The 90s sample-loop hip-hop you cited is the territory of Track by Artist."',
    },
]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def get_bigrams(text: str) -> list[tuple[str, str]]:
    words = re.findall(r"\b\w+\b", text.lower())
    return list(zip(words, words[1:]))


def is_content_bigram(bg: tuple[str, str]) -> bool:
    return bg[0] not in STOPWORDS or bg[1] not in STOPWORDS


def compute_bigram_doc_count(rows: list[dict[str, Any]]) -> Counter:
    cnt = Counter()
    for r in rows:
        seen = set(get_bigrams(r["predicted_response"]))
        for bg in seen:
            cnt[bg] += 1
    return cnt


def r74_audit(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Audit R73 rows by content-bigram-repeat density.

    Returns (per_row_audit, top_corpus_overused_phrases).
    """
    doc_count = compute_bigram_doc_count(rows)

    # Top overused content bigrams (appear in >= 8 rows, content-bigram)
    overused = [
        (bg, c) for bg, c in doc_count.items()
        if c >= 8 and is_content_bigram(bg)
    ]
    overused.sort(key=lambda x: -x[1])
    overused_phrases = [
        f"{bg[0]} {bg[1]}"
        for bg, c in overused[:BANNED_FROM_AUDIT_LIMIT]
    ]

    audit = []
    for idx, row in enumerate(rows):
        resp = row.get("predicted_response", "")
        bgs = get_bigrams(resp)
        content_bgs = [bg for bg in bgs if is_content_bigram(bg)]
        repeat_5 = sum(1 for bg in content_bgs if doc_count[bg] >= 5)
        repeat_8 = sum(1 for bg in content_bgs if doc_count[bg] >= 8)
        repeat_10 = sum(1 for bg in content_bgs if doc_count[bg] >= 10)
        # Density score: weighted by how-many-rows the bigram appears in
        density = sum(min(doc_count[bg], 20) for bg in content_bgs if doc_count[bg] >= 5)
        audit.append({
            "row_index": idx,
            "session_id": row["session_id"],
            "chars": len(resp),
            "word_count": len(resp.split()),
            "content_bigrams": len(content_bgs),
            "repeat_5plus": repeat_5,
            "repeat_8plus": repeat_8,
            "repeat_10plus": repeat_10,
            "density_score": density,
            "before_response": resp,
        })
    return audit, overused_phrases


def select_rows(audit: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    if not (MIN_REGEN_ROWS <= target <= MAX_REGEN_ROWS):
        raise ValueError(f"target must be in [{MIN_REGEN_ROWS}, {MAX_REGEN_ROWS}]")
    ranked = sorted(audit, key=lambda a: (-a["density_score"], -a["repeat_5plus"]))
    return ranked[:target]


def render_prompt(
    item: dict[str, Any],
    top_meta: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    row_index: int,
    current_response: str,
    architecture: dict[str, str],
    extra_banned: list[str],
) -> str:
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    album = top_meta.get("album_name") or "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"
    tags = r63.clean_tags(top_meta.get("tag_list") or [])

    return "\n".join([
        "Rewrite the following music recommender response. Goal: maintain the "
        "direct, concise style AND introduce more vocabulary variety vs the "
        "rest of the corpus.",
        "",
        "Hard requirements:",
        f"- {REGEN_WORD_TARGET[0]}-{REGEN_WORD_TARGET[1]} words. STRICT.",
        f"- Name the recommendation as {title} by {artist} in the first sentence.",
        "- Lead with a confident verdict or concrete musical claim — NO hedging.",
        "- Name at least 2 concrete musical attributes (genre, era, instrument, "
        "production technique, lyrical theme).",
        "- Reference at least 1 specific session detail.",
        "",
        f"Use the {architecture['name']} sentence pattern: {architecture['instruction']}",
        f"(Reference only — DO NOT copy: {architecture['example']})",
        "",
        "Banned phrases / patterns (do not use ANY of these — they appear "
        "too often in other rows of this corpus):",
        *[f"- {p}" for p in BANNED_PHRASES_R73],
        *[f"- {p}  (overused in current corpus)" for p in extra_banned],
        "- Em-dashes, parentheticals, or ending question marks.",
        "- Boilerplate: 'If you', 'You might', 'Here's a', 'For fans of'.",
        "- Filler: 'vibe', 'journey', 'soundscape', 'captures the essence'.",
        "- Do not output prompt labels, bullets, markdown, or quotes around the full text.",
        "",
        "Diversification guidance:",
        "- Prefer specific instrument names (e.g., 'Wurlitzer', 'sitar', 'distorted Rhodes') "
        "over generic terms ('keyboard', 'string').",
        "- Use era-specific descriptors that fit (e.g., 'mid-decade', 'pre-millennium', "
        "'late-70s') over generic 'modern' or 'classic'.",
        "- Vary verbs across rows: 'anchors', 'threads', 'frames', 'centers', 'paces' "
        "instead of repeating 'drives' / 'leans' / 'pulls' / 'sits'.",
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
        "Current R73 response. Keep true specifics, deliver in fresh vocabulary "
        "that doesn't overlap the banned/overused list above:",
        current_response,
    ])


def validate_regen_text(
    text: str, top_meta: dict[str, Any], extra_banned: list[str],
) -> list[str]:
    failures = []
    wc = len(text.split())
    if wc < REGEN_WORD_TARGET[0] or wc > REGEN_WORD_TARGET[1]:
        failures.append(f"word_count {wc} outside [{REGEN_WORD_TARGET[0]}, {REGEN_WORD_TARGET[1]}]")
    full_banned = list(BANNED_PHRASES_R73) + list(extra_banned)
    for phrase in full_banned:
        if re.search(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE):
            failures.append(f"banned_phrase: {phrase!r}")
    if "—" in text or "–" in text:
        failures.append("contains em-dash")
    if "(" in text or ")" in text:
        failures.append("contains parenthetical")
    if text.rstrip().endswith("?"):
        failures.append("ends with question mark")
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    if title.lower() not in text.lower():
        failures.append(f"missing_track_name: {title!r}")
    if artist.lower() not in text.lower():
        failures.append(f"missing_artist_name: {artist!r}")
    return failures


def call_opus(client, system, user_prompt, usage, model):
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
    client, item, top_meta, catalog, row_index, current_response,
    architecture, extra_banned, usage, model, max_retries=2,
):
    system = (
        "You write concise, personalized music recommendation responses with a "
        "direct, confident style and rich vocabulary. Follow every constraint "
        "EXACTLY. Output only the response text, no preamble."
    )
    last_response = ""
    last_failures = []
    for attempt in range(max_retries + 1):
        prompt = render_prompt(
            item, top_meta, catalog, row_index, current_response,
            architecture, extra_banned,
        )
        if attempt > 0:
            prompt += (
                f"\n\nPrior attempt rejected for: {', '.join(last_failures)}. "
                "Try again. Stricter on word count and banned phrases."
            )
        response = call_opus(client, system, prompt, usage, model)
        failures = validate_regen_text(response, top_meta, extra_banned)
        last_response = response
        last_failures = failures
        if not failures:
            return response, [], attempt + 1
    return last_response, last_failures, max_retries + 1


def assert_tracks_unchanged(new_rows, r73_rows):
    assert len(new_rows) == len(r73_rows)
    for i, (n, o) in enumerate(zip(new_rows, r73_rows)):
        if n["session_id"] != o["session_id"]:
            raise RuntimeError(f"row {i}: session_id mismatch")
        if n["turn_number"] != o["turn_number"]:
            raise RuntimeError(f"row {i}: turn_number mismatch")
        if n["predicted_track_ids"] != o["predicted_track_ids"]:
            raise RuntimeError(f"row {i}: predicted_track_ids mismatch")


def distinct2_corpus(rows):
    bgs = []
    for r in rows:
        bgs.extend(get_bigrams(r["predicted_response"]))
    return len(set(bgs)) / len(bgs) if bgs else 0


def distinct2_per_resp_avg(rows):
    vals = []
    for r in rows:
        bgs = get_bigrams(r["predicted_response"])
        if bgs:
            vals.append(len(set(bgs)) / len(bgs))
    return sum(vals) / len(vals) if vals else 0


def write_zip(rows):
    payload = json.dumps(rows, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    args = parser.parse_args()

    if args.model != MODEL_ID:
        raise SystemExit(f"Model override {args.model!r} refused; R74 requires {MODEL_ID!r}.")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY")

    import anthropic

    print(f"{ts()} R74 LexDiv polish (bigram-repeat-density audit)")
    print(f"{ts()} Loading R73 + Blind-A sessions + catalog")
    r73_rows = r63c.read_jsonl(R73_ROWS)
    if len(r73_rows) != 80:
        raise RuntimeError(f"Expected 80 R73 rows, found {len(r73_rows)}")
    r54c_rows = r63.load_r54c_rows()
    blind_by_sid = r63.load_blind_by_sid()
    catalog = r63.load_catalog()

    baseline_corpus = distinct2_corpus(r73_rows)
    baseline_perresp = distinct2_per_resp_avg(r73_rows)
    print(f"{ts()} R73 baseline LexDiv: corpus={baseline_corpus:.4f}  "
          f"per_resp_avg={baseline_perresp:.4f}")

    print(f"{ts()} Auditing R73 by bigram-repeat-density")
    audit, overused_phrases = r74_audit(r73_rows)
    print(f"  top {len(overused_phrases)} overused content-bigrams "
          f"(will be banned in regen):")
    for p in overused_phrases:
        print(f"    {p!r}")

    selected = select_rows(audit, args.target_rows)
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps({
        "all_audit": audit,
        "overused_phrases": overused_phrases,
        "selected_indices": [int(s["row_index"]) for s in selected],
    }, indent=2), encoding="utf-8")
    print(f"{ts()} Selected {len(selected)} rows for regen:")
    for n, item in enumerate(selected):
        print(f"  {n+1}/{len(selected)}  idx={item['row_index']:2d}  "
              f"density={item['density_score']:3d}  "
              f"reps5+={item['repeat_5plus']:2d}  "
              f"chars={item['chars']}")

    client = anthropic.Anthropic(api_key=api_key)
    usage = r63.UsageTotals()
    print(f"{ts()} Confirming model: {args.model}")
    r63.confirm_model(client, usage, args.model)
    print(f"{ts()} Model OK")

    final_rows = copy.deepcopy(r73_rows)
    r63c.write_jsonl(PERSISTED_ROWS, final_rows)

    attempts = []
    generation_failures = []
    accepted_indices = set()
    started_at = datetime.now().isoformat()
    start = time.time()

    for n_done, sel in enumerate(selected, start=1):
        idx = int(sel["row_index"])
        base_row = r73_rows[idx]
        sid = base_row["session_id"]
        top_id = base_row["predicted_track_ids"][0]
        top_meta = catalog[top_id]
        architecture = ARCHITECTURES[(n_done - 1) % len(ARCHITECTURES)]
        response, failures, n_attempts = generate_one(
            client, blind_by_sid[sid], top_meta, catalog, idx,
            base_row["predicted_response"], architecture, overused_phrases,
            usage, args.model,
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
                "architecture": architecture["name"],
                "failures": failures,
                "last_generated_response": response,
            })
        attempts.append({
            "row_index": idx,
            "session_id": sid,
            "density_score": sel["density_score"],
            "repeat_5plus": sel["repeat_5plus"],
            "attempts": n_attempts,
            "accepted": accepted,
            "word_count": len(final_rows[idx]["predicted_response"].split()),
            "architecture": architecture["name"],
        })
        r63c.write_jsonl(PERSISTED_ROWS, final_rows)
        print(f"{ts()} Row {n_done}/{len(selected)} idx={idx} accepted={accepted} "
              f"attempts={n_attempts} arch={architecture['name']} "
              f"api={usage.calls}", flush=True)

    elapsed = time.time() - start
    print(f"{ts()} Regen done in {elapsed:.0f}s, "
          f"accepted={len(accepted_indices)}/{len(selected)}")

    r63c.write_jsonl(FINAL_ROWS, final_rows)

    print(f"{ts()} Validating tracks bitwise identical to R73")
    assert_tracks_unchanged(final_rows, r73_rows)
    print("  PASS")

    final_corpus = distinct2_corpus(final_rows)
    final_perresp = distinct2_per_resp_avg(final_rows)
    corpus_delta = final_corpus - baseline_corpus
    perresp_delta = final_perresp - baseline_perresp
    print(f"{ts()} LexDiv:")
    print(f"  corpus     R73={baseline_corpus:.4f}  R74={final_corpus:.4f}  Δ={corpus_delta:+.4f}")
    print(f"  per_resp   R73={baseline_perresp:.4f}  R74={final_perresp:.4f}  Δ={perresp_delta:+.4f}")

    if corpus_delta >= 0 and perresp_delta >= LEXDIV_FLOOR_DELTA:
        gate = "PASS"
    elif corpus_delta >= LEXDIV_FLOOR_DELTA:
        gate = "BORDERLINE"
    else:
        gate = "FAIL"
    print(f"{ts()} LexDiv gate: {gate}")

    validation = r63.validate_submission(final_rows, r54c_rows, catalog)
    r63.abort_on_track_mismatch(validation)

    summary = {
        "experiment": "R74 LexDiv polish (bigram-repeat-density)",
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "started_at": started_at,
        "elapsed_s": elapsed,
        "target_rows": args.target_rows,
        "n_selected": len(selected),
        "n_accepted": len(accepted_indices),
        "selected_indices": [int(s["row_index"]) for s in selected],
        "accepted_indices": sorted(accepted_indices),
        "overused_phrases_banned": overused_phrases,
        "lexdiv": {
            "r73_corpus": baseline_corpus,
            "r74_corpus": final_corpus,
            "corpus_delta": corpus_delta,
            "r73_per_resp_avg": baseline_perresp,
            "r74_per_resp_avg": final_perresp,
            "per_resp_delta": perresp_delta,
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

    if gate in ("PASS", "BORDERLINE"):
        write_zip(final_rows)
        print(f"{ts()} Wrote {OUT_ZIP}")
        ship_status = "READY" if gate == "PASS" else "BORDERLINE_HOLD"
    else:
        ship_status = "DO_NOT_SHIP_LEXDIV_REGRESSED"

    sample_lines = []
    for idx in sorted(accepted_indices)[:4]:
        before = r73_rows[idx]["predicted_response"]
        after = final_rows[idx]["predicted_response"]
        sample_lines.append(f"### Row {idx}")
        sample_lines.append(f"**Before (R73, {len(before)} chars):** {before}")
        sample_lines.append("")
        sample_lines.append(f"**After (R74, {len(after)} chars):** {after}")
        sample_lines.append("")

    doc = "\n".join([
        "# R74 LexDiv polish",
        "",
        f"Model: `{args.model}`  Prompt: `{PROMPT_VERSION}`",
        f"Target rows: {args.target_rows}  Selected: {len(selected)}  Accepted: {len(accepted_indices)}",
        f"Elapsed: {elapsed:.0f}s  API calls: {usage.calls}",
        "",
        f"## LexDiv Gate: **{gate}**",
        "",
        f"- Corpus Distinct-2:    R73={baseline_corpus:.4f}  R74={final_corpus:.4f}  Δ={corpus_delta:+.4f}",
        f"- Per-response avg:     R73={baseline_perresp:.4f}  R74={final_perresp:.4f}  Δ={perresp_delta:+.4f}",
        "",
        f"## Ship: **{ship_status}**",
        "",
        "## Audit-derived banned phrases",
        "",
        *[f"- {p}" for p in overused_phrases],
        "",
        "## Selected rows (top by bigram repeat density)",
        "",
        "| Idx | Density | Reps≥5 | Chars | Accepted |",
        "|---:|---:|---:|---:|:---:|",
    ])
    for s in selected:
        doc += (
            f"\n| {s['row_index']} | {s['density_score']} | {s['repeat_5plus']} | "
            f"{s['chars']} | {'✓' if s['row_index'] in accepted_indices else '✗'} |"
        )
    doc += "\n\n## Before / After Samples\n\n" + "\n".join(sample_lines)
    OUT_DOC.write_text(doc, encoding="utf-8")
    print(f"{ts()} Saved {OUT_DOC}")


if __name__ == "__main__":
    main()
