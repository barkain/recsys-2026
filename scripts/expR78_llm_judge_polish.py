#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R78 LLM-judge polish — target LLM 4.95 with LexDiv ≥0.88, tracks fixed to R77.

R77 broke the LLM 4.85 ceiling → 4.90 with richer/more-specific vocabulary.
Goal of R78: push another step to 4.95 (semintelligence's level) while
preserving LexDiv ≥0.88 and tracks bitwise identical.

DIFFERENT AUDIT THAN R74/R77. R74/R77 targeted bigram-repeat density (LexDiv
optimizer). R78 targets LLM-judge content weakness signals:
- Lack of concrete musical attributes (genre, era, instrument, technique)
- Weak session reference (no explicit anchor to prior user statements)
- No top-1 justification (no causal "because/since/with" linking)
- Imperative closers ("Crank it loud", "Pour cocoa and press play")
- Vague descriptors ("warm", "vibrant", "energetic") without specificity

LexDiv constraint: corpus Distinct-2 must NOT drop below R77's baseline.
If R78's regen would degrade LexDiv (e.g., by reintroducing common phrases),
the script will detect that and warn.
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
R77_ROWS = OUT_DIR / "r77_ceiling_rows_final.jsonl"

PERSISTED_ROWS = OUT_DIR / "r78_llm_polish_rows_persisted.jsonl"
FINAL_ROWS = OUT_DIR / "r78_llm_polish_rows_final.jsonl"
OUT_ZIP = OUT_DIR / "r78_llm_polish_submission.zip"
OUT_METADATA = OUT_DIR / "r78_llm_polish_submission.metadata.json"
OUT_DOC = REPO / "docs" / "r78_llm_polish_result.md"
AUDIT_JSON = REPO / "exp" / "eval" / "expR78_llm_audit.json"

MODEL_ID = "claude-opus-4-7"
PROMPT_VERSION = (
    "R78 v1; LLM-judge polish on R77 base; emphasis on concrete specificity, "
    "strong session anchor, confident assertion, measured ending; "
    "LexDiv corpus floor ≥ R77"
)
MAX_TOKENS = 260
REGEN_WORD_TARGET = (65, 88)  # slightly higher floor than R74/R77 (was 60-82)
DEFAULT_TARGET_ROWS = 12
MIN_REGEN_ROWS = 10
MAX_REGEN_ROWS = 18

# Phrases that signal LLM-judge weakness (used in audit AND banned in regen)
VAGUE_DESCRIPTORS = [
    r"\bwarm\b", r"\bvibrant\b", r"\benergetic\b", r"\bcozy\b",
    r"\bbeautiful\b", r"\bgorgeous\b", r"\bgentle\b", r"\bcharming\b",
    r"\bmemorable\b", r"\bdelightful\b", r"\blovely\b",
]
VAGUE_RE = re.compile("|".join(VAGUE_DESCRIPTORS), re.IGNORECASE)

IMPERATIVE_CLOSERS = [
    r"crank it loud\b", r"press play\b", r"lace up\b",
    r"pour\b.*\bpress play\b", r"hit play\b", r"queue this up\b",
    r"give it a spin\b", r"throw it on\b",
]
IMPERATIVE_RE = re.compile("|".join(IMPERATIVE_CLOSERS), re.IGNORECASE)

# Causal-link phrases — presence is a positive signal
CAUSAL_PATTERNS = [
    r"\bbecause\b", r"\bsince\b", r"\bwhile\b", r"\bwhere\b", r"\bafter\b",
    r"\bafter that\b", r"\bbuilt on\b", r"\bgrounded in\b", r"\banchors\b",
    r"\bframes\b", r"\bthreads\b", r"\bcenters\b",
]
CAUSAL_RE = re.compile("|".join(CAUSAL_PATTERNS), re.IGNORECASE)

# Musical attribute words (positive signal)
ATTRIBUTE_WORDS = {
    "guitar", "bass", "drum", "drums", "piano", "vocal", "vocals", "synth",
    "synthesizer", "horn", "horns", "brass", "string", "strings",
    "percussion", "kick", "snare", "hi-hat", "fiddle", "banjo", "mandolin",
    "rhythm", "tempo", "groove", "beat", "riff", "hook", "melody",
    "harmony", "chord", "key", "minor", "major", "tone", "timbre",
    "production", "mix", "mastering", "lyrics", "lyric", "verse", "chorus",
    "bridge", "intro", "outro", "tremolo", "vibrato", "reverb", "delay",
    "fuzz", "distortion", "blast", "double-kick", "fingerpicked",
    "palm-muted", "arpeggio", "arpeggiated",
}
ATTR_RE = re.compile(r"\b(" + "|".join(ATTRIBUTE_WORDS) + r")\b", re.IGNORECASE)

# LLM-friendly archetypes (R77-style preserved, just clearer guidance)
ARCHITECTURES = [
    {
        "name": "verdict-with-anchor",
        "instruction": (
            "Open with a confident verdict that explicitly anchors to a "
            "specific prior session detail (named track, named artist, or "
            "stated preference). Then deliver 2-3 concrete musical attributes "
            "(instrument, production technique, lyrical theme)."
        ),
    },
    {
        "name": "production-detail-lead",
        "instruction": (
            "Open with a concrete production or instrumentation detail "
            "(e.g., 'Fingerpicked acoustic guitar and pedal steel'), then "
            "name the track and artist in the same sentence, then explain "
            "the session connection."
        ),
    },
    {
        "name": "album-era-anchor",
        "instruction": (
            "Open with the album/release context including year or era, "
            "then introduce the track and artist with a concrete reason. "
            "Close with a session-specific reference (prior track or "
            "stated preference)."
        ),
    },
    {
        "name": "explicit-because",
        "instruction": (
            "Build the response around an explicit causal link: '[Track] by "
            "[Artist] fits BECAUSE [specific musical or lyrical reason tied "
            "to user context].' Use 'because' or 'since' explicitly."
        ),
    },
    {
        "name": "trait-comparison",
        "instruction": (
            "Open with a sharp comparison or contrast to a prior session "
            "track ('Where [prior] [trait], [Track] by [Artist] [different "
            "trait]'), then specific musical detail."
        ),
    },
]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def get_bigrams(text: str) -> list[tuple[str, str]]:
    words = re.findall(r"\b\w+\b", text.lower())
    return list(zip(words, words[1:]))


def distinct2_corpus(rows):
    bgs = []
    for r in rows:
        bgs.extend(get_bigrams(r["predicted_response"]))
    return len(set(bgs)) / len(bgs) if bgs else 0


def distinct2_per_resp(rows):
    vals = []
    for r in rows:
        bgs = get_bigrams(r["predicted_response"])
        if bgs:
            vals.append(len(set(bgs)) / len(bgs))
    return sum(vals) / len(vals) if vals else 0


def r78_audit(rows):
    """Score each row by LLM-judge-weakness signals."""
    out = []
    for idx, r in enumerate(rows):
        resp = r.get("predicted_response", "")
        wc = len(resp.split())
        chars = len(resp)
        attr_hits = len(set(m.group(0).lower() for m in ATTR_RE.finditer(resp)))
        causal_hits = len(CAUSAL_RE.findall(resp))
        vague_hits = len(VAGUE_RE.findall(resp))
        imperative_hits = len(IMPERATIVE_RE.findall(resp))

        # Score: higher = weaker
        score = 0
        # Lack of concrete attributes
        if attr_hits < 3: score += 2
        elif attr_hits < 5: score += 1
        # Lack of causal reasoning
        if causal_hits < 1: score += 2
        elif causal_hits < 2: score += 1
        # Vague descriptors present
        if vague_hits >= 3: score += 2
        elif vague_hits >= 1: score += 1
        # Imperative closer present
        if imperative_hits >= 1: score += 2
        # Length signals
        if wc < 65: score += 1
        if wc > 95: score += 1

        out.append({
            "row_index": idx,
            "session_id": r["session_id"],
            "weakness_score": score,
            "chars": chars,
            "word_count": wc,
            "attr_hits": attr_hits,
            "causal_hits": causal_hits,
            "vague_hits": vague_hits,
            "imperative_hits": imperative_hits,
            "before_response": resp,
        })
    return out


def select_weak(audit, target):
    if not (MIN_REGEN_ROWS <= target <= MAX_REGEN_ROWS):
        raise ValueError(f"target must be in [{MIN_REGEN_ROWS}, {MAX_REGEN_ROWS}]")
    ranked = sorted(audit, key=lambda a: (-a["weakness_score"],
                                          -a["imperative_hits"],
                                          -a["vague_hits"]))
    return ranked[:target]


def render_prompt(item, top_meta, catalog, current_response, architecture):
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    album = top_meta.get("album_name") or "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"
    tags = r63.clean_tags(top_meta.get("tag_list") or [])

    return "\n".join([
        "Rewrite this music recommender response in a way an LLM judge would "
        "reward more highly. Goal: maximize specificity, confident "
        "assertion, and explicit session-context anchoring. Avoid vague or "
        "boilerplate phrasing.",
        "",
        "Hard requirements:",
        f"- {REGEN_WORD_TARGET[0]}-{REGEN_WORD_TARGET[1]} words. STRICT.",
        f"- Name the recommendation as {title} by {artist}.",
        "- Name AT LEAST 3 concrete musical attributes (instrument, "
        "production technique, lyrical theme, era, genre, sub-genre).",
        "- Include AT LEAST ONE explicit causal link to the session: "
        "use 'because', 'since', 'after', or 'where' to tie the "
        "recommendation to a SPECIFIC prior session detail (named track, "
        "named artist, or stated preference).",
        "- Close with a measured, declarative final sentence — NOT an "
        "imperative instruction. Avoid endings like 'Crank it loud', "
        "'Press play', 'Lace up and go'.",
        "",
        f"Use the {architecture['name']} pattern: {architecture['instruction']}",
        "",
        "Banned vague descriptors (do not use ANY of these):",
        "- 'warm', 'vibrant', 'energetic', 'cozy', 'beautiful', 'gorgeous'",
        "- 'gentle', 'charming', 'memorable', 'delightful', 'lovely'",
        "- 'captures the essence', 'vibe', 'journey', 'soundscape'",
        "",
        "Banned phrases (still excluded from prior sprints):",
        "- 'comes off', 'fits that', 'carries that', 'matches the'",
        "- 'leans into', 'you're looking', 'you're chasing', 'perfect for'",
        "- 'exactly what', 'makes it a', 'pulled from'",
        "",
        "Structure rules:",
        "- No em-dashes or parentheticals.",
        "- No imperative closers.",
        "- No ending question marks.",
        "- Output only the response text — no labels, bullets, markdown, "
        "or quote wrappers.",
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
        "Current R77 response. Improve LLM-judge friendliness while keeping "
        "all true specifics; deliver in measured, declarative prose:",
        current_response,
    ])


def validate_regen(text, top_meta):
    failures = []
    wc = len(text.split())
    if wc < REGEN_WORD_TARGET[0] or wc > REGEN_WORD_TARGET[1]:
        failures.append(f"word_count {wc} outside [{REGEN_WORD_TARGET[0]}, {REGEN_WORD_TARGET[1]}]")
    banned = [
        "comes off", "fits that", "carries that", "matches the",
        "leans into", "you're looking", "you're chasing",
        "perfect for", "exactly what", "makes it a", "pulled from",
        "captures the essence", "vibe", "journey", "soundscape",
    ]
    for p in banned:
        if re.search(rf"\b{re.escape(p)}\b", text, re.IGNORECASE):
            failures.append(f"banned_phrase: {p!r}")
    # Vague descriptors
    vague_present = VAGUE_RE.findall(text)
    if vague_present:
        failures.append(f"vague_descriptors: {vague_present[:3]}")
    # Imperative closer
    if IMPERATIVE_RE.search(text):
        failures.append("imperative_closer")
    # Structure
    if "—" in text or "–" in text:
        failures.append("em-dash")
    if "(" in text or ")" in text:
        failures.append("parenthetical")
    if text.rstrip().endswith("?"):
        failures.append("ends with question")
    # Must contain track + artist
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    if title.lower() not in text.lower():
        failures.append(f"missing_track: {title!r}")
    if artist.lower() not in text.lower():
        failures.append(f"missing_artist: {artist!r}")
    # Causal link required
    if not CAUSAL_RE.search(text):
        failures.append("no_causal_link")
    return failures


def call_opus(client, system, user_prompt, usage, model):
    message = client.messages.create(
        model=model, max_tokens=MAX_TOKENS, system=system,
        messages=[{"role": "user", "content": user_prompt}],
    )
    usage.add(getattr(message, "usage", None))
    if getattr(message, "stop_reason", None) != "end_turn":
        raise RuntimeError(f"stop_reason={getattr(message, 'stop_reason', None)}")
    return r63.normalize_ws("".join(getattr(part, "text", "") for part in message.content))


def generate_one(client, item, top_meta, catalog, current_response,
                 architecture, usage, model, max_retries=3):
    system = (
        "You write concise, specific music recommendation responses with "
        "confident assertion and explicit session anchoring. Output only the "
        "response text. Follow every constraint EXACTLY."
    )
    last_resp = ""
    last_fails = []
    for attempt in range(max_retries + 1):
        prompt = render_prompt(item, top_meta, catalog, current_response, architecture)
        if attempt > 0:
            prompt += (f"\n\nPrior rejected for: {', '.join(last_fails)}. "
                       "Try again. Stricter on word count, causal link, banned phrases.")
        resp = call_opus(client, system, prompt, usage, model)
        fails = validate_regen(resp, top_meta)
        last_resp = resp
        last_fails = fails
        if not fails:
            return resp, [], attempt + 1
    return last_resp, last_fails, max_retries + 1


def assert_tracks_unchanged(new_rows, base_rows):
    assert len(new_rows) == len(base_rows)
    for i, (n, o) in enumerate(zip(new_rows, base_rows)):
        if n["session_id"] != o["session_id"]:
            raise RuntimeError(f"row {i}: session_id mismatch")
        if n["turn_number"] != o["turn_number"]:
            raise RuntimeError(f"row {i}: turn_number mismatch")
        if n["predicted_track_ids"] != o["predicted_track_ids"]:
            raise RuntimeError(f"row {i}: predicted_track_ids mismatch")


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
        raise SystemExit(f"Model override refused; R78 requires {MODEL_ID!r}.")

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY")

    import anthropic

    print(f"{ts()} R78 LLM-judge polish")
    print(f"{ts()} Loading R77 + Blind-A sessions + catalog")
    r77_rows = r63c.read_jsonl(R77_ROWS)
    if len(r77_rows) != 80:
        raise RuntimeError(f"Expected 80 R77 rows, found {len(r77_rows)}")
    r54c_rows = r63.load_r54c_rows()
    blind_by_sid = r63.load_blind_by_sid()
    catalog = r63.load_catalog()

    baseline_corpus = distinct2_corpus(r77_rows)
    baseline_perresp = distinct2_per_resp(r77_rows)
    print(f"{ts()} R77 baseline LexDiv: corpus={baseline_corpus:.4f}  "
          f"per_resp={baseline_perresp:.4f}")

    print(f"{ts()} Auditing R77 for LLM-judge weakness")
    audit = r78_audit(r77_rows)
    selected = select_weak(audit, args.target_rows)
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps({
        "all_audit": audit,
        "selected_indices": [int(a["row_index"]) for a in selected],
    }, indent=2))
    print(f"{ts()} Selected {len(selected)} rows for regen:")
    for n, s in enumerate(selected):
        print(f"  {n+1}/{len(selected)}  idx={s['row_index']:2d}  "
              f"score={s['weakness_score']}  attr={s['attr_hits']}  "
              f"causal={s['causal_hits']}  vague={s['vague_hits']}  "
              f"imp={s['imperative_hits']}  wc={s['word_count']}")

    client = anthropic.Anthropic(api_key=api_key)
    usage = r63.UsageTotals()
    print(f"{ts()} Confirming model: {args.model}")
    r63.confirm_model(client, usage, args.model)
    print(f"{ts()} Model OK")

    final_rows = copy.deepcopy(r77_rows)
    r63c.write_jsonl(PERSISTED_ROWS, final_rows)

    attempts = []
    fails_summary = []
    accepted = set()
    start = time.time()
    started_at = datetime.now().isoformat()

    for n, sel in enumerate(selected, start=1):
        idx = int(sel["row_index"])
        base = r77_rows[idx]
        sid = base["session_id"]
        top_id = base["predicted_track_ids"][0]
        top_meta = catalog[top_id]
        arch = ARCHITECTURES[(n - 1) % len(ARCHITECTURES)]
        resp, fails, n_att = generate_one(
            client, blind_by_sid[sid], top_meta, catalog,
            base["predicted_response"], arch, usage, args.model,
        )
        ok = not fails
        if ok:
            final_rows[idx]["predicted_response"] = resp
            accepted.add(idx)
        else:
            final_rows[idx]["predicted_response"] = base["predicted_response"]
            fails_summary.append({
                "row_index": idx, "session_id": sid,
                "architecture": arch["name"], "failures": fails,
                "last_attempt": resp,
            })
        attempts.append({
            "row_index": idx, "session_id": sid,
            "weakness_score": sel["weakness_score"],
            "n_attempts": n_att, "accepted": ok,
            "architecture": arch["name"],
            "word_count": len(final_rows[idx]["predicted_response"].split()),
        })
        r63c.write_jsonl(PERSISTED_ROWS, final_rows)
        print(f"{ts()} Row {n}/{len(selected)} idx={idx} accepted={ok} "
              f"attempts={n_att} arch={arch['name']} api={usage.calls}",
              flush=True)

    elapsed = time.time() - start
    print(f"{ts()} Regen done in {elapsed:.0f}s, accepted={len(accepted)}/{len(selected)}")

    r63c.write_jsonl(FINAL_ROWS, final_rows)

    print(f"{ts()} Validating tracks bitwise identical to R77")
    assert_tracks_unchanged(final_rows, r77_rows)
    print("  PASS")

    final_corpus = distinct2_corpus(final_rows)
    final_perresp = distinct2_per_resp(final_rows)
    corpus_delta = final_corpus - baseline_corpus
    perresp_delta = final_perresp - baseline_perresp
    print(f"{ts()} LexDiv:")
    print(f"  corpus     R77={baseline_corpus:.4f}  R78={final_corpus:.4f}  Δ={corpus_delta:+.4f}")
    print(f"  per_resp   R77={baseline_perresp:.4f}  R78={final_perresp:.4f}  Δ={perresp_delta:+.4f}")

    # Hard gate: corpus must NOT drop
    if corpus_delta >= -0.001:
        lexdiv_gate = "PASS"
    elif corpus_delta >= -0.005:
        lexdiv_gate = "BORDERLINE"
    else:
        lexdiv_gate = "FAIL"
    print(f"{ts()} LexDiv corpus gate: {lexdiv_gate} (delta={corpus_delta:+.4f}, "
          f"floor=-0.001 for PASS)")

    validation = r63.validate_submission(final_rows, r54c_rows, catalog)
    r63.abort_on_track_mismatch(validation)

    summary = {
        "experiment": "R78 LLM-judge polish",
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "started_at": started_at,
        "elapsed_s": elapsed,
        "target_rows": args.target_rows,
        "n_selected": len(selected),
        "n_accepted": len(accepted),
        "selected_indices": [int(s["row_index"]) for s in selected],
        "accepted_indices": sorted(accepted),
        "lexdiv": {
            "r77_corpus": baseline_corpus,
            "r78_corpus": final_corpus,
            "corpus_delta": corpus_delta,
            "r77_per_resp": baseline_perresp,
            "r78_per_resp": final_perresp,
            "per_resp_delta": perresp_delta,
            "gate": lexdiv_gate,
        },
        "validation": validation,
        "attempts": attempts,
        "generation_failures": fails_summary,
        "usage": {
            "calls": usage.calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
    }
    OUT_METADATA.write_text(json.dumps(summary, indent=2))
    print(f"{ts()} Saved {OUT_METADATA}")

    ship_status = "READY" if lexdiv_gate == "PASS" else (
        "BORDERLINE_HOLD" if lexdiv_gate == "BORDERLINE" else "DO_NOT_SHIP_LEXDIV_DROP"
    )
    if lexdiv_gate in ("PASS", "BORDERLINE"):
        write_zip(final_rows)
        print(f"{ts()} Wrote {OUT_ZIP}")
    else:
        print(f"{ts()} Zip NOT written; LexDiv dropped too much")

    # Sample before/after
    sample_lines = []
    for idx in sorted(accepted)[:4]:
        before = r77_rows[idx]["predicted_response"]
        after = final_rows[idx]["predicted_response"]
        sample_lines.append(f"### Row {idx}")
        sample_lines.append(f"**Before R77 ({len(before)} chars):** {before}")
        sample_lines.append("")
        sample_lines.append(f"**After R78 ({len(after)} chars):** {after}")
        sample_lines.append("")

    doc = "\n".join([
        "# R78 LLM-judge polish",
        "",
        f"Model: `{args.model}`",
        f"Target rows: {args.target_rows}  Selected: {len(selected)}  Accepted: {len(accepted)}",
        f"Elapsed: {elapsed:.0f}s  API calls: {usage.calls}",
        "",
        f"## LexDiv Gate: **{lexdiv_gate}**",
        "",
        f"- Corpus Distinct-2: R77={baseline_corpus:.4f}  R78={final_corpus:.4f}  Δ={corpus_delta:+.4f}",
        f"- Per-response avg: R77={baseline_perresp:.4f}  R78={final_perresp:.4f}  Δ={perresp_delta:+.4f}",
        "",
        f"## Ship status: **{ship_status}**",
        "",
        "## Audit signals used",
        "",
        "- attr_hits: count of concrete musical attribute words (target ≥ 3)",
        "- causal_hits: count of explicit causal links (target ≥ 1)",
        "- vague_hits: count of vague descriptors (penalize)",
        "- imperative_hits: count of imperative closers (penalize)",
        "- Word count outside [65, 88] penalized",
        "",
        "## Selected rows (top by LLM weakness score)",
        "",
        "| Idx | Score | Attr | Causal | Vague | Imp | WC | Accepted |",
        "|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for s in selected:
        doc += (
            f"\n| {s['row_index']} | {s['weakness_score']} | "
            f"{s['attr_hits']} | {s['causal_hits']} | "
            f"{s['vague_hits']} | {s['imperative_hits']} | "
            f"{s['word_count']} | {'✓' if s['row_index'] in accepted else '✗'} |"
        )
    doc += "\n\n## Before / After samples\n\n" + "\n".join(sample_lines)
    OUT_DOC.write_text(doc, encoding="utf-8")
    print(f"{ts()} Saved {OUT_DOC}")


if __name__ == "__main__":
    main()
