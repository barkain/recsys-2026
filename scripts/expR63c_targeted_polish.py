#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R63c: targeted response-side polish on top of R63b.

Only the weakest R63b response rows are regenerated. Track IDs remain copied
from R63b/R54c and are validated bitwise against R54c before packaging.

Outputs:
  exp/inference/blind_a/r63c_rows_persisted.jsonl
  exp/inference/blind_a/r63c_rows_final.jsonl
  exp/inference/blind_a/r63c_targeted_polish_submission.zip
  exp/inference/blind_a/r63c_targeted_polish_submission.metadata.json
  docs/r63c_targeted_polish_result.md
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

OUT_DIR = REPO / "exp" / "inference" / "blind_a"
R63_ROWS = OUT_DIR / "r63_rows_persisted.jsonl"
R63_METADATA = OUT_DIR / "r63_response_only_submission.metadata.json"
R63B_ROWS = OUT_DIR / "r63b_rows_final.jsonl"
R63B_METADATA = OUT_DIR / "r63b_targeted_polish_submission.metadata.json"
PERSISTED_ROWS = OUT_DIR / "r63c_rows_persisted.jsonl"
FINAL_ROWS = OUT_DIR / "r63c_rows_final.jsonl"
OUT_ZIP = OUT_DIR / "r63c_targeted_polish_submission.zip"
OUT_METADATA = OUT_DIR / "r63c_targeted_polish_submission.metadata.json"
OUT_DOC = REPO / "docs" / "r63c_targeted_polish_result.md"

MODEL_ID = "claude-opus-4-7"
PROMPT_VERSION = (
    "R63c targeted polish v1; selected weakest R63b rows only; "
    "top-1 justification must connect >=2 specific session details"
)
TARGET_REGEN_COUNT = 15
MIN_REGEN_ROWS = 10
MAX_REGEN_ROWS = 20
MAX_ATTEMPTS = 4  # initial attempt + up to 3 retries
MAX_TOKENS = 320
REGEN_WORD_TARGET = (75, 95)
BASE_WORD_TARGET = (65, 95)
LOCAL_LEXDIV_WARN_FLOOR = 0.82

GENRE_WORDS = {
    "americana",
    "ambient",
    "alternative",
    "bebop",
    "bluegrass",
    "blues",
    "cabaret",
    "classical",
    "country",
    "dance",
    "darkwave",
    "disco",
    "downtempo",
    "electro",
    "electronic",
    "folk",
    "funk",
    "gospel",
    "hard",
    "hip",
    "hop",
    "house",
    "indie",
    "jazz",
    "jpop",
    "kpop",
    "latin",
    "metal",
    "pop",
    "punk",
    "rap",
    "reggae",
    "rock",
    "score",
    "shoegaze",
    "soul",
    "soundtrack",
    "synthwave",
    "techno",
}

ATTRIBUTE_WORDS = {
    "acoustic",
    "arrangement",
    "arpeggio",
    "arpeggios",
    "banjo",
    "bass",
    "bassline",
    "beat",
    "beats",
    "brass",
    "breakbeat",
    "cadence",
    "choral",
    "chorus",
    "distorted",
    "distortion",
    "drone",
    "drones",
    "drum",
    "drums",
    "electric",
    "feedback",
    "fiddle",
    "flow",
    "flows",
    "guitar",
    "guitars",
    "harmonies",
    "harmony",
    "hi-hat",
    "hook",
    "horn",
    "horns",
    "keys",
    "kick",
    "loop",
    "loops",
    "lyric",
    "lyrical",
    "lyrics",
    "melodic",
    "melody",
    "organ",
    "orchestral",
    "pads",
    "percussion",
    "piano",
    "production",
    "reverb",
    "rhythm",
    "riff",
    "riffs",
    "sample",
    "samples",
    "sax",
    "saxophone",
    "slide",
    "snare",
    "steel",
    "strings",
    "synth",
    "synthesizer",
    "synths",
    "texture",
    "textures",
    "tremolo",
    "trumpet",
    "vocal",
    "vocals",
    "voice",
}

SESSION_STOPWORDS = {
    "about",
    "after",
    "album",
    "and",
    "artist",
    "could",
    "feel",
    "feeling",
    "feels",
    "find",
    "from",
    "get",
    "good",
    "has",
    "have",
    "here",
    "just",
    "keep",
    "like",
    "listen",
    "listening",
    "love",
    "maybe",
    "more",
    "music",
    "need",
    "recommendation",
    "recommended",
    "really",
    "right",
    "same",
    "should",
    "something",
    "song",
    "sound",
    "sounds",
    "that",
    "the",
    "them",
    "there",
    "this",
    "track",
    "want",
    "wanted",
    "what",
    "when",
    "where",
    "with",
    "would",
    "your",
}

GENERIC_PHRASE_RX = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\benergy\b",
        r"\bmomentum\b",
        r"\bspirit\b",
        r"\bterritory\b",
        r"\bsweet spot\b",
        r"\bdrawn to\b",
        r"\bgravitating toward\b",
        r"\bslots neatly\b",
        r"\bkeeps that\b",
        r"\bshould sit well\b",
        r"\bworth hearing\b",
    ]
]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def tokenize_alpha(text: Any) -> list[str]:
    return [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", str(text or ""))
    ]


def response_words(text: str) -> list[str]:
    return [tok.lower() for tok in re.findall(r"\b[\w'-]+\b", text)]


def compact_text(text: str, limit: int = 260) -> str:
    clean = r63.normalize_ws(text)
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def session_tokens(
    item: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> set[str]:
    tokens: list[str] = []
    for turn in item.get("conversations") or []:
        role = str(turn.get("role"))
        content = str(turn.get("content") or "")
        if role == "user":
            tokens.extend(tokenize_alpha(content))
        elif role == "music" and content in catalog:
            meta = catalog[content]
            tokens.extend(tokenize_alpha(meta.get("track_name")))
            tokens.extend(tokenize_alpha(meta.get("artist_name")))
            tokens.extend(tokenize_alpha(meta.get("album_name")))
    return {
        tok
        for tok in tokens
        if len(tok) > 3 and tok not in SESSION_STOPWORDS
    }


def metadata_terms(top_meta: dict[str, Any]) -> tuple[set[str], set[str], set[str], set[str]]:
    title_terms = set(tokenize_alpha(top_meta.get("track_name")))
    artist_terms = set(tokenize_alpha(top_meta.get("artist_name")))
    album_terms = set(tokenize_alpha(top_meta.get("album_name")))
    tag_terms: set[str] = set()
    for tag in top_meta.get("tag_list") or []:
        tag_terms.update(tokenize_alpha(tag))
    return title_terms, artist_terms, album_terms, tag_terms


def weakness_audit(
    rows: list[dict[str, Any]],
    blind_by_sid: dict[str, dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        tracks = row["predicted_track_ids"]
        top_meta = catalog[tracks[0]]
        title_terms, artist_terms, album_terms, tag_terms = metadata_terms(top_meta)
        words = response_words(row["predicted_response"])
        word_set = set(words)
        attr_hits = sorted(
            word_set & (ATTRIBUTE_WORDS | GENRE_WORDS | tag_terms)
        )
        meta_hits = sorted(word_set & (album_terms | tag_terms))
        item = blind_by_sid[row["session_id"]]
        overlap = sorted(
            (
                word_set
                - title_terms
                - artist_terms
                - album_terms
                - SESSION_STOPWORDS
            )
            & session_tokens(item, catalog)
        )
        wc = r63.word_count(row["predicted_response"])
        has_causal_link = bool(
            re.search(
                r"\b(because|since|with|while|where|built|leans|matches|"
                r"carries|keeps|anchors|fits|turns|gives|pulls|pairs|"
                r"rides|echoing|answers|works|makes|lets|connects)\b",
                row["predicted_response"],
                re.IGNORECASE,
            )
        )

        generic_reasoning = 0
        if len(attr_hits) < 2:
            generic_reasoning += 2
        elif len(attr_hits) == 2:
            generic_reasoning += 1
        if len(meta_hits) + len(attr_hits) + min(len(overlap), 2) < 4:
            generic_reasoning += 1
        if any(rx.search(row["predicted_response"]) for rx in GENERIC_PHRASE_RX) and len(attr_hits) < 4:
            generic_reasoning += 1

        no_user_session_ref = 0
        if len(overlap) == 0:
            no_user_session_ref = 2
        elif len(overlap) == 1:
            no_user_session_ref = 1
        if len(overlap) == 0 and re.search(
            r"\b(you mentioned|you wanted|you asked|you connected|you "
            r"remembered|this chat|latest|earlier|previous|accepted|"
            r"rejected|found|pressed|called for|craving|chasing|looking "
            r"for|exploring|kept describing|asked about)\b",
            row["predicted_response"],
            re.IGNORECASE,
        ):
            no_user_session_ref = 1

        no_top1_justification = 0
        if not has_causal_link:
            no_top1_justification += 1
        if len(attr_hits) < 2:
            no_top1_justification += 1
        if not ((album_terms | tag_terms) & word_set) and len(attr_hits) < 3:
            no_top1_justification += 1

        overly_short_flat = 0
        if wc < 70:
            overly_short_flat += 2
        elif wc < 75:
            overly_short_flat += 1
        if len(attr_hits) < 3 or len(overlap) < 1:
            overly_short_flat += 1

        weakness_score = (
            generic_reasoning
            + no_user_session_ref
            + no_top1_justification
            + overly_short_flat
        )
        audit.append({
            "row_index": idx,
            "session_id": row["session_id"],
            "top1_track_id": tracks[0],
            "top1_track": top_meta.get("track_name", ""),
            "top1_artist": top_meta.get("artist_name", ""),
            "word_count": wc,
            "weakness_score": weakness_score,
            "components": {
                "generic_reasoning": generic_reasoning,
                "no_user_session_ref": no_user_session_ref,
                "no_top1_justification": no_top1_justification,
                "overly_short_flat": overly_short_flat,
            },
            "audit_signals": {
                "concrete_attribute_hits": attr_hits[:12],
                "metadata_hits": meta_hits[:12],
                "session_token_overlap": overlap[:12],
            },
            "before_response": row["predicted_response"],
            "before_snapshot": compact_text(row["predicted_response"]),
        })
    return audit


def select_weak_rows(audit: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    if target < MIN_REGEN_ROWS or target > MAX_REGEN_ROWS:
        raise ValueError(
            f"target must be between {MIN_REGEN_ROWS} and {MAX_REGEN_ROWS}: {target}"
        )
    ranked = sorted(
        audit,
        key=lambda a: (
            -int(a["weakness_score"]),
            int(a["word_count"]),
            int(a["row_index"]),
        ),
    )
    return ranked[:target]


def render_prompt(
    item: dict[str, Any],
    top_meta: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    row_index: int,
    r63b_response: str,
    retry_feedback: list[str] | None = None,
) -> str:
    retry_feedback = retry_feedback or []
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    album = top_meta.get("album_name") or "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"
    tags = r63.clean_tags(top_meta.get("tag_list") or [])
    style_note = [
        "Use a direct music-critical explanation, not a catalog blurb.",
        "Tie the recommendation to the latest user constraint before broad praise.",
        "Let one prior track or named preference carry the personalization.",
        "Use concrete nouns for the sound and avoid empty adjectives.",
        "Make the top-pick logic explicit in the first two sentences.",
        "Prefer session evidence over universal claims about the track.",
        "Mention the album or era only if it helps explain this match.",
        "Keep the voice conversational but compact.",
    ][row_index % 8]

    pieces = [
        "Write exactly one replacement recommender response for this Blind-A row.",
        "",
        "Hard requirements:",
        f"- 75-95 words.",
        f"- The first sentence must name the recommendation as {title} by {artist}.",
        "- Name 2-3 concrete musical attributes in prose: genre, era, a signature "
        "production element, lyrical theme, or specific instrument.",
        "- Reference at least one specific session detail, such as a stated user "
        "preference, a previous track or artist, or the conversation context.",
        "- Justify why this exact top-1 track is the best pick: use a concrete "
        "'this specifically, because ...' explanation, without sounding formulaic. "
        "Connect at least two specific session details, not just one.",
        "- Do not end with a question.",
        "- Do not use boilerplate openers such as 'If you're looking for', "
        "'You might enjoy', or 'Here's a track that'.",
        "- Do not use these filler words/phrases except when literally part of "
        "the track title or artist name: vibe, journey, soundscape, captures "
        "the essence.",
        "- Do not output prompt labels, bullets, metadata prefixes, markdown, "
        "or quotes around the whole answer.",
        "- Avoid crutches like 'perfect for', 'right in', 'lands', 'delivers', "
        "'you asked for', 'you described', 'exactly what', and 'makes it a'.",
        f"- Style note: {style_note}",
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
        "Previous R63b response being polished. Keep any true specifics, but make "
        "the new response more grounded and concrete:",
        r63b_response,
    ]
    if retry_feedback:
        pieces.extend([
            "",
            "The previous draft failed these checks; fix them in the new draft:",
            "; ".join(retry_feedback),
        ])
    return "\n".join(pieces)


def validate_regenerated_text(resp: str, top_meta: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    clean = r63.normalize_ws(resp)
    wc = r63.word_count(clean)
    if wc < REGEN_WORD_TARGET[0] or wc > REGEN_WORD_TARGET[1]:
        failures.append(
            f"word_count_{wc}_outside_{REGEN_WORD_TARGET[0]}_{REGEN_WORD_TARGET[1]}"
        )
    if not clean:
        failures.append("empty")
    if r63.PREFIX_LEAK_RX.search(clean) or r63.TAG_LINE_RX.search(clean):
        failures.append("prefix_or_tag_leak")
    if r63.TRAILING_QUESTION_RX.search(clean):
        failures.append("trailing_question")
    boiler_hits = r63.boilerplate_hits(clean, top_meta)
    if boiler_hits:
        failures.append("boilerplate_or_filler:" + ",".join(boiler_hits[:3]))

    sent = r63.opening_window(clean)
    if not r63.contains_text(sent, top_meta["track_name"]):
        failures.append("first_sentence_missing_track")
    artist_parts = [a.strip() for a in top_meta["artist_name"].split(",") if a.strip()]
    if top_meta["artist_name"] and not any(
        r63.contains_text(sent, artist) for artist in artist_parts
    ):
        failures.append("first_sentence_missing_artist")
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
    r63b_response: str,
    usage: r63.UsageTotals,
    model: str,
) -> tuple[str, int, list[str]]:
    system = (
        "You write concise, personalized music recommendation responses. "
        "Follow every constraint exactly. Output only the response text."
    )
    failures: list[str] = []
    last = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = render_prompt(
            item,
            top_meta,
            catalog,
            row_index,
            r63b_response,
            retry_feedback=failures,
        )
        last = call_opus(client, system, prompt, usage, model=model)
        failures = validate_regenerated_text(last, top_meta)
        if not failures:
            return last, attempt, []
        time.sleep(0.4)
    return last, MAX_ATTEMPTS, failures


def load_json_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_zip(rows: list[dict[str, Any]]) -> None:
    payload = json.dumps(rows, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)


def per_regen_validation(
    rows: list[dict[str, Any]],
    selected_indices: set[int],
    accepted_indices: set[int],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    violations = []
    for idx in sorted(selected_indices):
        row = rows[idx]
        top_meta = catalog[row["predicted_track_ids"][0]]
        wc = r63.word_count(row["predicted_response"])
        if idx in accepted_indices:
            min_words, max_words = REGEN_WORD_TARGET
            gate_name = "regenerated_word_range_75_95"
        else:
            min_words, max_words = BASE_WORD_TARGET
            gate_name = "fallback_original_word_range_65_95"
        fs_failures = [
            f
            for f in validate_regenerated_text(row["predicted_response"], top_meta)
            if f.startswith("first_sentence_missing")
        ]
        word_ok = min_words <= wc <= max_words
        if not word_ok or fs_failures:
            violations.append({
                "row_index": idx,
                "session_id": row["session_id"],
                "word_count": wc,
                "word_gate": gate_name,
                "word_range_ok": word_ok,
                "first_sentence_failures": fs_failures,
            })
    return {
        "selected_rows": len(selected_indices),
        "accepted_regenerated_rows": len(accepted_indices),
        "fallback_original_rows": len(selected_indices - accepted_indices),
        "violations": violations,
        "passed": not violations,
    }


def selected_samples(
    selected: list[dict[str, Any]],
    r63b_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    samples = []
    for entry in selected[:limit]:
        idx = int(entry["row_index"])
        top_id = final_rows[idx]["predicted_track_ids"][0]
        meta = catalog[top_id]
        samples.append({
            "row_index": idx,
            "session_id": final_rows[idx]["session_id"],
            "top1_track_id": top_id,
            "top1_track": meta.get("track_name", ""),
            "top1_artist": meta.get("artist_name", ""),
            "weakness_score": entry["weakness_score"],
            "weakness_components": entry["components"],
            "r63b_response": r63b_rows[idx]["predicted_response"],
            "r63c_response": final_rows[idx]["predicted_response"],
        })
    return samples


def gate_statuses(
    validation: dict[str, Any],
    regen_validation: dict[str, Any],
) -> dict[str, Any]:
    statuses = {}
    for name, passed in validation["gates"].items():
        if name in {"lexdiv_floor_0_83", "lexdiv_target_r54c_0_8381"}:
            status = "warn" if not passed else "pass"
            blocking = False
        else:
            status = "pass" if passed else "fail"
            blocking = True
        statuses[name] = {
            "status": status,
            "passed": bool(passed),
            "blocking": blocking,
        }
    statuses["selected_rows_word_range_and_first_sentence"] = {
        "status": "pass" if regen_validation["passed"] else "fail",
        "passed": bool(regen_validation["passed"]),
        "blocking": True,
    }
    statuses["local_lexdiv_floor_0_82"] = {
        "status": "pass" if validation["lexdiv_distinct2"] >= LOCAL_LEXDIV_WARN_FLOOR else "warn",
        "passed": validation["lexdiv_distinct2"] >= LOCAL_LEXDIV_WARN_FLOOR,
        "blocking": False,
    }
    return statuses


def ready_to_submit(
    validation: dict[str, Any],
    regen_validation: dict[str, Any],
) -> bool:
    nonblocking = {
        "lexdiv_floor_0_83",
        "lexdiv_target_r54c_0_8381",
    }
    structural_ok = all(
        passed
        for name, passed in validation["gates"].items()
        if name not in nonblocking
    )
    return structural_ok and bool(regen_validation["passed"])


def gate_table(validation: dict[str, Any], regen_validation: dict[str, Any]) -> str:
    statuses = gate_statuses(validation, regen_validation)
    lines = ["| Gate | Result |", "|---|---|"]
    for name, info in statuses.items():
        lines.append(f"| `{name}` | {info['status'].upper()} |")
    return "\n".join(lines)


def write_metadata(
    final_rows: list[dict[str, Any]],
    r63b_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    failures_after_retries: list[dict[str, Any]],
    validation: dict[str, Any],
    regen_validation: dict[str, Any],
    usage: r63.UsageTotals,
    model: str,
    started_at: str,
    completed_at: str,
    r63_metadata: dict[str, Any] | None,
    r63b_metadata: dict[str, Any] | None,
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    accepted_indices = {
        int(item["row_index"])
        for item in attempts
        if item.get("accepted_regenerated_response")
    }
    lexdiv = validation["lexdiv_distinct2"]
    label = (
        "R63c targeted response polish | base=R63b | tracks=R54c | "
        f"{len(accepted_indices)} rows regenerated | LexDiv={lexdiv:.4f} | "
        "purpose=push LLM judge 4.85 -> 4.90+"
    )
    current_usage = usage.as_dict()
    r63_usage_reference = (
        (r63_metadata or {}).get("cumulative_usage_including_prior_aborted_runs")
        or (r63_metadata or {}).get("usage")
    )
    r63b_usage_reference = (r63b_metadata or {}).get("usage")
    cumulative_usage = r63.combine_usage_dicts(
        r63.combine_usage_dicts(current_usage, r63b_usage_reference),
        r63_usage_reference,
    )
    metadata = {
        "submission_label": label,
        "created_at": completed_at,
        "generation_timestamp": completed_at,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "base": {
            "rows_file": str(R63B_ROWS.relative_to(REPO)),
            "submission_zip": "exp/inference/blind_a/r63b_targeted_polish_submission.zip",
            "metadata_file": str(R63B_METADATA.relative_to(REPO)),
            "blind_result": {
                "composite": 0.6219,
                "llm_judge": 4.85,
                "lexdiv": 0.8389,
                "tracks": "bitwise R54c",
            },
        },
        "source_tracks": {
            "artifact": str(r63.R54C_ZIP.relative_to(REPO)),
            "constraint": "R63c predicted_track_ids copied from R63b/R54c, unchanged.",
        },
        "persisted_rows_file": str(PERSISTED_ROWS.relative_to(REPO)),
        "final_rows_file": str(FINAL_ROWS.relative_to(REPO)),
        "submission_zip": str(OUT_ZIP.relative_to(REPO)),
        "lexdiv_distinct2": lexdiv,
        "lexdiv_policy": {
            "local_warning_floor": LOCAL_LEXDIV_WARN_FLOOR,
            "warn_if_below_floor": lexdiv < LOCAL_LEXDIV_WARN_FLOOR,
            "blind_scorer_note": (
                "R63b local LexDiv was 0.8260 while blind LexDiv was 0.8389; "
                "local LexDiv >=~0.82 is treated as acceptable information."
            ),
        },
        "selection": {
            "policy": (
                "Composite weakness score over generic reasoning, missing session "
                "reference, missing top-1 justification, and overly short/flat prose. "
                "Selected only the top ranked weak rows after re-auditing merged R63b; "
                "all other R63b responses kept."
            ),
            "target_selected_rows": TARGET_REGEN_COUNT,
            "selected_row_indices": [int(item["row_index"]) for item in selected],
            "selected_rows": selected,
        },
        "generation": {
            "started_at": started_at,
            "completed_at": completed_at,
            "selected_rows_attempted": len(selected),
            "accepted_regenerated_rows": len(accepted_indices),
            "fallback_original_rows": regen_validation["fallback_original_rows"],
            "non_selected_rows_kept_from_r63b": 80 - len(selected),
            "max_attempts_per_selected_row": MAX_ATTEMPTS,
            "word_target_regenerated_rows": list(REGEN_WORD_TARGET),
            "word_target_fallback_original_rows": list(BASE_WORD_TARGET),
            "failures_after_retries": failures_after_retries,
            "attempts": attempts,
            "rows_persisted_immediately_after_each_targeted_row": True,
            "packaging_source": str(FINAL_ROWS.relative_to(REPO)),
            "run_policy": "one targeted generation pass; no whole-80 regeneration",
        },
        "usage": current_usage,
        "cumulative_usage": cumulative_usage,
        "r63_usage_reference": r63_usage_reference,
        "r63b_usage_reference": r63b_usage_reference,
        "validation_results": gate_statuses(validation, regen_validation),
        "ready_to_submit": ready_to_submit(validation, regen_validation),
        "r54c_track_hash_match": validation["track_hash_comparison"],
        "validation": validation,
        "regenerated_row_validation": regen_validation,
        "before_after_samples": selected_samples(
            selected, r63b_rows, final_rows, catalog
        ),
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def write_result_doc(
    metadata: dict[str, Any],
    validation: dict[str, Any],
    regen_validation: dict[str, Any],
    selected: list[dict[str, Any]],
    samples: list[dict[str, Any]],
) -> None:
    comp = validation["track_hash_comparison"]
    sample_lines = []
    for sample in samples:
        sample_lines.extend([
            f"### Row {sample['row_index']}: {sample['top1_track']} by {sample['top1_artist']}",
            f"- session_id: `{sample['session_id']}`",
            f"- weakness score: `{sample['weakness_score']}`",
            f"- R63b: {sample['r63b_response']}",
            f"- R63c: {sample['r63c_response']}",
            "",
        ])
    sample_block = "\n".join(sample_lines)

    selected_lines = []
    for item in selected:
        components = ", ".join(
            f"{k}={v}" for k, v in item["components"].items() if v
        )
        selected_lines.append(
            f"- row `{item['row_index']}` / `{item['session_id']}`: "
            f"score={item['weakness_score']}, wc={item['word_count']}, "
            f"{item['top1_track']} by {item['top1_artist']} "
            f"({components or 'low residual weakness'}; "
            f"previously_regenerated_in_r63b={item.get('previously_regenerated_in_r63b', False)})"
        )

    doc = f"""# R63c Targeted Polish Result

## Gate Table
{gate_table(validation, regen_validation)}

## Track Hash Comparison (R54c vs R63c)
```text
Track Hash Comparison (R54c vs R63c):
  rows compared: {comp['rows_compared']}
  rows with matching track sequence: {comp['rows_with_matching_track_sequence']}
  rows with mismatched track sequence: {comp['rows_with_mismatched_track_sequence']}
  total tracks compared: {comp['total_tracks_compared']}
  per-position mismatches: {comp['per_position_mismatches']}
```

## Summary
- Submission label: `{metadata['submission_label']}`
- Model used: `{metadata['model']}`
- Submission artifact: `{OUT_ZIP.relative_to(REPO)}`
- Metadata: `{OUT_METADATA.relative_to(REPO)}`
- Persisted rows: `{PERSISTED_ROWS.relative_to(REPO)}`
- Final rows: `{FINAL_ROWS.relative_to(REPO)}`
- Selected weak rows: {len(selected)}
- Accepted regenerated rows: {metadata['generation']['accepted_regenerated_rows']}
- Fallback original rows: {metadata['generation']['fallback_original_rows']}
- Non-selected rows kept from R63b: {metadata['generation']['non_selected_rows_kept_from_r63b']}
- LexDiv (Distinct-2, local audit): {validation['lexdiv_distinct2']:.4f}
- Local LexDiv below 0.82 warning floor: {'YES' if validation['lexdiv_distinct2'] < LOCAL_LEXDIV_WARN_FLOOR else 'NO'}
- Max repeated opener cluster: {validation['opener_max_cluster']}
- Opus API calls for R63c run: {metadata['usage']['calls']}
- Estimated R63c run cost: ${metadata['usage']['estimated_cost_usd']:.4f}
- Cumulative API calls including R63 + R63b prior runs: {metadata['cumulative_usage']['calls']}
- Estimated cumulative cost including R63 + R63b prior runs: ${metadata['cumulative_usage']['estimated_cost_usd']:.4f}
- Ready to submit manually to Codabench: {'YES' if metadata['ready_to_submit'] else 'NO'}

## Selection Rationale
Rows were ranked by a composite weakness score over generic reasoning, missing
session reference, missing top-1 justification, and short or flat prose. Only
the selected rows below were regenerated; every other R63b response was kept.

{chr(10).join(selected_lines)}

## Repeated Opener Clusters
```json
{json.dumps(validation['repeated_opener_clusters'], indent=2, ensure_ascii=False)}
```

## Sample Comparisons
{sample_block}
"""
    OUT_DOC.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--target-rows", type=int, default=TARGET_REGEN_COUNT)
    args = parser.parse_args()

    if args.model != MODEL_ID:
        raise SystemExit(
            f"Refusing model override {args.model!r}; R63c requires {MODEL_ID!r}."
        )
    if args.target_rows < MIN_REGEN_ROWS or args.target_rows > MAX_REGEN_ROWS:
        raise SystemExit(
            f"--target-rows must be {MIN_REGEN_ROWS}-{MAX_REGEN_ROWS}; got {args.target_rows}"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY. Pause.")

    import anthropic

    print(f"{ts()} R63c targeted response polish")
    print(f"{ts()} Loading R63b rows, R54c tracks, Blind-A sessions, and catalog")
    r63b_rows = read_jsonl(R63B_ROWS)
    if len(r63b_rows) != 80:
        raise RuntimeError(f"Expected 80 R63b rows, found {len(r63b_rows)}")
    r54c_rows = r63.load_r54c_rows()
    blind_by_sid = r63.load_blind_by_sid()
    catalog = r63.load_catalog()
    r63_metadata = load_json_metadata(R63_METADATA)
    r63b_metadata = load_json_metadata(R63B_METADATA)

    print(f"{ts()} Auditing merged R63b responses for targeted weak-row selection")
    audit = weakness_audit(r63b_rows, blind_by_sid, catalog)
    selected = select_weak_rows(audit, args.target_rows)
    prior_r63b_selected = set(
        (r63b_metadata or {}).get("selection", {}).get("selected_row_indices", [])
    )
    for item in selected:
        item["previously_regenerated_in_r63b"] = int(item["row_index"]) in prior_r63b_selected
    selected_indices = {int(item["row_index"]) for item in selected}
    print(
        f"{ts()} Selected {len(selected)} weak rows: "
        f"{[int(item['row_index']) for item in selected]}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    usage = r63.UsageTotals()
    print(f"{ts()} Confirming Anthropic model availability: {args.model}")
    r63.confirm_model(client, usage, args.model)
    print(f"{ts()} Model check passed")

    final_rows = copy.deepcopy(r63b_rows)
    write_jsonl(PERSISTED_ROWS, final_rows)
    attempts: list[dict[str, Any]] = []
    failures_after_retries: list[dict[str, Any]] = []
    accepted_indices: set[int] = set()
    started_at = datetime.now().isoformat()
    start = time.time()

    selected_by_idx = {int(item["row_index"]): item for item in selected}
    for n_done, idx in enumerate(sorted(selected_indices), start=1):
        base_row = r63b_rows[idx]
        sid = base_row["session_id"]
        top_id = base_row["predicted_track_ids"][0]
        if sid not in blind_by_sid:
            raise RuntimeError(f"Missing Blind-A session_id={sid}")
        if top_id not in catalog:
            raise RuntimeError(f"Missing catalog metadata for top_id={top_id}")
        top_meta = catalog[top_id]
        resp, n_attempts, failures = generate_one(
            client,
            blind_by_sid[sid],
            top_meta,
            catalog,
            idx,
            base_row["predicted_response"],
            usage,
            args.model,
        )
        accepted = not failures
        if accepted:
            final_rows[idx]["predicted_response"] = resp
            accepted_indices.add(idx)
        else:
            final_rows[idx]["predicted_response"] = base_row["predicted_response"]
            failures_after_retries.append({
                "row_index": idx,
                "session_id": sid,
                "top1_track_id": top_id,
                "top1_track": top_meta.get("track_name", ""),
                "top1_artist": top_meta.get("artist_name", ""),
                "failures": failures,
                "last_generated_response": resp,
                "kept_original_r63b_response": True,
            })
        attempts.append({
            "row_index": idx,
            "session_id": sid,
            "weakness_score": selected_by_idx[idx]["weakness_score"],
            "attempts": n_attempts,
            "accepted_regenerated_response": accepted,
            "word_count": r63.word_count(final_rows[idx]["predicted_response"]),
            "top1_track": top_meta.get("track_name", ""),
            "top1_artist": top_meta.get("artist_name", ""),
        })
        write_jsonl(PERSISTED_ROWS, final_rows)
        print(
            f"{ts()} Targeted row {n_done}/{len(selected)} idx={idx} "
            f"accepted={accepted} attempts={n_attempts} api_calls={usage.calls}",
            flush=True,
        )

    print(f"{ts()} Writing final merged rows from disk state")
    final_rows = read_jsonl(PERSISTED_ROWS)
    write_jsonl(FINAL_ROWS, final_rows)

    print(f"{ts()} Validating merged payload")
    validation = r63.validate_submission(final_rows, r54c_rows, catalog)
    r63.abort_on_track_mismatch(validation)
    regen_validation = per_regen_validation(
        final_rows, selected_indices, accepted_indices, catalog
    )
    completed_at = datetime.now().isoformat()

    metadata = write_metadata(
        final_rows,
        r63b_rows,
        selected,
        attempts,
        failures_after_retries,
        validation,
        regen_validation,
        usage,
        args.model,
        started_at,
        completed_at,
        r63_metadata,
        r63b_metadata,
        catalog,
    )
    write_result_doc(
        metadata,
        validation,
        regen_validation,
        selected,
        metadata["before_after_samples"],
    )

    if not metadata["ready_to_submit"]:
        print(json.dumps({
            "validation": validation,
            "regenerated_row_validation": regen_validation,
        }, indent=2, ensure_ascii=False))
        raise SystemExit("Structural validation failed. Zip was not written.")

    print(f"{ts()} Packaging from {FINAL_ROWS.relative_to(REPO)}")
    rows_for_zip = read_jsonl(FINAL_ROWS)
    write_zip(rows_for_zip)
    elapsed = time.time() - start
    print(f"{ts()} Wrote {OUT_ZIP}")
    print(f"{ts()} Wrote {OUT_METADATA}")
    print(f"{ts()} Wrote {OUT_DOC}")
    print(
        "R63c DONE: "
        f"regenerated={metadata['generation']['accepted_regenerated_rows']} rows, "
        f"fallback_to_r63b={metadata['generation']['fallback_original_rows']}, "
        f"model={args.model}, "
        f"LexDiv={validation['lexdiv_distinct2']:.4f}, "
        "track_match="
        f"{validation['track_hash_comparison']['rows_with_matching_track_sequence']}/80, "
        f"ready_to_submit={'YES' if metadata['ready_to_submit'] else 'NO'}, "
        f"elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
