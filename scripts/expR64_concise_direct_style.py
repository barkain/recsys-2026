#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R64: concise-direct response style variant on fixed R63c-repair tracks.

This is a clean response-only style experiment. It regenerates all 80 Blind-A
responses, keeps track IDs bitwise identical to R63c-repair/R54c, persists rows
immediately, and packages only from the persisted JSONL if local gates pass.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.expR63_e1_response_only as r63

OUT_DIR = REPO / "exp" / "inference" / "blind_a"
R63C_REPAIR_ZIP = OUT_DIR / "r63c_repair_polish_submission.zip"
R63C_REPAIR_METADATA = OUT_DIR / "r63c_repair_polish_submission.metadata.json"
PERSISTED_ROWS = OUT_DIR / "r64_rows_persisted.jsonl"
OUT_ZIP = OUT_DIR / "r64_concise_direct_submission.zip"
OUT_METADATA = OUT_DIR / "r64_concise_direct_submission.metadata.json"
OUT_DOC = REPO / "docs" / "r64_concise_direct_result.md"

MODEL_ID = "claude-opus-4-7"
PROMPT_VERSION = (
    "R64 concise-direct v1; full 80-row response regen; "
    "recommendation-card style 35-55w target, 60w ceiling"
)
MAX_TOKENS = 220
MAX_ATTEMPTS = 4
WORD_TARGET = (35, 55)
WORD_HARD = (35, 60)
LEXDIV_FLOOR = 0.830
LEXDIV_TARGET = 0.835
MAX_FAILED_ROWS_FOR_PACKAGING = 5

SUBMISSION_LABEL_TEMPLATE = (
    "R64 concise-direct response style variant | base=R63c-repair | tracks=R54c | "
    "full 80-row regen, recommendation-card style 35-55w | LexDiv={lexdiv:.4f} | "
    "purpose=disambiguate LLM 4.85 style ceiling"
)

EXTRA_FORBIDDEN_RX = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bsonic\s+landscape\b",
        r"\btapestry\b",
        r"\baural\b",
        r"\bimmersive\b",
        r"\bevocative\b",
        r"\btransports\s+you\b",
        r"\btakes\s+you\s+on\b",
        r"\bfor\s+your\s+playlist\b",
        r"\bgreat\s+addition\b",
        r"\bstandout\s+addition\b",
    ]
]

STYLE_PLANS = [
    {
        "first": "Use 'fits' only if it is the cleanest verb; otherwise say it 'matches' the stated need.",
        "evidence": "Cite one earlier track, artist, or accepted/rejected direction.",
        "music": "Name rhythm, guitar, synth, vocal, sample, or arrangement in plain language.",
    },
    {
        "first": "Frame the pick as the strongest answer to the user's latest constraint.",
        "evidence": "Point to a concrete phrase or preference from the session.",
        "music": "Describe tempo, groove, instrumentation, or production texture.",
    },
    {
        "first": "Say the track works because it preserves one useful trait from the session.",
        "evidence": "Use one prior music turn as the evidence point.",
        "music": "Keep the attribute sentence factual and compact.",
    },
    {
        "first": "Make the first sentence a direct top-pick claim, not a general compliment.",
        "evidence": "Mention what the user wanted more of or wanted less of.",
        "music": "Name a concrete sound source such as bass, drums, guitar, keys, or vocals.",
    },
    {
        "first": "Tie the choice to the listener goal before any sound description.",
        "evidence": "Refer to the conversation's most recent turn when possible.",
        "music": "Use one specific musical noun and one clear modifier.",
    },
    {
        "first": "State that the track suits the request because of a single clear match.",
        "evidence": "Use one session anchor without retelling the whole chat.",
        "music": "Identify a production, arrangement, or performance detail.",
    },
    {
        "first": "Present the track as a direct continuation of the best-matching thread.",
        "evidence": "Name one prior artist or track if the session includes one.",
        "music": "Describe the musical attribute without genre-label padding.",
    },
    {
        "first": "Use 'is the right pick' only if the rest of the sentence gives the reason.",
        "evidence": "Keep the evidence sentence centered on the user's behavior or wording.",
        "music": "Pick one attribute that explains the match, not a generic mood word.",
    },
    {
        "first": "Start with title and artist, then a concise because-clause.",
        "evidence": "Use a concrete session contrast if the user rejected something.",
        "music": "Name the hook, pulse, vocal delivery, or instrumental lead.",
    },
    {
        "first": "Make the match logic about specificity, not broad genre membership.",
        "evidence": "Reference only one session fact, but make it identifiable.",
        "music": "Use album or era only if it clarifies the sound.",
    },
    {
        "first": "Say the recommendation answers the request with one exact reason.",
        "evidence": "Avoid 'you asked for'; paraphrase the user's need instead.",
        "music": "Use concrete nouns over adjectives.",
    },
    {
        "first": "Give the first sentence a compact recommendation-card cadence.",
        "evidence": "Mention one earlier match that made the direction clear.",
        "music": "Anchor the sound in beat, melody, vocals, or instrumentation.",
    },
    {
        "first": "Make the title and artist the subject of the first sentence.",
        "evidence": "Connect the choice to one user preference or conversation goal.",
        "music": "Name one audible detail a listener could verify.",
    },
    {
        "first": "Use a firm but plain top-1 justification in sentence one.",
        "evidence": "Use the user's latest refinement as the evidence point.",
        "music": "Keep the musical attribute sentence under 18 words if possible.",
    },
    {
        "first": "Explain the fit through restraint, intensity, continuity, or contrast as appropriate.",
        "evidence": "Point to one accepted direction or one mismatch being avoided.",
        "music": "Name the arrangement element that creates that effect.",
    },
    {
        "first": "Write the first sentence as a direct recommendation, not an essay opener.",
        "evidence": "Use profile, goal, or prior turn evidence when the chat is sparse.",
        "music": "Mention the concrete sonic feature last.",
    },
]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            clean = line.strip()
            if not clean:
                continue
            row = json.loads(clean)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def load_rows_from_zip(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source submission zip: {path}")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        if names != ["prediction.json"]:
            raise ValueError(f"Unexpected zip contents for {path}: {names}")
        rows = json.loads(zf.read("prediction.json"))
    if not isinstance(rows, list):
        raise TypeError(f"{path} prediction.json root is not a list")
    return rows


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def clean_stale_package() -> None:
    for path in (OUT_ZIP, OUT_METADATA):
        if path.exists():
            path.unlink()


def reset_persisted_rows() -> None:
    if PERSISTED_ROWS.exists():
        PERSISTED_ROWS.unlink()


def usage_from_dict(data: dict[str, Any] | None) -> r63.UsageTotals:
    usage = r63.UsageTotals()
    if not data:
        return usage
    usage.calls = int(data.get("calls", 0) or 0)
    usage.input_tokens = int(data.get("input_tokens", 0) or 0)
    usage.output_tokens = int(data.get("output_tokens", 0) or 0)
    usage.cache_creation_input_tokens = int(
        data.get("cache_creation_input_tokens", 0) or 0
    )
    usage.cache_read_input_tokens = int(data.get("cache_read_input_tokens", 0) or 0)
    return usage


def row_track_hash(row: dict[str, Any]) -> str:
    return sha256("\n".join(row.get("predicted_track_ids") or []).encode("utf-8")).hexdigest()


def normalize_for_counts(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text.lower())


def distinct2(rows: list[dict[str, Any]]) -> float:
    bigrams: list[tuple[str, str]] = []
    for row in rows:
        toks = normalize_for_counts(str(row.get("predicted_response") or ""))
        bigrams.extend(zip(toks, toks[1:]))
    return len(set(bigrams)) / max(len(bigrams), 1)


def opener_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cnt: Counter[str] = Counter()
    for row in rows:
        words = r63.normalize_for_cluster(str(row.get("predicted_response") or "")).split()[:5]
        opener = " ".join(words)
        if opener:
            cnt[opener] += 1
    return [
        {"opener": opener, "count": count}
        for opener, count in cnt.most_common()
        if count > 1
    ]


def extra_forbidden_hits(resp: str, top_meta: dict[str, Any]) -> list[str]:
    required_text = " ".join([
        str(top_meta.get("track_name") or ""),
        str(top_meta.get("artist_name") or ""),
    ])
    required_norm = r63.normalize_for_cluster(required_text)
    hits: list[str] = []
    for rx in EXTRA_FORBIDDEN_RX:
        if not rx.search(resp):
            continue
        pattern_words = re.sub(r"\\b|\\s\+", " ", rx.pattern)
        pattern_words = r63.normalize_ws(re.sub(r"[^A-Za-z0-9' -]", " ", pattern_words))
        if pattern_words and pattern_words.lower() in required_norm:
            continue
        hits.append(rx.pattern)
    return hits


def boilerplate_hits(resp: str, top_meta: dict[str, Any]) -> list[str]:
    return r63.boilerplate_hits(resp, top_meta) + extra_forbidden_hits(resp, top_meta)


def first_sentence(text: str) -> str:
    return r63.first_sentence(text)


def validate_response_text(resp: str, top_meta: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    clean = r63.normalize_ws(resp)
    wc = r63.word_count(clean)
    if wc < WORD_HARD[0] or wc > WORD_HARD[1]:
        failures.append(f"word_count_{wc}_outside_{WORD_HARD[0]}_{WORD_HARD[1]}")
    if not clean:
        failures.append("empty")
    if r63.PREFIX_LEAK_RX.search(clean) or r63.TAG_LINE_RX.search(clean):
        failures.append("prefix_or_tag_leak")
    if r63.TRAILING_QUESTION_RX.search(clean):
        failures.append("trailing_question")
    style_hits = boilerplate_hits(clean, top_meta)
    if style_hits:
        failures.append("boilerplate_or_forbidden_style:" + ",".join(style_hits[:4]))

    sent = r63.opening_window(clean)
    title = str(top_meta.get("track_name") or "")
    artist = str(top_meta.get("artist_name") or "")
    if title and not r63.contains_text(sent, title):
        failures.append("first_sentence_missing_track")
    artist_parts = [a.strip() for a in artist.split(",") if a.strip()]
    if artist_parts and not any(r63.contains_text(sent, a) for a in artist_parts):
        failures.append("first_sentence_missing_artist")
    return failures


def render_prompt(
    item: dict[str, Any],
    top_meta: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    row_index: int,
    retry_feedback: list[str] | None = None,
) -> str:
    retry_feedback = retry_feedback or []
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    album = top_meta.get("album_name") or "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"
    tags = r63.clean_tags(top_meta.get("tag_list") or [])
    plan = STYLE_PLANS[row_index % len(STYLE_PLANS)]

    pieces = [
        "Write exactly one concise music recommendation-card response.",
        "",
        "Hard style requirements:",
        f"- 35-55 words target; never below {WORD_HARD[0]} or above {WORD_HARD[1]} words.",
        "- Prefer exactly 3 sentences. Use a 4th only if the response would be unclear.",
        f"- Sentence 1 must name the top pick exactly as {title} by {artist} and say why it fits.",
        "- Sentence 2 must give one concrete session/user evidence point.",
        "- Sentence 3 must give one concrete musical attribute.",
        "- Direct recommendation-card prose, not an essay.",
        "- No metaphor, no literary flourish, no tag line, no genre prefix, no markdown.",
        "- Do not use these words or phrases except when literally part of the track or artist name: "
        "vibe, journey, soundscape, captures the essence.",
        "- Do not end with a question.",
        "- Do not use boilerplate openers such as 'If you're looking for', 'You might enjoy', "
        "'Here's a track that', or 'Great choice'.",
        "- Avoid repeated opener templates and repeated crutches such as 'perfect for', "
        "'right in', 'you asked for', 'you described', 'exactly what', and 'makes it a'.",
        "",
        "Variation plan for this row:",
        f"- First sentence: {plan['first']}",
        f"- Evidence sentence: {plan['evidence']}",
        f"- Music sentence: {plan['music']}",
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
        "Output only the response text.",
    ]
    if retry_feedback:
        pieces.extend([
            "",
            "The previous draft failed these checks; fix them in the new draft:",
            "; ".join(retry_feedback),
        ])
    return "\n".join(pieces)


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
    usage: r63.UsageTotals,
    model: str,
) -> tuple[str, int, list[str]]:
    system = (
        "You write direct, concise music recommendation-card responses. "
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
            retry_feedback=failures,
        )
        last = call_opus(client, system, prompt, usage, model=model)
        failures = validate_response_text(last, top_meta)
        if not failures:
            return last, attempt, []
        time.sleep(0.4)
    return last, MAX_ATTEMPTS, failures


def compare_track_sequences(
    rows: list[dict[str, Any]],
    ref_rows: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    rows_compared = min(len(rows), len(ref_rows))
    matching = 0
    mismatched = 0
    session_mismatches = 0
    turn_mismatches = 0
    per_position_mismatches = 0
    per_row: list[dict[str, Any]] = []
    for idx in range(rows_compared):
        row = rows[idx]
        ref = ref_rows[idx]
        tracks = list(row.get("predicted_track_ids") or [])
        ref_tracks = list(ref.get("predicted_track_ids") or [])
        session_match = row.get("session_id") == ref.get("session_id")
        turn_match = row.get("turn_number") == ref.get("turn_number")
        track_match = tracks == ref_tracks
        if session_match and turn_match and track_match:
            matching += 1
        else:
            mismatched += 1
        if not session_match:
            session_mismatches += 1
        if not turn_match:
            turn_mismatches += 1
        if not track_match:
            per_position_mismatches += sum(a != b for a, b in zip(tracks, ref_tracks))
            per_position_mismatches += abs(len(tracks) - len(ref_tracks))
        per_row.append({
            "row_index": idx,
            "session_id": row.get("session_id"),
            "reference_session_id": ref.get("session_id"),
            "r64_sha256": row_track_hash(row),
            f"{label}_sha256": row_track_hash(ref),
            "session_match": session_match,
            "turn_match": turn_match,
            "track_sequence_match": track_match,
            "matching": session_match and turn_match and track_match,
        })

    length_mismatch = abs(len(rows) - len(ref_rows))
    mismatched += length_mismatch
    return {
        "reference_label": label,
        "rows_compared": rows_compared,
        "row_count_difference": len(rows) - len(ref_rows),
        "rows_with_matching_session_turn_and_track_sequence": matching,
        "rows_with_mismatch": mismatched,
        "session_mismatches": session_mismatches + length_mismatch,
        "turn_mismatches": turn_mismatches + length_mismatch,
        "per_position_track_mismatches": per_position_mismatches,
        "total_tracks_compared": sum(len(row.get("predicted_track_ids") or []) for row in rows),
        "per_row": per_row,
    }


def validate_submission(
    rows: list[dict[str, Any]],
    r63c_repair_rows: list[dict[str, Any]],
    r54c_rows: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sids = [row.get("session_id") for row in rows]
    duplicate_session_count = len(sids) - len(set(sids))
    track_rows_wrong_len = 0
    rows_with_duplicate_tracks = 0
    invalid_uuid_count = 0
    invalid_catalog_count = 0
    prefix_leak_count = 0
    trailing_question_count = 0
    boilerplate_count = 0
    hard_word_range_violations = 0
    target_word_range_violations = 0
    first_sentence_name_violations = 0
    empty_count = 0
    sentence_count_violations = 0
    row_word_counts: list[dict[str, Any]] = []
    response_failures: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        tracks = list(row.get("predicted_track_ids") or [])
        resp = str(row.get("predicted_response") or "")
        clean = r63.normalize_ws(resp)
        if len(tracks) != 20:
            track_rows_wrong_len += 1
        if len(set(tracks)) != len(tracks):
            rows_with_duplicate_tracks += 1
        invalid_uuid_count += sum(1 for tid in tracks if not r63.UUID_RX.match(str(tid)))
        invalid_catalog_count += sum(1 for tid in tracks if str(tid) not in catalog)
        if not clean:
            empty_count += 1
        if r63.PREFIX_LEAK_RX.search(clean) or r63.TAG_LINE_RX.search(clean):
            prefix_leak_count += 1
        if r63.TRAILING_QUESTION_RX.search(clean):
            trailing_question_count += 1

        top_meta = catalog.get(tracks[0]) if tracks else None
        if top_meta and boilerplate_hits(clean, top_meta):
            boilerplate_count += 1
        wc = r63.word_count(clean)
        if wc < WORD_HARD[0] or wc > WORD_HARD[1]:
            hard_word_range_violations += 1
        if wc < WORD_TARGET[0] or wc > WORD_TARGET[1]:
            target_word_range_violations += 1
        sentence_count = len([s for s in re.split(r"(?<=[.!?])\s+", clean) if s.strip()])
        if sentence_count < 3 or sentence_count > 4:
            sentence_count_violations += 1
        row_word_counts.append({
            "row_index": idx,
            "session_id": row.get("session_id"),
            "word_count": wc,
            "sentence_count_estimate": sentence_count,
        })
        if top_meta:
            failures = validate_response_text(clean, top_meta)
            fs_failures = [f for f in failures if f.startswith("first_sentence_missing")]
            if fs_failures:
                first_sentence_name_violations += 1
            if failures:
                response_failures.append({
                    "row_index": idx,
                    "session_id": row.get("session_id"),
                    "failures": failures,
                    "word_count": wc,
                })

    lexdiv = distinct2(rows)
    clusters = opener_clusters(rows)
    opener_max = max([c["count"] for c in clusters], default=1)
    r63c_comp = compare_track_sequences(rows, r63c_repair_rows, "r63c_repair")
    r54c_comp = compare_track_sequences(rows, r54c_rows, "r54c")
    rows_track_match_r63c = (
        r63c_comp["rows_with_matching_session_turn_and_track_sequence"] == 80
        and r63c_comp["rows_with_mismatch"] == 0
        and r63c_comp["per_position_track_mismatches"] == 0
    )
    rows_track_match_r54c = (
        r54c_comp["rows_with_matching_session_turn_and_track_sequence"] == 80
        and r54c_comp["rows_with_mismatch"] == 0
        and r54c_comp["per_position_track_mismatches"] == 0
    )

    gates = {
        "rows_80": len(rows) == 80,
        "unique_sessions_80": len(set(sids)) == 80 and duplicate_session_count == 0,
        "tracks_20_each": track_rows_wrong_len == 0,
        "total_tracks_1600": sum(len(row.get("predicted_track_ids") or []) for row in rows) == 1600,
        "no_duplicate_tracks_within_row": rows_with_duplicate_tracks == 0,
        "valid_uuid_track_ids": invalid_uuid_count == 0,
        "valid_catalog_track_ids": invalid_catalog_count == 0,
        "track_ids_exactly_equal_to_r63c_repair_per_position": rows_track_match_r63c,
        "track_ids_exactly_equal_to_r54c_per_position": rows_track_match_r54c,
        "prefix_leak_count_0": prefix_leak_count == 0,
        "trailing_question_count_0": trailing_question_count == 0,
        "boilerplate_count_0": boilerplate_count == 0,
        "word_count_35_60_hard_band": hard_word_range_violations == 0,
        "target_word_count_35_55": target_word_range_violations == 0,
        "first_sentence_names_top1_track_and_artist": first_sentence_name_violations == 0,
        "sentence_count_3_or_4": sentence_count_violations == 0,
        "local_lexdiv_floor_0_830": lexdiv >= LEXDIV_FLOOR,
        "local_lexdiv_target_0_835": lexdiv >= LEXDIV_TARGET,
        "opener_cluster_max_le_5": opener_max <= 5,
    }
    return {
        "gates": gates,
        "counts": {
            "rows": len(rows),
            "unique_sessions": len(set(sids)),
            "duplicate_session_count": duplicate_session_count,
            "track_rows_wrong_len": track_rows_wrong_len,
            "total_tracks": sum(len(row.get("predicted_track_ids") or []) for row in rows),
            "rows_with_duplicate_tracks": rows_with_duplicate_tracks,
            "invalid_uuid_track_ids": invalid_uuid_count,
            "invalid_catalog_track_ids": invalid_catalog_count,
            "prefix_leak_count": prefix_leak_count,
            "trailing_question_count": trailing_question_count,
            "boilerplate_count": boilerplate_count,
            "empty_response_count": empty_count,
            "hard_word_range_violations": hard_word_range_violations,
            "target_word_range_violations": target_word_range_violations,
            "first_sentence_name_violations": first_sentence_name_violations,
            "sentence_count_violations": sentence_count_violations,
        },
        "word_counts": row_word_counts,
        "response_failures": response_failures,
        "lexdiv_distinct2": lexdiv,
        "repeated_opener_clusters": clusters,
        "opener_max_cluster": opener_max,
        "track_hash_comparison": {
            "r63c_repair": r63c_comp,
            "r54c": r54c_comp,
            "track_match_summary": {
                "r63c_repair_rows_matching": r63c_comp[
                    "rows_with_matching_session_turn_and_track_sequence"
                ],
                "r54c_rows_matching": r54c_comp[
                    "rows_with_matching_session_turn_and_track_sequence"
                ],
            },
        },
    }


WARNING_GATES = {
    "target_word_count_35_55",
    "sentence_count_3_or_4",
    "local_lexdiv_target_0_835",
}


def hard_gates_pass(validation: dict[str, Any]) -> bool:
    return all(
        bool(passed)
        for name, passed in validation["gates"].items()
        if name not in WARNING_GATES
    )


def ready_label(
    validation: dict[str, Any],
    rows_failed_after_retries: int,
) -> str:
    if rows_failed_after_retries > MAX_FAILED_ROWS_FOR_PACKAGING:
        return "NO_TOO_MANY_FAILS"
    if float(validation["lexdiv_distinct2"]) < LEXDIV_FLOOR:
        return "NO_LEXDIV_BELOW_FLOOR"
    if not hard_gates_pass(validation):
        return "NO_VALIDATION_FAILED"
    if float(validation["lexdiv_distinct2"]) < LEXDIV_TARGET:
        return "YES_WARN_BORDERLINE"
    return "YES_PASS"


def gate_statuses(
    validation: dict[str, Any],
    rows_failed_after_retries: int,
) -> dict[str, Any]:
    statuses: dict[str, Any] = {}
    for name, passed in validation["gates"].items():
        if passed:
            status = "pass"
            blocking = False
        elif name in WARNING_GATES:
            status = "warn"
            blocking = False
        else:
            status = "fail"
            blocking = True
        statuses[name] = {
            "status": status,
            "passed": bool(passed),
            "blocking": blocking,
        }
    statuses["rows_failed_after_retries_le_5"] = {
        "status": "pass" if rows_failed_after_retries <= MAX_FAILED_ROWS_FOR_PACKAGING else "fail",
        "passed": rows_failed_after_retries <= MAX_FAILED_ROWS_FOR_PACKAGING,
        "blocking": True,
    }
    return statuses


def gate_table(
    validation: dict[str, Any],
    rows_failed_after_retries: int,
) -> str:
    lines = ["| Gate | Result |", "|---|---|"]
    statuses = gate_statuses(validation, rows_failed_after_retries)
    for name, info in statuses.items():
        lines.append(f"| `{name}` | {str(info['status']).upper()} |")
    return "\n".join(lines)


def write_zip_from_persisted_rows() -> list[dict[str, Any]]:
    rows = read_jsonl(PERSISTED_ROWS)
    payload = json.dumps(rows, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)
    return rows


def sample_comparisons(
    r63c_repair_rows: list[dict[str, Any]],
    r64_rows: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    limit: int = 10,
) -> list[dict[str, Any]]:
    samples = []
    for idx in range(min(limit, len(r63c_repair_rows), len(r64_rows))):
        top_id = r64_rows[idx]["predicted_track_ids"][0]
        meta = catalog.get(top_id, {})
        samples.append({
            "row_index": idx,
            "session_id": r64_rows[idx]["session_id"],
            "top1_track_id": top_id,
            "top1_track": meta.get("track_name", ""),
            "top1_artist": meta.get("artist_name", ""),
            "r63c_repair_response": r63c_repair_rows[idx]["predicted_response"],
            "r64_response": r64_rows[idx]["predicted_response"],
            "r64_word_count": r63.word_count(r64_rows[idx]["predicted_response"]),
        })
    return samples


def write_metadata(
    rows: list[dict[str, Any]],
    r63c_repair_rows: list[dict[str, Any]],
    r54c_rows: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    usage: r63.UsageTotals,
    generation_report: dict[str, Any],
    model: str,
    r63c_repair_metadata: dict[str, Any] | None,
    packaged: bool,
    ready: str,
) -> dict[str, Any]:
    lexdiv = float(validation["lexdiv_distinct2"])
    metadata = {
        "submission_label": SUBMISSION_LABEL_TEMPLATE.format(lexdiv=lexdiv),
        "created_at": datetime.now().isoformat(),
        "generation_timestamp": datetime.now().isoformat(),
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "base": {
            "name": "R63c-repair production",
            "submission_zip": str(R63C_REPAIR_ZIP.relative_to(REPO)),
            "metadata_file": str(R63C_REPAIR_METADATA.relative_to(REPO)),
            "blind_result": {
                "composite": 0.6224,
                "llm_judge": 4.85,
                "lexdiv": 0.8438,
                "local_lexdiv": (r63c_repair_metadata or {}).get("lexdiv_distinct2"),
            },
        },
        "source_tracks": {
            "r63c_repair_artifact": str(R63C_REPAIR_ZIP.relative_to(REPO)),
            "r54c_artifact": str(r63.R54C_ZIP.relative_to(REPO)),
            "constraint": "R64 predicted_track_ids copied from R63c-repair/R54c, unchanged.",
        },
        "persisted_rows_file": str(PERSISTED_ROWS.relative_to(REPO)),
        "submission_zip": str(OUT_ZIP.relative_to(REPO)),
        "packaged": packaged,
        "ready_to_submit": ready,
        "lexdiv_distinct2": lexdiv,
        "lexdiv_policy": {
            "hard_floor": LEXDIV_FLOOR,
            "target": LEXDIV_TARGET,
            "do_not_package_below_floor": True,
            "borderline_warning_band": [LEXDIV_FLOOR, LEXDIV_TARGET],
        },
        "generation": generation_report,
        "usage": usage.as_dict(),
        "validation_results": gate_statuses(
            validation,
            int(generation_report["rows_failed_after_retries"]),
        ),
        "r64_track_hash_match": validation["track_hash_comparison"],
        "validation": validation,
        "before_after_samples": sample_comparisons(r63c_repair_rows, rows, catalog),
        "decision_tree_post_blind_result": {
            "blind_llm_gte_4_90": "R64 becomes production.",
            "blind_llm_eq_4_85": "Compare LexDiv/composite, keep better of R63c-repair vs R64.",
            "blind_llm_lt_4_85": "Archive R64.",
        },
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def write_result_doc(
    metadata: dict[str, Any],
    validation: dict[str, Any],
    generation_report: dict[str, Any],
    samples: list[dict[str, Any]],
) -> None:
    lexdiv = float(validation["lexdiv_distinct2"])
    r63c_comp = validation["track_hash_comparison"]["r63c_repair"]
    r54c_comp = validation["track_hash_comparison"]["r54c"]
    sample_lines: list[str] = []
    for sample in samples:
        sample_lines.extend([
            f"### Row {sample['row_index']}: {sample['top1_track']} by {sample['top1_artist']}",
            f"- session_id: `{sample['session_id']}`",
            f"- R63c-repair: {sample['r63c_repair_response']}",
            f"- R64 ({sample['r64_word_count']}w): {sample['r64_response']}",
            "",
        ])
    verdict = metadata["ready_to_submit"]
    packaged = metadata["packaged"]
    usage = metadata.get("usage") or {}
    if usage.get("token_usage_unavailable_after_package_revalidation"):
        cost_line = (
            "unavailable; package-only revalidation preserved call count but not "
            "the original token detail"
        )
    else:
        cost_line = f"${float(usage.get('estimated_cost_usd', 0.0)):.4f}"
    doc = f"""# R64 Concise-Direct Result

## Gate Table
{gate_table(validation, int(generation_report['rows_failed_after_retries']))}

## Track Hash Comparison
```text
R64 vs R63c-repair:
  rows compared: {r63c_comp['rows_compared']}
  rows matching session/turn/track sequence: {r63c_comp['rows_with_matching_session_turn_and_track_sequence']}
  rows with mismatch: {r63c_comp['rows_with_mismatch']}
  per-position track mismatches: {r63c_comp['per_position_track_mismatches']}

R64 vs R54c:
  rows compared: {r54c_comp['rows_compared']}
  rows matching session/turn/track sequence: {r54c_comp['rows_with_matching_session_turn_and_track_sequence']}
  rows with mismatch: {r54c_comp['rows_with_mismatch']}
  per-position track mismatches: {r54c_comp['per_position_track_mismatches']}
```

## Summary
- Submission label: `{metadata['submission_label']}`
- Model used: `{metadata['model']}`
- Verdict: `{verdict}`
- Packaged: {'YES' if packaged else 'NO'}
- Submission artifact: `{OUT_ZIP.relative_to(REPO)}`{' (not written)' if not packaged else ''}
- Metadata: `{OUT_METADATA.relative_to(REPO)}`
- Persisted rows: `{PERSISTED_ROWS.relative_to(REPO)}`
- Local LexDiv: {lexdiv:.4f}
- LexDiv hard floor: {LEXDIV_FLOOR:.3f} ({'PASS' if lexdiv >= LEXDIV_FLOOR else 'FAIL'})
- LexDiv target: {LEXDIV_TARGET:.3f} ({'PASS' if lexdiv >= LEXDIV_TARGET else 'WARN'})
- Rows failed after retries: {generation_report['rows_failed_after_retries']}
- Average attempts per row: {generation_report['retries_per_row_avg']:.4f}
- Max repeated opener cluster: {validation['opener_max_cluster']}
- Prefix leaks: {validation['counts']['prefix_leak_count']}
- Trailing questions: {validation['counts']['trailing_question_count']}
- Boilerplate/forbidden-style rows: {validation['counts']['boilerplate_count']}
- Hard word-band violations: {validation['counts']['hard_word_range_violations']}
- Target word-band warnings: {validation['counts']['target_word_range_violations']}
- Opus API calls: {usage.get('calls', 0)}
- Estimated run cost: {cost_line}

## Decision Tree
- If blind LLM >= 4.90: R64 becomes production.
- If blind LLM = 4.85: compare LexDiv/composite, keep better of R63c-repair vs R64.
- If blind LLM < 4.85: archive R64.

## Repeated Opener Clusters
```json
{json.dumps(validation['repeated_opener_clusters'], indent=2, ensure_ascii=False)}
```

## Failed Rows After Retries
```json
{json.dumps(generation_report['failures_after_retries'], indent=2, ensure_ascii=False)}
```

## Before/After Samples
{chr(10).join(sample_lines)}
"""
    OUT_DOC.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing partial r64_rows_persisted.jsonl if present.",
    )
    parser.add_argument(
        "--package-existing",
        action="store_true",
        help="Validate/package the existing persisted rows without new model calls.",
    )
    args = parser.parse_args()

    if args.model != MODEL_ID:
        raise SystemExit(
            f"Refusing model override {args.model!r}; R64 requires {MODEL_ID!r}."
        )
    if args.package_existing:
        if OUT_ZIP.exists():
            OUT_ZIP.unlink()
        print(f"{ts()} R64 package-existing validation from persisted rows")
        r63c_repair_rows = load_rows_from_zip(R63C_REPAIR_ZIP)
        r54c_rows = r63.load_r54c_rows()
        catalog = r63.load_catalog()
        r63c_repair_metadata = load_json(R63C_REPAIR_METADATA)
        prior_metadata = load_json(OUT_METADATA) or {}
        rows = read_jsonl(PERSISTED_ROWS)
        validation = validate_submission(rows, r63c_repair_rows, r54c_rows, catalog)
        failures_after_retries = validation["response_failures"]
        generation_report = dict(prior_metadata.get("generation") or {})
        attempts = list(generation_report.get("attempts") or [])
        failing_indices = {int(item["row_index"]) for item in failures_after_retries}
        for item in attempts:
            if "row_index" in item:
                item["accepted_after_style_gates"] = int(item["row_index"]) not in failing_indices
        generation_report.update({
            "rows_generated": len(rows),
            "rows_failed_after_retries": len(failures_after_retries),
            "failures_after_retries": failures_after_retries,
            "attempts": attempts,
            "packaging_revalidated_at": datetime.now().isoformat(),
            "packaging_revalidation_note": (
                "Validation/package-only pass from persisted rows; no generated text changed."
            ),
        })
        if "retries_per_row_avg" not in generation_report and attempts:
            total_attempts = sum(int(item.get("attempts", 1) or 1) for item in attempts)
            generation_report["retries_per_row_avg"] = (
                total_attempts - len(attempts)
            ) / max(len(attempts), 1)
        if "retries_per_row_avg" not in generation_report:
            generation_report["retries_per_row_avg"] = 0.0
        ready = ready_label(validation, len(failures_after_retries))
        packaged = (
            ready in {"YES_PASS", "YES_WARN_BORDERLINE"}
            and len(failures_after_retries) <= MAX_FAILED_ROWS_FOR_PACKAGING
            and hard_gates_pass(validation)
        )
        usage = usage_from_dict(prior_metadata.get("usage"))
        metadata = write_metadata(
            rows,
            r63c_repair_rows,
            r54c_rows,
            catalog,
            validation,
            usage,
            generation_report,
            args.model,
            r63c_repair_metadata,
            packaged,
            ready,
        )
        write_result_doc(
            metadata,
            validation,
            generation_report,
            metadata["before_after_samples"],
        )
        if packaged:
            write_zip_from_persisted_rows()
            print(f"{ts()} Wrote {OUT_ZIP}")
        else:
            if OUT_ZIP.exists():
                OUT_ZIP.unlink()
            print(f"{ts()} Packaging skipped; ready_to_submit={ready}")
        print(f"{ts()} Wrote {OUT_METADATA}")
        print(f"{ts()} Wrote {OUT_DOC}")
        print(
            "R64 DONE: "
            f"model={args.model}, "
            f"LexDiv={validation['lexdiv_distinct2']:.4f}, "
            f"retries_per_row_avg={generation_report['retries_per_row_avg']:.4f}, "
            f"rows_failed_after_retries={len(failures_after_retries)}, "
            "track_match="
            f"{validation['track_hash_comparison']['r63c_repair']['rows_with_matching_session_turn_and_track_sequence']}/80, "
            f"ready_to_submit={ready}"
        )
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY. Pause.")

    import anthropic

    clean_stale_package()
    if not args.resume:
        reset_persisted_rows()

    print(f"{ts()} R64 concise-direct full response regeneration")
    print(f"{ts()} Loading R63c-repair rows, R54c rows, Blind-A sessions, and catalog")
    r63c_repair_rows = load_rows_from_zip(R63C_REPAIR_ZIP)
    if len(r63c_repair_rows) != 80:
        raise RuntimeError(f"Expected 80 R63c-repair rows, found {len(r63c_repair_rows)}")
    r54c_rows = r63.load_r54c_rows()
    blind_by_sid = r63.load_blind_by_sid()
    catalog = r63.load_catalog()
    r63c_repair_metadata = load_json(R63C_REPAIR_METADATA)

    client = anthropic.Anthropic(api_key=api_key)
    usage = r63.UsageTotals()
    print(f"{ts()} Confirming Anthropic model availability: {args.model}")
    r63.confirm_model(client, usage, args.model)
    print(f"{ts()} Model check passed")

    rows: list[dict[str, Any]] = []
    if args.resume and PERSISTED_ROWS.exists():
        rows = read_jsonl(PERSISTED_ROWS)
        print(f"{ts()} Resuming from {len(rows)} persisted rows")
    write_jsonl_atomic(PERSISTED_ROWS, rows)

    attempts: list[dict[str, Any]] = []
    failures_after_retries: list[dict[str, Any]] = []
    started_at = datetime.now().isoformat()
    start = time.time()

    for idx in range(len(rows), len(r63c_repair_rows)):
        base = r63c_repair_rows[idx]
        sid = base["session_id"]
        top_id = base["predicted_track_ids"][0]
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
            usage,
            args.model,
        )
        if failures:
            failures_after_retries.append({
                "row_index": idx,
                "session_id": sid,
                "top1_track_id": top_id,
                "top1_track": top_meta.get("track_name", ""),
                "top1_artist": top_meta.get("artist_name", ""),
                "failures": failures,
                "last_generated_response": resp,
                "word_count": r63.word_count(resp),
            })
        rows.append({
            "session_id": sid,
            "turn_number": base["turn_number"],
            "predicted_track_ids": list(base["predicted_track_ids"]),
            "predicted_response": resp,
        })
        attempts.append({
            "row_index": idx,
            "session_id": sid,
            "attempts": n_attempts,
            "accepted_after_style_gates": not failures,
            "word_count": r63.word_count(resp),
            "top1_track": top_meta.get("track_name", ""),
            "top1_artist": top_meta.get("artist_name", ""),
        })
        write_jsonl_atomic(PERSISTED_ROWS, rows)
        if (idx + 1) % 5 == 0:
            elapsed = time.time() - start
            avg_attempts = sum(item["attempts"] for item in attempts) / max(len(attempts), 1)
            print(
                f"{ts()} Generated {idx + 1}/80 rows; "
                f"api_calls={usage.calls}; avg_attempts={avg_attempts:.2f}; "
                f"failed_after_retries={len(failures_after_retries)}; elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(f"{ts()} Re-reading persisted rows for validation and packaging gates")
    rows = read_jsonl(PERSISTED_ROWS)
    validation = validate_submission(rows, r63c_repair_rows, r54c_rows, catalog)
    completed_at = datetime.now().isoformat()
    total_row_attempts = sum(item["attempts"] for item in attempts)
    generation_report = {
        "started_at": started_at,
        "completed_at": completed_at,
        "rows_generated": len(rows),
        "rows_failed_after_retries": len(failures_after_retries),
        "failures_after_retries": failures_after_retries,
        "attempts": attempts,
        "max_attempts_per_row": MAX_ATTEMPTS,
        "word_target": list(WORD_TARGET),
        "word_hard_band": list(WORD_HARD),
        "retries_per_row_avg": (
            (total_row_attempts - len(attempts)) / max(len(attempts), 1)
        ),
        "attempts_per_row_avg": total_row_attempts / max(len(attempts), 1),
        "rows_persisted_immediately_after_each_generation": True,
        "packaging_source": str(PERSISTED_ROWS.relative_to(REPO)),
        "run_policy": (
            "one full generation pass; per-row retries only; no iterative repair; "
            "no fallback to R63c responses"
        ),
    }
    ready = ready_label(validation, len(failures_after_retries))
    packaged = (
        ready in {"YES_PASS", "YES_WARN_BORDERLINE"}
        and len(failures_after_retries) <= MAX_FAILED_ROWS_FOR_PACKAGING
        and hard_gates_pass(validation)
    )

    metadata = write_metadata(
        rows,
        r63c_repair_rows,
        r54c_rows,
        catalog,
        validation,
        usage,
        generation_report,
        args.model,
        r63c_repair_metadata,
        packaged,
        ready,
    )
    write_result_doc(
        metadata,
        validation,
        generation_report,
        metadata["before_after_samples"],
    )

    if packaged:
        print(f"{ts()} Packaging from persisted rows: {PERSISTED_ROWS.relative_to(REPO)}")
        write_zip_from_persisted_rows()
        print(f"{ts()} Wrote {OUT_ZIP}")
    else:
        print(f"{ts()} Packaging skipped; ready_to_submit={ready}")
        if OUT_ZIP.exists():
            OUT_ZIP.unlink()

    print(f"{ts()} Wrote {PERSISTED_ROWS}")
    print(f"{ts()} Wrote {OUT_METADATA}")
    print(f"{ts()} Wrote {OUT_DOC}")
    elapsed = time.time() - start
    print(
        "R64 DONE: "
        f"model={args.model}, "
        f"LexDiv={validation['lexdiv_distinct2']:.4f}, "
        f"retries_per_row_avg={generation_report['retries_per_row_avg']:.4f}, "
        f"rows_failed_after_retries={len(failures_after_retries)}, "
        "track_match="
        f"{validation['track_hash_comparison']['r63c_repair']['rows_with_matching_session_turn_and_track_sequence']}/80, "
        f"ready_to_submit={ready}, "
        f"elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
