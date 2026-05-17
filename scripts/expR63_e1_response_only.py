#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R63 / E1: response-only Blind-A submission on top of R54c tracks.

Tracks are copied from the R54c production zip and must remain bitwise
identical per session and per position. Only `predicted_response` is
regenerated, using Claude Opus 4.7 via the Anthropic API.

Outputs:
  exp/inference/blind_a/r63_response_only_submission.zip
  exp/inference/blind_a/r63_response_only_submission.metadata.json
  docs/r63_e1_response_only_result.md
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
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "4"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R54C_ZIP = REPO / "exp" / "inference" / "blind_a" / "r54c_polish_submission.zip"
OUT_DIR = REPO / "exp" / "inference" / "blind_a"
OUT_ZIP = OUT_DIR / "r63_response_only_submission.zip"
OUT_METADATA = OUT_DIR / "r63_response_only_submission.metadata.json"
PERSISTED_ROWS = OUT_DIR / "r63_rows_persisted.jsonl"
OUT_DOC = REPO / "docs" / "r63_e1_response_only_result.md"

MODEL_ID = "claude-opus-4-7"
PROMPT_VERSION = "R63 pass-3 final prompt, packaging-only re-run"
MAX_TOKENS = 256
TEMPERATURE = None  # Opus 4.7 rejects explicit temperature; use provider default.
MAX_ATTEMPTS = 4
WORD_TARGET = (65, 95)
PROMPT_WORD_AIM = (65, 70)
WORD_TOO_SHORT = 30
LEXDIV_FLOOR = 0.83
R54C_LEXDIV_TARGET = 0.8381

# Pricing source: https://www.anthropic.com/research/claude-opus-4-7
OPUS_47_INPUT_USD_PER_MTOK = 5.00
OPUS_47_OUTPUT_USD_PER_MTOK = 25.00
PRICING_SOURCE = "https://www.anthropic.com/research/claude-opus-4-7"

UUID_RX = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
TRAILING_QUESTION_RX = re.compile(r"\?\s*$")
PREFIX_LEAK_RX = re.compile(
    r"^\s*(system|user|assistant|response|recommended response|final response|"
    r"track|artist|album|tags|genre|mood|musical attributes|conversation|"
    r"context|user/session|session evidence|prompt|output)\s*[:\-]",
    re.IGNORECASE,
)
TAG_LINE_RX = re.compile(
    r"^\s*(genre|mood|tags?|artist|track|album|attributes?)\s*[:\-]",
    re.IGNORECASE,
)
FORBIDDEN_OPENERS = [
    r"if\s+you(?:'| a)re\s+looking\s+for",
    r"you\s+might\s+enjoy",
    r"here(?:'| i)s\s+a\s+track\s+that",
    r"here\s+is\s+a\s+track\s+that",
]
FILLER_PATTERNS = [
    r"\bvibe\b",
    r"\bjourney\b",
    r"\bsoundscape\b",
    r"\bcaptures\s+the\s+essence\b",
]
BOILERPLATE_PATTERNS = [
    r"\bgreat\s+choice\b",
    r"\bwonderful\s+choice\b",
    r"\bperfect\s+choice\b",
    r"\bhope\s+you\s+enjoy\b",
    r"\bi\s+hope\s+you\b",
    r"\benjoy\s+listening\b",
    r"\bhappy\s+listening\b",
    r"\blet\s+me\s+know\b",
    r"\bwould\s+you\s+like\b",
    r"\bis\s+there\s+anything\b",
    r"\bdo\s+you\s+want\b",
    r"\bas\s+an\s+ai\b",
    r"\bi(?:'| a)m\s+sorry\b",
    r"\bi\s+apologize\b",
]
COMPILED_FORBIDDEN_OPENERS = [re.compile(p, re.IGNORECASE) for p in FORBIDDEN_OPENERS]
COMPILED_FILLER = [re.compile(p, re.IGNORECASE) for p in FILLER_PATTERNS]
COMPILED_BOILERPLATE_ONLY = [
    re.compile(p, re.IGNORECASE) for p in BOILERPLATE_PATTERNS
]
COMPILED_BOILERPLATE = (
    COMPILED_FORBIDDEN_OPENERS + COMPILED_FILLER + COMPILED_BOILERPLATE_ONLY
)

STRUCTURAL_GATE_NAMES = {
    "rows_80",
    "unique_sessions_80",
    "tracks_20_each",
    "total_tracks_1600",
    "no_duplicate_tracks_within_row",
    "valid_uuid_track_ids",
    "track_ids_exactly_equal_to_r54c_per_position",
    "prefix_leak_count_0",
    "trailing_question_count_0",
    "boilerplate_count_0",
}
WARNING_GATE_NAMES = {
    "lexdiv_floor_0_83",
    "lexdiv_target_r54c_0_8381",
    "opener_cluster_max_le_5",
}


@dataclass
class UsageTotals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.calls += 1
        if usage is None:
            return
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        self.cache_creation_input_tokens += int(
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )
        self.cache_read_input_tokens += int(
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )

    def cost_usd(self) -> float:
        input_billable = self.input_tokens + self.cache_creation_input_tokens
        return (
            input_billable / 1_000_000 * OPUS_47_INPUT_USD_PER_MTOK
            + self.output_tokens / 1_000_000 * OPUS_47_OUTPUT_USD_PER_MTOK
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "estimated_cost_usd": round(self.cost_usd(), 6),
            "pricing": {
                "input_usd_per_million_tokens": OPUS_47_INPUT_USD_PER_MTOK,
                "output_usd_per_million_tokens": OPUS_47_OUTPUT_USD_PER_MTOK,
                "source": PRICING_SOURCE,
            },
        }


def combine_usage_dicts(current: dict[str, Any], prior: dict[str, Any] | None) -> dict[str, Any]:
    if not prior:
        return current
    calls = int(current.get("calls", 0) or 0) + int(prior.get("calls", 0) or 0)
    input_tokens = int(current.get("input_tokens", 0) or 0) + int(
        prior.get("input_tokens", 0) or 0
    )
    output_tokens = int(current.get("output_tokens", 0) or 0) + int(
        prior.get("output_tokens", 0) or 0
    )
    cache_creation = int(current.get("cache_creation_input_tokens", 0) or 0) + int(
        prior.get("cache_creation_input_tokens", 0) or 0
    )
    cache_read = int(current.get("cache_read_input_tokens", 0) or 0) + int(
        prior.get("cache_read_input_tokens", 0) or 0
    )
    cost = (
        (input_tokens + cache_creation) / 1_000_000 * OPUS_47_INPUT_USD_PER_MTOK
        + output_tokens / 1_000_000 * OPUS_47_OUTPUT_USD_PER_MTOK
    )
    return {
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "estimated_cost_usd": round(cost, 6),
        "pricing": current.get("pricing", {}),
    }


def load_prior_aborted_usage() -> dict[str, Any] | None:
    if not OUT_METADATA.exists():
        return None
    try:
        data = json.loads(OUT_METADATA.read_text())
    except Exception:
        return None
    gates = data.get("validation", {}).get("gates", {})
    if gates and all(bool(v) for v in gates.values()):
        return None
    usage = data.get("cumulative_usage_including_prior_aborted_runs") or data.get("usage")
    return usage if isinstance(usage, dict) else None


def reset_persisted_rows() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if PERSISTED_ROWS.exists():
        PERSISTED_ROWS.unlink()


def write_persisted_rows(rows: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PERSISTED_ROWS.with_name(f"{PERSISTED_ROWS.name}.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, PERSISTED_ROWS)


def load_persisted_rows() -> list[dict[str, Any]]:
    if not PERSISTED_ROWS.exists():
        raise FileNotFoundError(f"Missing persisted rows file: {PERSISTED_ROWS}")
    rows = []
    with PERSISTED_ROWS.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(
                    f"Persisted row {line_no} is {type(row).__name__}, expected object"
                )
            rows.append(row)
    return rows


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def normalize_for_cluster(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("\u2018", "'").replace("\u2019", "'")
    s = re.sub(r"[\"'`.,;:!?()\[\]{}]", "", s)
    return re.sub(r"\s+", " ", s)


def word_count(s: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", s))


def first_sentence(s: str) -> str:
    parts = re.split(r"(?<=[.!])\s+", s.strip(), maxsplit=1)
    return parts[0] if parts else s.strip()


def opening_window(s: str, n_words: int = 30) -> str:
    words = s.strip().split()
    return " ".join(words[:n_words])


def contains_text(haystack: str, needle: str) -> bool:
    if not needle:
        return False
    return normalize_for_cluster(needle) in normalize_for_cluster(haystack)


def boilerplate_hits(resp: str, top_meta: dict[str, Any] | None = None) -> list[str]:
    """Return banned opener/boilerplate/filler hits.

    A filler word is allowed only when it is part of the required track or
    artist name, e.g. Kendrick Lamar's "Bitch, Don't Kill My Vibe".
    """
    hits = []
    for rx in COMPILED_FORBIDDEN_OPENERS + COMPILED_BOILERPLATE_ONLY:
        if rx.search(resp):
            hits.append(rx.pattern)

    required_text = ""
    if top_meta:
        required_text = " ".join([
            str(top_meta.get("track_name") or ""),
            str(top_meta.get("artist_name") or ""),
        ])
    required_norm = normalize_for_cluster(required_text)
    for rx in COMPILED_FILLER:
        if not rx.search(resp):
            continue
        filler_token = normalize_for_cluster(rx.pattern)
        filler_token = re.sub(r"\\b|\\s\+", " ", filler_token)
        filler_token = normalize_ws(filler_token)
        if filler_token and filler_token in required_norm:
            continue
        hits.append(rx.pattern)
    return hits


def one(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def list_text(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    return [str(value)] if value else []


def load_r54c_rows() -> list[dict[str, Any]]:
    if not R54C_ZIP.exists():
        raise FileNotFoundError(f"Missing R54c production zip: {R54C_ZIP}")
    with zipfile.ZipFile(R54C_ZIP) as zf:
        names = zf.namelist()
        if names != ["prediction.json"]:
            raise ValueError(f"Unexpected R54c zip contents: {names}")
        rows = json.loads(zf.read("prediction.json"))
    if not isinstance(rows, list):
        raise TypeError("R54c prediction.json root is not a list")
    return rows


def load_blind_by_sid() -> dict[str, dict[str, Any]]:
    from datasets import DownloadConfig, load_dataset

    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Blind-A",
        split="test",
        download_config=DownloadConfig(local_files_only=True),
    )
    return {str(item["session_id"]): dict(item) for item in ds}


def load_catalog() -> dict[str, dict[str, Any]]:
    from datasets import DownloadConfig, load_dataset

    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        download_config=DownloadConfig(local_files_only=True),
    )["all_tracks"]
    catalog: dict[str, dict[str, Any]] = {}
    for row in ds:
        tid = str(row["track_id"])
        catalog[tid] = {
            "track_id": tid,
            "track_name": one(row.get("track_name")),
            "artist_name": ", ".join(list_text(row.get("artist_name"))),
            "album_name": one(row.get("album_name")),
            "release_date": str(row.get("release_date") or ""),
            "tag_list": list_text(row.get("tag_list")),
        }
    return catalog


def compact_profile(profile: Any) -> str:
    if not isinstance(profile, dict):
        return str(profile or "")
    fields = []
    for key in ("age_group", "country_name", "gender", "preferred_musical_culture"):
        val = profile.get(key)
        if val:
            fields.append(f"{key}={val}")
    return ", ".join(fields)


def compact_goal(goal: Any) -> str:
    if not isinstance(goal, dict):
        return str(goal or "")
    return str(goal.get("listener_goal") or goal)


def conversation_lines(item: dict[str, Any], catalog: dict[str, dict[str, Any]]) -> str:
    rows = []
    conv = item.get("conversations") or []
    for idx, turn in sorted(
        enumerate(conv), key=lambda x: (int(x[1].get("turn_number", 0)), x[0])
    ):
        role = str(turn.get("role", "unknown"))
        content = str(turn.get("content", ""))
        if role == "music" and content in catalog:
            meta = catalog[content]
            content = f"{meta['track_name']} by {meta['artist_name']}"
        rows.append(f"{role}: {content}")
    return "\n".join(rows)


def clean_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        t = normalize_ws(tag)
        if not t or len(t) > 32:
            continue
        low = t.lower()
        if low in seen:
            continue
        if any(p.search(low) for p in COMPILED_BOILERPLATE):
            continue
        if re.fullmatch(r"[0-9\s.,/+-]+", low):
            continue
        seen.add(low)
        out.append(t)
        if len(out) >= 10:
            break
    return out


def render_prompt(
    item: dict[str, Any],
    top_meta: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    row_index: int,
    retry_feedback: list[str] | None = None,
    avoid_openers: list[str] | None = None,
    avoid_phrases: list[str] | None = None,
) -> str:
    retry_feedback = retry_feedback or []
    avoid_openers = avoid_openers or []
    avoid_phrases = avoid_phrases or []
    tags = clean_tags(top_meta.get("tag_list") or [])
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    album = top_meta.get("album_name") or "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"

    style_note = [
        "Open with a precise claim about the match, then cite one session detail.",
        "Use compact music-critical language and avoid generic recommender phrasing.",
        "Let the user's latest constraint drive the explanation before describing the sound.",
        "Mention the musical texture in concrete terms, then tie it back to the request.",
        "Use active verbs and avoid stock phrases like 'perfect for' or 'right in'.",
        "Keep the prose lean, specific, and different from a normal catalog blurb.",
        "Anchor the recommendation in the user's accepted or rejected examples.",
        "Favor concrete nouns over broad adjectives such as great, nice, or beautiful.",
        "Make the first sentence self-contained and the second sentence evidence-led.",
        "Use one vivid musical detail, but do not over-describe the track.",
        "Avoid repeating 'you asked for' or 'you described'; paraphrase the need instead.",
        "Keep the explanation conversational but not chatty.",
    ][row_index % 12]

    pieces = [
        "Write exactly one natural-language recommender response for this row.",
        "",
        "Hard requirements:",
        f"- 65-95 words; aim for {PROMPT_WORD_AIM[0]}-{PROMPT_WORD_AIM[1]} words.",
        f"- The first sentence must name the recommendation as {title} by {artist}.",
        "- Mention why it fits using concrete evidence from the conversation.",
        "- Mention 1-2 musical attributes as prose, not as label/value tags.",
        "- Do not end with a question.",
        "- Do not use boilerplate openers such as 'If you're looking for', "
        "'You might enjoy', or 'Here's a track that'.",
        "- Do not use these filler words/phrases except when literally part of "
        "the track title or artist name: vibe, journey, soundscape, captures "
        "the essence.",
        "- Do not output prompt labels, bullets, metadata prefixes, markdown, or quotes "
        "around the whole answer.",
        "- For lexical diversity, avoid crutches like 'perfect for', 'right in', "
        "'lands', 'delivers', 'you asked for', 'you described', 'exactly what', "
        "and 'makes it a' unless there is no cleaner wording.",
        f"- Style note: {style_note}",
        "",
        f"User profile: {compact_profile(item.get('user_profile'))}",
        f"Conversation goal: {compact_goal(item.get('conversation_goal'))}",
        "Conversation:",
        conversation_lines(item, catalog),
        "",
        "Top recommendation metadata:",
        f"Track: {title}",
        f"Artist: {artist}",
        f"Album: {album}",
        f"Release date: {release}",
        f"Tags: {', '.join(tags) if tags else '(none)'}",
    ]
    if avoid_openers:
        pieces.extend([
            "",
            "Avoid starting with any of these already overused first-five-word patterns:",
            "; ".join(avoid_openers),
        ])
    if avoid_phrases:
        pieces.extend([
            "",
            "Lexical diversity repair: avoid reusing these over-common two-word "
            "phrases unless they are literally part of the track or artist name:",
            "; ".join(avoid_phrases),
        ])
    if retry_feedback:
        pieces.extend([
            "",
            "The previous draft failed these checks; fix them in the new draft:",
            "; ".join(retry_feedback),
        ])
    return "\n".join(pieces)


def validate_response_text(resp: str, top_meta: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    clean = normalize_ws(resp)
    wc = word_count(clean)
    if wc < WORD_TARGET[0] or wc > WORD_TARGET[1]:
        failures.append(f"word_count_{wc}_outside_{WORD_TARGET[0]}_{WORD_TARGET[1]}")
    if wc < WORD_TOO_SHORT:
        failures.append("too_short_under_30")
    if not clean:
        failures.append("empty")
    if PREFIX_LEAK_RX.search(clean) or TAG_LINE_RX.search(clean):
        failures.append("prefix_or_tag_leak")
    if TRAILING_QUESTION_RX.search(clean):
        failures.append("trailing_question")
    boiler_hits = boilerplate_hits(clean, top_meta)
    if boiler_hits:
        failures.append("boilerplate_or_filler:" + ",".join(boiler_hits[:3]))

    sent = opening_window(clean)
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    if not contains_text(sent, title):
        failures.append("first_sentence_missing_track")
    artist_parts = [a.strip() for a in artist.split(",") if a.strip()]
    if artist and not any(contains_text(sent, a) for a in artist_parts):
        failures.append("first_sentence_missing_artist")
    return failures


def call_opus(
    client: Any,
    system: str,
    user_prompt: str,
    usage: UsageTotals,
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
    return normalize_ws("".join(getattr(part, "text", "") for part in message.content))


def confirm_model(client: Any, usage: UsageTotals, model: str) -> None:
    message = client.messages.create(
        model=model,
        max_tokens=8,
        system="Reply with exactly OK.",
        messages=[{"role": "user", "content": "ping"}],
    )
    usage.add(getattr(message, "usage", None))
    text = normalize_ws("".join(getattr(part, "text", "") for part in message.content))
    if "OK" not in text.upper():
        raise RuntimeError(f"Unexpected model check response from {model!r}: {text!r}")


def generate_response(
    client: Any,
    item: dict[str, Any],
    top_meta: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    row_index: int,
    usage: UsageTotals,
    model: str,
    avoid_openers: list[str] | None = None,
    avoid_phrases: list[str] | None = None,
) -> tuple[str, int, list[str]]:
    system = (
        "You write concise, personalized music recommendation responses for a "
        "Blind-A submission. Follow every formatting constraint exactly. Output only "
        "the response text, with no labels or analysis."
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
            avoid_openers=avoid_openers,
            avoid_phrases=avoid_phrases,
        )
        last = call_opus(client, system, prompt, usage, model=model)
        failures = validate_response_text(last, top_meta)
        if not failures:
            return last, attempt, []
        time.sleep(0.4)
    return last, MAX_ATTEMPTS, failures


def row_track_hash(row: dict[str, Any]) -> str:
    return sha256("\n".join(row["predicted_track_ids"]).encode("utf-8")).hexdigest()


def distinct2(rows: list[dict[str, Any]]) -> float:
    bigrams = []
    for row in rows:
        toks = re.findall(r"\b[\w'-]+\b", row["predicted_response"].lower())
        bigrams.extend(zip(toks, toks[1:]))
    return len(set(bigrams)) / max(len(bigrams), 1)


def response_bigrams(resp: str) -> list[tuple[str, str]]:
    toks = re.findall(r"\b[\w'-]+\b", resp.lower())
    return list(zip(toks, toks[1:]))


def common_bigram_phrases(rows: list[dict[str, Any]], limit: int = 30) -> list[str]:
    counts = Counter()
    for row in rows:
        counts.update(response_bigrams(row["predicted_response"]))
    skip = {
        "by", "and", "the", "a", "an", "of", "to", "in", "on", "for", "with",
        "from", "it", "this", "that",
    }
    phrases = []
    for (a, b), count in counts.most_common():
        if count < 3:
            break
        if a == "by" or b == "by":
            continue
        if a in skip and b in skip:
            continue
        phrase = f"{a} {b}"
        phrases.append(phrase)
        if len(phrases) >= limit:
            break
    return phrases


def lexdiv_repair_indices(rows: list[dict[str, Any]], n_rows: int = 20) -> list[int]:
    counts = Counter()
    per_row = []
    for row in rows:
        bigrams = response_bigrams(row["predicted_response"])
        per_row.append(bigrams)
        counts.update(bigrams)
    scored = []
    for idx, bigrams in enumerate(per_row):
        # Unique row bigrams prevent long repeated phrases in a single response
        # from dominating the repair target.
        score = sum(max(counts[bg] - 1, 0) for bg in set(bigrams))
        scored.append((score, idx))
    return [idx for score, idx in sorted(scored, reverse=True)[:n_rows] if score > 0]


def opener_clusters(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cnt = Counter()
    for row in rows:
        words = normalize_for_cluster(row["predicted_response"]).split()[:5]
        opener = " ".join(words)
        cnt[opener] += 1
    return [
        {"opener": opener, "count": count}
        for opener, count in cnt.most_common()
        if count > 1
    ]


def validate_submission(
    rows: list[dict[str, Any]],
    r54c_rows: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    by_sid_r54c = {row["session_id"]: row for row in r54c_rows}
    sids = [row.get("session_id") for row in rows]
    duplicate_session_count = len(sids) - len(set(sids))

    track_rows_wrong_len = 0
    rows_with_duplicate_tracks = 0
    invalid_uuid_count = 0
    invalid_catalog_count = 0
    prefix_leak_count = 0
    trailing_question_count = 0
    boilerplate_count = 0
    empty_or_too_short_count = 0
    target_word_range_violations = 0
    first_sentence_name_violations = 0
    per_position_mismatches = 0
    rows_with_mismatched_sequence = 0
    rows_with_matching_sequence = 0
    track_hash_rows = []

    for row in rows:
        sid = row["session_id"]
        tracks = row.get("predicted_track_ids") or []
        resp = row.get("predicted_response") or ""
        if len(tracks) != 20:
            track_rows_wrong_len += 1
        if len(set(tracks)) != len(tracks):
            rows_with_duplicate_tracks += 1
        invalid_uuid_count += sum(1 for tid in tracks if not UUID_RX.match(str(tid)))
        invalid_catalog_count += sum(1 for tid in tracks if str(tid) not in catalog)

        r54c = by_sid_r54c.get(sid)
        r63_hash = row_track_hash(row)
        r54c_hash = row_track_hash(r54c) if r54c else None
        seq_match = bool(r54c and tracks == r54c["predicted_track_ids"])
        if seq_match:
            rows_with_matching_sequence += 1
        else:
            rows_with_mismatched_sequence += 1
            if r54c:
                per_position_mismatches += sum(
                    a != b for a, b in zip(tracks, r54c["predicted_track_ids"])
                )
                per_position_mismatches += abs(len(tracks) - len(r54c["predicted_track_ids"]))
        track_hash_rows.append({
            "session_id": sid,
            "r54c_sha256": r54c_hash,
            "r63_sha256": r63_hash,
            "matching": seq_match,
        })

        clean = normalize_ws(resp)
        if PREFIX_LEAK_RX.search(clean) or TAG_LINE_RX.search(clean):
            prefix_leak_count += 1
        if TRAILING_QUESTION_RX.search(clean):
            trailing_question_count += 1
        top_meta = catalog.get(tracks[0]) if tracks else None
        if boilerplate_hits(clean, top_meta):
            boilerplate_count += 1
        wc = word_count(clean)
        if not clean or wc < WORD_TOO_SHORT:
            empty_or_too_short_count += 1
        if wc < WORD_TARGET[0] or wc > WORD_TARGET[1]:
            target_word_range_violations += 1
        if tracks and tracks[0] in catalog:
            if validate_response_text(clean, catalog[tracks[0]]):
                # Only count first sentence name violations here; other text checks
                # are already represented by the hard gate counters above.
                fs_failures = [
                    f for f in validate_response_text(clean, catalog[tracks[0]])
                    if f.startswith("first_sentence_missing")
                ]
                if fs_failures:
                    first_sentence_name_violations += 1

    lexdiv = distinct2(rows)
    clusters = opener_clusters(rows)
    opener_max = max([c["count"] for c in clusters], default=1)

    gates = {
        "rows_80": len(rows) == 80,
        "unique_sessions_80": len(set(sids)) == 80 and duplicate_session_count == 0,
        "tracks_20_each": track_rows_wrong_len == 0,
        "total_tracks_1600": sum(len(row.get("predicted_track_ids") or []) for row in rows) == 1600,
        "no_duplicate_tracks_within_row": rows_with_duplicate_tracks == 0,
        "valid_uuid_track_ids": invalid_uuid_count == 0,
        "track_ids_exactly_equal_to_r54c_per_position": (
            rows_with_matching_sequence == 80
            and rows_with_mismatched_sequence == 0
            and per_position_mismatches == 0
        ),
        "prefix_leak_count_0": prefix_leak_count == 0,
        "trailing_question_count_0": trailing_question_count == 0,
        "boilerplate_count_0": boilerplate_count == 0,
        "empty_or_too_short_count_0": empty_or_too_short_count == 0,
        "target_word_range_65_95": target_word_range_violations == 0,
        "first_sentence_names_top1_track_and_artist": first_sentence_name_violations == 0,
        "lexdiv_floor_0_83": lexdiv >= LEXDIV_FLOOR,
        "lexdiv_target_r54c_0_8381": lexdiv >= R54C_LEXDIV_TARGET,
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
            "empty_or_too_short_count": empty_or_too_short_count,
            "target_word_range_violations": target_word_range_violations,
            "first_sentence_name_violations": first_sentence_name_violations,
        },
        "lexdiv_distinct2": lexdiv,
        "repeated_opener_clusters": clusters,
        "opener_max_cluster": opener_max,
        "track_hash_comparison": {
            "rows_compared": len(rows),
            "rows_with_matching_track_sequence": rows_with_matching_sequence,
            "rows_with_mismatched_track_sequence": rows_with_mismatched_sequence,
            "total_tracks_compared": sum(len(row.get("predicted_track_ids") or []) for row in rows),
            "per_position_mismatches": per_position_mismatches,
            "per_row": track_hash_rows,
        },
    }


def abort_on_track_mismatch(validation: dict[str, Any]) -> None:
    comp = validation["track_hash_comparison"]
    if (
        comp["rows_with_mismatched_track_sequence"] != 0
        or comp["per_position_mismatches"] != 0
        or comp["rows_with_matching_track_sequence"] != 80
    ):
        raise RuntimeError(
            "ABORT: R63 track sequence differs from R54c. "
            f"rows_match={comp['rows_with_matching_track_sequence']} "
            f"rows_mismatch={comp['rows_with_mismatched_track_sequence']} "
            f"per_position_mismatches={comp['per_position_mismatches']}"
        )


def write_zip_from_persisted_rows() -> list[dict[str, Any]]:
    rows = load_persisted_rows()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rows, indent=2, ensure_ascii=False)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload)
    return rows


def sample_comparisons(
    r54c_rows: list[dict[str, Any]],
    r63_rows: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    r63_by_sid = {row["session_id"]: row for row in r63_rows}
    samples = []
    for row in r54c_rows[:5]:
        sid = row["session_id"]
        top_id = row["predicted_track_ids"][0]
        meta = catalog.get(top_id, {})
        samples.append({
            "session_id": sid,
            "top1_track_id": top_id,
            "top1_track": meta.get("track_name", ""),
            "top1_artist": meta.get("artist_name", ""),
            "r54c_response": row["predicted_response"],
            "r63_response": r63_by_sid[sid]["predicted_response"],
        })
    return samples


def write_metadata(
    rows: list[dict[str, Any]],
    r54c_rows: list[dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
    validation: dict[str, Any],
    usage: UsageTotals,
    generation_report: dict[str, Any],
    model: str,
    prior_aborted_usage: dict[str, Any] | None,
) -> None:
    current_usage = usage.as_dict()
    cumulative_usage = combine_usage_dicts(current_usage, prior_aborted_usage)
    generation_timestamp = datetime.now().isoformat()
    lexdiv = validation["lexdiv_distinct2"]
    metadata = {
        "submission_label": (
            "R63 response-only Opus exploratory | tracks=R54c | "
            f"LexDiv={lexdiv:.4f} | purpose=test blind Gemini LLM judge headroom"
        ),
        "created_at": generation_timestamp,
        "generation_timestamp": generation_timestamp,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "source_tracks": {
            "artifact": str(R54C_ZIP.relative_to(REPO)),
            "constraint": "R63 predicted_track_ids copied from R54c, unchanged.",
        },
        "persisted_full_rows_file": str(PERSISTED_ROWS.relative_to(REPO)),
        "lexdiv_distinct2": lexdiv,
        "lexdiv_below_floor": lexdiv < LEXDIV_FLOOR,
        "generation": generation_report,
        "usage": current_usage,
        "prior_aborted_run_usage": prior_aborted_usage,
        "cumulative_usage_including_prior_aborted_runs": cumulative_usage,
        "validation_results": gate_statuses(validation),
        "r54c_track_hash_match": validation["track_hash_comparison"],
        "validation": validation,
        "before_after_samples": sample_comparisons(r54c_rows, rows, catalog),
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


def gate_statuses(validation: dict[str, Any]) -> dict[str, Any]:
    results = {}
    for name, passed in validation["gates"].items():
        if passed:
            status = "pass"
        elif name in WARNING_GATE_NAMES:
            status = "warn"
        else:
            status = "fail"
        results[name] = {
            "status": status,
            "passed": bool(passed),
            "blocking": name in STRUCTURAL_GATE_NAMES,
        }
    return {
        "all_gates": results,
        "structural_gates_pass": hard_gates_pass(validation),
        "lexdiv_gate_policy": "warn_non_blocking",
        "opener_cluster_policy": (
            "warn_non_blocking_after_one_targeted_regeneration_if_needed"
        ),
    }


def gate_table(validation: dict[str, Any]) -> str:
    lines = ["| Gate | Result |", "|---|---|"]
    for name, passed in validation["gates"].items():
        if passed:
            result = "PASS"
        elif name in WARNING_GATE_NAMES:
            result = "WARN"
        else:
            result = "FAIL"
        lines.append(f"| `{name}` | {result} |")
    return "\n".join(lines)


def write_result_doc(
    validation: dict[str, Any],
    usage: UsageTotals,
    samples: list[dict[str, Any]],
    model: str,
    prior_aborted_usage: dict[str, Any] | None,
) -> None:
    comp = validation["track_hash_comparison"]
    current_usage = usage.as_dict()
    cumulative_usage = combine_usage_dicts(current_usage, prior_aborted_usage)
    sample_lines = []
    for sample in samples:
        sample_lines.extend([
            f"### {sample['top1_track']} by {sample['top1_artist']}",
            f"- session_id: `{sample['session_id']}`",
            f"- R54c: {sample['r54c_response']}",
            f"- R63: {sample['r63_response']}",
            "",
        ])
    doc = f"""# R63 E1 Response-Only Result

## Gate Table
{gate_table(validation)}

## Track Hash Comparison (R54c vs R63)
```text
Track Hash Comparison (R54c vs R63):
  rows compared: {comp['rows_compared']}
  rows with matching track sequence: {comp['rows_with_matching_track_sequence']}
  rows with mismatched track sequence: {comp['rows_with_mismatched_track_sequence']}
  total tracks compared: {comp['total_tracks_compared']}
  per-position mismatches: {comp['per_position_mismatches']}
```

## Summary
- Model used: `{model}`
- Submission artifact: `{OUT_ZIP.relative_to(REPO)}`
- Metadata: `{OUT_METADATA.relative_to(REPO)}`
- Persisted full rows: `{PERSISTED_ROWS.relative_to(REPO)}`
- LexDiv (Distinct-2): {validation['lexdiv_distinct2']:.4f}
- LexDiv below 0.83 floor: {'YES' if validation['lexdiv_distinct2'] < LEXDIV_FLOOR else 'NO'} (warning only)
- Max repeated opener cluster: {validation['opener_max_cluster']}
- Opus API calls for successful artifact run: {usage.calls}
- Total Opus API calls including aborted validation run(s): {cumulative_usage['calls']}
- Tokens for successful artifact run: input={usage.input_tokens}, output={usage.output_tokens}
- Total cost estimate including aborted validation run(s): ${cumulative_usage['estimated_cost_usd']:.4f} using ${OPUS_47_INPUT_USD_PER_MTOK:.2f}/MTok input and ${OPUS_47_OUTPUT_USD_PER_MTOK:.2f}/MTok output from {PRICING_SOURCE}
- Ready to submit manually to Codabench: {'YES' if hard_gates_pass(validation) else 'NO'} (structural gates only)

## Repeated Opener Clusters
```json
{json.dumps(validation['repeated_opener_clusters'], indent=2, ensure_ascii=False)}
```

## Sample Comparisons
{''.join(sample_lines)}
"""
    OUT_DOC.write_text(doc, encoding="utf-8")


def hard_gates_pass(validation: dict[str, Any]) -> bool:
    return all(validation["gates"].get(name, False) for name in STRUCTURAL_GATE_NAMES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    if args.model != MODEL_ID:
        raise SystemExit(
            f"Refusing model override {args.model!r}; R63 E1 requires {MODEL_ID!r}."
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY. Pause.")

    import anthropic

    print(f"{ts()} R63 E1 response-only generation")
    print(f"{ts()} Loading R54c production tracks, Blind-A rows, and catalog")
    prior_aborted_usage = load_prior_aborted_usage()
    if prior_aborted_usage:
        print(
            f"{ts()} Found prior aborted R63 usage: "
            f"calls={prior_aborted_usage.get('calls')}"
        )
    r54c_rows = load_r54c_rows()
    blind_by_sid = load_blind_by_sid()
    catalog = load_catalog()
    client = anthropic.Anthropic(api_key=api_key)
    usage = UsageTotals()

    print(f"{ts()} Confirming Anthropic model availability: {args.model}")
    confirm_model(client, usage, args.model)
    print(f"{ts()} Model check passed")

    reset_persisted_rows()
    print(f"{ts()} Persisting rows as they complete: {PERSISTED_ROWS}")
    rows: list[dict[str, Any]] = []
    generation_attempts = []
    failures_after_retries = []
    generation_started_at = datetime.now().isoformat()
    start = time.time()
    for i, base in enumerate(r54c_rows):
        sid = base["session_id"]
        top_id = base["predicted_track_ids"][0]
        if sid not in blind_by_sid:
            raise RuntimeError(f"Missing Blind-A session_id={sid}")
        if top_id not in catalog:
            raise RuntimeError(f"Missing catalog metadata for top_id={top_id}")
        resp, attempts, failures = generate_response(
            client,
            blind_by_sid[sid],
            catalog[top_id],
            catalog,
            i,
            usage,
            model=args.model,
        )
        if failures:
            failures_after_retries.append({
                "session_id": sid,
                "top1_id": top_id,
                "failures": failures,
                "response": resp,
            })
        rows.append({
            "session_id": sid,
            "turn_number": base["turn_number"],
            "predicted_track_ids": list(base["predicted_track_ids"]),
            "predicted_response": resp,
        })
        write_persisted_rows(rows)
        generation_attempts.append({"session_id": sid, "attempts": attempts})
        if (i + 1) % 5 == 0:
            elapsed = time.time() - start
            print(
                f"{ts()} Generated {i + 1}/80 responses; "
                f"api_calls={usage.calls}; elapsed={elapsed:.1f}s",
                flush=True,
            )

    if failures_after_retries:
        print(f"{ts()} Initial generation left {len(failures_after_retries)} failed rows")

    print(f"{ts()} Re-reading persisted rows before validation/packaging")
    rows = load_persisted_rows()
    validation = validate_submission(rows, r54c_rows, catalog)
    abort_on_track_mismatch(validation)

    oversized = [
        c for c in validation["repeated_opener_clusters"]
        if c["count"] > 5
    ]
    opener_repair_report: dict[str, Any] = {
        "policy": "one targeted regeneration pass only; ship even if still >5",
        "needed": bool(oversized),
        "before_max_cluster": validation["opener_max_cluster"],
        "before_oversized_clusters": oversized,
        "rows_regenerated": 0,
        "candidate_row_indices": [],
        "after_max_cluster": validation["opener_max_cluster"],
        "after_oversized_clusters": [],
        "shipped_with_remaining_oversized_cluster": False,
    }
    if oversized:
        print(f"{ts()} Repairing repeated opener buckets once: {oversized}")
        avoid = [c["opener"] for c in validation["repeated_opener_clusters"]]
        by_opener: dict[str, list[int]] = {}
        for idx, row in enumerate(rows):
            opener = " ".join(normalize_for_cluster(row["predicted_response"]).split()[:5])
            by_opener.setdefault(opener, []).append(idx)
        to_repair = []
        for cluster in oversized:
            to_repair.extend(by_opener[cluster["opener"]][5:])
        opener_repair_report["candidate_row_indices"] = sorted(set(to_repair))
        for idx in sorted(set(to_repair)):
            base = r54c_rows[idx]
            sid = base["session_id"]
            top_id = base["predicted_track_ids"][0]
            resp, attempts, failures = generate_response(
                client,
                blind_by_sid[sid],
                catalog[top_id],
                catalog,
                idx,
                usage,
                model=args.model,
                avoid_openers=avoid,
            )
            if failures:
                failures_after_retries.append({
                    "session_id": sid,
                    "top1_id": top_id,
                    "failures": failures,
                    "response": resp,
                    "opener_cluster_repair": True,
                })
            rows[idx]["predicted_response"] = resp
            write_persisted_rows(rows)
            opener_repair_report["rows_regenerated"] += 1
            generation_attempts.append({
                "session_id": sid,
                "attempts": attempts,
                "opener_cluster_repair": True,
            })
        rows = load_persisted_rows()
        validation = validate_submission(rows, r54c_rows, catalog)
        abort_on_track_mismatch(validation)
        remaining_oversized = [
            c for c in validation["repeated_opener_clusters"]
            if c["count"] > 5
        ]
        opener_repair_report["after_max_cluster"] = validation["opener_max_cluster"]
        opener_repair_report["after_oversized_clusters"] = remaining_oversized
        opener_repair_report["shipped_with_remaining_oversized_cluster"] = bool(
            remaining_oversized
        )
        if remaining_oversized:
            print(
                f"{ts()} WARNING: opener cluster still >5 after one targeted pass; "
                f"shipping per run policy: {remaining_oversized}",
                flush=True,
            )

    print(f"{ts()} Final packaging re-read from persisted rows")
    rows = load_persisted_rows()
    validation = validate_submission(rows, r54c_rows, catalog)
    abort_on_track_mismatch(validation)
    if validation["lexdiv_distinct2"] < LEXDIV_FLOOR:
        print(
            f"{ts()} WARNING: LexDiv {validation['lexdiv_distinct2']:.4f} "
            f"is below {LEXDIV_FLOOR:.2f}; non-blocking for this exploratory run",
            flush=True,
        )

    generation_report = {
        "generation_started_at": generation_started_at,
        "generation_completed_at": datetime.now().isoformat(),
        "initial_responses_generated": len(r54c_rows),
        "responses_generated_including_opener_repair": len(generation_attempts),
        "failures_after_retries": failures_after_retries,
        "attempts": generation_attempts,
        "max_attempts_per_generation": MAX_ATTEMPTS,
        "word_target": list(WORD_TARGET),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "persisted_rows_file": str(PERSISTED_ROWS.relative_to(REPO)),
        "rows_persisted_immediately_after_generation": True,
        "packaging_source": "persisted_rows_file_only",
        "lexdiv_repair_rounds": [],
        "lexdiv_repair_policy": "disabled for packaging-only re-run",
        "opener_cluster_repair": opener_repair_report,
    }
    write_metadata(
        rows,
        r54c_rows,
        catalog,
        validation,
        usage,
        generation_report,
        args.model,
        prior_aborted_usage,
    )
    write_result_doc(
        validation,
        usage,
        sample_comparisons(r54c_rows, rows, catalog),
        args.model,
        prior_aborted_usage,
    )

    if not hard_gates_pass(validation):
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        raise SystemExit("Structural validation failed. Zip was not written.")

    write_zip_from_persisted_rows()
    print(f"{ts()} Wrote {OUT_ZIP}")
    print(f"{ts()} Wrote {OUT_METADATA}")
    print(f"{ts()} Wrote {OUT_DOC}")
    print(
        "R63 E1 DONE: "
        f"model={args.model}, "
        f"LexDiv={validation['lexdiv_distinct2']:.4f}, "
        "track_match="
        f"{validation['track_hash_comparison']['rows_with_matching_track_sequence']}/80, "
        f"opener_max_cluster={validation['opener_max_cluster']}, "
        f"ready_to_submit={'YES' if hard_gates_pass(validation) else 'NO'}"
    )


if __name__ == "__main__":
    main()
