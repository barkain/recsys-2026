#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R63c repair: one-shot LexDiv recovery pass over the 15 R63c rows only.

R63b remains production unless this repair clears the LexDiv packaging gate.
Tracks are copied from the current R63c rows and validated bitwise against R54c.
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
R63_METADATA = OUT_DIR / "r63_response_only_submission.metadata.json"
R63B_ROWS = OUT_DIR / "r63b_rows_final.jsonl"
R63B_METADATA = OUT_DIR / "r63b_targeted_polish_submission.metadata.json"
R63C_ROWS = OUT_DIR / "r63c_rows_final.jsonl"
R63C_METADATA = OUT_DIR / "r63c_targeted_polish_submission.metadata.json"
PERSISTED_ROWS = OUT_DIR / "r63c_repair_rows_persisted.jsonl"
FINAL_ROWS = OUT_DIR / "r63c_repair_rows_final.jsonl"
OUT_ZIP = OUT_DIR / "r63c_repair_polish_submission.zip"
OUT_METADATA = OUT_DIR / "r63c_repair_polish_submission.metadata.json"
OUT_DOC = REPO / "docs" / "r63c_repair_polish_result.md"

MODEL_ID = "claude-opus-4-7"
PROMPT_VERSION = (
    "R63c repair v1; one attempt over same 15 R63c rows; "
    "diversified sentence architecture for LexDiv recovery"
)
MAX_TOKENS = 340
REGEN_WORD_TARGET = (75, 95)
BASE_WORD_TARGET = (65, 95)
LEXDIV_PASS_FLOOR = 0.8255
LEXDIV_BORDERLINE_FLOOR = 0.8240

ARCHITECTURES = [
    {
        "name": "contrast sentence",
        "instruction": (
            "Open with a contrast: unlike a prior track, artist, or direction from "
            "the session, this recommendation leans into the more relevant trait."
        ),
        "example": 'Unlike X, this leans into Y...',
    },
    {
        "name": "genre-first sentence",
        "instruction": (
            "Open with the genre, style, or era as the subject, then name the "
            "track and artist inside that first sentence."
        ),
        "example": 'Synthwave at its most reflective...',
    },
    {
        "name": "memory-reference sentence",
        "instruction": (
            "Open by picking up a remembered track, artist, artwork clue, or user "
            "preference from the session, then connect it to this track."
        ),
        "example": 'If you remembered Y, this picks up where that left off...',
    },
    {
        "name": "artist-context sentence",
        "instruction": (
            "Open with the artist, album, or composer context and the concrete "
            "element the track is built around."
        ),
        "example": 'X built this around Y...',
    },
    {
        "name": "energy/mood sentence",
        "instruction": (
            "Open with the track's energy, pacing, or mood, then name the track "
            "and artist before the sentence ends."
        ),
        "example": 'A patient burn from start to finish...',
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


def selected_from_r63c_metadata(
    r63c_metadata: dict[str, Any],
    r63c_rows: list[dict[str, Any]],
    blind_by_sid: dict[str, dict[str, Any]],
    catalog: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    selected_indices = [
        int(idx) for idx in r63c_metadata["selection"]["selected_row_indices"]
    ]
    audit_by_idx = {
        int(item["row_index"]): item
        for item in r63c.weakness_audit(r63c_rows, blind_by_sid, catalog)
    }
    selected = []
    for order, idx in enumerate(selected_indices):
        item = copy.deepcopy(audit_by_idx[idx])
        item["repair_order"] = order
        item["repair_architecture"] = ARCHITECTURES[order % len(ARCHITECTURES)]["name"]
        selected.append(item)
    return selected


def render_prompt(
    item: dict[str, Any],
    top_meta: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    row_index: int,
    r63c_response: str,
    architecture: dict[str, str],
) -> str:
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    album = top_meta.get("album_name") or "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"
    tags = r63.clean_tags(top_meta.get("tag_list") or [])

    return "\n".join([
        "Write exactly one replacement recommender response for this Blind-A row.",
        "",
        "Hard requirements:",
        f"- {REGEN_WORD_TARGET[0]}-{REGEN_WORD_TARGET[1]} words.",
        f"- The first sentence must name the recommendation as {title} by {artist}.",
        "- Do not make every row begin with 'Track by Artist'; vary the sentence shape.",
        "- Name 2-3 concrete musical attributes in prose: genre, era, a signature "
        "production element, lyrical theme, or specific instrument.",
        "- Reference at least two specific session details, such as a stated user "
        "preference, previous track, previous artist, artwork clue, or conversation context.",
        "- Justify why this exact top-1 track is the best pick with concrete session evidence.",
        "",
        "Sentence architecture palette requested for this repair:",
        '* contrast sentence (e.g., "Unlike X, this leans into Y...")',
        '* genre-first sentence (e.g., "Synthwave at its most reflective...")',
        '* memory-reference sentence (e.g., "If you remembered Y, this picks up where that left off...")',
        '* artist-context sentence (e.g., "X built this around Y...")',
        '* energy/mood sentence (e.g., "A patient burn from start to finish...")',
        "",
        f"For this row, use the {architecture['name']} pattern: {architecture['instruction']}",
        f"Do not copy the example wording: {architecture['example']}",
        "",
        "Anti-boilerplate rules:",
        "- Do not end with a question.",
        "- Do not use boilerplate openers such as 'If you're looking for', "
        "'You might enjoy', or 'Here's a track that'.",
        "- Do not use these filler words/phrases except when literally part of "
        "the track title or artist name: vibe, journey, soundscape, captures the essence.",
        "- Do not output prompt labels, bullets, metadata prefixes, markdown, or quotes around the whole answer.",
        "- Avoid crutches like 'perfect for', 'right in', 'lands', 'delivers', "
        "'you asked for', 'you described', 'exactly what', and 'makes it a'.",
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
        "Current R63c response being repaired. Keep true specifics, but change "
        "the sentence architecture and avoid preserving repeated connective "
        "phrasing verbatim:",
        r63c_response,
    ])


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
    r63c_response: str,
    architecture: dict[str, str],
    usage: r63.UsageTotals,
    model: str,
) -> tuple[str, list[str]]:
    system = (
        "You write concise, personalized music recommendation responses. "
        "Follow every constraint exactly. Output only the response text."
    )
    prompt = render_prompt(
        item,
        top_meta,
        catalog,
        row_index,
        r63c_response,
        architecture,
    )
    response = call_opus(client, system, prompt, usage, model=model)
    failures = r63c.validate_regenerated_text(response, top_meta)
    return response, failures


def gate_status(validation: dict[str, Any], regen_validation: dict[str, Any]) -> str:
    nonblocking = {"lexdiv_floor_0_83", "lexdiv_target_r54c_0_8381"}
    structural_ok = all(
        passed
        for name, passed in validation["gates"].items()
        if name not in nonblocking
    )
    structural_ok = structural_ok and bool(regen_validation["passed"])
    if not structural_ok:
        return "FAIL"
    lexdiv = float(validation["lexdiv_distinct2"])
    if lexdiv >= LEXDIV_PASS_FLOOR:
        return "PASS"
    if lexdiv >= LEXDIV_BORDERLINE_FLOOR:
        return "BORDERLINE"
    return "FAIL"


def ready_label(gate: str) -> str:
    if gate == "PASS":
        return "YES"
    if gate == "BORDERLINE":
        return "BORDERLINE"
    return "NO_KEEP_R63B"


def validation_statuses(
    validation: dict[str, Any],
    regen_validation: dict[str, Any],
    gate: str,
) -> dict[str, Any]:
    statuses = r63c.gate_statuses(validation, regen_validation)
    lexdiv = float(validation["lexdiv_distinct2"])
    statuses["repair_lexdiv_pass_floor_0_8255"] = {
        "status": "pass" if lexdiv >= LEXDIV_PASS_FLOOR else "warn",
        "passed": lexdiv >= LEXDIV_PASS_FLOOR,
        "blocking": gate == "FAIL",
    }
    statuses["repair_lexdiv_borderline_floor_0_8240"] = {
        "status": "pass" if lexdiv >= LEXDIV_BORDERLINE_FLOOR else "fail",
        "passed": lexdiv >= LEXDIV_BORDERLINE_FLOOR,
        "blocking": True,
    }
    return statuses


def selected_samples(
    selected: list[dict[str, Any]],
    r63c_rows: list[dict[str, Any]],
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
            "repair_architecture": entry["repair_architecture"],
            "r63c_response": r63c_rows[idx]["predicted_response"],
            "repair_response": final_rows[idx]["predicted_response"],
        })
    return samples


def write_metadata(
    final_rows: list[dict[str, Any]],
    r63c_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    generation_failures: list[dict[str, Any]],
    validation: dict[str, Any],
    regen_validation: dict[str, Any],
    usage: r63.UsageTotals,
    gate: str,
    model: str,
    started_at: str,
    completed_at: str,
    r63c_metadata: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    lexdiv = float(validation["lexdiv_distinct2"])
    current_usage = usage.as_dict()
    cumulative_usage = r63.combine_usage_dicts(
        current_usage,
        r63c_metadata.get("cumulative_usage") or r63c_metadata.get("usage"),
    )
    label = (
        "R63c-repair targeted response polish | base=R63b | tracks=R54c | "
        "15 rows regenerated with diversified sentence architecture | "
        f"LexDiv={lexdiv:.4f} | "
        "purpose=push LLM judge 4.85 \u2192 4.90+ with preserved LexDiv"
    )
    metadata = {
        "submission_label": label,
        "created_at": completed_at,
        "generation_timestamp": completed_at,
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "base": {
            "rows_file": str(R63C_ROWS.relative_to(REPO)),
            "metadata_file": str(R63C_METADATA.relative_to(REPO)),
            "r63b_rows_file": str(R63B_ROWS.relative_to(REPO)),
            "r63b_metadata_file": str(R63B_METADATA.relative_to(REPO)),
            "note": "R63b remains production unless this repair clears the LexDiv gate.",
        },
        "source_tracks": {
            "artifact": str(r63.R54C_ZIP.relative_to(REPO)),
            "constraint": "R63c-repair predicted_track_ids copied from R63c/R54c, unchanged.",
        },
        "persisted_rows_file": str(PERSISTED_ROWS.relative_to(REPO)),
        "final_rows_file": str(FINAL_ROWS.relative_to(REPO)),
        "submission_zip": str(OUT_ZIP.relative_to(REPO)),
        "lexdiv_distinct2": lexdiv,
        "repair_gate": gate,
        "ready_to_submit": ready_label(gate),
        "lexdiv_policy": {
            "pass_floor": LEXDIV_PASS_FLOOR,
            "borderline_floor": LEXDIV_BORDERLINE_FLOOR,
            "hard_stop_below": LEXDIV_BORDERLINE_FLOOR,
            "r63b_local_reference": 0.8260,
            "r63c_local_before_repair": r63c_metadata.get("lexdiv_distinct2"),
        },
        "selection": {
            "policy": "Same 15 rows regenerated in R63c; other 65 rows kept bitwise at row-response level.",
            "selected_row_indices": [int(item["row_index"]) for item in selected],
            "selected_rows": selected,
            "architecture_assignments": [
                {
                    "row_index": int(item["row_index"]),
                    "architecture": item["repair_architecture"],
                }
                for item in selected
            ],
        },
        "generation": {
            "started_at": started_at,
            "completed_at": completed_at,
            "selected_rows_attempted": len(selected),
            "accepted_regenerated_rows": sum(
                1 for item in attempts if item["accepted_regenerated_response"]
            ),
            "fallback_to_r63c_rows": sum(
                1 for item in attempts if not item["accepted_regenerated_response"]
            ),
            "non_selected_rows_kept_from_r63c": 80 - len(selected),
            "max_attempts_per_selected_row": 1,
            "word_target_regenerated_rows": list(REGEN_WORD_TARGET),
            "word_target_fallback_rows": list(BASE_WORD_TARGET),
            "generation_failures": generation_failures,
            "attempts": attempts,
            "rows_persisted_immediately_after_each_targeted_row": True,
            "packaging_source": str(FINAL_ROWS.relative_to(REPO)),
            "run_policy": "one repair generation pass; no second repair pass",
        },
        "usage": current_usage,
        "cumulative_usage": cumulative_usage,
        "r63c_usage_reference": r63c_metadata.get("usage"),
        "r63c_cumulative_usage_reference": r63c_metadata.get("cumulative_usage"),
        "validation_results": validation_statuses(validation, regen_validation, gate),
        "r54c_track_hash_match": validation["track_hash_comparison"],
        "validation": validation,
        "regenerated_row_validation": regen_validation,
        "before_after_samples": selected_samples(selected, r63c_rows, final_rows, catalog),
    }
    OUT_METADATA.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def gate_table(statuses: dict[str, Any]) -> str:
    lines = ["| Gate | Result |", "|---|---|"]
    for name, info in statuses.items():
        lines.append(f"| `{name}` | {str(info['status']).upper()} |")
    return "\n".join(lines)


def write_result_doc(
    metadata: dict[str, Any] | None,
    validation: dict[str, Any],
    regen_validation: dict[str, Any],
    selected: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    generation_failures: list[dict[str, Any]],
    gate: str,
    packaged: bool,
) -> None:
    lexdiv = float(validation["lexdiv_distinct2"])
    ready = ready_label(gate)
    statuses = validation_statuses(validation, regen_validation, gate)
    comp = validation["track_hash_comparison"]

    selected_lines = []
    for item in selected:
        selected_lines.append(
            f"- row `{item['row_index']}` / `{item['session_id']}`: "
            f"score={item['weakness_score']}, wc={item['word_count']}, "
            f"{item['top1_track']} by {item['top1_artist']} "
            f"(architecture={item['repair_architecture']})"
        )

    attempt_lines = []
    for item in attempts:
        attempt_lines.append(
            f"- row `{item['row_index']}`: accepted={item['accepted_regenerated_response']}, "
            f"wc={item['word_count']}, architecture={item['repair_architecture']}"
        )

    sample_lines = []
    for sample in samples:
        sample_lines.extend([
            f"### Row {sample['row_index']}: {sample['top1_track']} by {sample['top1_artist']}",
            f"- session_id: `{sample['session_id']}`",
            f"- weakness score: `{sample['weakness_score']}`",
            f"- architecture: `{sample['repair_architecture']}`",
            f"- R63c: {sample['r63c_response']}",
            f"- R63c-repair: {sample['repair_response']}",
            "",
        ])

    hard_stop_note = ""
    if gate == "FAIL" and lexdiv < LEXDIV_BORDERLINE_FLOOR:
        hard_stop_note = (
            "\n**R63c repair failed LexDiv hard stop, R63b remains production.**\n"
        )
    elif gate == "BORDERLINE":
        hard_stop_note = (
            "\n**Borderline:** packaged with warning; user decides whether to submit.\n"
        )

    label = metadata["submission_label"] if metadata else "(not packaged)"
    usage = metadata["usage"] if metadata else {}
    cumulative = metadata["cumulative_usage"] if metadata else {}

    doc = f"""# R63c Repair Polish Result
{hard_stop_note}
## Gate Table
{gate_table(statuses)}

## Track Hash Comparison (R54c vs R63c-repair)
```text
Track Hash Comparison (R54c vs R63c-repair):
  rows compared: {comp['rows_compared']}
  rows with matching track sequence: {comp['rows_with_matching_track_sequence']}
  rows with mismatched track sequence: {comp['rows_with_mismatched_track_sequence']}
  total tracks compared: {comp['total_tracks_compared']}
  per-position mismatches: {comp['per_position_mismatches']}
```

## Summary
- Submission label: `{label}`
- Model used: `{MODEL_ID}`
- Repair gate: `{gate}`
- Ready to submit: `{ready}`
- Packaged: {'YES' if packaged else 'NO'}
- Submission artifact: `{OUT_ZIP.relative_to(REPO)}`{' (not written)' if not packaged else ''}
- Metadata: `{OUT_METADATA.relative_to(REPO)}`{' (not written)' if not packaged else ''}
- Persisted rows: `{PERSISTED_ROWS.relative_to(REPO)}`
- Final rows: `{FINAL_ROWS.relative_to(REPO)}`
- Result doc: `{OUT_DOC.relative_to(REPO)}`
- Selected repair rows: {len(selected)}
- Accepted regenerated rows: {sum(1 for item in attempts if item['accepted_regenerated_response'])}
- Fallback to R63c rows: {sum(1 for item in attempts if not item['accepted_regenerated_response'])}
- Non-selected rows kept from R63c: {80 - len(selected)}
- LexDiv (Distinct-2, local audit): {lexdiv:.4f}
- LexDiv pass floor: {LEXDIV_PASS_FLOOR:.4f}
- LexDiv borderline floor: {LEXDIV_BORDERLINE_FLOOR:.4f}
- R63c local before repair: 0.8191
- R63b local reference: 0.8260
- Max repeated opener cluster: {validation['opener_max_cluster']}
- Opus API calls for repair run: {usage.get('calls', '(not packaged)')}
- Estimated repair run cost: {('$' + format(usage.get('estimated_cost_usd', 0.0), '.4f')) if usage else '(not packaged)'}
- Cumulative API calls including R63 + R63b + R63c + repair: {cumulative.get('calls', '(not packaged)')}
- Estimated cumulative cost: {('$' + format(cumulative.get('estimated_cost_usd', 0.0), '.4f')) if cumulative else '(not packaged)'}

## Selection Rationale
Same 15 rows that R63c regenerated were repaired. The other 65 rows were not touched.

{chr(10).join(selected_lines)}

## Sentence Architecture Assignments
{chr(10).join(attempt_lines)}

## Generation Failures
```json
{json.dumps(generation_failures, indent=2, ensure_ascii=False)}
```

## Repeated Opener Clusters
```json
{json.dumps(validation['repeated_opener_clusters'], indent=2, ensure_ascii=False)}
```

## Sample Comparisons
{chr(10).join(sample_lines)}
"""
    OUT_DOC.write_text(doc, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    if args.model != MODEL_ID:
        raise SystemExit(
            f"Refusing model override {args.model!r}; R63c repair requires {MODEL_ID!r}."
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY. Pause.")

    import anthropic

    clean_stale_package()
    print(f"{ts()} R63c repair polish")
    print(f"{ts()} Loading R63c rows, R54c tracks, Blind-A sessions, and catalog")
    r63c_rows = r63c.read_jsonl(R63C_ROWS)
    if len(r63c_rows) != 80:
        raise RuntimeError(f"Expected 80 R63c rows, found {len(r63c_rows)}")
    r54c_rows = r63.load_r54c_rows()
    blind_by_sid = r63.load_blind_by_sid()
    catalog = r63.load_catalog()
    r63c_metadata = load_metadata(R63C_METADATA)
    if not r63c_metadata:
        raise RuntimeError(f"Missing or invalid R63c metadata: {R63C_METADATA}")

    selected = selected_from_r63c_metadata(r63c_metadata, r63c_rows, blind_by_sid, catalog)
    selected_indices = {int(item["row_index"]) for item in selected}
    print(
        f"{ts()} Repairing same {len(selected)} R63c rows: "
        f"{[int(item['row_index']) for item in selected]}"
    )

    client = anthropic.Anthropic(api_key=api_key)
    usage = r63.UsageTotals()
    print(f"{ts()} Confirming Anthropic model availability: {args.model}")
    r63.confirm_model(client, usage, args.model)
    print(f"{ts()} Model check passed")

    final_rows = copy.deepcopy(r63c_rows)
    r63c.write_jsonl(PERSISTED_ROWS, final_rows)
    attempts: list[dict[str, Any]] = []
    generation_failures: list[dict[str, Any]] = []
    accepted_indices: set[int] = set()
    started_at = datetime.now().isoformat()
    start = time.time()

    selected_by_idx = {int(item["row_index"]): item for item in selected}
    for n_done, selected_item in enumerate(selected, start=1):
        idx = int(selected_item["row_index"])
        base_row = r63c_rows[idx]
        sid = base_row["session_id"]
        top_id = base_row["predicted_track_ids"][0]
        top_meta = catalog[top_id]
        architecture = ARCHITECTURES[(n_done - 1) % len(ARCHITECTURES)]
        response, failures = generate_one(
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
                "kept_original_r63c_response": True,
            })
        attempts.append({
            "row_index": idx,
            "session_id": sid,
            "weakness_score": selected_by_idx[idx]["weakness_score"],
            "attempts": 1,
            "accepted_regenerated_response": accepted,
            "word_count": r63.word_count(final_rows[idx]["predicted_response"]),
            "top1_track": top_meta.get("track_name", ""),
            "top1_artist": top_meta.get("artist_name", ""),
            "repair_architecture": architecture["name"],
        })
        r63c.write_jsonl(PERSISTED_ROWS, final_rows)
        print(
            f"{ts()} Repair row {n_done}/{len(selected)} idx={idx} "
            f"accepted={accepted} architecture={architecture['name']} api_calls={usage.calls}",
            flush=True,
        )

    print(f"{ts()} Writing final repaired rows from persisted state")
    final_rows = r63c.read_jsonl(PERSISTED_ROWS)
    r63c.write_jsonl(FINAL_ROWS, final_rows)

    print(f"{ts()} Validating repaired payload")
    validation = r63.validate_submission(final_rows, r54c_rows, catalog)
    r63.abort_on_track_mismatch(validation)
    regen_validation = r63c.per_regen_validation(
        final_rows, selected_indices, accepted_indices, catalog
    )
    gate = gate_status(validation, regen_validation)
    completed_at = datetime.now().isoformat()
    packaged = gate in {"PASS", "BORDERLINE"}

    metadata = None
    if packaged:
        metadata = write_metadata(
            final_rows,
            r63c_rows,
            selected,
            attempts,
            generation_failures,
            validation,
            regen_validation,
            usage,
            gate,
            args.model,
            started_at,
            completed_at,
            r63c_metadata,
            catalog,
        )
        write_zip(final_rows)

    samples = selected_samples(selected, r63c_rows, final_rows, catalog)
    write_result_doc(
        metadata,
        validation,
        regen_validation,
        selected,
        samples,
        attempts,
        generation_failures,
        gate,
        packaged,
    )

    elapsed = time.time() - start
    print(f"{ts()} Wrote {PERSISTED_ROWS}")
    print(f"{ts()} Wrote {FINAL_ROWS}")
    if packaged:
        print(f"{ts()} Wrote {OUT_ZIP}")
        print(f"{ts()} Wrote {OUT_METADATA}")
    else:
        print(f"{ts()} LexDiv hard stop; package not written")
    print(f"{ts()} Wrote {OUT_DOC}")
    print(
        "R63c REPAIR DONE: "
        f"LexDiv={validation['lexdiv_distinct2']:.4f}, "
        f"gate={gate}, "
        f"ready_to_submit={ready_label(gate)}, "
        f"elapsed={elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
