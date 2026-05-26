"""R84c Blind-B fresh-response generator.

Generates fresh R78-style responses for every session in an arbitrary blind
split. Used when there's no prior submission (R78/R84c) to reuse responses
from — i.e., Blind-B day.

Why R78-style and not R87-style: R87's evidence-injection variant slightly
regressed LexDiv on R84c base (0.8720 -> 0.8706). R78-style prompting is
the empirically best response generator we have. See
[[project_r87_outcome]] for the data.

Inputs:
- --tracks-json: routed tracks JSON (from expR84c_blind_replay.py)
- --blind-cache: blind source cache (built by expR55_blind_source_cache.py)
- --output-zip: final submission ZIP path

Validation gates:
- track IDs unchanged from input
- non-empty response per case
- word count 60..110
- no banned phrases (R78 inherited list)

Flags:
- --dry-run: identify cases needing regen + print prompts, no API calls
- --limit N: only regen first N cases (for smoke testing)

Run:
  ANTHROPIC_RECSYS_API_KEY=... uv run python scripts/expR84c_blind_b_responses.py \
      --tracks-json exp/inference/blind_b/r84c_replay_blind_b_track_lists.json \
      --blind-cache cache/blind_b/source_cache.pkl \
      --output-zip exp/inference/blind_b/r84c_blind_b_submission.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL_ID = "claude-opus-4-7"
MAX_TOKENS = 280
WORD_MIN = 60
WORD_MAX = 110

# R78-style banned phrases — proven set across R78/R84c/R86 sprints
VAGUE_RE = re.compile(
    r"\b(warm|vibrant|energetic|cozy|beautiful|gorgeous|gentle|charming|"
    r"memorable|delightful|lovely)\b", re.IGNORECASE,
)
IMPERATIVE_RE = re.compile(
    r"crank it loud\b|press play\b|lace up\b|hit play\b|"
    r"queue this up\b|give it a spin\b|throw it on\b", re.IGNORECASE,
)
BANNED_INHERITED = [
    "comes off", "fits that", "carries that", "captures the", "captures that",
    "leans into", "matches the",
    "you're looking", "you're chasing", "you're trying", "you're after",
    "you described", "you mentioned",
    "perfect for", "exactly what", "makes it a",
    "vibe", "journey", "soundscape",
]


def ts(): return f"[{datetime.now():%H:%M:%S}]"


def head_sha():
    g = shutil.which("git")
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip() if g else "no-git"


def short_ref(tid, meta):
    m = meta.get(tid, {})
    n = m.get("track_name", [])
    a = m.get("artist_name", [])
    name = n[0] if isinstance(n, list) and n else str(n)
    artist = a[0] if isinstance(a, list) and a else str(a)
    return f"{name} by {artist}"


def clean_tags(tags):
    if not isinstance(tags, list):
        return []
    seen = set()
    out = []
    for t in tags:
        if not t:
            continue
        s = str(t).strip()
        low = s.lower()
        if low in seen or re.fullmatch(r"[0-9\s.,/+-]+", low):
            continue
        seen.add(low)
        out.append(s)
        if len(out) >= 10:
            break
    return out


def build_prompt(case, top_meta, played_refs):
    """R78-style prompt — proven best across sprints. Same shape as
    expR84c_response_regen.py."""
    tags = clean_tags(top_meta.get("tag_list") or [])
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    if isinstance(title, list): title = title[0] if title else "(unknown)"
    if isinstance(artist, list): artist = artist[0] if artist else "(unknown)"
    album = top_meta.get("album_name")
    if isinstance(album, list): album = album[0] if album else "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"

    conv_lines = []
    for h in case["history"]:
        role = h.get("role", "")
        content = str(h.get("content", ""))
        if role == "user":
            conv_lines.append(f"USER: {content}")
        elif role == "music":
            ref = played_refs.get(content, content[:8])
            conv_lines.append(f"MUSIC: {ref}")
    conv_lines.append(f"USER: {case['user_query']}")
    conv_text = "\n".join(conv_lines[-12:])

    parts = [
        "Write exactly one natural-language recommender response for this row.",
        "",
        "Hard requirements:",
        "- 65-95 words; aim for 70-85 words.",
        f"- The first sentence must name the recommendation as {title} by {artist}.",
        "- Mention why it fits using concrete evidence from the conversation.",
        "- Include 1-2 musical attributes (instrument, era, production technique, "
        "rhythm) as prose, not as labels.",
        "- Include at least one causal/anchoring connector (because, since, where, "
        "after, anchors, threads, frames).",
        "- Do not end with a question or an imperative closer.",
        "- Do not use vague descriptors: warm, vibrant, energetic, cozy, beautiful, "
        "gorgeous, gentle, charming, memorable, delightful, lovely.",
        "- Do not use boilerplate openers like 'If you're looking for', "
        "'You might enjoy', 'Here's a track that'.",
        "- Do not output prompt labels, bullets, metadata prefixes, markdown, or quotes "
        "around the whole answer.",
        "- Avoid crutches: 'perfect for', 'right in', 'lands', 'delivers', "
        "'you asked for', 'you described', 'exactly what', 'makes it a'.",
        "",
        "Conversation:",
        conv_text,
        "",
        "Top recommendation metadata:",
        f"Track: {title}",
        f"Artist: {artist}",
        f"Album: {album}",
        f"Release date: {release}",
        f"Tags: {', '.join(tags) if tags else '(none)'}",
    ]
    return "\n".join(parts)


def call_opus(client, prompt, model=MODEL_ID, max_tokens=MAX_TOKENS):
    msg = client.messages.create(
        model=model, max_tokens=max_tokens,
        system=("You are a music recommendation writer. Be specific, concrete, "
                "and concise. Follow all hard requirements exactly."),
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip(), msg


def validate(text):
    issues = []
    if not text:
        issues.append("empty")
        return issues
    if VAGUE_RE.search(text):
        issues.append("vague_descriptor")
    if IMPERATIVE_RE.search(text):
        issues.append("imperative_closer")
    low = text.lower()
    for p in BANNED_INHERITED:
        if p in low:
            issues.append(f"banned:{p}")
            break
    n_words = len(text.split())
    if n_words < WORD_MIN or n_words > WORD_MAX:
        issues.append(f"word_count={n_words}")
    return issues


def load_catalog_meta():
    from datasets import Dataset, load_dataset  # type: ignore
    hf_cache = Path.home() / ".cache/huggingface/datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    if matches:
        ds = Dataset.from_file(str(matches[-1]))
    else:
        ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["all_tracks"]
    cols = ds.to_dict()
    meta = {}
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        meta[tid] = {k: cols[k][i] for k in cols}
    return meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks-json", type=Path, required=True,
                   help="Routed tracks JSON from expR84c_blind_replay.py")
    p.add_argument("--blind-cache", type=Path, required=True,
                   help="Blind source cache (with user_query, history, music_turns)")
    p.add_argument("--output-zip", type=Path, required=True,
                   help="Output submission ZIP path")
    p.add_argument("--blind-name", default="blind_b",
                   help="Used for audit/log labeling only (default: blind_b)")
    p.add_argument("--dry-run", action="store_true",
                   help="No API calls; identify cases + print first 2 prompts")
    p.add_argument("--limit", type=int, default=None,
                   help="Only generate first N responses (for smoke test)")
    p.add_argument("--retry-on-issues", type=int, default=1)
    p.add_argument("--output-rows", type=Path, default=None,
                   help="JSONL log of per-row generation. Default: "
                        "alongside --output-zip with _rows.jsonl suffix.")
    args = p.parse_args()

    t0 = time.time()
    print(f"{ts()} R84c {args.blind_name} fresh-response generator (R78-style)")
    print("=" * 70)
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not args.dry_run and not api_key:
        print("ERROR: ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY required.")
        sys.exit(1)

    # Load inputs
    if not args.tracks_json.exists():
        print(f"ERROR: tracks JSON missing: {args.tracks_json}")
        sys.exit(1)
    if not args.blind_cache.exists():
        print(f"ERROR: blind cache missing: {args.blind_cache}")
        sys.exit(1)
    with open(args.tracks_json) as f:
        track_lists = json.load(f)
    with open(args.blind_cache, "rb") as f:
        blind = pickle.load(f)
    print(f"  tracks: {len(track_lists)} cases")
    print(f"  blind cases: {len(blind)} sessions in cache")

    # Sanity: every tracks_json sid should be in blind cache
    missing_in_cache = [t["session_id"] for t in track_lists if t["session_id"] not in blind]
    if missing_in_cache:
        print(f"ERROR: {len(missing_in_cache)} tracks sessions not in blind cache (first 5):")
        for sid in missing_in_cache[:5]:
            print(f"  {sid}")
        sys.exit(1)

    # Determine which cases need regen (here, ALL of them for fresh-response mode)
    todo = track_lists
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"\n  Will generate {len(todo)} responses with {MODEL_ID}")

    # Load catalog meta
    print(f"\n{ts()} Loading catalog metadata...")
    meta = load_catalog_meta()
    print(f"  {len(meta)} tracks")

    # Build played_refs per sid
    played_refs_per_sid = {
        sid: {tid: short_ref(tid, meta)
              for tid in blind[sid].get("music_turns", []) if tid in meta}
        for sid in blind
    }

    if args.dry_run:
        print(f"\n[dry-run] Would generate {len(todo)} responses.")
        print(f"\n  First 2 prompts:")
        for tl in todo[:2]:
            sid = tl["session_id"]
            top1 = tl["predicted_track_ids"][0]
            top_meta = meta.get(top1)
            if not top_meta:
                print(f"  !! {sid[:8]}: no top1 meta in catalog ({top1})")
                continue
            prompt = build_prompt(blind[sid], top_meta, played_refs_per_sid.get(sid, {}))
            print(f"\n  --- {sid[:8]} t{tl['turn_number']} top1={top1[:12]} ---")
            print(prompt[:1500])
            if len(prompt) > 1500:
                print(f"  ...[{len(prompt) - 1500} more chars]")
        return

    # Generate
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    new_responses: dict[str, str] = {}
    regen_log = []
    print(f"\n{ts()} Generating responses...")
    for n, tl in enumerate(todo):
        sid = tl["session_id"]
        case = blind[sid]
        top1 = tl["predicted_track_ids"][0] if tl["predicted_track_ids"] else None
        if not top1:
            print(f"  !! {sid[:8]}: empty track list, skipping")
            continue
        top_meta = meta.get(top1)
        if not top_meta:
            print(f"  !! {sid[:8]}: no top1 meta in catalog, skipping")
            continue
        prompt = build_prompt(case, top_meta, played_refs_per_sid.get(sid, {}))
        t_call = time.time()
        try:
            resp, _ = call_opus(client, prompt)
        except Exception as e:
            print(f"  !! {n+1}/{len(todo)} {sid[:8]}: API error: {e}")
            continue
        issues = validate(resp)
        if issues and args.retry_on_issues:
            retry_prompt = prompt + (
                f"\n\nIssues with prior attempt: {', '.join(issues)}. "
                "Rewrite exactly addressing those issues."
            )
            try:
                resp2, _ = call_opus(client, retry_prompt)
                if len(validate(resp2)) < len(issues):
                    resp = resp2
                    issues = validate(resp)
            except Exception:
                pass
        new_responses[sid] = resp
        regen_log.append({
            "session_id": sid, "turn_number": int(tl["turn_number"]),
            "top1_track_id": top1,
            "n_words": len(resp.split()),
            "issues_after": issues,
            "elapsed_s": time.time() - t_call,
        })
        print(f"  {n+1}/{len(todo)} {sid[:8]} "
              f"({len(resp.split())} words, issues={issues}, "
              f"{time.time()-t_call:.1f}s)")

    # Persist log
    rows_path = args.output_rows or args.output_zip.with_suffix(".rows.jsonl")
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rows_path, "w") as f:
        for row in regen_log:
            f.write(json.dumps(row) + "\n")
    print(f"\n  Saved regen log -> {rows_path}")

    # Build submission with validation
    final_items = []
    track_hash_input = []
    track_hash_output = []
    n_empty = 0
    n_invalid = 0
    for tl in track_lists:
        sid = tl["session_id"]
        track_hash_input.append((sid, int(tl["turn_number"]), tuple(tl["predicted_track_ids"])))
        resp = new_responses.get(sid, "")
        if not resp:
            n_empty += 1
        elif validate(resp):
            n_invalid += 1
        final_items.append({
            "session_id": sid,
            "turn_number": int(tl["turn_number"]),
            "predicted_track_ids": tl["predicted_track_ids"],
            "predicted_response": resp,
        })
        track_hash_output.append((sid, int(tl["turn_number"]), tuple(tl["predicted_track_ids"])))

    # Verify tracks unchanged
    track_hash_in = hashlib.sha256(json.dumps(sorted(track_hash_input), sort_keys=True).encode()).hexdigest()
    track_hash_out = hashlib.sha256(json.dumps(sorted(track_hash_output), sort_keys=True).encode()).hexdigest()
    tracks_match = (track_hash_in == track_hash_out)
    print(f"\n  Track hash unchanged: {'✓' if tracks_match else 'FAILED'}")
    print(f"  Empty responses: {n_empty}/{len(final_items)}")
    print(f"  Invalid responses (still have issues): {n_invalid}/{len(final_items)}")

    if not tracks_match:
        print("FATAL: track IDs changed during response generation. Refusing to write submission.")
        sys.exit(2)

    # Package
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    payload_str = json.dumps(final_items, indent=2)
    with zipfile.ZipFile(args.output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload_str)
    sub_sha = hashlib.sha256(open(args.output_zip, "rb").read()).hexdigest()
    print(f"\n  Wrote {args.output_zip} ({args.output_zip.stat().st_size} bytes)")
    print(f"  sha256: {sub_sha}")

    # Metadata
    meta_out = {
        "experiment": f"R84c {args.blind_name} fresh-response (R78-style)",
        "created_at": datetime.now().isoformat(),
        "head_sha": head_sha(),
        "blind_name": args.blind_name,
        "model": MODEL_ID,
        "prompt_style": "R78 (per R87 lesson: evidence-injection hurt LexDiv)",
        "n_cases": len(final_items),
        "n_regenerated": len([r for r in new_responses if r is not None]),
        "n_empty": n_empty,
        "n_invalid": n_invalid,
        "track_hash_sha256": track_hash_out,
        "submission_sha256": sub_sha,
    }
    meta_path = args.output_zip.with_suffix(".metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"  Metadata -> {meta_path}")
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
