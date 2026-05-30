"""R84c response regeneration — only for sessions where top-1 changed vs R78.

Reuses R78 responses for the 63 unchanged-top-1 sessions; regenerates with
Opus 4.7 (R78-style prompt) for the 17 changed-top-1 sessions, then packages
the final R84c blind submission ZIP.

Run:
  ANTHROPIC_API_KEY=sk-... uv run python scripts/expR84c_response_regen.py
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

# Inputs
R84C_TRACKS_JSON = REPO / "exp" / "inference" / "blind_a" / "r84c_blind_track_lists.json"
R78_SUB = REPO / "exp" / "inference" / "blind_a" / "r78_llm_polish_submission.zip"
BLIND_SRC = REPO / "cache" / "blind_a" / "source_cache.pkl"

# Outputs
OUT_DIR = REPO / "exp" / "inference" / "blind_a"
OUT_ZIP = OUT_DIR / "r84c_selective_submission.zip"
OUT_METADATA = OUT_DIR / "r84c_selective_submission.metadata.json"
OUT_REGEN_ROWS = OUT_DIR / "r84c_regen_rows.jsonl"
OUT_DIFF_MD = REPO / "docs" / "r84c_blind_diff.md"

MODEL_ID = "claude-opus-4-7"
MAX_TOKENS = 280

# R78-style banned phrases (same audit rules as R78)
VAGUE_RE = re.compile(
    r"\b(warm|vibrant|energetic|cozy|beautiful|gorgeous|gentle|charming|"
    r"memorable|delightful|lovely)\b", re.IGNORECASE,
)
IMPERATIVE_RE = re.compile(
    r"crank it loud\b|press play\b|lace up\b|hit play\b|"
    r"queue this up\b|give it a spin\b|throw it on\b", re.IGNORECASE,
)


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    g = shutil.which("git")
    if g is None:
        return "no-git"
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip()


def load_catalog_meta():
    """Just (track_id -> {name, artist, album, tags}) — for prompt context only."""
    from datasets import Dataset, load_dataset  # type: ignore
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
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


def build_prompt(case, top_meta, played_meta_refs):
    """R78-style prompt for one case + new top-1 track."""
    tags = clean_tags(top_meta.get("tag_list") or [])
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    if isinstance(title, list):
        title = title[0] if title else "(unknown)"
    if isinstance(artist, list):
        artist = artist[0] if artist else "(unknown)"
    album = top_meta.get("album_name")
    if isinstance(album, list):
        album = album[0] if album else "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"

    # Build conversation context
    conv_lines = []
    for h in case["history"]:
        role = h.get("role", "")
        content = str(h.get("content", ""))
        if role == "user":
            conv_lines.append(f"USER: {content}")
        elif role == "music":
            ref = short_ref(content, top_meta if content == "" else
                            {content: top_meta} if False else {})
            # Use played_meta_refs lookup
            ref = played_meta_refs.get(content, content[:8])
            conv_lines.append(f"MUSIC: {ref}")
    conv_lines.append(f"USER: {case['user_query']}")
    conv_text = "\n".join(conv_lines[-12:])  # last 12 turns

    parts = [
        "Write exactly one natural-language recommender response for this row.",
        "",
        "Hard requirements:",
        "- 65-95 words; aim for 70-85 words.",
        f"- The first sentence must name the recommendation as {title} by {artist}.",
        "- Mention why it fits using concrete evidence from the conversation.",
        "- Include 1-2 musical attributes (instrument, era, production technique, "
        "rhythm) as prose, not as label/value tags.",
        "- Include at least one causal/anchoring connector (because, since, where, "
        "while, after, anchors, threads, frames).",
        "- Do not end with a question or an imperative closer (no 'crank it loud', "
        "'press play', 'queue this up', etc.).",
        "- Do not use vague descriptors: warm, vibrant, energetic, cozy, beautiful, "
        "gorgeous, gentle, charming, memorable, delightful, lovely.",
        "- Do not use boilerplate openers such as 'If you're looking for', "
        "'You might enjoy', or 'Here's a track that'.",
        "- Do not output prompt labels, bullets, metadata prefixes, markdown, or quotes "
        "around the whole answer.",
        "- Avoid crutches like 'perfect for', 'right in', 'lands', 'delivers', "
        "'you asked for', 'you described', 'exactly what', 'makes it a' unless there "
        "is no cleaner wording.",
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
        system="You are a music recommendation writer. Be specific, concrete, "
               "and concise. Follow all hard requirements exactly.",
        messages=[{"role": "user", "content": prompt}],
    )
    out = msg.content[0].text.strip()
    return out, msg


def validate(text):
    """Lightweight checks — just warn, don't reject."""
    issues = []
    if VAGUE_RE.search(text):
        issues.append("vague_descriptor")
    if IMPERATIVE_RE.search(text):
        issues.append("imperative_closer")
    n_words = len(text.split())
    if n_words < 55 or n_words > 110:
        issues.append(f"word_count={n_words}")
    return issues


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retry-on-issues", type=int, default=1,
                   help="Number of regen retries if validate finds issues")
    p.add_argument("--dry-run", action="store_true", help="No API calls; just show targets")
    args = p.parse_args()

    t0 = time.time()
    print(f"{ts()} R84c response regeneration")
    print("=" * 70)

    # API key
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not args.dry_run and not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set. Re-run with `export ANTHROPIC_API_KEY=...`")
        sys.exit(1)

    # Load inputs
    with open(R84C_TRACKS_JSON) as f:
        r84c_tracks = json.load(f)
    with zipfile.ZipFile(R78_SUB) as z:
        r78_items = json.loads(z.read("prediction.json"))
    r78_by_key = {(i["session_id"], int(i["turn_number"])): i for i in r78_items}
    print(f"  R84c tracks: {len(r84c_tracks)} cases")
    print(f"  R78 responses: {len(r78_items)} cases")
    assert len(r78_items) == 80 and len(r84c_tracks) == 80

    with open(BLIND_SRC, "rb") as f:
        blind = pickle.load(f)

    # Identify changed-top-1 cases
    changed_indices = []
    for idx, r84c in enumerate(r84c_tracks):
        key = (r84c["session_id"], int(r84c["turn_number"]))
        r78 = r78_by_key.get(key)
        if not r78:
            continue
        r84c_top1 = r84c["predicted_track_ids"][0] if r84c["predicted_track_ids"] else None
        r78_top1 = r78["predicted_track_ids"][0] if r78["predicted_track_ids"] else None
        if r84c_top1 != r78_top1:
            changed_indices.append(idx)
    print(f"\n  Cases needing regeneration: {len(changed_indices)}/80")
    for idx in changed_indices:
        r84c = r84c_tracks[idx]
        r78 = r78_by_key[(r84c["session_id"], int(r84c["turn_number"]))]
        print(f"    {r84c['session_id'][:8]} t{r84c['turn_number']} "
              f"R78_top1={r78['predicted_track_ids'][0][:8]} → "
              f"R84c_top1={r84c['predicted_track_ids'][0][:8]} "
              f"(R84={r84c.get('_routed_r84')}, margin={r84c.get('_r54_margin'):.3f})")

    if args.dry_run:
        print(f"\n  [dry-run] would call Opus {len(changed_indices)} times.")
        return

    # Load catalog
    print(f"\n{ts()} Loading catalog...")
    meta = load_catalog_meta()
    print(f"  {len(meta)} tracks")

    # Build played_meta_refs per case (helps prompt readability)
    played_refs_per_case = {}
    for r84c in r84c_tracks:
        sid = r84c["session_id"]
        if sid not in blind:
            continue
        case_played = blind[sid]["music_turns"]
        played_refs_per_case[sid] = {
            tid: short_ref(tid, meta) for tid in case_played if tid in meta
        }

    # Regen
    print(f"\n{ts()} Regenerating {len(changed_indices)} responses with {MODEL_ID}...")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    regen_rows = []
    regen_responses = {}
    for n, idx in enumerate(changed_indices):
        r84c = r84c_tracks[idx]
        sid = r84c["session_id"]
        case = blind[sid]
        new_top1 = r84c["predicted_track_ids"][0]
        top_meta = meta.get(new_top1)
        if not top_meta:
            print(f"  !! missing meta for {new_top1}, skipping")
            continue
        prompt = build_prompt(case, top_meta, played_refs_per_case.get(sid, {}))
        t_call = time.time()
        try:
            resp, msg = call_opus(client, prompt)
        except Exception as e:
            print(f"  !! {n+1}/{len(changed_indices)} {sid[:8]}: API error: {e}")
            continue
        issues = validate(resp)
        if issues and args.retry_on_issues:
            retry_prompt = prompt + f"\n\nIssues with prior attempt: {', '.join(issues)}. " \
                                   "Rewrite addressing those issues exactly."
            try:
                resp2, msg2 = call_opus(client, retry_prompt)
                issues2 = validate(resp2)
                if len(issues2) < len(issues):
                    resp = resp2
                    issues = issues2
            except Exception as e:
                print(f"     retry failed: {e}")
        regen_responses[sid] = resp
        regen_rows.append({
            "session_id": sid, "turn_number": int(r84c["turn_number"]),
            "new_top1": new_top1, "response": resp,
            "issues_after_regen": issues,
            "_routed_r84": r84c.get("_routed_r84"),
            "_r54_margin": r84c.get("_r54_margin"),
            "n_words": len(resp.split()),
            "elapsed_s": time.time() - t_call,
        })
        print(f"  {n+1}/{len(changed_indices)} {sid[:8]} "
              f"({len(resp.split())} words, issues={issues}, "
              f"{time.time()-t_call:.1f}s)")

    # Save regen rows
    OUT_REGEN_ROWS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REGEN_ROWS, "w") as f:
        for row in regen_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\n  Saved regen rows -> {OUT_REGEN_ROWS}")

    # Build final submission
    final_items = []
    n_reused = 0
    n_regen = 0
    for r84c in r84c_tracks:
        sid = r84c["session_id"]
        key = (sid, int(r84c["turn_number"]))
        if sid in regen_responses:
            resp = regen_responses[sid]
            n_regen += 1
        else:
            resp = r78_by_key[key]["predicted_response"]
            n_reused += 1
        final_items.append({
            "session_id": sid,
            "turn_number": int(r84c["turn_number"]),
            "predicted_track_ids": r84c["predicted_track_ids"],
            "predicted_response": resp,
        })
    print(f"\n  Final assembly: {n_regen} regen responses + {n_reused} reused R78")

    # Save submission ZIP
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload_str = json.dumps(final_items, indent=2)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload_str)
    sha = hashlib.sha256(open(OUT_ZIP, "rb").read()).hexdigest()
    print(f"  Wrote {OUT_ZIP} ({OUT_ZIP.stat().st_size} bytes)")
    print(f"  sha256: {sha}")

    # Metadata
    meta_out = {
        "experiment": "R84c selective deployment Blind-A candidate",
        "created_at": datetime.now().isoformat(),
        "head_sha": head_sha(),
        "model": MODEL_ID,
        "n_cases": len(final_items),
        "n_responses_regenerated": n_regen,
        "n_responses_reused_from_r78": n_reused,
        "tracks_from": "R84c selective routing (R84 if R54c LR top-1 margin < 0.5 OR >= 2.0, else R54c)",
        "responses_changed_when": "R84c top-1 != R78 top-1",
        "submission_sha256": sha,
        "submission_path": str(OUT_ZIP.relative_to(REPO)),
    }
    with open(OUT_METADATA, "w") as f:
        json.dump(meta_out, f, indent=2)
    print(f"  Metadata -> {OUT_METADATA}")

    # Diff report
    md = [
        "# R84c Blind-A Candidate — Submission Diff",
        "",
        f"HEAD: `{meta_out['head_sha'][:10]}`",
        f"Submission: `{OUT_ZIP.name}` ({OUT_ZIP.stat().st_size} bytes)",
        f"sha256: `{sha}`",
        "",
        "## Composition",
        f"- 80 cases total",
        f"- Tracks from R84c selective routing (margin < 0.5 OR >= 2.0 → R84)",
        f"- Responses: **{n_regen} regenerated** (changed top-1 vs R78), "
        f"**{n_reused} reused from R78** (unchanged top-1)",
        "",
        "## Regenerated sessions",
        "",
        "| session_id | turn | new top-1 | routed | margin | words | issues |",
        "|---|---:|---|---|---:|---:|---|",
    ]
    for row in regen_rows:
        md.append(
            f"| `{row['session_id'][:12]}` | {row['turn_number']} | "
            f"`{row['new_top1'][:12]}` | "
            f"{'R84' if row['_routed_r84'] else 'R54'} | "
            f"{row['_r54_margin']:.3f} | {row['n_words']} | "
            f"{','.join(row['issues_after_regen']) or '—'} |"
        )
    md += [
        "",
        "## Submission gate",
        "",
        "- Predeclared rule: `r54c_top1_margin < 0.5 OR margin >= 2.0` → R84 LR",
        "- Audit (from `expR84c_blind_audit.json`): churn 17/80 ✓, overlap 16.31/20 ✓",
        "- Production R78 (composite 0.6302, #4) UNTOUCHED",
        "",
        "**NOT auto-uploaded.** User decides whether to submit.",
    ]
    OUT_DIFF_MD.write_text("\n".join(md) + "\n")
    print(f"\n{ts()} Diff -> {OUT_DIFF_MD}")
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
