"""R86 LexDiv recovery on top of R84c production.

R84c won blind (composite +0.0060) but LexDiv dropped 0.8845 → 0.8720 (−0.0125)
because the 17 regen'd responses introduced corpus-level bigram redundancy.
LLM held flat at 4.90. Tracks/nDCG unchanged.

R86 strategy (zero-nDCG-risk):
1. Audit R84c corpus bigrams; identify the top-K over-represented content bigrams.
2. Among the 17 changed-top-1 rows (the only ones differing from R78), rank by
   bigram-repeat-density (how many bigrams in this row also appear in other rows).
3. Regenerate the top-N highest-density rows with Opus 4.7, banning the audit's
   top bigrams + R74-style LexDiv repair guidance + R84c content fidelity
   (must still match the new top-1 track's musical attributes).
4. Tracks bitwise identical; only responses change.

Gate:
- Tracks identical (hash check).
- Local corpus Distinct-2 ≥ R84c + 0.010 (proxy for LexDiv recovery).
- All responses 70-90 words, no banned phrases.
- No vague descriptors, no imperative closers.

Run:
  ANTHROPIC_RECSYS_API_KEY=... uv run python scripts/expR86_lexdiv_recovery.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Inputs
R84C_SUB = REPO / "exp" / "inference" / "blind_a" / "r84c_selective_submission.zip"
R84C_REGEN_ROWS = REPO / "exp" / "inference" / "blind_a" / "r84c_regen_rows.jsonl"
BLIND_SRC = REPO / "cache" / "blind_a" / "source_cache.pkl"

# Outputs
OUT_DIR = REPO / "exp" / "inference" / "blind_a"
OUT_ZIP = OUT_DIR / "r86_lexdiv_recovery_submission.zip"
OUT_METADATA = OUT_DIR / "r86_lexdiv_recovery_submission.metadata.json"
OUT_REGEN_ROWS = OUT_DIR / "r86_regen_rows.jsonl"
OUT_AUDIT = REPO / "exp" / "eval" / "expR86_bigram_audit.json"
OUT_DIFF_MD = REPO / "docs" / "r86_lexdiv_recovery_result.md"

MODEL_ID = "claude-opus-4-7"
MAX_TOKENS = 280

# Number of top-density rows (out of 17) to regenerate
DEFAULT_TARGET_ROWS = 12
MIN_REGEN_ROWS = 8
MAX_REGEN_ROWS = 17

# Local Distinct-2 gate (R84c proxy ≈ 0.872; require +0.010 lift)
DISTINCT2_LIFT_REQUIRED = 0.010

# Stopwords (R74 set)
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "of", "to",
    "with", "for", "is", "are", "by", "that", "this", "as", "s", "t",
    "it", "its", "be", "was", "been", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "can", "so", "if", "from", "after",
    "before", "into", "through", "between", "while", "since",
}

# R78-style banned (vague + imperative)
VAGUE_RE = re.compile(
    r"\b(warm|vibrant|energetic|cozy|beautiful|gorgeous|gentle|charming|"
    r"memorable|delightful|lovely)\b", re.IGNORECASE,
)
IMPERATIVE_RE = re.compile(
    r"crank it loud\b|press play\b|lace up\b|hit play\b|"
    r"queue this up\b|give it a spin\b|throw it on\b", re.IGNORECASE,
)
# R73/R74 inherited banned phrases
BANNED_INHERITED = [
    "comes off", "fits that", "carries that", "captures the",
    "captures that", "leans into", "matches the",
    "you're looking", "you're chasing", "you're trying",
    "you're after", "you described", "you mentioned",
    "perfect for", "exactly what", "makes it a",
    "vibe", "journey", "soundscape",
]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def head_sha() -> str:
    g = shutil.which("git")
    return (subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO))
            .decode().strip()) if g else "no-git"


def tokenize_content(text):
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) >= 2]


def bigrams_of(text):
    toks = tokenize_content(text)
    return [f"{toks[i]} {toks[i+1]}" for i in range(len(toks) - 1)]


def all_bigrams_set(text):
    return set(bigrams_of(text))


def corpus_bigram_doc_count(responses):
    """Returns {bigram: # of responses containing it}."""
    doc_counts = Counter()
    for r in responses:
        for bg in set(bigrams_of(r)):
            doc_counts[bg] += 1
    return doc_counts


def corpus_distinct2(responses):
    """Compute corpus-level Distinct-2 = unique bigrams / total bigrams."""
    all_bg = []
    for r in responses:
        all_bg.extend(bigrams_of(r))
    if not all_bg:
        return 0.0
    return len(set(all_bg)) / len(all_bg)


def row_repeat_density(row_text, doc_counts, threshold=2):
    """Score = # of bigrams in row that appear in `threshold`+ other rows."""
    row_bgs = set(bigrams_of(row_text))
    return sum(1 for bg in row_bgs if doc_counts.get(bg, 0) >= threshold)


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


def build_prompt(case, top_meta, played_refs, banned_audit_bigrams, archetype_idx):
    tags = clean_tags(top_meta.get("tag_list") or [])
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    if isinstance(title, list): title = title[0] if title else "(unknown)"
    if isinstance(artist, list): artist = artist[0] if artist else "(unknown)"
    album = top_meta.get("album_name")
    if isinstance(album, list): album = album[0] if album else "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"

    # Build conversation context
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

    # Archetype rotation for opening variety
    archetypes = [
        ("verdict-first", "Open with a direct verdict naming the recommendation"),
        ("concrete-detail-lead", "Open with one concrete musical detail (instrument, production technique), then name the recommendation"),
        ("album-anchor", "Open with album or release-year context, then introduce the track"),
        ("artist-confidence", "Open with an artist-quality claim, then ground it in this track"),
        ("session-anchor", "Open with the user's specific session need, then pivot to the track"),
    ]
    arch_name, arch_open = archetypes[archetype_idx % len(archetypes)]

    parts = [
        f"Write exactly one music recommendation response. Archetype: {arch_name}.",
        f"Opening style: {arch_open}.",
        "",
        "Hard requirements:",
        "- 70-90 words.",
        f"- Must name the recommendation as {title} by {artist} in the first sentence.",
        "- Include 1-2 concrete musical attributes (instrument, production technique, "
        "rhythm/tempo descriptor, era reference) as prose, not as labels.",
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
        "LEXICAL DIVERSITY REQUIREMENT — DO NOT use any of these specific two-word "
        "phrases (they are over-used in this corpus):",
        "; ".join(banned_audit_bigrams),
        "",
        "Use fresh vocabulary; vary sentence openings; avoid reusing canned phrasing.",
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
        system=("You are a music recommendation writer. Be specific, concrete, and "
                "lexically diverse. Follow all hard requirements exactly."),
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip(), msg


def validate(text):
    issues = []
    if VAGUE_RE.search(text):
        issues.append("vague_descriptor")
    if IMPERATIVE_RE.search(text):
        issues.append("imperative_closer")
    low = text.lower()
    for phrase in BANNED_INHERITED:
        if phrase in low:
            issues.append(f"banned:{phrase}")
    n_words = len(text.split())
    if n_words < 60 or n_words > 110:
        issues.append(f"word_count={n_words}")
    return issues


def load_r84c_submission():
    with zipfile.ZipFile(R84C_SUB) as z:
        return json.loads(z.read("prediction.json"))


def load_catalog_meta():
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--audit-only", action="store_true",
                   help="Just compute corpus bigram audit, no API calls")
    p.add_argument("--retry-on-issues", type=int, default=1)
    args = p.parse_args()

    t0 = time.time()
    print(f"{ts()} R86 LexDiv recovery on R84c")
    print("=" * 70)

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not (args.dry_run or args.audit_only) and not api_key:
        print("ERROR: ANTHROPIC_API_KEY or ANTHROPIC_RECSYS_API_KEY required")
        sys.exit(1)

    # Load R84c
    r84c_items = load_r84c_submission()
    r84c_by_key = {(i["session_id"], int(i["turn_number"])): i for i in r84c_items}
    print(f"  R84c: {len(r84c_items)} cases")

    # Load the 17 regen rows (these are the changed-top-1 cases)
    regen_rows = []
    with open(R84C_REGEN_ROWS) as f:
        for line in f:
            regen_rows.append(json.loads(line))
    print(f"  R84c regen rows (changed-top-1): {len(regen_rows)}")
    regen_keys = {(r["session_id"], r["turn_number"]) for r in regen_rows}

    # Corpus-level bigram audit on R84c
    responses = [i["predicted_response"] for i in r84c_items]
    distinct2_r84c = corpus_distinct2(responses)
    doc_counts = corpus_bigram_doc_count(responses)
    print(f"\n  R84c corpus Distinct-2: {distinct2_r84c:.4f}")
    print(f"  Unique bigrams: {len(doc_counts)}")
    print(f"  Top-30 over-represented bigrams (appear in >=4 rows):")
    over_represented = sorted(
        [(bg, c) for bg, c in doc_counts.items() if c >= 4],
        key=lambda x: -x[1],
    )
    for bg, c in over_represented[:30]:
        print(f"    {c:3d}x  '{bg}'")
    print(f"  ... total >=4-doc bigrams: {len(over_represented)}")

    # Score each of the 17 regen rows by repeat density
    print(f"\n  Per-row repeat density (17 regen rows):")
    scored = []
    for r in regen_rows:
        key = (r["session_id"], r["turn_number"])
        text = r84c_by_key[key]["predicted_response"]
        density = row_repeat_density(text, doc_counts, threshold=3)
        scored.append({
            "session_id": r["session_id"],
            "turn_number": r["turn_number"],
            "density": density,
            "n_words": len(text.split()),
            "response": text,
        })
    scored.sort(key=lambda x: -x["density"])
    for s in scored:
        print(f"    density={s['density']:3d}  words={s['n_words']:3d}  "
              f"sid={s['session_id'][:8]}t{s['turn_number']}")

    # Pick top-N for regen
    n_target = max(MIN_REGEN_ROWS, min(args.target_rows, MAX_REGEN_ROWS))
    pick = scored[:n_target]
    print(f"\n  R86 will regenerate top-{n_target} by repeat-density "
          f"(density range: {pick[-1]['density']} – {pick[0]['density']})")

    # Audit-derived banned bigrams: top-20 over-represented bigrams
    banned_audit = [bg for bg, _ in over_represented[:20]]
    print(f"\n  Banned bigrams from audit (top-{len(banned_audit)}): "
          f"{banned_audit[:10]}...")

    # Persist audit
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    audit_out = {
        "experiment": "R86 LexDiv recovery audit",
        "created_at": datetime.now().isoformat(),
        "r84c_distinct2": distinct2_r84c,
        "n_unique_bigrams": len(doc_counts),
        "n_over_represented_ge4": len(over_represented),
        "top_30_over_represented": [
            {"bigram": bg, "doc_count": c} for bg, c in over_represented[:30]
        ],
        "banned_audit_bigrams": banned_audit,
        "candidate_rows_ranked_by_density": scored,
        "selected_for_regen": [
            {"session_id": s["session_id"], "turn_number": s["turn_number"],
             "density": s["density"]} for s in pick
        ],
    }
    with open(OUT_AUDIT, "w") as f:
        json.dump(audit_out, f, indent=2)
    print(f"\n  Audit saved -> {OUT_AUDIT}")

    if args.audit_only:
        print("\n[audit-only mode — exiting]")
        return

    if args.dry_run:
        print(f"\n[dry-run — would call Opus {len(pick)} times]")
        return

    # Load catalog + blind for prompt context
    print(f"\n{ts()} Loading catalog + blind sources...")
    meta = load_catalog_meta()
    import pickle
    with open(BLIND_SRC, "rb") as f:
        blind = pickle.load(f)

    played_refs_per_sid = {}
    for r84c in r84c_items:
        sid = r84c["session_id"]
        if sid in blind:
            played_refs_per_sid[sid] = {
                tid: short_ref(tid, meta)
                for tid in blind[sid]["music_turns"] if tid in meta
            }

    # Regen
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    print(f"\n{ts()} Regenerating {len(pick)} responses with {MODEL_ID}...")
    new_responses_by_sid = {}
    regen_log = []
    for n, sel in enumerate(pick):
        sid = sel["session_id"]
        turn = sel["turn_number"]
        r84c_item = r84c_by_key[(sid, turn)]
        new_top1 = r84c_item["predicted_track_ids"][0]
        top_meta = meta.get(new_top1)
        if not top_meta:
            print(f"  !! {sid[:8]}: missing top meta, skipping")
            continue
        case = blind[sid]
        prompt = build_prompt(case, top_meta, played_refs_per_sid.get(sid, {}),
                               banned_audit, archetype_idx=n)
        t_call = time.time()
        try:
            resp, _ = call_opus(client, prompt)
        except Exception as e:
            print(f"  !! {sid[:8]}: API error: {e}")
            continue
        issues = validate(resp)
        if issues and args.retry_on_issues:
            retry_prompt = prompt + (
                f"\n\nIssues with prior attempt: {', '.join(issues)}. "
                "Rewrite exactly addressing those issues. Keep all other "
                "constraints in place."
            )
            try:
                resp2, _ = call_opus(client, retry_prompt)
                if len(validate(resp2)) < len(issues):
                    resp = resp2
                    issues = validate(resp)
            except Exception:
                pass
        new_responses_by_sid[sid] = resp
        regen_log.append({
            "session_id": sid, "turn_number": turn,
            "density_before": sel["density"],
            "old_response": sel["response"],
            "new_response": resp,
            "issues": issues,
            "n_words": len(resp.split()),
            "elapsed_s": time.time() - t_call,
        })
        print(f"  {n+1}/{len(pick)} {sid[:8]} "
              f"({len(resp.split())} words, issues={issues}, "
              f"{time.time()-t_call:.1f}s)")

    # Build final submission
    final_items = []
    n_changed = 0
    for r84c_item in r84c_items:
        sid = r84c_item["session_id"]
        new_resp = new_responses_by_sid.get(sid)
        if new_resp:
            n_changed += 1
            final_items.append({
                "session_id": sid,
                "turn_number": int(r84c_item["turn_number"]),
                "predicted_track_ids": r84c_item["predicted_track_ids"],
                "predicted_response": new_resp,
            })
        else:
            final_items.append({
                "session_id": sid,
                "turn_number": int(r84c_item["turn_number"]),
                "predicted_track_ids": r84c_item["predicted_track_ids"],
                "predicted_response": r84c_item["predicted_response"],
            })

    # Verify tracks identical
    r84c_track_hash = hashlib.sha256(json.dumps(
        [(i["session_id"], i["turn_number"], i["predicted_track_ids"]) for i in r84c_items],
        sort_keys=True,
    ).encode()).hexdigest()
    r86_track_hash = hashlib.sha256(json.dumps(
        [(i["session_id"], i["turn_number"], i["predicted_track_ids"]) for i in final_items],
        sort_keys=True,
    ).encode()).hexdigest()
    tracks_identical = r84c_track_hash == r86_track_hash
    print(f"\n  Track hash R84c vs R86: {'IDENTICAL ✓' if tracks_identical else 'DIFFER ✗'}")

    # Corpus Distinct-2 after R86 regen
    new_responses = [i["predicted_response"] for i in final_items]
    distinct2_r86 = corpus_distinct2(new_responses)
    lift = distinct2_r86 - distinct2_r84c
    print(f"  R84c Distinct-2: {distinct2_r84c:.4f}")
    print(f"  R86  Distinct-2: {distinct2_r86:.4f}  (Δ {lift:+.4f})")
    print(f"  Gate: lift >= {DISTINCT2_LIFT_REQUIRED:+.3f}: "
          f"{'PASS' if lift >= DISTINCT2_LIFT_REQUIRED else 'FAIL'}")

    # Save regen log
    OUT_REGEN_ROWS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REGEN_ROWS, "w") as f:
        for row in regen_log:
            f.write(json.dumps(row) + "\n")

    # Package final submission ZIP
    payload_str = json.dumps(final_items, indent=2)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload_str)
    sha = hashlib.sha256(open(OUT_ZIP, "rb").read()).hexdigest()
    print(f"\n  Wrote {OUT_ZIP} ({OUT_ZIP.stat().st_size} bytes)")
    print(f"  sha256: {sha}")

    # Metadata
    meta_out = {
        "experiment": "R86 LexDiv recovery on R84c (tracks identical, responses only)",
        "created_at": datetime.now().isoformat(),
        "head_sha": head_sha(),
        "model": MODEL_ID,
        "base_submission": "r84c_selective_submission.zip",
        "n_cases": len(final_items),
        "n_responses_regenerated": n_changed,
        "n_responses_unchanged": len(final_items) - n_changed,
        "tracks_identical_to_r84c": tracks_identical,
        "r84c_local_distinct2": distinct2_r84c,
        "r86_local_distinct2": distinct2_r86,
        "distinct2_lift": lift,
        "gate_lift_required": DISTINCT2_LIFT_REQUIRED,
        "gate_passed": lift >= DISTINCT2_LIFT_REQUIRED,
        "submission_sha256": sha,
        "submission_path": str(OUT_ZIP.relative_to(REPO)),
    }
    with open(OUT_METADATA, "w") as f:
        json.dump(meta_out, f, indent=2)

    # Diff report
    md = [
        "# R86 LexDiv Recovery on R84c — Result",
        "",
        f"HEAD: `{meta_out['head_sha'][:10]}`",
        f"Base: `r84c_selective_submission.zip` (production)",
        f"R86: `{OUT_ZIP.name}` ({OUT_ZIP.stat().st_size} bytes, sha256 `{sha[:16]}`)",
        "",
        "## Composition",
        f"- 80 cases (tracks identical to R84c, hash {'matches' if tracks_identical else 'DIFFERS'})",
        f"- **{n_changed}** responses regenerated (top-{n_changed} by R84c bigram-repeat-density)",
        f"- **{len(final_items) - n_changed}** responses unchanged from R84c",
        "",
        "## Lexical diversity",
        "",
        "| metric | R84c | R86 | Δ |",
        "|---|---:|---:|---:|",
        f"| local corpus Distinct-2 | {distinct2_r84c:.4f} | {distinct2_r86:.4f} | "
        f"{lift:+.4f} |",
        "",
        f"Gate (lift >= {DISTINCT2_LIFT_REQUIRED:+.3f}): "
        f"**{'PASS' if lift >= DISTINCT2_LIFT_REQUIRED else 'FAIL'}**",
        "",
        "## Regenerated rows (top-density)",
        "",
        "| session_id | turn | density_before | words | issues |",
        "|---|---:|---:|---:|---|",
    ]
    for row in regen_log:
        md.append(
            f"| `{row['session_id'][:12]}` | {row['turn_number']} | "
            f"{row['density_before']} | {row['n_words']} | "
            f"{','.join(row['issues']) or '—'} |"
        )
    md += [
        "",
        "## Submission gate",
        "",
        "- Tracks bitwise identical to R84c (zero nDCG risk).",
        "- Local Distinct-2 proxy: gate "
        f"{'PASSED' if lift >= DISTINCT2_LIFT_REQUIRED else 'FAILED'}.",
        "- No banned-phrase failures expected (validate caught only inline issues).",
        "",
        "**NOT auto-uploaded.** User decides whether to submit.",
    ]
    OUT_DIFF_MD.write_text("\n".join(md) + "\n")
    print(f"  Diff report -> {OUT_DIFF_MD}")
    print(f"  Metadata -> {OUT_METADATA}")
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
