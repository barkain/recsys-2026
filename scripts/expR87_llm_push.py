"""R87 response-side push — target LLM 4.95 on R84c base.

Base: R84c production (composite 0.6362). Tracks bitwise identical.

Audit each of 80 R84c responses for LLM-judge weakness signals:
- missing causal connector (because/since/where/while/after/etc.)
- missing concrete musical attribute (instrument, era, production technique)
- missing session/history anchor (reference to user's stated need)
- corpus-level bigram repetition with other rows
- vague descriptors (residual after R78/R86)

Pick top-N (default 12) weakest rows, regenerate with stronger evidence
prompt that demands all three lift signals explicitly. Validate locally.

Run:
  ANTHROPIC_RECSYS_API_KEY=... uv run python scripts/expR87_llm_push.py
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
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Inputs
R84C_SUB = REPO / "exp" / "inference" / "blind_a" / "r84c_selective_submission.zip"
BLIND_SRC = REPO / "cache" / "blind_a" / "source_cache.pkl"

# Outputs
OUT_DIR = REPO / "exp" / "inference" / "blind_a"
OUT_ZIP = OUT_DIR / "r87_llm_push_submission.zip"
OUT_METADATA = OUT_DIR / "r87_llm_push_submission.metadata.json"
OUT_REGEN_ROWS = OUT_DIR / "r87_regen_rows.jsonl"
OUT_AUDIT = REPO / "exp" / "eval" / "expR87_audit.json"
OUT_DIFF_MD = REPO / "docs" / "r87_llm_push_result.md"

MODEL_ID = "claude-opus-4-7"
MAX_TOKENS = 300
DEFAULT_TARGET_ROWS = 12
MIN_REGEN_ROWS = 8
MAX_REGEN_ROWS = 15

# Stopwords for content-bigram audit
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "of", "to",
    "with", "for", "is", "are", "by", "that", "this", "as", "s", "t",
    "it", "its", "be", "was", "been", "has", "have", "had", "do", "does",
    "did", "will", "would", "could", "can", "so", "if", "from", "after",
    "before", "into", "through", "between", "while", "since",
}

# Banned residual phrases (R78/R86 inherited + R87 specific)
VAGUE_RE = re.compile(
    r"\b(warm|vibrant|energetic|cozy|beautiful|gorgeous|gentle|charming|"
    r"memorable|delightful|lovely|wonderful|amazing|stunning|terrific|"
    r"exceptional|outstanding|great|nice|good)\b", re.IGNORECASE,
)
IMPERATIVE_RE = re.compile(
    r"crank it loud\b|press play\b|lace up\b|hit play\b|"
    r"queue this up\b|give it a spin\b|throw it on\b", re.IGNORECASE,
)
BANNED_INHERITED = [
    "comes off", "fits that", "carries that", "captures the",
    "captures that", "leans into", "matches the",
    "you're looking", "you're chasing", "you're trying",
    "you're after", "you described", "you mentioned",
    "perfect for", "exactly what", "makes it a",
    "vibe", "journey", "soundscape",
    "rather than",  # top R84c overused bigram (12x)
]

# Causal connectors (positive signal)
CAUSAL_PATTERNS = [
    r"\bbecause\b", r"\bsince\b", r"\bwhere\b", r"\bafter\b",
    r"\bwhen\b", r"\bbuilt on\b", r"\bgrounded in\b", r"\banchors\b",
    r"\bframes\b", r"\bthreads\b", r"\bcenters\b", r"\bopens with\b",
    r"\bdrives\b", r"\bweaves\b", r"\bripens\b", r"\bextends\b",
    r"\btraces\b", r"\bdelivers\b",  # delivers is borderline
]
CAUSAL_RE = re.compile("|".join(CAUSAL_PATTERNS), re.IGNORECASE)

# Concrete musical attributes (positive signal)
ATTRIBUTE_WORDS = {
    "guitar", "bass", "drum", "drums", "piano", "vocal", "vocals", "synth",
    "synthesizer", "horn", "horns", "brass", "string", "strings",
    "percussion", "kick", "snare", "hi-hat", "fiddle", "banjo", "mandolin",
    "fingerpicked", "fingerstyle", "palm-muted", "arpeggiated", "syncopated",
    "rhythm", "tempo", "groove", "beat", "riff", "hook", "melody",
    "harmony", "chord", "key", "minor", "major", "tone", "timbre",
    "production", "mix", "mastering", "lyric", "verse", "chorus",
    "bridge", "intro", "outro", "tremolo", "vibrato", "reverb", "delay",
    "fuzz", "distortion", "double-kick", "blast", "drum-machine",
    "808", "909", "moog", "rhodes", "tape", "lo-fi", "hi-fi",
    "boom-bap", "trap", "polyrhythm", "blast-beat",
}

# Session-anchor markers (positive signal)
SESSION_ANCHOR_PATTERNS = [
    r"\byou (asked|wanted|requested|sought|liked|praised|locked|noted)",
    r"\bafter (your|the) (previous|earlier|prior|opening)",
    r"\b(matching|mirroring|extending|continuing) your\b",
    r"\bthe (thread|chain|arc) (you|of)\b",
    r"\b(builds|builds on|extends|continues) (the|that|your)\b",
]
SESSION_ANCHOR_RE = re.compile("|".join(SESSION_ANCHOR_PATTERNS), re.IGNORECASE)


def ts(): return f"[{datetime.now():%H:%M:%S}]"


def head_sha():
    g = shutil.which("git")
    return subprocess.check_output([g, "rev-parse", "HEAD"], cwd=str(REPO)).decode().strip() if g else "no-git"


def tokenize_content(text):
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    return [t for t in toks if t not in STOPWORDS and len(t) >= 2]


def bigrams_of(text):
    t = tokenize_content(text)
    return [f"{t[i]} {t[i+1]}" for i in range(len(t) - 1)]


def corpus_distinct2(responses):
    bgs = []
    for r in responses:
        bgs.extend(bigrams_of(r))
    return len(set(bgs)) / max(1, len(bgs))


def corpus_bigram_doc_count(responses):
    dc = Counter()
    for r in responses:
        for bg in set(bigrams_of(r)):
            dc[bg] += 1
    return dc


def audit_response(text, doc_counts):
    """Score for LLM-judge weakness; higher = weaker = needs regen."""
    score = 0
    diagnosis = []
    # Missing causal connector
    if not CAUSAL_RE.search(text):
        score += 3
        diagnosis.append("no_causal")
    # Missing musical attribute
    text_toks = set(re.findall(r"[a-zA-Z'-]+", text.lower()))
    if not text_toks & ATTRIBUTE_WORDS:
        score += 3
        diagnosis.append("no_attribute")
    # Missing session anchor
    if not SESSION_ANCHOR_RE.search(text):
        score += 2
        diagnosis.append("no_session_anchor")
    # Vague descriptors
    if VAGUE_RE.search(text):
        score += 2
        diagnosis.append("vague_descriptor")
    # Banned phrases
    low = text.lower()
    for p in BANNED_INHERITED:
        if p in low:
            score += 1
            diagnosis.append(f"banned:{p}")
            break
    # Imperative closers
    if IMPERATIVE_RE.search(text):
        score += 2
        diagnosis.append("imperative")
    # Corpus bigram density (over-represented)
    row_bgs = set(bigrams_of(text))
    density = sum(1 for bg in row_bgs if doc_counts.get(bg, 0) >= 3)
    if density >= 5:
        score += 2
        diagnosis.append(f"high_bigram_density({density})")
    elif density >= 3:
        score += 1
        diagnosis.append(f"mid_bigram_density({density})")
    return score, diagnosis


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


def build_prompt(case, top_meta, played_refs, diagnosis, banned_corpus_bigrams):
    """R87 stronger-evidence prompt."""
    title = top_meta["track_name"]
    artist = top_meta["artist_name"]
    if isinstance(title, list): title = title[0] if title else "(unknown)"
    if isinstance(artist, list): artist = artist[0] if artist else "(unknown)"
    album = top_meta.get("album_name")
    if isinstance(album, list): album = album[0] if album else "(unknown)"
    release = top_meta.get("release_date") or "(unknown)"
    tags = clean_tags(top_meta.get("tag_list") or [])

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

    diag_str = ", ".join(diagnosis) if diagnosis else "general polish"

    parts = [
        "Write exactly one natural-language music recommendation response.",
        f"PRIOR ATTEMPT WEAKNESSES: {diag_str}.",
        "",
        "Hard requirements — ALL must hold:",
        "- 70-90 words.",
        f"- First sentence: name the recommendation as {title} by {artist}.",
        "- THREE specific lift signals required (each as a distinct sentence "
        "fragment, not jammed together):",
        "  1. Why top-1 fits the request (anchor to user's stated need, "
        "naming one specific thing they asked for).",
        "  2. ONE concrete musical attribute (instrument name, production "
        "technique, era reference, rhythmic descriptor, or vocal quality) "
        "— used as prose, not as a label.",
        "  3. ONE causal/anchoring connector linking attribute back to the "
        "request (use 'because', 'since', 'where', 'after', 'anchors', "
        "'threads', 'frames', 'extends', 'builds on').",
        "- If the conversation history reveals a contrast (something the user "
        "rejected or moved away from), reference it once with 'rather than' "
        "or 'instead of' — but ONLY if it fits naturally.",
        "- Do not end with a question.",
        "- Do not use imperative closers (no 'crank it loud', 'press play', "
        "'queue this up', 'give it a spin').",
        "- Do not use vague descriptors: warm, vibrant, energetic, cozy, "
        "beautiful, gorgeous, gentle, charming, memorable, delightful, "
        "lovely, wonderful, amazing, great, nice.",
        "- Do not use boilerplate openers ('If you're looking for', "
        "'You might enjoy', 'Here's a track that').",
        "- Do not use crutches: 'perfect for', 'right in', 'lands', "
        "'delivers', 'you asked for', 'you described', 'exactly what', "
        "'makes it a'.",
        "",
        "Lexical diversity — DO NOT use any of these specific two-word "
        "phrases (over-represented in the corpus):",
        "; ".join(banned_corpus_bigrams),
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
                "and lexically diverse. Follow all hard requirements exactly."),
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip(), msg


def validate(text, doc_counts):
    issues = []
    if VAGUE_RE.search(text):
        issues.append("vague_descriptor")
    if IMPERATIVE_RE.search(text):
        issues.append("imperative_closer")
    low = text.lower()
    for p in BANNED_INHERITED:
        if p in low:
            issues.append(f"banned:{p}")
            break
    if not CAUSAL_RE.search(text):
        issues.append("no_causal_connector")
    toks = set(re.findall(r"[a-zA-Z'-]+", text.lower()))
    if not toks & ATTRIBUTE_WORDS:
        issues.append("no_concrete_attribute")
    n_words = len(text.split())
    if n_words < 60 or n_words > 110:
        issues.append(f"word_count={n_words}")
    return issues


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
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--retry-on-issues", type=int, default=1)
    args = p.parse_args()

    t0 = time.time()
    print(f"{ts()} R87 response-side push (target LLM 4.95 on R84c base)")
    print("=" * 70)

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not (args.dry_run or args.audit_only) and not api_key:
        print("ERROR: ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY missing")
        sys.exit(1)

    # Load R84c
    with zipfile.ZipFile(R84C_SUB) as z:
        r84c_items = json.loads(z.read("prediction.json"))
    r84c_by_key = {(i["session_id"], int(i["turn_number"])): i for i in r84c_items}
    print(f"  R84c: {len(r84c_items)} cases")

    # Corpus audit on R84c
    responses = [i["predicted_response"] for i in r84c_items]
    distinct2_r84c = corpus_distinct2(responses)
    doc_counts = corpus_bigram_doc_count(responses)
    print(f"\n  R84c corpus Distinct-2: {distinct2_r84c:.4f}")

    # Top over-represented bigrams (for banned list in prompt)
    banned_corpus = [
        bg for bg, c in sorted(doc_counts.items(), key=lambda x: -x[1])[:20]
        if c >= 3
    ]
    print(f"  banned corpus bigrams (>=3 docs, top-20): {banned_corpus[:10]}...")

    # Score each row by weakness
    print(f"\n{ts()} Auditing all 80 responses for LLM-judge weakness...")
    scored_rows = []
    for item in r84c_items:
        text = item["predicted_response"]
        score, diag = audit_response(text, doc_counts)
        scored_rows.append({
            "session_id": item["session_id"],
            "turn_number": int(item["turn_number"]),
            "response": text,
            "audit_score": score,
            "diagnosis": diag,
            "n_words": len(text.split()),
        })
    scored_rows.sort(key=lambda x: -x["audit_score"])
    print(f"  Top-{min(20, len(scored_rows))} weakest by audit score:")
    for r in scored_rows[:20]:
        print(f"    sid={r['session_id'][:8]}t{r['turn_number']} "
              f"score={r['audit_score']:>3} words={r['n_words']:>3} "
              f"diag={','.join(r['diagnosis'])}")

    n_target = max(MIN_REGEN_ROWS, min(args.target_rows, MAX_REGEN_ROWS))
    pick = scored_rows[:n_target]
    print(f"\n  R87 will regenerate top-{n_target} (score range: "
          f"{pick[-1]['audit_score']} – {pick[0]['audit_score']})")

    audit_out = {
        "experiment": "R87 LLM push audit",
        "created_at": datetime.now().isoformat(),
        "r84c_distinct2": distinct2_r84c,
        "n_unique_bigrams": len(doc_counts),
        "banned_corpus_bigrams_top20": banned_corpus,
        "scored_rows": scored_rows,
        "selected_for_regen": [
            {"session_id": r["session_id"], "turn_number": r["turn_number"],
             "score": r["audit_score"], "diagnosis": r["diagnosis"]}
            for r in pick
        ],
    }
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_AUDIT, "w") as f:
        json.dump(audit_out, f, indent=2)
    print(f"\n  Audit -> {OUT_AUDIT}")

    if args.audit_only:
        print("\n[audit-only — exiting]")
        return
    if args.dry_run:
        print(f"\n[dry-run — would call Opus {len(pick)} times]")
        return

    # Load catalog + blind
    print(f"\n{ts()} Loading catalog + blind...")
    meta = load_catalog_meta()
    with open(BLIND_SRC, "rb") as f:
        blind = pickle.load(f)
    played_refs_per_sid = {
        sid: {tid: short_ref(tid, meta)
              for tid in blind[sid].get("music_turns", []) if tid in meta}
        for sid in blind
    }

    # Regen
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    print(f"\n{ts()} Regenerating {len(pick)} responses with {MODEL_ID}...")
    new_responses = {}
    regen_log = []
    for n, sel in enumerate(pick):
        sid = sel["session_id"]
        case = blind.get(sid)
        if not case:
            print(f"  !! {sid[:8]}: no blind case, skipping")
            continue
        r84c_item = r84c_by_key[(sid, sel["turn_number"])]
        top1 = r84c_item["predicted_track_ids"][0]
        top_meta = meta.get(top1)
        if not top_meta:
            print(f"  !! {sid[:8]}: no top1 meta, skipping")
            continue
        prompt = build_prompt(case, top_meta, played_refs_per_sid.get(sid, {}),
                              sel["diagnosis"], banned_corpus)
        t_call = time.time()
        try:
            resp, _ = call_opus(client, prompt)
        except Exception as e:
            print(f"  !! {n+1}/{len(pick)} {sid[:8]}: API error: {e}")
            continue
        issues = validate(resp, doc_counts)
        if issues and args.retry_on_issues:
            retry_prompt = prompt + (
                f"\n\nPrior issues: {', '.join(issues)}. "
                "Rewrite exactly addressing all of these issues while keeping all "
                "other hard requirements."
            )
            try:
                resp2, _ = call_opus(client, retry_prompt)
                if len(validate(resp2, doc_counts)) < len(issues):
                    resp = resp2
                    issues = validate(resp, doc_counts)
            except Exception:
                pass
        new_responses[sid] = resp
        regen_log.append({
            "session_id": sid, "turn_number": sel["turn_number"],
            "audit_score_before": sel["audit_score"],
            "diagnosis_before": sel["diagnosis"],
            "old_response": sel["response"],
            "new_response": resp,
            "issues_after": issues,
            "n_words": len(resp.split()),
            "elapsed_s": time.time() - t_call,
        })
        print(f"  {n+1}/{len(pick)} {sid[:8]} "
              f"({len(resp.split())} words, issues={issues}, "
              f"{time.time()-t_call:.1f}s)")

    # Save regen log
    OUT_REGEN_ROWS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_REGEN_ROWS, "w") as f:
        for row in regen_log:
            f.write(json.dumps(row) + "\n")
    print(f"\n  Saved regen log -> {OUT_REGEN_ROWS}")

    # Build final submission
    final_items = []
    n_changed = 0
    for r84c in r84c_items:
        sid = r84c["session_id"]
        if sid in new_responses:
            n_changed += 1
            final_items.append({
                "session_id": sid,
                "turn_number": int(r84c["turn_number"]),
                "predicted_track_ids": r84c["predicted_track_ids"],
                "predicted_response": new_responses[sid],
            })
        else:
            final_items.append({
                "session_id": sid,
                "turn_number": int(r84c["turn_number"]),
                "predicted_track_ids": r84c["predicted_track_ids"],
                "predicted_response": r84c["predicted_response"],
            })

    # Verify tracks identical
    r84c_track_hash = hashlib.sha256(json.dumps(
        [(i["session_id"], i["turn_number"], i["predicted_track_ids"]) for i in r84c_items],
        sort_keys=True,
    ).encode()).hexdigest()
    r87_track_hash = hashlib.sha256(json.dumps(
        [(i["session_id"], i["turn_number"], i["predicted_track_ids"]) for i in final_items],
        sort_keys=True,
    ).encode()).hexdigest()
    tracks_identical = r84c_track_hash == r87_track_hash
    print(f"\n  Track hash R84c vs R87: {'IDENTICAL ✓' if tracks_identical else 'DIFFER ✗'}")

    # Distinct-2 proxy
    distinct2_r87 = corpus_distinct2([i["predicted_response"] for i in final_items])
    print(f"  R84c local Distinct-2: {distinct2_r84c:.4f}")
    print(f"  R87  local Distinct-2: {distinct2_r87:.4f}  (Δ {distinct2_r87 - distinct2_r84c:+.4f})")
    distinct2_gate = distinct2_r87 >= distinct2_r84c
    print(f"  Gate (R87 D-2 >= R84c): {'PASS' if distinct2_gate else 'fail'}")

    # Issues across regen
    n_issues_total = sum(len(r["issues_after"]) for r in regen_log)
    n_clean = sum(1 for r in regen_log if not r["issues_after"])
    print(f"  Regen validation: {n_clean}/{len(regen_log)} clean "
          f"({n_issues_total} total issues across {len(regen_log) - n_clean} rows)")

    # Save submission
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload_str = json.dumps(final_items, indent=2)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("prediction.json", payload_str)
    sha = hashlib.sha256(open(OUT_ZIP, "rb").read()).hexdigest()
    print(f"\n  Wrote {OUT_ZIP} ({OUT_ZIP.stat().st_size} bytes)")
    print(f"  sha256: {sha}")

    meta_out = {
        "experiment": "R87 LLM-judge push on R84c (tracks identical)",
        "created_at": datetime.now().isoformat(),
        "head_sha": head_sha(),
        "model": MODEL_ID,
        "base_submission": "r84c_selective_submission.zip",
        "n_cases": len(final_items),
        "n_responses_regenerated": n_changed,
        "n_responses_unchanged": len(final_items) - n_changed,
        "tracks_identical_to_r84c": tracks_identical,
        "r84c_local_distinct2": distinct2_r84c,
        "r87_local_distinct2": distinct2_r87,
        "lexdiv_local_gate_pass": distinct2_gate,
        "regen_clean": n_clean,
        "regen_total": len(regen_log),
        "submission_sha256": sha,
        "submission_path": str(OUT_ZIP.relative_to(REPO)),
    }
    with open(OUT_METADATA, "w") as f:
        json.dump(meta_out, f, indent=2)

    md = [
        "# R87 LLM-Judge Push on R84c — Result",
        "",
        f"HEAD: `{meta_out['head_sha'][:10]}`",
        f"Base: `r84c_selective_submission.zip` (production, composite 0.6362)",
        f"R87: `{OUT_ZIP.name}` ({OUT_ZIP.stat().st_size} bytes, sha256 `{sha[:16]}`)",
        "",
        "## Composition",
        f"- 80 cases total",
        f"- **{n_changed}** responses regenerated (weakest by LLM-judge audit)",
        f"- **{len(final_items) - n_changed}** responses unchanged from R84c",
        f"- Tracks bitwise identical: {'YES ✓' if tracks_identical else 'NO ✗'}",
        "",
        "## Local LexDiv (proxy)",
        "",
        "| metric | R84c | R87 | Δ |",
        "|---|---:|---:|---:|",
        f"| local Distinct-2 | {distinct2_r84c:.4f} | {distinct2_r87:.4f} | "
        f"{distinct2_r87 - distinct2_r84c:+.4f} |",
        "",
        f"Gate (R87 ≥ R84c): **{'PASS' if distinct2_gate else 'FAIL'}**",
        f"Caveat: per feedback_local_distinct2_doesnt_predict_lexdiv, local "
        "D-2 doesn't predict competition LexDiv. Use as directional only.",
        "",
        "## Regenerated rows",
        "",
        "| session_id | turn | audit_score_before | diagnosis | words | issues_after |",
        "|---|---:|---:|---|---:|---|",
    ]
    for row in regen_log:
        md.append(
            f"| `{row['session_id'][:12]}` | {row['turn_number']} | "
            f"{row['audit_score_before']} | "
            f"{','.join(row['diagnosis_before'])} | "
            f"{row['n_words']} | "
            f"{','.join(row['issues_after']) or '—'} |"
        )
    md += [
        "",
        "## Submission gate",
        "",
        "- Tracks bitwise identical to R84c (zero nDCG risk).",
        f"- Local Distinct-2 proxy: {'PASS' if distinct2_gate else 'FAIL'}.",
        f"- Regen validation: {n_clean}/{len(regen_log)} rows clean.",
        "- Expected risk: LLM judge could go up (4.90 → 4.95 target) or stay "
        "flat (R78 ceiling). LexDiv could regress at competition scorer "
        "despite local-D-2 maintained.",
        "",
        "**NOT auto-uploaded.** User decides whether to submit.",
        "",
        "**Codabench scorer still has Gemini deprecation issue as of R86 "
        "submission (2026-05-25). Verify scorer is fixed before submitting.**",
    ]
    OUT_DIFF_MD.write_text("\n".join(md) + "\n")
    print(f"  Diff report -> {OUT_DIFF_MD}")
    print(f"  Metadata -> {OUT_METADATA}")
    print(f"\nTotal elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
