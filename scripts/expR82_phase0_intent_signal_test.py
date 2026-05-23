#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R82 Phase 0 — LLM intent feature signal test (cheap, no training).

For each fold-0 h7 case:
1. Build LLM intent JSON from the query + conversation (Opus 4.7).
2. For GT and R54c top-20 false positives, compute candidate-level
   intent-match scores using existing track metadata (tags, artist, year).
3. Test pairwise separation: does GT have higher avg match score than
   R54c top-1 false positives? Than top-20 false positives?

Gate: if GT does NOT significantly outscore R54c false positives, archive.

No model training. Just intent extraction + scalar match features.
~200 LLM calls. Total cost: ~$1-3.
"""
from __future__ import annotations
import json
import math
import os
import pickle
import re
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Load training_pairs from R79 for fold mapping + R54c top-20 per case
TRAINING_PAIRS = REPO / "cache" / "r79" / "training_pairs.pkl"
R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"

OUT_DIR = REPO / "cache" / "r82"
OUT_INTENTS = OUT_DIR / "intents_fold0_h7.json"
OUT_FEATURES = OUT_DIR / "candidate_features_fold0_h7.json"
OUT_RESULT = REPO / "exp" / "eval" / "expR82_phase0_signal_test.json"
OUT_DOC = REPO / "docs" / "r82_phase0_signal_test.md"

MODEL_ID = "claude-opus-4-7"
MAX_TOKENS = 600
DEFAULT_SAMPLE = 50  # Start with 50 for cheap signal check; expand if positive

INTENT_SYSTEM = (
    "You extract structured listening intent from a music recommendation "
    "conversation. Output ONLY a JSON object with keys: mood (1-3 short "
    "descriptors), genre (1-3 genres or 'any'), era (decade or 'any', e.g. "
    "'1990s', '2010s'), language (e.g. 'english', 'spanish', 'any'), "
    "energy ('low', 'medium', 'high', 'any'), artist_relation ('same as "
    "previous', 'different but similar', 'completely new', 'any'), "
    "novelty ('familiar', 'discover', 'any'), constraints (list of "
    "specific requirements or empty list). No other text. Output must "
    "be valid JSON."
)


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def build_query_prompt(case: dict, history_text: str, played_summary: str) -> str:
    """Build the query text passed to the LLM for intent extraction."""
    return (
        f"User query: {case['user_query']}\n"
        f"Played so far (artists): {played_summary}\n"
        f"Recent conversation: {history_text}\n\n"
        f"Extract listening intent as JSON."
    )


def parse_intent_json(text: str) -> dict | None:
    """Best-effort JSON parser; returns None on failure."""
    # Trim markdown if present
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                return None
        return None


def extract_year(release_date: str) -> int | None:
    if not release_date or len(release_date) < 4:
        return None
    try:
        return int(release_date[:4])
    except (ValueError, TypeError):
        return None


def year_to_decade(year: int) -> str:
    return f"{(year // 10) * 10}s"


def normalize(s) -> str:
    if isinstance(s, list):
        s = s[0] if s else ""
    if not isinstance(s, str):
        s = str(s)
    return s.lower().strip()


def compute_match_features(
    intent: dict | None, track_meta: dict, query_played_artists: list[str]
) -> dict:
    """Compute per-candidate scalar match features.

    Higher score = better match to intent.
    """
    if intent is None:
        return {
            "mood_match": 0.0, "genre_match": 0.0, "era_match": 0.0,
            "language_match": 0.0, "energy_match": 0.0, "artist_rel_match": 0.0,
            "constraints_match": 0.0, "total": 0.0,
        }

    track_tags = [normalize(t) for t in (track_meta.get("tag_list") or []) if t]
    raw_artist = track_meta.get("artist_name") or ""
    if isinstance(raw_artist, list):
        raw_artist = raw_artist[0] if raw_artist else ""
    track_artist = normalize(raw_artist)

    track_year = extract_year(track_meta.get("release_date") or "")
    track_decade = year_to_decade(track_year) if track_year else None

    # Mood match: any of intent's mood descriptors appear in track tags
    mood = intent.get("mood") or []
    if isinstance(mood, str):
        mood = [mood]
    mood_norm = [normalize(m) for m in mood if m]
    mood_match = 0.0
    if mood_norm and track_tags:
        mood_hits = sum(1 for m in mood_norm if any(m in t or t in m for t in track_tags))
        mood_match = mood_hits / max(len(mood_norm), 1)

    # Genre match
    genre = intent.get("genre") or []
    if isinstance(genre, str):
        genre = [genre]
    genre_norm = [normalize(g) for g in genre if g and normalize(g) != "any"]
    genre_match = 0.0
    if genre_norm and track_tags:
        # tags often contain genre info
        genre_hits = sum(1 for g in genre_norm if any(g in t or t in g for t in track_tags))
        genre_match = genre_hits / max(len(genre_norm), 1)
    elif not genre_norm or (len(genre_norm) == 1 and genre_norm[0] == "any"):
        genre_match = 0.5  # neutral if no preference

    # Era match
    era = normalize(intent.get("era") or "")
    era_match = 0.0
    if era and era != "any" and track_decade:
        era_match = 1.0 if track_decade in era or era in track_decade else 0.0
    elif era == "any" or not era:
        era_match = 0.5

    # Language match (heuristic: check track tags for language hints or use artist name)
    language = normalize(intent.get("language") or "")
    language_match = 0.5  # neutral default
    if language and language != "any" and track_tags:
        if language in track_tags or any(language in t for t in track_tags):
            language_match = 1.0

    # Energy match (look for energy keywords in tags)
    energy = normalize(intent.get("energy") or "")
    energy_keywords = {
        "high": ["energetic", "fast", "upbeat", "intense", "powerful", "aggressive", "dance"],
        "medium": ["mid-tempo", "moderate", "balanced", "rhythmic"],
        "low": ["slow", "mellow", "soft", "calm", "ambient", "quiet", "soothing"],
    }
    energy_match = 0.5
    if energy and energy != "any" and track_tags:
        kws = energy_keywords.get(energy, [])
        energy_match = 1.0 if any(kw in t for t in track_tags for kw in kws) else 0.0

    # Artist relation match
    artist_rel = normalize(intent.get("artist_relation") or "")
    artist_rel_match = 0.5
    played_artists_norm = [normalize(a) for a in query_played_artists if a]
    if artist_rel == "same as previous":
        artist_rel_match = 1.0 if track_artist in played_artists_norm else 0.0
    elif artist_rel == "different but similar":
        artist_rel_match = 1.0 if track_artist not in played_artists_norm else 0.3
    elif artist_rel == "completely new":
        artist_rel_match = 1.0 if track_artist not in played_artists_norm else 0.0

    # Constraints (skip for now — too specific to model)
    constraints_match = 0.5

    total = (mood_match + genre_match + era_match + language_match +
             energy_match + artist_rel_match + constraints_match)
    return {
        "mood_match": mood_match,
        "genre_match": genre_match,
        "era_match": era_match,
        "language_match": language_match,
        "energy_match": energy_match,
        "artist_rel_match": artist_rel_match,
        "constraints_match": constraints_match,
        "total": total,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                        help="Number of h7 fold-0 cases to sample (cheap mode = 50)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_RECSYS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ANTHROPIC_API_KEY/ANTHROPIC_RECSYS_API_KEY")

    import anthropic

    t0 = time.time()
    print(f"{ts()} R82 Phase 0 — LLM intent feature signal test")
    print(f"  sample={args.sample} cases (h7 fold-0)")
    print("=" * 70)

    print(f"{ts()} Loading R79 training pairs (for case metadata + R54c top-20) ...",
          flush=True)
    with open(TRAINING_PAIRS, "rb") as f:
        data = pickle.load(f)
    pairs_all = data["training_pairs"]
    fold0_h7 = [p for p in pairs_all if p["fold"] == 0 and p["is_h7"]]
    print(f"  fold-0 h7 cases: {len(fold0_h7)}")

    # Sample N cases for cheap signal test
    import random
    random.seed(0)
    sample = random.sample(fold0_h7, min(args.sample, len(fold0_h7)))
    print(f"  sampled: {len(sample)}")

    print(f"{ts()} Loading catalog from HuggingFace ...", flush=True)
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig())["all_tracks"]
    catalog = {}
    for item in ds:
        tid = str(item["track_id"])
        catalog[tid] = {
            "track_name": item.get("track_name"),
            "artist_name": item.get("artist_name"),
            "album_name": item.get("album_name"),
            "tag_list": item.get("tag_list") or item.get("tags") or [],
            "release_date": item.get("release_date") or "",
        }
    print(f"  catalog: {len(catalog)} tracks")

    # Build played artists per case from training_pairs
    payload = pickle.load(open(R12_CACHE, "rb"))
    cases = payload["cases"]
    track_artist_map = payload.get("track_artist", {})

    # ---- Extract LLM intent per case (or load cache) ----
    intents = {}
    n_failed = 0
    api_calls = 0
    in_tokens = 0
    out_tokens = 0

    if OUT_INTENTS.exists():
        print(f"\n{ts()} === Loading cached intents from {OUT_INTENTS} ===")
        cached = json.loads(OUT_INTENTS.read_text())
        # cached keys are case_idx as strings
        intents = {int(k): v for k, v in cached.items()}
        # Filter to current sample's case_ids
        sample_ids = {sp["case_idx"] for sp in sample}
        intents = {k: v for k, v in intents.items() if k in sample_ids}
        n_failed = sum(1 for v in intents.values() if not v.get("parsed"))
        print(f"  cached intents loaded: {len(intents)} (failed: {n_failed})")

    if not intents or len(intents) < len(sample):
        client = anthropic.Anthropic(api_key=api_key)
        print(f"\n{ts()} === Extracting LLM intent for {len(sample) - len(intents)} new cases ===")
        for ki, sp in enumerate(sample):
            if sp["case_idx"] in intents:
                continue
        case_idx = sp["case_idx"]
        case = cases[case_idx]
        history_text = sp.get("history_text", "")[:2000]  # truncate to keep prompt small
        # Played artists
        played_artists = []
        for pt in sp.get("played_tracks", []):
            a = track_artist_map.get(pt)
            if isinstance(a, str):
                played_artists.append(a)
            elif isinstance(a, list) and a:
                played_artists.append(a[0])
        played_summary = ", ".join(played_artists[-5:]) if played_artists else "(none)"

        prompt = build_query_prompt(case, history_text, played_summary)
        try:
            msg = client.messages.create(
                model=MODEL_ID, max_tokens=MAX_TOKENS,
                system=INTENT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            api_calls += 1
            if msg.usage:
                in_tokens += msg.usage.input_tokens
                out_tokens += msg.usage.output_tokens
            text = "".join(p.text for p in msg.content if hasattr(p, "text"))
            intent = parse_intent_json(text)
            if intent is None:
                n_failed += 1
                print(f"    case {ki+1}/{len(sample)} idx={case_idx} FAILED to parse JSON",
                      flush=True)
                intents[case_idx] = {"raw": text, "parsed": None,
                                     "played_artists": played_artists}
            else:
                intents[case_idx] = {"raw": text, "parsed": intent,
                                     "played_artists": played_artists}
        except Exception as e:
            n_failed += 1
            print(f"    case {ki+1}/{len(sample)} idx={case_idx} API ERROR: {str(e)[:100]}",
                  flush=True)
            intents[case_idx] = {"raw": "", "parsed": None,
                                 "played_artists": played_artists}
        if (ki + 1) % 10 == 0:
            print(f"    extracted {ki+1}/{len(sample)} (api_calls={api_calls})", flush=True)

    print(f"  intents extracted: {len(intents) - n_failed}/{len(sample)}")
    print(f"  API calls: {api_calls}  in_tokens: {in_tokens}  out_tokens: {out_tokens}")
    cost_est = in_tokens * 15 / 1e6 + out_tokens * 75 / 1e6
    print(f"  cost estimate: ${cost_est:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_INTENTS.write_text(json.dumps(intents, indent=2))
    print(f"  saved → {OUT_INTENTS}")

    # ---- Compute match features for GT + R54c top-20 ----
    print(f"\n{ts()} === Compute candidate match features ===")
    per_case_features = []
    for sp in sample:
        case_idx = sp["case_idx"]
        intent_data = intents[case_idx]
        intent = intent_data["parsed"]
        played_artists = intent_data["played_artists"]
        gt = sp["gt"]
        top20 = sp["oof_top20"]
        gt_in_top20 = gt in set(top20)

        # GT features
        gt_meta = catalog.get(gt, {})
        gt_feats = compute_match_features(intent, gt_meta, played_artists) if intent else None

        # R54c top-20 (excluding GT)
        false_positives = [t for t in top20 if t != gt]
        fp_feats = []
        for fp in false_positives:
            fp_meta = catalog.get(fp, {})
            fp_feats.append(compute_match_features(intent, fp_meta, played_artists) if intent else None)

        per_case_features.append({
            "case_idx": case_idx,
            "intent_parsed": intent is not None,
            "gt_in_top20": gt_in_top20,
            "gt_features": gt_feats,
            "false_positive_features": fp_feats,
            "n_false_positives": len(false_positives),
        })

    OUT_FEATURES.write_text(json.dumps(per_case_features, indent=2))
    print(f"  saved → {OUT_FEATURES}")

    # ---- Pairwise separation test ----
    print(f"\n{ts()} === Pairwise separation: GT vs R54c false positives ===")
    valid_cases = [c for c in per_case_features if c["intent_parsed"] and c["gt_features"]]
    n_valid = len(valid_cases)
    print(f"  valid cases (intent parsed): {n_valid}/{len(sample)}")

    if n_valid < 20:
        print(f"  NOT ENOUGH VALID CASES — likely LLM JSON parse failures")
        verdict = "ARCHIVE_PARSE_FAILED"
    else:
        # Test 1: avg total match score (GT vs top-1 FP)
        gt_totals = [c["gt_features"]["total"] for c in valid_cases]
        top1_fp_totals = [c["false_positive_features"][0]["total"]
                          for c in valid_cases if c["false_positive_features"]]
        all_fp_totals = [fp["total"] for c in valid_cases for fp in c["false_positive_features"]]

        gt_mean = float(np.mean(gt_totals))
        top1_fp_mean = float(np.mean(top1_fp_totals))
        all_fp_mean = float(np.mean(all_fp_totals))

        # Test 2: per-case pairwise win rate (GT > top-1 FP)
        gt_wins_top1 = sum(1 for c in valid_cases
                           if c["false_positive_features"] and
                           c["gt_features"]["total"] > c["false_positive_features"][0]["total"])
        gt_ties_top1 = sum(1 for c in valid_cases
                           if c["false_positive_features"] and
                           c["gt_features"]["total"] == c["false_positive_features"][0]["total"])
        n_with_fp = sum(1 for c in valid_cases if c["false_positive_features"])
        gt_win_rate_top1 = gt_wins_top1 / max(n_with_fp, 1)

        # Test 3: per-case: GT outranks ALL R54c FPs (best-case)
        gt_beats_all_fps = 0
        gt_beats_any_fp = 0
        for c in valid_cases:
            if not c["false_positive_features"]:
                continue
            gt_score = c["gt_features"]["total"]
            fp_scores = [fp["total"] for fp in c["false_positive_features"]]
            if gt_score > max(fp_scores):
                gt_beats_all_fps += 1
            if gt_score > min(fp_scores):
                gt_beats_any_fp += 1
        beats_all_rate = gt_beats_all_fps / max(n_with_fp, 1)
        beats_any_rate = gt_beats_any_fp / max(n_with_fp, 1)

        # Per-feature breakdown
        feature_names = ["mood_match", "genre_match", "era_match", "language_match",
                         "energy_match", "artist_rel_match"]
        per_feature_gt_means = {f: float(np.mean([c["gt_features"][f] for c in valid_cases]))
                                for f in feature_names}
        per_feature_fp_means = {f: float(np.mean([fp[f] for c in valid_cases
                                                  for fp in c["false_positive_features"]]))
                                for f in feature_names}

        print(f"\n  Total match score:")
        print(f"    GT mean:                  {gt_mean:.3f}")
        print(f"    R54c top-1 FP mean:       {top1_fp_mean:.3f}")
        print(f"    R54c all FP mean:         {all_fp_mean:.3f}")
        print(f"  GT > top-1 FP per case: {gt_wins_top1}/{n_with_fp} = {gt_win_rate_top1:.3f}")
        print(f"  GT > all FPs per case:  {gt_beats_all_fps}/{n_with_fp} = {beats_all_rate:.3f}")
        print(f"  GT > any FP per case:   {gt_beats_any_fp}/{n_with_fp} = {beats_any_rate:.3f}")
        print(f"\n  Per-feature GT vs FP means:")
        for f in feature_names:
            print(f"    {f:20}  GT={per_feature_gt_means[f]:.3f}  FP={per_feature_fp_means[f]:.3f}  "
                  f"Δ={per_feature_gt_means[f]-per_feature_fp_means[f]:+.3f}")

        # Gate: GT must outscore top-1 FP > 50% (chance baseline) by margin
        # AND GT total mean > FP total mean by ≥0.1
        gate_win_rate = gt_win_rate_top1 >= 0.55  # 5pt above chance
        gate_mean_gap = (gt_mean - all_fp_mean) >= 0.1
        if gate_win_rate and gate_mean_gap:
            verdict = "PROCEED_PHASE_1"
        elif gt_win_rate_top1 >= 0.52 and (gt_mean - all_fp_mean) >= 0.05:
            verdict = "PROCEED_EXPLORATORY"
        else:
            verdict = "ARCHIVE"

    print(f"\n  VERDICT: {verdict}", flush=True)

    out = {
        "experiment": "R82 Phase 0 — LLM intent feature signal test",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "verdict": verdict,
        "sample_size": len(sample),
        "n_intents_parsed": n_valid if 'n_valid' in dir() else 0,
        "api_calls": api_calls,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "cost_estimate_usd": cost_est,
    }
    if verdict not in ["ARCHIVE_PARSE_FAILED"]:
        out["metrics"] = {
            "gt_mean_total": gt_mean,
            "top1_fp_mean_total": top1_fp_mean,
            "all_fp_mean_total": all_fp_mean,
            "gt_win_rate_vs_top1_fp": gt_win_rate_top1,
            "gt_beats_all_fps_rate": beats_all_rate,
            "gt_beats_any_fp_rate": beats_any_rate,
            "per_feature_gt_means": per_feature_gt_means,
            "per_feature_fp_means": per_feature_fp_means,
        }

    OUT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESULT.write_text(json.dumps(out, indent=2))
    print(f"\n{ts()} Saved → {OUT_RESULT}")

    md = [
        "# R82 Phase 0 — LLM intent feature signal test",
        "",
        f"Elapsed: {out['elapsed_s']:.0f}s  API cost: ${cost_est:.2f}",
        f"Sample: {len(sample)} cases (h7 fold-0)",
        f"Intents parsed: {out.get('n_intents_parsed', 0)}/{len(sample)}",
        "",
        f"## Verdict: **{verdict}**",
        "",
    ]
    if "metrics" in out:
        m = out["metrics"]
        md += [
            "## Pairwise separation: GT vs R54c false positives",
            "",
            f"- GT mean total match score: **{m['gt_mean_total']:.3f}**",
            f"- R54c top-1 FP mean: **{m['top1_fp_mean_total']:.3f}**",
            f"- All FPs mean: **{m['all_fp_mean_total']:.3f}**",
            f"- GT > top-1 FP rate: **{m['gt_win_rate_vs_top1_fp']:.3f}** (chance = 0.5)",
            f"- GT > all FPs rate: **{m['gt_beats_all_fps_rate']:.3f}**",
            f"- GT > any FP rate: **{m['gt_beats_any_fp_rate']:.3f}**",
            "",
            "## Per-feature comparison",
            "",
            "| Feature | GT mean | FP mean | Δ |",
            "|---|---:|---:|---:|",
        ]
        for f in m["per_feature_gt_means"]:
            md.append(f"| {f} | {m['per_feature_gt_means'][f]:.3f} | "
                      f"{m['per_feature_fp_means'][f]:.3f} | "
                      f"{m['per_feature_gt_means'][f] - m['per_feature_fp_means'][f]:+.3f} |")
    OUT_DOC.write_text("\n".join(md) + "\n")
    print(f"{ts()} Saved → {OUT_DOC}")


if __name__ == "__main__":
    main()
