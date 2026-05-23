#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R82 Phase 0 — Eval only: load cached intents + run signal test.

Skips the LLM call (uses cache/r82/intents_fold0_h7.json) and
re-runs feature computation + pairwise separation test.
"""
from __future__ import annotations
import json
import pickle
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # type: ignore[reportMissingImports]

OUT_INTENTS = REPO / "cache" / "r82" / "intents_fold0_h7.json"
OUT_FEATURES = REPO / "cache" / "r82" / "candidate_features_fold0_h7.json"
OUT_RESULT = REPO / "exp" / "eval" / "expR82_phase0_signal_test.json"
OUT_DOC = REPO / "docs" / "r82_phase0_signal_test.md"
TRAINING_PAIRS = REPO / "cache" / "r79" / "training_pairs.pkl"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def normalize(s):
    if isinstance(s, list):
        s = s[0] if s else ""
    if not isinstance(s, str):
        s = str(s)
    return s.lower().strip()


def extract_year(release_date):
    if not release_date or len(str(release_date)) < 4:
        return None
    try:
        return int(str(release_date)[:4])
    except (ValueError, TypeError):
        return None


def year_to_decade(year):
    return f"{(year // 10) * 10}s"


def compute_match_features(intent, track_meta, query_played_artists):
    if intent is None:
        return {"mood_match": 0, "genre_match": 0, "era_match": 0,
                "language_match": 0, "energy_match": 0, "artist_rel_match": 0,
                "constraints_match": 0, "total": 0}

    track_tags = [normalize(t) for t in (track_meta.get("tag_list") or []) if t]
    raw_artist = track_meta.get("artist_name") or ""
    track_artist = normalize(raw_artist)
    track_year = extract_year(track_meta.get("release_date") or "")
    track_decade = year_to_decade(track_year) if track_year else None

    # Mood
    mood = intent.get("mood") or []
    if isinstance(mood, str): mood = [mood]
    mood_norm = [normalize(m) for m in mood if m]
    mood_match = 0.0
    if mood_norm and track_tags:
        hits = sum(1 for m in mood_norm if any(m in t or t in m for t in track_tags))
        mood_match = hits / max(len(mood_norm), 1)

    # Genre
    genre = intent.get("genre") or []
    if isinstance(genre, str): genre = [genre]
    genre_norm = [normalize(g) for g in genre if g and normalize(g) != "any"]
    genre_match = 0.0
    if genre_norm and track_tags:
        hits = sum(1 for g in genre_norm if any(g in t or t in g for t in track_tags))
        genre_match = hits / max(len(genre_norm), 1)
    elif not genre_norm:
        genre_match = 0.5

    # Era
    era = normalize(intent.get("era") or "")
    era_match = 0.0
    if era and era != "any" and track_decade:
        era_match = 1.0 if track_decade in era or era in track_decade else 0.0
    elif era == "any" or not era:
        era_match = 0.5

    # Language
    language = normalize(intent.get("language") or "")
    language_match = 0.5
    if language and language != "any" and track_tags:
        if language in track_tags or any(language in t for t in track_tags):
            language_match = 1.0

    # Energy
    energy = normalize(intent.get("energy") or "")
    energy_kw = {
        "high": ["energetic", "fast", "upbeat", "intense", "powerful", "aggressive", "dance"],
        "medium": ["mid-tempo", "moderate", "balanced", "rhythmic"],
        "low": ["slow", "mellow", "soft", "calm", "ambient", "quiet", "soothing"],
    }
    energy_match = 0.5
    if energy and energy != "any" and track_tags:
        kws = energy_kw.get(energy, [])
        energy_match = 1.0 if any(kw in t for t in track_tags for kw in kws) else 0.0

    # Artist relation
    artist_rel = normalize(intent.get("artist_relation") or "")
    played_artists_norm = [normalize(a) for a in query_played_artists if a]
    artist_rel_match = 0.5
    if artist_rel == "same as previous":
        artist_rel_match = 1.0 if track_artist in played_artists_norm else 0.0
    elif artist_rel in ("different but similar", "completely new"):
        artist_rel_match = 1.0 if track_artist not in played_artists_norm else 0.0

    constraints_match = 0.5
    total = (mood_match + genre_match + era_match + language_match +
             energy_match + artist_rel_match + constraints_match)
    return {
        "mood_match": mood_match, "genre_match": genre_match,
        "era_match": era_match, "language_match": language_match,
        "energy_match": energy_match, "artist_rel_match": artist_rel_match,
        "constraints_match": constraints_match, "total": total,
    }


def main():
    t0 = time.time()
    print(f"{ts()} R82 Phase 0 — Eval only (using cached intents)")

    intents_raw = json.loads(OUT_INTENTS.read_text())
    intents = {int(k): v for k, v in intents_raw.items()}
    print(f"  loaded {len(intents)} cached intents")

    with open(TRAINING_PAIRS, "rb") as f:
        data = pickle.load(f)
    pairs_all = data["training_pairs"]
    sample = [p for p in pairs_all if p["case_idx"] in intents]
    print(f"  matched {len(sample)} training pair cases")

    # Catalog
    from datasets import DownloadConfig, load_dataset
    print(f"{ts()} Loading catalog from HuggingFace ...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                      download_config=DownloadConfig())["all_tracks"]
    catalog = {}
    for item in ds:
        catalog[str(item["track_id"])] = {
            "artist_name": item.get("artist_name"),
            "tag_list": item.get("tag_list") or item.get("tags") or [],
            "release_date": item.get("release_date") or "",
        }
    print(f"  catalog: {len(catalog)} tracks")

    print(f"\n{ts()} Compute candidate match features ...")
    per_case = []
    for sp in sample:
        case_idx = sp["case_idx"]
        idata = intents[case_idx]
        intent = idata.get("parsed")
        played_artists = idata.get("played_artists", [])
        gt = sp["gt"]
        top20 = sp["oof_top20"]
        gt_meta = catalog.get(gt, {})
        gt_feats = compute_match_features(intent, gt_meta, played_artists) if intent else None
        fp_feats = []
        for fp in top20:
            if fp == gt: continue
            fp_meta = catalog.get(fp, {})
            fp_feats.append(compute_match_features(intent, fp_meta, played_artists) if intent else None)
        per_case.append({
            "case_idx": case_idx,
            "intent_parsed": intent is not None,
            "gt_in_top20": gt in set(top20),
            "gt_features": gt_feats,
            "false_positive_features": fp_feats,
            "n_false_positives": len(fp_feats),
        })

    OUT_FEATURES.write_text(json.dumps(per_case, indent=2))
    print(f"  saved → {OUT_FEATURES}")

    # Pairwise separation
    print(f"\n{ts()} === Pairwise separation: GT vs R54c false positives ===")
    valid = [c for c in per_case if c["intent_parsed"] and c["gt_features"]]
    print(f"  valid cases: {len(valid)}/{len(per_case)}")

    if len(valid) < 20:
        print("  NOT ENOUGH VALID CASES — archive")
        verdict = "ARCHIVE_PARSE_FAILED"
        out = {"verdict": verdict, "elapsed_s": time.time() - t0,
               "sample_size": len(per_case), "n_valid": len(valid)}
    else:
        gt_totals = [c["gt_features"]["total"] for c in valid]
        top1_fp_totals = [c["false_positive_features"][0]["total"]
                          for c in valid if c["false_positive_features"]]
        all_fp_totals = [fp["total"] for c in valid for fp in c["false_positive_features"]]

        gt_mean = float(np.mean(gt_totals))
        top1_fp_mean = float(np.mean(top1_fp_totals)) if top1_fp_totals else 0.0
        all_fp_mean = float(np.mean(all_fp_totals)) if all_fp_totals else 0.0

        n_with_fp = sum(1 for c in valid if c["false_positive_features"])
        gt_wins_top1 = sum(1 for c in valid if c["false_positive_features"]
                           and c["gt_features"]["total"] > c["false_positive_features"][0]["total"])
        gt_win_rate_top1 = gt_wins_top1 / max(n_with_fp, 1)

        gt_beats_all_fps = sum(
            1 for c in valid if c["false_positive_features"]
            and c["gt_features"]["total"] > max(fp["total"] for fp in c["false_positive_features"]))
        gt_beats_any_fp = sum(
            1 for c in valid if c["false_positive_features"]
            and c["gt_features"]["total"] > min(fp["total"] for fp in c["false_positive_features"]))
        beats_all_rate = gt_beats_all_fps / max(n_with_fp, 1)
        beats_any_rate = gt_beats_any_fp / max(n_with_fp, 1)

        feat_names = ["mood_match", "genre_match", "era_match", "language_match",
                      "energy_match", "artist_rel_match"]
        gt_per_f = {f: float(np.mean([c["gt_features"][f] for c in valid])) for f in feat_names}
        fp_per_f = {f: float(np.mean([fp[f] for c in valid for fp in c["false_positive_features"]]))
                    for f in feat_names}

        print(f"\n  Total match score:")
        print(f"    GT mean:                  {gt_mean:.3f}")
        print(f"    R54c top-1 FP mean:       {top1_fp_mean:.3f}")
        print(f"    R54c all FP mean:         {all_fp_mean:.3f}")
        print(f"  GT > top-1 FP per case:    {gt_wins_top1}/{n_with_fp} = {gt_win_rate_top1:.3f}")
        print(f"  GT > all FPs per case:     {gt_beats_all_fps}/{n_with_fp} = {beats_all_rate:.3f}")
        print(f"  GT > any FP per case:      {gt_beats_any_fp}/{n_with_fp} = {beats_any_rate:.3f}")
        print(f"\n  Per-feature GT vs FP means:")
        for f in feat_names:
            print(f"    {f:20}  GT={gt_per_f[f]:.3f}  FP={fp_per_f[f]:.3f}  "
                  f"Δ={gt_per_f[f]-fp_per_f[f]:+.3f}")

        gate_win = gt_win_rate_top1 >= 0.55
        gate_gap = (gt_mean - all_fp_mean) >= 0.1
        if gate_win and gate_gap:
            verdict = "PROCEED_PHASE_1"
        elif gt_win_rate_top1 >= 0.52 and (gt_mean - all_fp_mean) >= 0.05:
            verdict = "PROCEED_EXPLORATORY"
        else:
            verdict = "ARCHIVE"

        out = {
            "verdict": verdict, "elapsed_s": time.time() - t0,
            "sample_size": len(per_case), "n_valid": len(valid),
            "metrics": {
                "gt_mean_total": gt_mean,
                "top1_fp_mean_total": top1_fp_mean,
                "all_fp_mean_total": all_fp_mean,
                "gt_win_rate_vs_top1_fp": gt_win_rate_top1,
                "gt_beats_all_fps_rate": beats_all_rate,
                "gt_beats_any_fp_rate": beats_any_rate,
                "per_feature_gt_means": gt_per_f,
                "per_feature_fp_means": fp_per_f,
            },
        }

    print(f"\n  VERDICT: {verdict}", flush=True)

    OUT_RESULT.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESULT.write_text(json.dumps(out, indent=2))
    print(f"{ts()} Saved → {OUT_RESULT}")

    md = [
        "# R82 Phase 0 — LLM intent feature signal test",
        "",
        f"Sample: {len(per_case)} cases, valid (intent parsed): {out.get('n_valid', 0)}",
        f"## Verdict: **{verdict}**",
        "",
    ]
    if "metrics" in out:
        m = out["metrics"]
        md += [
            "## Pairwise separation: GT vs R54c false positives",
            "",
            f"- GT mean total: **{m['gt_mean_total']:.3f}**",
            f"- R54c top-1 FP mean: **{m['top1_fp_mean_total']:.3f}**",
            f"- All FPs mean: **{m['all_fp_mean_total']:.3f}**",
            f"- GT > top-1 FP rate: **{m['gt_win_rate_vs_top1_fp']:.3f}** (chance = 0.5)",
            f"- GT > all FPs rate: **{m['gt_beats_all_fps_rate']:.3f}**",
            f"- GT > any FP rate: **{m['gt_beats_any_fp_rate']:.3f}**",
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
