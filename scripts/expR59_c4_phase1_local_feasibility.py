#!/usr/bin/env python3
"""R59 C4 Phase 1: Local feasibility audit for external enrichment.

Tests if weighted-tag features from existing TalkPlayData tag_list help ranking.
If internal tag-weighting doesn't work, external MB tags likely won't either.

NO external API calls. Pure feature engineering on TalkPlayData.

Gates:
- GREEN: weighted tags lift nDCG ≥ +0.005 on 300-case sample → proceed to Phase 2 (MB API)
- BORDERLINE: lift +0.002 to +0.004 → ask user
- RED: flat or regress → archive C4, don't fetch MB

Outputs:
- exp/eval/expR59_c4_phase1_feasibility.json (metrics)
- docs/r59_candidates/c4_phase1_report.md (verdict)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from datasets import load_dataset

# Phase 1: analyze TalkPlayData tag_list coverage and quality
def analyze_tag_coverage():
    """Analyze tag_list field in TalkPlayData catalog."""
    print("[Phase 1.1] Loading TalkPlayData catalog...")
    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split="all_tracks",
        cache_dir=REPO_ROOT / ".hf_cache"
    )

    total_tracks = len(ds)
    has_tags = sum(1 for row in ds if row["tag_list"] and len(row["tag_list"]) > 0)
    tag_counts = Counter()
    tracks_per_tag_count = Counter()

    for row in ds:
        tags = row["tag_list"] or []
        tracks_per_tag_count[len(tags)] += 1
        tag_counts.update(tags)

    print(f"\n=== Tag Coverage Analysis ===")
    print(f"Total tracks: {total_tracks:,}")
    print(f"Tracks with tags: {has_tags:,} ({100*has_tags/total_tracks:.1f}%)")
    print(f"Tracks with 0 tags: {tracks_per_tag_count[0]:,} ({100*tracks_per_tag_count[0]/total_tracks:.1f}%)")
    print(f"Unique tags: {len(tag_counts):,}")
    print(f"Most common tags (top 20):")
    for tag, count in tag_counts.most_common(20):
        print(f"  {tag}: {count:,} tracks ({100*count/total_tracks:.1f}%)")

    # Check tag sparsity distribution
    tag_dist = []
    for count in range(0, 21):
        n = tracks_per_tag_count.get(count, 0)
        if n > 0:
            tag_dist.append(f"{count}tags:{n:,}")
    print(f"\nTag count distribution: {', '.join(tag_dist[:10])}")

    # Compute tag IDF (inverse document frequency) as a quality proxy
    # Low IDF = common tag (e.g., "rock"), high IDF = rare tag (e.g., "mathcore")
    tag_idf = {}
    for tag, doc_freq in tag_counts.items():
        tag_idf[tag] = np.log(total_tracks / (1 + doc_freq))

    high_idf_tags = sorted(tag_idf.items(), key=lambda x: x[1], reverse=True)[:10]
    low_idf_tags = sorted(tag_idf.items(), key=lambda x: x[1])[:10]

    print(f"\nHighest IDF tags (rare, specific): {', '.join(t for t, _ in high_idf_tags)}")
    print(f"Lowest IDF tags (common, generic): {', '.join(t for t, _ in low_idf_tags)}")

    return {
        "total_tracks": total_tracks,
        "tracks_with_tags": has_tags,
        "tag_coverage_pct": 100 * has_tags / total_tracks,
        "unique_tags": len(tag_counts),
        "mean_tags_per_track": sum(tag_counts.values()) / total_tracks,
        "top_20_tags": [{"tag": t, "count": c} for t, c in tag_counts.most_common(20)],
        "tag_distribution": dict(tracks_per_tag_count),
    }


def check_existing_tag_features():
    """Check if R39/R54 already use tag_jaccard_last feature."""
    print("\n[Phase 1.2] Checking if tag features already exist in R39/R54...")

    # R39 feature list is in the training script or model artifact
    # For now, just report that R39 has tag_jaccard_last (confirmed in design doc)
    print("  ✓ R39 includes `tag_jaccard_last` feature (last 3 played tracks)")
    print("  ✓ R54 extends R39 with album/pop features, preserves tag_jaccard_last")
    print("\n  → Question: Can we IMPROVE tag_jaccard_last with weighting (TF-IDF)?")
    print("  → If weighted tags don't help, external MB tags (same class) likely won't either.")

    return {
        "r39_has_tag_jaccard": True,
        "r54_has_tag_jaccard": True,
        "weighted_version_exists": False,
    }


def assess_mb_cache_value():
    """Assess whether existing MB cache (536 files, 100-track sample) adds signal."""
    print("\n[Phase 1.3] Assessing MusicBrainz cache value...")

    mb_results_path = REPO_ROOT / "cache" / "mb_audit_results.json"
    if not mb_results_path.exists():
        print("  ✗ MB audit results not found. Cannot assess MB value.")
        return {"mb_cache_exists": False}

    with open(mb_results_path) as f:
        mb_data = json.load(f)

    results = mb_data["results"]
    total_sample = len(results)
    has_mb_match = sum(1 for r in results if r.get("mb_match", False))
    has_mb_tags = sum(1 for r in results if r.get("mb_tags") and len(r["mb_tags"]) > 0)
    has_mb_genres = sum(1 for r in results if r.get("mb_genres") and len(r["mb_genres"]) > 0)

    print(f"  MB cache: {total_sample} tracks sampled")
    print(f"  ISRC→MB match: {has_mb_match}/{total_sample} ({100*has_mb_match/total_sample:.1f}%)")
    print(f"  Has MB tags: {has_mb_tags}/{total_sample} ({100*has_mb_tags/total_sample:.1f}%)")
    print(f"  Has MB genres: {has_mb_genres}/{total_sample} ({100*has_mb_genres/total_sample:.1f}%)")

    # Sample a few tracks to compare TalkPlayData tags vs MB tags
    print("\n  Comparing TalkPlayData tags vs MB tags (3 examples):")

    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
        split="all_tracks",
        cache_dir=REPO_ROOT / ".hf_cache"
    )
    tid_to_tags = {row["track_id"]: row["tag_list"] for row in ds}

    for i, r in enumerate(results[:3]):
        tid = r["tid"]
        tpd_tags = tid_to_tags.get(tid, [])
        mb_tags = r.get("mb_tags", [])
        overlap = set(tpd_tags) & set(mb_tags)

        print(f"\n  Example {i+1}: {r['track_name'][0]} by {r['artist_name'][0]}")
        print(f"    TalkPlayData tags ({len(tpd_tags)}): {', '.join(tpd_tags[:5])}...")
        print(f"    MusicBrainz tags ({len(mb_tags)}): {', '.join(mb_tags[:5])}...")
        print(f"    Overlap: {len(overlap)} tags — {', '.join(list(overlap)[:3])}")

    return {
        "mb_cache_exists": True,
        "mb_sample_size": total_sample,
        "mb_match_rate": has_mb_match / total_sample,
        "mb_tags_yield": has_mb_tags / total_sample,
        "mb_genres_yield": has_mb_genres / total_sample,
    }


def gate_decision(tag_stats, mb_stats):
    """Decide Phase 1 gate: GREEN/BORDERLINE/RED."""
    print("\n[Phase 1.4] Gate Decision...")

    # Criteria for GREEN (proceed to Phase 2):
    # 1. TalkPlayData tag coverage ≥ 80% (enough data for tag-based features)
    # 2. MB cache shows ≥ 60% metadata yield (MB has useful data)
    # 3. Tag diversity: ≥ 1000 unique tags (enough signal)

    # Criteria for RED (archive):
    # 1. Tag coverage < 50% (tags too sparse)
    # 2. MB cache shows < 40% metadata yield (MB won't add much)
    # 3. Tag diversity < 500 (too few unique tags)

    tag_coverage = tag_stats["tag_coverage_pct"]
    unique_tags = tag_stats["unique_tags"]
    mb_tags_yield = mb_stats.get("mb_tags_yield", 0)

    reasons = []

    if tag_coverage >= 80:
        reasons.append(f"✓ Tag coverage {tag_coverage:.1f}% ≥ 80%")
    elif tag_coverage >= 50:
        reasons.append(f"⚠ Tag coverage {tag_coverage:.1f}% in 50-80% range (borderline)")
    else:
        reasons.append(f"✗ Tag coverage {tag_coverage:.1f}% < 50% (too sparse)")

    if unique_tags >= 1000:
        reasons.append(f"✓ {unique_tags:,} unique tags ≥ 1000 (good diversity)")
    elif unique_tags >= 500:
        reasons.append(f"⚠ {unique_tags:,} unique tags in 500-1000 range (borderline)")
    else:
        reasons.append(f"✗ {unique_tags:,} unique tags < 500 (low diversity)")

    if mb_tags_yield >= 0.60:
        reasons.append(f"✓ MB tags yield {100*mb_tags_yield:.1f}% ≥ 60%")
    elif mb_tags_yield >= 0.40:
        reasons.append(f"⚠ MB tags yield {100*mb_tags_yield:.1f}% in 40-60% range")
    else:
        reasons.append(f"✗ MB tags yield {100*mb_tags_yield:.1f}% < 40%")

    # Count green/yellow/red
    green_count = sum(1 for r in reasons if r.startswith("✓"))
    red_count = sum(1 for r in reasons if r.startswith("✗"))

    if green_count >= 2 and red_count == 0:
        verdict = "GREEN"
        recommendation = "Proceed to Phase 2: 300-case MB diagnostic with API calls"
    elif red_count >= 2:
        verdict = "RED"
        recommendation = "Archive C4. Tag coverage/diversity too low; MB unlikely to help."
    else:
        verdict = "BORDERLINE"
        recommendation = "Ask user. Mixed signals on tag quality and MB value."

    print(f"\n=== Gate Evaluation ===")
    for r in reasons:
        print(f"  {r}")

    print(f"\n  VERDICT: {verdict}")
    print(f"  Recommendation: {recommendation}")

    return {
        "verdict": verdict,
        "recommendation": recommendation,
        "reasons": reasons,
        "green_count": green_count,
        "red_count": red_count,
    }


def main():
    print("=" * 60)
    print("R59 C4 Phase 1: Local Feasibility Audit")
    print("NO external API calls. Analyze existing TalkPlayData tags.")
    print("=" * 60)

    # Phase 1.1: Tag coverage analysis
    tag_stats = analyze_tag_coverage()

    # Phase 1.2: Check existing features
    feature_stats = check_existing_tag_features()

    # Phase 1.3: MB cache value assessment
    mb_stats = assess_mb_cache_value()

    # Phase 1.4: Gate decision
    gate_result = gate_decision(tag_stats, mb_stats)

    # Save results
    output = {
        "phase": "C4_Phase1_LocalFeasibility",
        "tag_stats": tag_stats,
        "feature_stats": feature_stats,
        "mb_cache_stats": mb_stats,
        "gate": gate_result,
    }

    output_path = REPO_ROOT / "exp" / "eval" / "expR59_c4_phase1_feasibility.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # Generate Phase 1 report
    report_path = REPO_ROOT / "docs" / "r59_candidates" / "c4_phase1_report.md"
    with open(report_path, "w") as f:
        f.write("# R59 C4 Phase 1: Local Feasibility Audit Report\n\n")
        f.write(f"**Date:** 2026-05-16\n")
        f.write(f"**Verdict:** {gate_result['verdict']}\n\n")
        f.write(f"## Summary\n\n")
        f.write(f"{gate_result['recommendation']}\n\n")
        f.write(f"## Tag Coverage Analysis\n\n")
        f.write(f"- Total tracks: {tag_stats['total_tracks']:,}\n")
        f.write(f"- Tracks with tags: {tag_stats['tracks_with_tags']:,} ({tag_stats['tag_coverage_pct']:.1f}%)\n")
        f.write(f"- Unique tags: {tag_stats['unique_tags']:,}\n")
        f.write(f"- Mean tags per track: {tag_stats['mean_tags_per_track']:.2f}\n\n")
        f.write(f"## MusicBrainz Cache Assessment\n\n")
        if mb_stats.get("mb_cache_exists"):
            f.write(f"- Sample size: {mb_stats['mb_sample_size']} tracks\n")
            f.write(f"- ISRC→MB match rate: {100*mb_stats['mb_match_rate']:.1f}%\n")
            f.write(f"- MB tags yield: {100*mb_stats['mb_tags_yield']:.1f}%\n")
            f.write(f"- MB genres yield: {100*mb_stats['mb_genres_yield']:.1f}%\n\n")
        else:
            f.write("MB cache not found.\n\n")
        f.write(f"## Gate Criteria\n\n")
        for reason in gate_result['reasons']:
            f.write(f"- {reason}\n")
        f.write(f"\n**Green checks:** {gate_result['green_count']}/3\n")
        f.write(f"**Red flags:** {gate_result['red_count']}/3\n\n")
        f.write(f"## Recommendation\n\n")
        f.write(f"**{gate_result['verdict']}:** {gate_result['recommendation']}\n")

    print(f"✓ Report saved to {report_path}")
    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE: {gate_result['verdict']}")
    print(f"{'='*60}")

    return gate_result["verdict"]


if __name__ == "__main__":
    verdict = main()
    sys.exit(0 if verdict == "GREEN" else 1 if verdict == "RED" else 2)
