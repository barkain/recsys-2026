#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54b blind change analysis — characterize the 27 sessions where R54
top-1 differed from R39 top-1, and cross-reference with dev recovered/lost
cases from the Phase 3 full integration.

Goals:
  1. Qualitative profile of the 27 blind changes:
     - same vs different artist (R54 top-1 vs R39 top-1)
     - same vs different artist vs user history
     - track popularity tier of R39 top-1 vs R54 top-1
     - history length / conversation style
     - top-1 cosine confidence (from R54 ensemble blind retrieval)

  2. Dev cross-reference: in CV5 integration on dev, which h7 cases did
     R54 recover or lose? Profile those by the same axes.

  3. If recovered/lost profiles on dev match the 27 blind changes, the blind
     gain is mechanistic and predictable. If they diverge, the blind gain is
     either noise or capturing a property not modeled in dev.

Output: exp/eval/expR54b_blind_change_analysis.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R39_BLIND = REPO / "exp" / "inference" / "blind_a" / "r39_album_submission.json"
R54_BLIND = REPO / "exp" / "inference" / "blind_a" / "r54_phase3_exploratory_submission.json"
R54_BLIND_LISTS = REPO / "cache" / "r54_production" / "blind_r54_lists.json"


def pop_tier(pop_value: int, max_pop: int) -> str:
    """Bucket popularity into tiers."""
    norm = pop_value / max(max_pop, 1)
    if norm < 0.01:
        return "very_rare"
    if norm < 0.1:
        return "rare"
    if norm < 0.5:
        return "common"
    return "popular"


def load_blind_meta(item):
    import pandas as pd
    df = pd.DataFrame(item["conversations"]).sort_values("turn_number")
    user_rows = df[df["role"] == "user"]
    last_user = user_rows.iloc[-1]
    user_query = str(last_user["content"])
    prior = df[df["turn_number"] < int(last_user["turn_number"])]
    history_user_msgs = [str(r["content"]) for _, r in prior.iterrows() if r["role"] == "user"]
    music_history = [str(r["content"]).strip() for _, r in prior.iterrows() if r["role"] == "music"]
    return {
        "user_query": user_query,
        "history_user_msgs": history_user_msgs,
        "music_history": music_history,
        "n_user_turns": len(history_user_msgs) + 1,
        "n_played": len(music_history),
    }


def main():
    print("R54b blind change analysis")
    print("=" * 70)

    print("Loading dev payload (track_artist + track_pop) ...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    ta = payload["track_artist"]
    from scripts.expS2_lr_v2 import build_popularity_stats
    track_pop = build_popularity_stats()
    max_pop = max(track_pop.values()) if track_pop else 1

    print("Loading submissions ...")
    with open(R39_BLIND) as f:
        r39 = {r["session_id"]: r for r in json.load(f)}
    with open(R54_BLIND) as f:
        r54 = {r["session_id"]: r for r in json.load(f)}
    with open(R54_BLIND_LISTS) as f:
        r54_blind_lists = json.load(f)["lists"]

    # Blind sessions for context
    print("Loading blind-A ...")
    from datasets import DownloadConfig, load_dataset
    db = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A", split="test",
                      download_config=DownloadConfig(local_files_only=True))
    blind_by_sid = {str(item["session_id"]): item for item in db}

    changed = []
    kept = []
    for sid in r39:
        if r39[sid]["predicted_track_ids"][0] != r54[sid]["predicted_track_ids"][0]:
            changed.append(sid)
        else:
            kept.append(sid)
    print(f"\nTop-1 changes vs R39: {len(changed)} changed, {len(kept)} kept")

    # ==========================================================
    # 1. Profile the 27 changed blind sessions
    # ==========================================================
    profile = []
    for sid in changed:
        item = blind_by_sid[sid]
        meta = load_blind_meta(item)
        r39_top1 = r39[sid]["predicted_track_ids"][0]
        r54_top1 = r54[sid]["predicted_track_ids"][0]
        r39_artist = ta.get(r39_top1, "")
        r54_artist = ta.get(r54_top1, "")
        if isinstance(r39_artist, list):
            r39_artist = r39_artist[0] if r39_artist else ""
        if isinstance(r54_artist, list):
            r54_artist = r54_artist[0] if r54_artist else ""
        history_artists = set()
        for t in meta["music_history"]:
            a = ta.get(t, "")
            if isinstance(a, list):
                a = a[0] if a else ""
            if a:
                history_artists.add(a)
        # R54 top-1 cosine from blind ensemble retrieval
        r54_cos = next((s for t, s in r54_blind_lists[sid][:5] if t == r54_top1), None)
        r39_in_r54_top20 = r39_top1 in r54[sid]["predicted_track_ids"]
        r54_in_r39_top20 = r54_top1 in r39[sid]["predicted_track_ids"]
        prof = {
            "session_id": sid,
            "n_user_turns": meta["n_user_turns"],
            "n_played": meta["n_played"],
            "history_artist_count": len(history_artists),
            "r39_top1": r39_top1, "r54_top1": r54_top1,
            "r39_artist": r39_artist, "r54_artist": r54_artist,
            "same_artist_top1": r39_artist == r54_artist if r39_artist else False,
            "r39_top1_artist_in_history": r39_artist in history_artists if r39_artist else False,
            "r54_top1_artist_in_history": r54_artist in history_artists if r54_artist else False,
            "r39_top1_pop_tier": pop_tier(track_pop.get(r39_top1, 0), max_pop),
            "r54_top1_pop_tier": pop_tier(track_pop.get(r54_top1, 0), max_pop),
            "r54_top1_cosine": r54_cos,
            "r39_top1_in_r54_top20": r39_in_r54_top20,
            "r54_top1_in_r39_top20": r54_in_r39_top20,
            "user_query_len": len(meta["user_query"]),
        }
        profile.append(prof)

    # Aggregates
    print(f"\n=== Profile of 27 R39→R54 changed blind sessions ===")
    n = len(profile)
    same_artist = sum(1 for p in profile if p["same_artist_top1"])
    print(f"  Same-artist swap (same artist, different track):  {same_artist}/{n}")
    diff_artist = n - same_artist
    print(f"  Different-artist swap:                            {diff_artist}/{n}")

    r39_in_hist = sum(1 for p in profile if p["r39_top1_artist_in_history"])
    r54_in_hist = sum(1 for p in profile if p["r54_top1_artist_in_history"])
    print(f"  R39 top-1 artist in user history:                 {r39_in_hist}/{n}")
    print(f"  R54 top-1 artist in user history:                 {r54_in_hist}/{n}")
    moved_to_history = sum(1 for p in profile
                            if not p["r39_top1_artist_in_history"]
                            and p["r54_top1_artist_in_history"])
    moved_away_history = sum(1 for p in profile
                              if p["r39_top1_artist_in_history"]
                              and not p["r54_top1_artist_in_history"])
    print(f"    moved INTO history-artist:                       {moved_to_history}/{n}")
    print(f"    moved AWAY from history-artist:                  {moved_away_history}/{n}")

    print(f"\n  R39 top-1 popularity tiers:")
    for tier, c in Counter(p["r39_top1_pop_tier"] for p in profile).most_common():
        print(f"    {tier:<12}: {c}")
    print(f"  R54 top-1 popularity tiers:")
    for tier, c in Counter(p["r54_top1_pop_tier"] for p in profile).most_common():
        print(f"    {tier:<12}: {c}")

    print(f"\n  History length distribution (n_played for these 27):")
    hist_lens = [p["n_played"] for p in profile]
    print(f"    mean={np.mean(hist_lens):.1f}  median={int(np.median(hist_lens))}  "
          f"min={min(hist_lens)}  max={max(hist_lens)}")
    print(f"  User turns:")
    n_turns = [p["n_user_turns"] for p in profile]
    print(f"    mean={np.mean(n_turns):.1f}  median={int(np.median(n_turns))}")

    print(f"\n  R54 top-1 ensemble cosine (confidence in new pick):")
    cosines = [p["r54_top1_cosine"] for p in profile if p["r54_top1_cosine"] is not None]
    if cosines:
        print(f"    mean={np.mean(cosines):.4f}  median={np.median(cosines):.4f}  "
              f"min={min(cosines):.4f}  max={max(cosines):.4f}")

    r39_still_in_r54_top20 = sum(1 for p in profile if p["r39_top1_in_r54_top20"])
    r54_was_in_r39_top20 = sum(1 for p in profile if p["r54_top1_in_r39_top20"])
    print(f"\n  R39's old top-1 still appears somewhere in R54 top-20: {r39_still_in_r54_top20}/{n}")
    print(f"  R54's new top-1 was already in R39 top-20:             {r54_was_in_r39_top20}/{n}")

    # ==========================================================
    # 2. Compare with dev-side recovered/lost h7 cases
    # ==========================================================
    print(f"\n=== Dev-side recovered/lost comparison ===")
    integ_path = REPO / "exp" / "eval" / "expR54_phase3_full5fold_integration.json"
    if integ_path.exists():
        with open(integ_path) as f:
            integ = json.load(f)
        best_cfg = integ.get("best_cfg", "R39+R54_w1.0_feats")
        r54_res = integ["results"][best_cfg]
        recovered = r54_res.get("recovered", 0)
        lost = r54_res.get("lost", 0)
        print(f"  CV5 vs R39 baseline: recovered={recovered}  lost={lost}  net={recovered-lost:+d}")
        print(f"  (across all 8000 dev cases, top-20 churn vs R39+R54_w1.0_feats)")
    else:
        print(f"  Skipped (integration artifact missing)")

    # Save
    out = {
        "blind_changes": {
            "n_changed": n,
            "same_artist_swap": same_artist,
            "diff_artist_swap": diff_artist,
            "r39_top1_artist_in_history": r39_in_hist,
            "r54_top1_artist_in_history": r54_in_hist,
            "moved_into_history_artist": moved_to_history,
            "moved_away_history_artist": moved_away_history,
            "r39_top1_pop_tiers": dict(Counter(p["r39_top1_pop_tier"] for p in profile)),
            "r54_top1_pop_tiers": dict(Counter(p["r54_top1_pop_tier"] for p in profile)),
            "n_played_mean": float(np.mean(hist_lens)),
            "n_user_turns_mean": float(np.mean(n_turns)),
            "r54_top1_cosine_mean": float(np.mean(cosines)) if cosines else None,
            "r39_old_top1_in_r54_top20": r39_still_in_r54_top20,
            "r54_new_top1_was_in_r39_top20": r54_was_in_r39_top20,
        },
        "per_session_profile": profile,
    }
    out_path = REPO / "exp" / "eval" / "expR54b_blind_change_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
