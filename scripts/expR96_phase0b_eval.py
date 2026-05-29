#!/usr/bin/env python3
"""R96 Phase 0b eval: E5 unique-recovery on the complete-union absent set (CPU).

Consumes E5 OOF lists (oof_e5_lists.json from expR96_e5_phase0b.py) and measures
the ONLY thing that matters: how much GT that is absent from the complete union
(R96 Phase 0a.2, 1784 dev cases) does E5 recover, and at what rank. Because the
absent set is by definition outside the union, every absent-set recovery is a
UNIQUE recovery.

Reports absent-set recovery @20/30/100/300, rank distribution, history-depth and
same-artist(in-history) / diff-artist (= new-artist) splits, plus overall E5
recall for context, and the go/stop flags from the user's stop conditions.
"""
from __future__ import annotations

import argparse
import ast
import json
import pickle
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl"
ABSENT = REPO / "exp" / "eval" / "expR96_absent_set.json"
PAYLOAD = REPO / "cache" / "r54_phase3_payload_maps.pkl"
R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
DEFAULT_E5 = REPO / "cache" / "r96_e5" / "phase0b" / "oof_e5_lists.json"
OUT = REPO / "exp" / "eval" / "expR96_phase0b_eval.json"

DEPTHS = [20, 30, 100, 300]


def ids(seq, k=None):
    out = [x[0] if isinstance(x, (list, tuple)) else x for x in (seq or [])]
    return out[:k] if k else out


def artist_of(track_artist, t):
    raw = track_artist.get(t)
    if raw is None:
        return ""
    try:
        v = ast.literal_eval(raw) if isinstance(raw, str) else raw
    except (ValueError, SyntaxError):
        return str(raw).strip().lower()
    return (str(v[0]) if isinstance(v, (list, tuple)) and v else str(v)).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5-lists", type=Path, default=DEFAULT_E5)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    ref = pickle.load(open(REF, "rb"))
    absent = json.load(open(ABSENT))["absent"]          # {case_idx(str): {gt, history_depth}}
    absent_idx = {int(k): v for k, v in absent.items()}
    track_artist = pickle.load(open(PAYLOAD, "rb"))["track_artist"]
    e5 = json.load(open(args.e5_lists))["lists"]         # list[N] of [(tid,score)]
    payload = pickle.load(open(R12_CACHE, "rb"))
    played_by_idx = {i: ids(c.get("music_turns")) for i, c in enumerate(payload["cases"])}

    N = len(ref)
    overall = Counter()
    abs_rec = Counter()                                  # absent-set recovery @depth
    rankdist = Counter()
    hist_split = {b: Counter() for b in ["h0", "h1-2", "h3-6", "h7+"]}
    artist_split = Counter()                             # same(in-history) / diff(new artist)
    n_absent = len(absent_idx)

    for e in ref:
        i = e["case_idx"]
        gt = e["gt_track_id"]
        lst = ids(e5[i]) if i < len(e5) and e5[i] else []
        pos = lst.index(gt) if gt in lst else None
        for D in DEPTHS:
            if pos is not None and pos < D:
                overall[D] += 1

        if i not in absent_idx:
            continue
        # absent (= unique) recovery analysis
        for D in DEPTHS:
            if pos is not None and pos < D:
                abs_rec[D] += 1
        if pos is not None:
            b = ("top20" if pos < 20 else "20-30" if pos < 30 else "30-50" if pos < 50
                 else "50-100" if pos < 100 else "100-300")
            rankdist[b] += 1
            hd = absent_idx[i]["history_depth"]
            hb = "h0" if hd == 0 else "h1-2" if hd <= 2 else "h3-6" if hd <= 6 else "h7+"
            in30 = pos < 30
            hist_split[hb]["rec30" if in30 else "rec_deep"] += 1
            played_artists = {artist_of(track_artist, t) for t in played_by_idx.get(i, [])}
            played_artists.discard("")
            same = artist_of(track_artist, gt) in played_artists
            artist_split[("same_artist" if same else "new_artist") + ("_top30" if in30 else "_deep")] += 1

    def pct(x, d=N):
        return round(x / d, 4)

    rec30 = abs_rec[30]
    deep = sum(rankdist[k] for k in ["30-50", "50-100", "100-300"])
    total_abs_rec = sum(rankdist.values())
    report = {
        "experiment": "R96 Phase 0b E5 unique-recovery eval",
        "e5_lists": str(args.e5_lists),
        "n_dev": N,
        "n_absent_target": n_absent,
        "overall_e5_recall": {str(D): pct(overall[D]) for D in DEPTHS},
        "absent_set_recovery": {str(D): {"n": abs_rec[D], "of_absent": round(abs_rec[D] / n_absent, 4)}
                                 for D in DEPTHS},
        "absent_recovery_rank_distribution": dict(rankdist),
        "absent_recovery_history_split": {k: dict(v) for k, v in hist_split.items()},
        "absent_recovery_artist_split": dict(artist_split),
        "go_stop": {
            "unique_top30_recoveries": rec30,
            "frac_of_recovered_that_are_top30": round(rec30 / max(1, total_abs_rec), 4),
            "frac_of_recovered_that_are_deep_50_300": round(
                sum(rankdist[k] for k in ["50-100", "100-300"]) / max(1, total_abs_rec), 4),
            "verdict_hint": ("STOP: recoveries mostly deep (50-300)"
                             if total_abs_rec and (deep / total_abs_rec) > 0.5 and rec30 < 30
                             else "PROCEED: nontrivial top-30 absent-set recoveries"
                             if rec30 >= 30 else "WEAK: few top-30 recoveries"),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)

    print(f"n_dev={N}  absent_target={n_absent}")
    print(f"overall E5 recall: " + "  ".join(f"@{D}={pct(overall[D]):.3f}" for D in DEPTHS))
    print("absent-set (unique) recovery:")
    for D in DEPTHS:
        print(f"  @{D}: {abs_rec[D]:4d}  ({abs_rec[D]/n_absent:.3f} of absent)")
    print(f"rank dist of recoveries: {dict(rankdist)}")
    print(f"history split: {report['absent_recovery_history_split']}")
    print(f"artist split:  {report['absent_recovery_artist_split']}")
    print(f"\nGO/STOP: {report['go_stop']['verdict_hint']}  "
          f"(top30 unique recoveries={rec30}, "
          f"deep_frac={report['go_stop']['frac_of_recovered_that_are_deep_50_300']})")
    try:
        print(f"Wrote {args.out.relative_to(REPO)}")
    except ValueError:
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
