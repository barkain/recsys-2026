#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R104 — behavioral candidate recall audit (item-item graph + session-NN).

The pivot (user directive): stop polishing semantic matches; the remaining nDCG
signal is behavioral/session/item-item, i.e. "what track did the user actually
play next", not "what text is nearest". This script is the GATING first step:
build a behavioral candidate universe from TRAIN listening sessions and measure
whether it adds USABLE ground-truth recall (top-30) over the current semantic
union@300. Only if it does is the R105 behavior ranker worth building.

Leakage: behavioral indices are built from TRAIN sessions only (15,199 sessions,
disjoint from the 1,000 dev sessions — asserted at runtime). The GT next-track of
a dev case is never observed during index construction, so train->dev co-occurrence
is OOF-clean. (Dev-OOF sessions could be added later; train-only is the clean
conservative first cut.)

Behavioral sources (per dev case, given played history H and excluding H):
  cooc_next   : track t in H -> tracks that immediately FOLLOWED t in train
                (recency-weighted over H, conditional-prob normalized).
  cooc_sess   : track t in H -> tracks in the SAME train session as t (symmetric).
  artist_next : artist of H tracks -> tracks that followed that artist.
  tag_next    : tags of H tracks  -> tracks that followed that tag.
  album_next  : album of H tracks -> tracks that followed that album.
  session_nn  : train sessions most similar to H (track-set Jaccard) -> their tracks.
  popularity  : global train next-track frequency (cold-start fallback).

Audit (vs semantic union@300, RRF over A/B/C/D/F/ALS/R21/R54):
  - semantic recall@30/@300, behavior recall@30/@300, combined(RRF) recall@30
  - UNIQUE recoveries: GT in behavior top-30 but NOT in semantic top-30 (and the
    stronger variant: not in semantic top-300 either)
  - per-source recall@30 (which behavioral signal carries the recall)
  - rank distribution of recovered GTs (usable vs buried)
  - breakdown by n_prior_music bucket (esp. h7 = full history) and same/diff artist
  - GT-in-train-catalog coverage ceiling (behavioral recall upper bound)

Gate (user spec): +5% ABSOLUTE recall@30 over the union, OR >=50 unique dev GTs in
behavior top-30 not in the semantic union — AND the added GTs sit at usable ranks.

Output: exp/eval/expR104_behavior_recall.json (aggregate) +
        exp/eval/expR104_percase.json (per-case, feeds R105/R106).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from heapq import nlargest
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.expR59_c3_pool_admission_diagnostic as c3  # noqa: E402
from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expR59_c3_phase2_frozen_lr_conversion import (  # noqa: E402
    load_supporting_maps, same_artist_case,
)

RRF_K = 20
POOL_K = 300
TOP30 = 30
TOP20 = 20
N_FOLDS = 5

DECAY = 0.8          # recency weight for history aggregation (most-recent = 1.0)
SRC_CAP = 300        # cap per behavioral source list
NN_TRACK_CAP = 4000  # max train sessions scanned per history track (popular-track guard)
NN_TOP_SESS = 50     # nearest train sessions used per case
PRUNE_NEXT = 400     # keep top-N next-tracks per key in the count indices (memory)

TRAIN_GLOB = (".hf_cache/datasets/talkpl-ai___talk_play_data-challenge-dataset/"
              "default/*/*/talk_play_data-challenge-dataset-train.arrow")

OUT_JSON = REPO / "exp" / "eval" / "expR104_behavior_recall.json"
OUT_PERCASE = REPO / "exp" / "eval" / "expR104_percase.json"

BEH_SOURCES = ["cooc_next", "cooc_sess", "artist_next", "tag_next",
               "album_next", "session_nn", "popularity"]
# RRF weights for the behavior union (equal-ish; popularity down-weighted as it is
# query-independent and would otherwise dominate cold cases).
BEH_W = {"cooc_next": 1.0, "cooc_sess": 1.0, "artist_next": 1.0, "tag_next": 0.7,
         "album_next": 0.7, "session_nn": 1.0, "popularity": 0.3}


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


# --------------------------------------------------------------------------- #
# Train behavioral indices
# --------------------------------------------------------------------------- #
def session_tracks(conv) -> list[str]:
    """Ordered list of music track_ids in a train session's conversation."""
    out = []
    for row in conv:
        if row.get("role") == "music":
            c = row.get("content")
            if c:
                out.append(str(c))
    return out


def build_train_indices(track_artist, track_tags, track_album):
    from datasets import Dataset
    paths = sorted(Path(REPO).glob(TRAIN_GLOB))
    if not paths:
        sys.exit(f"train arrow not found under {TRAIN_GLOB}")
    ds = Dataset.from_file(str(paths[-1]))
    print(f"{ts()} train sessions: {len(ds)}", flush=True)

    track_next: dict[str, Counter] = defaultdict(Counter)
    track_cooc: dict[str, Counter] = defaultdict(Counter)
    artist_next: dict[str, Counter] = defaultdict(Counter)
    tag_next: dict[str, Counter] = defaultdict(Counter)
    album_next: dict[str, Counter] = defaultdict(Counter)
    popularity: Counter = Counter()
    track2sessions: dict[str, list[int]] = defaultdict(list)
    sess_tracks: list[list[str]] = []
    sess_trackset: list[frozenset] = []
    train_sids: set[str] = set()

    t0 = time.time()
    n_trans = 0
    for sidx in range(len(ds)):
        row = ds[sidx]
        train_sids.add(str(row["session_id"]))
        tracks = session_tracks(row["conversations"])
        sess_tracks.append(tracks)
        tset = frozenset(tracks)
        sess_trackset.append(tset)
        for t in tset:
            track2sessions[t].append(sidx)
        for i, t in enumerate(tracks):
            popularity[t] += 1
            # immediate next + downstream-in-session
            if i + 1 < len(tracks):
                nt = tracks[i + 1]
                track_next[t][nt] += 1
                n_trans += 1
                art = track_artist.get(t, "")
                if art:
                    artist_next[art][nt] += 1
                alb = track_album.get(t, "")
                if alb:
                    album_next[alb][nt] += 1
                for tg in (track_tags.get(t, ()) or ()):
                    tag_next[tg][nt] += 1
            for j in range(len(tracks)):
                if j != i:
                    track_cooc[t][tracks[j]] += 1
        if (sidx + 1) % 4000 == 0:
            print(f"  {sidx + 1}/{len(ds)} ({time.time() - t0:.0f}s) "
                  f"trans={n_trans}", flush=True)

    # prune the heavy *_next indices to top-N targets per key (memory + speed)
    def prune(idx):
        for k in list(idx.keys()):
            if len(idx[k]) > PRUNE_NEXT:
                idx[k] = Counter(dict(idx[k].most_common(PRUNE_NEXT)))
    for idx in (track_next, track_cooc, artist_next, tag_next, album_next):
        prune(idx)

    print(f"{ts()} indices built: track_next={len(track_next)} "
          f"artist_next={len(artist_next)} tag_next={len(tag_next)} "
          f"album_next={len(album_next)} transitions={n_trans} "
          f"({time.time() - t0:.0f}s)", flush=True)
    return {
        "track_next": track_next, "track_cooc": track_cooc,
        "artist_next": artist_next, "tag_next": tag_next, "album_next": album_next,
        "popularity": popularity, "track2sessions": track2sessions,
        "sess_tracks": sess_tracks, "sess_trackset": sess_trackset,
        "train_sids": train_sids, "n_trans": n_trans,
    }


# --------------------------------------------------------------------------- #
# Per-case behavioral candidate lists
# --------------------------------------------------------------------------- #
def _ranked(score: dict[str, float], played: set[str], cap: int = SRC_CAP) -> list[str]:
    if not score:
        return []
    items = [(t, s) for t, s in score.items() if t not in played and s > 0]
    items.sort(key=lambda x: -x[1])
    return [t for t, _ in items[:cap]]


def _cond_next(keys_weighted, idx) -> dict[str, float]:
    """Sum over (key, w): w * P(next | key).  conditional-prob normalized."""
    score: dict[str, float] = defaultdict(float)
    for key, w in keys_weighted:
        ctr = idx.get(key)
        if not ctr:
            continue
        tot = sum(ctr.values())
        if tot <= 0:
            continue
        for nt, cnt in ctr.items():
            score[nt] += w * (cnt / tot)
    return score


def behavior_sources_for_case(case, idx, track_artist, track_tags, track_album,
                              pop_top):
    H = case["music_turns"]
    played = set(H)
    k = len(H)
    out = {s: [] for s in BEH_SOURCES}

    if k > 0:
        # recency weights: most recent track highest
        tw = [(H[p], DECAY ** (k - 1 - p)) for p in range(k)]
        out["cooc_next"] = _ranked(_cond_next(tw, idx["track_next"]), played)
        out["cooc_sess"] = _ranked(_cond_next(tw, idx["track_cooc"]), played)

        # artist continuation (recency-weighted over distinct history artists)
        art_w: dict[str, float] = defaultdict(float)
        for t, w in tw:
            a = track_artist.get(t, "")
            if a:
                art_w[a] = max(art_w[a], w)
        out["artist_next"] = _ranked(
            _cond_next(list(art_w.items()), idx["artist_next"]), played)

        alb_w: dict[str, float] = defaultdict(float)
        for t, w in tw:
            al = track_album.get(t, "")
            if al:
                alb_w[al] = max(alb_w[al], w)
        out["album_next"] = _ranked(
            _cond_next(list(alb_w.items()), idx["album_next"]), played)

        tag_w: dict[str, float] = defaultdict(float)
        for t, w in tw:
            for tg in (track_tags.get(t, ()) or ()):
                tag_w[tg] = max(tag_w[tg], w)
        out["tag_next"] = _ranked(
            _cond_next(list(tag_w.items()), idx["tag_next"]), played)

        # session-NN over track-set Jaccard
        out["session_nn"] = _session_nn(H, played, idx)

    # popularity fallback (always available; query-independent)
    out["popularity"] = [t for t in pop_top if t not in played][:SRC_CAP]
    return out


def _session_nn(H, played, idx):
    track2sessions = idx["track2sessions"]
    sess_trackset = idx["sess_trackset"]
    hist = set(H)
    overlap: Counter = Counter()
    for t in hist:
        for sidx in track2sessions.get(t, ())[:NN_TRACK_CAP]:
            overlap[sidx] += 1
    if not overlap:
        return []

    def jac(sidx):
        st = sess_trackset[sidx]
        inter = overlap[sidx]
        return inter / (len(hist) + len(st) - inter)

    top = nlargest(NN_TOP_SESS, overlap.keys(), key=jac)
    cand: dict[str, float] = defaultdict(float)
    for sidx in top:
        j = jac(sidx)
        for t in sess_trackset[sidx]:
            if t not in played:
                cand[t] += j
    return _ranked(cand, played)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def rank_in(lst, gt, cap=None):
    """1-based rank of gt in lst (optionally only within first `cap`), else -1."""
    seg = lst if cap is None else lst[:cap]
    try:
        return seg.index(gt) + 1
    except ValueError:
        return -1


def main():
    t0 = time.time()
    print(f"{ts()} R104 — behavioral candidate recall audit")
    print("=" * 72)

    # --- semantic baseline (union@300) ---
    payload, r21_source, r54_source, r54_scores = c3.load_payloads()
    cases = payload["cases"]
    n = len(cases)
    factors, track_ids, track_to_idx = c3.load_als_cache()
    maps, track_pop_map, track_album = load_supporting_maps()
    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]

    case_index = c3.build_case_index(
        payload, r21_source, r54_source, r54_scores,
        factors, track_ids, track_to_idx)
    sem_pools = case_index["baseline_pools"]          # RRF@300, ordered

    # fold map
    import pickle
    w0 = pickle.load(open(REPO / "exp" / "eval" / "expR68_r54_reference_stats.pkl", "rb"))
    case_fold = [-1] * n
    for row in w0:
        case_fold[int(row["case_idx"])] = int(row["fold_idx"])

    # --- train behavioral indices ---
    idx = build_train_indices(track_artist, track_tags, track_album)

    # leakage guard: dev sessions must be disjoint from train sessions
    dev_sids = {c["session_id"] for c in cases}
    overlap_sids = dev_sids & idx["train_sids"]
    print(f"{ts()} dev∩train session overlap: {len(overlap_sids)} "
          f"(MUST be 0 for OOF-clean)", flush=True)
    assert not overlap_sids, f"LEAKAGE: {len(overlap_sids)} dev sessions appear in train"

    train_catalog = set(idx["popularity"].keys())
    pop_top = [t for t, _ in idx["popularity"].most_common(SRC_CAP)]

    # --- per-case audit ---
    print(f"\n{ts()} === per-case behavioral recall ({n} cases) ===", flush=True)
    results = []
    t_loop = time.time()
    for i, case in enumerate(cases):
        gt = case["gt"]
        sem = sem_pools[i]
        sem_r30 = rank_in(sem, gt, TOP30)
        sem_r300 = rank_in(sem, gt)

        bsrc = behavior_sources_for_case(case, idx, track_artist, track_tags,
                                         track_album, pop_top)
        beh_union = weighted_rrf(bsrc, BEH_W, POOL_K, RRF_K)
        beh_r30 = rank_in(beh_union, gt, TOP30)
        beh_r300 = rank_in(beh_union, gt)

        # combined: RRF over semantic union + behavioral sources
        comb_src = dict(bsrc)
        comb_src["SEM"] = sem
        comb_w = {**BEH_W, "SEM": 2.0}                     # semantic stays primary
        comb_union = weighted_rrf(comb_src, comb_w, POOL_K, RRF_K)
        comb_r30 = rank_in(comb_union, gt, TOP30)

        per_src_rank = {s: rank_in(bsrc[s], gt, TOP30) for s in BEH_SOURCES}

        results.append({
            "case_idx": i, "fold": case_fold[i],
            "n_prior_music": int(case["n_prior_music"]),
            "same_artist": bool(same_artist_case(case, track_artist)),
            "gt_in_train": gt in train_catalog,
            "sem_r30": sem_r30, "sem_r300": sem_r300,
            "beh_r30": beh_r30, "beh_r300": beh_r300,
            "comb_r30": comb_r30,
            "per_src_rank": per_src_rank,
            # for R105/R106 pool building: shallow behavior candidates + sem top-30
            "beh_top60": beh_union[:60],
            "sem_top30": sem[:TOP30],
        })
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t_loop:.0f}s)", flush=True)

    # --- aggregate ---
    def rate(rows, key, cap=TOP30):
        if not rows:
            return 0.0
        return float(np.mean([1.0 if (0 < r[key] <= cap) else 0.0 for r in rows]))

    def subset_block(rows):
        sem30 = rate(rows, "sem_r30")
        beh30 = rate(rows, "beh_r30")
        comb30 = rate(rows, "comb_r30")
        # unique recoveries: GT in behavior top-30 but not in semantic top-30
        uniq30 = sum(1 for r in rows
                     if 0 < r["beh_r30"] <= TOP30 and not (0 < r["sem_r30"] <= TOP30))
        # stronger: not even in semantic top-300
        uniq30_vs300 = sum(1 for r in rows
                           if 0 < r["beh_r30"] <= TOP30 and not (0 < r["sem_r300"] <= POOL_K))
        return {
            "n": len(rows),
            "sem_recall30": sem30, "beh_recall30": beh30, "comb_recall30": comb30,
            "comb_minus_sem": comb30 - sem30,
            "unique_beh30_not_sem30": uniq30,
            "unique_beh30_not_sem300": uniq30_vs300,
            "gt_in_train_rate": float(np.mean([1.0 if r["gt_in_train"] else 0.0 for r in rows])),
        }

    h7 = [r for r in results if r["n_prior_music"] == 7]
    same = [r for r in results if r["same_artist"]]
    diff = [r for r in results if not r["same_artist"]]
    by_bucket = {str(b): subset_block([r for r in results if r["n_prior_music"] == b])
                 for b in range(8)}

    agg = {
        "all": subset_block(results),
        "h7": subset_block(h7),
        "same_artist": subset_block(same),
        "diff_artist": subset_block(diff),
    }

    # per-source recall@30 (which behavioral signal carries it), all + h7
    per_source = {}
    for s in BEH_SOURCES:
        per_source[s] = {
            "all_recall30": float(np.mean([1.0 if 0 < r["per_src_rank"][s] <= TOP30 else 0.0
                                           for r in results])),
            "h7_recall30": float(np.mean([1.0 if 0 < r["per_src_rank"][s] <= TOP30 else 0.0
                                          for r in h7])) if h7 else 0.0,
            # unique vs semantic top-30 (this source alone surfaces GT sem misses)
            "h7_unique_vs_sem30": sum(1 for r in h7
                                      if 0 < r["per_src_rank"][s] <= TOP30
                                      and not (0 < r["sem_r30"] <= TOP30)),
        }

    # rank distribution of recovered GTs (behavior surfaces, semantic top-30 misses)
    recovered = [r for r in results
                 if 0 < r["beh_r30"] <= TOP30 and not (0 < r["sem_r30"] <= TOP30)]
    rec_ranks = [r["beh_r30"] for r in recovered]
    rank_dist = {
        "n_recovered": len(recovered),
        "at_rank_1_5": sum(1 for x in rec_ranks if x <= 5),
        "at_rank_6_10": sum(1 for x in rec_ranks if 6 <= x <= 10),
        "at_rank_11_20": sum(1 for x in rec_ranks if 11 <= x <= 20),
        "at_rank_21_30": sum(1 for x in rec_ranks if 21 <= x <= 30),
        "median_rank": float(np.median(rec_ranks)) if rec_ranks else 0.0,
    }

    # --- gate ---
    all_b = agg["all"]
    gate_recall = all_b["comb_minus_sem"] >= 0.05
    gate_unique = all_b["unique_beh30_not_sem30"] >= 50
    # "usable": at least half the recovered GTs at rank<=20
    usable = (rank_dist["at_rank_1_5"] + rank_dist["at_rank_6_10"]
              + rank_dist["at_rank_11_20"]) >= 0.5 * max(rank_dist["n_recovered"], 1)
    passed = (gate_recall or gate_unique) and usable
    verdict = "PROCEED_R105" if passed else "ARCHIVE_R104"

    # --- report ---
    print(f"\n{ts()} === AGGREGATE (vs semantic union@300) ===")
    print(f"  {'subset':12} {'n':>5} {'sem@30':>7} {'beh@30':>7} {'comb@30':>8} "
          f"{'Δcomb':>7} {'uniqβ⊄sem30':>11} {'uniqβ⊄sem300':>12} {'gt∈train':>9}")
    for name in ["all", "h7", "same_artist", "diff_artist"]:
        b = agg[name]
        print(f"  {name:12} {b['n']:5d} {b['sem_recall30']:7.4f} {b['beh_recall30']:7.4f} "
              f"{b['comb_recall30']:8.4f} {b['comb_minus_sem']:+7.4f} "
              f"{b['unique_beh30_not_sem30']:11d} {b['unique_beh30_not_sem300']:12d} "
              f"{b['gt_in_train_rate']:9.4f}")
    print(f"\n  per-source recall@30 (all / h7 / h7-unique-vs-sem30):")
    for s in BEH_SOURCES:
        ps = per_source[s]
        print(f"    {s:12} {ps['all_recall30']:.4f} / {ps['h7_recall30']:.4f} "
              f"/ {ps['h7_unique_vs_sem30']}")
    print(f"\n  recovered-GT rank distribution (behavior@30, semantic@30 miss):")
    print(f"    n={rank_dist['n_recovered']}  r1-5={rank_dist['at_rank_1_5']} "
          f"r6-10={rank_dist['at_rank_6_10']} r11-20={rank_dist['at_rank_11_20']} "
          f"r21-30={rank_dist['at_rank_21_30']}  median={rank_dist['median_rank']:.0f}")
    print(f"\n  by n_prior_music bucket (sem@30 / beh@30 / comb-Δ / uniq⊄sem30):")
    for b in range(8):
        bb = by_bucket[str(b)]
        print(f"    h{b}: n={bb['n']:4d}  {bb['sem_recall30']:.4f} / {bb['beh_recall30']:.4f} "
              f"/ {bb['comb_minus_sem']:+.4f} / {bb['unique_beh30_not_sem30']}")
    print(f"\n  GATES:")
    print(f"    recall  (Δcomb@30 ≥ +0.05):      {gate_recall}  ({all_b['comb_minus_sem']:+.4f})")
    print(f"    unique  (≥50 GT β@30 ⊄ sem@30):  {gate_unique}  ({all_b['unique_beh30_not_sem30']})")
    print(f"    usable  (≥50% recovered ≤rank20):{usable}")
    print(f"\n  VERDICT: {verdict}", flush=True)

    out = {
        "experiment": "R104 — behavioral candidate recall audit (item-item + session-NN)",
        "created_at": datetime.now().isoformat(),
        "elapsed_s": time.time() - t0,
        "n_cases": n, "n_train_sessions": len(idx["sess_tracks"]),
        "n_train_transitions": idx["n_trans"],
        "beh_sources": BEH_SOURCES, "beh_weights": BEH_W,
        "params": {"DECAY": DECAY, "SRC_CAP": SRC_CAP, "NN_TOP_SESS": NN_TOP_SESS,
                   "NN_TRACK_CAP": NN_TRACK_CAP, "PRUNE_NEXT": PRUNE_NEXT},
        "aggregate": agg, "by_bucket": by_bucket, "per_source": per_source,
        "recovered_rank_dist": rank_dist,
        "gates": {"recall_+5pct": gate_recall, "unique_ge_50": gate_unique,
                  "usable": usable},
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    print(f"\n{ts()} saved {OUT_JSON}")

    # per-case dump (compact) for R105/R106
    json.dump({"n": n, "results": [
        {k: r[k] for k in ("case_idx", "fold", "n_prior_music", "same_artist",
                           "gt_in_train", "sem_r30", "sem_r300", "beh_r30",
                           "beh_r300", "comb_r30", "per_src_rank", "beh_top60",
                           "sem_top30")}
        for r in results]}, open(OUT_PERCASE, "w"))
    print(f"{ts()} dumped per-case -> {OUT_PERCASE}")
    print(f"{ts()} total elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
