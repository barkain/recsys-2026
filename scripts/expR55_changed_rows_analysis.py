#!/usr/bin/env python3
# ruff: noqa: E402,T201,S301
"""Diagnose the 10 R55-vs-R54c top-1 changes after blind result.

R55 result on Blind-A (2026-05-16):
  nDCG@20 0.4925 → 0.4858 (-0.0067)
  LLM     4.70   → 4.75   (+0.05)
  Composite 0.6106 → 0.6108 (flat)

Production stays R54c/R54b. This script characterises the 10 sessions where
R55 changed the top-1 so we can spot which ones were good swaps (kept for a
potential R55/R54 hybrid) and which were bad swaps (clearly to be rejected).

For each changed session:
  - user query, history (last 3 user/music turns)
  - R54c top-1 metadata vs R55 top-1 metadata (name, artist, tags)
  - R54c response (first 200 chars) — and whether it references each of the
    two competing tracks (alignment-fix heuristic)
  - R55 response (regenerated or reused)
  - LR scores: what the existing LR assigned to R54c's top-1 vs R55's top-1
    when fed each ranker's source-list configuration

Outputs:
  exp/eval/expR55_changed_rows_analysis.json
"""
from __future__ import annotations

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.expF1_cfbpr_retrieval import weighted_rrf  # noqa: E402
from scripts.expR54_phase3_blind_submission import (  # noqa: E402
    POOL_K,
    RRF_K,
    SW,
    _featurize_row,
    load_track_albums,
)
from scripts.expR54c_response_polish import load_catalog_dict  # noqa: E402

SRC_CACHE = REPO / "cache" / "blind_a" / "source_cache.pkl"
R54_BLIND = REPO / "cache" / "r54_production" / "blind_r54_lists.json"
R55_BLIND = REPO / "cache" / "r55_production" / "blind_r55_lists.json"
R54C_SUB = REPO / "exp" / "inference" / "blind_a" / "r54c_polish_submission.json"
R55_SUB = REPO / "exp" / "inference" / "blind_a" / "r55_submission.json"
LR_MODEL = REPO / "cache" / "r54_phase3_lr_model.txt"
ALS_CACHE = REPO / "cache" / "r54_phase3_als.npz"
POP_CACHE = REPO / "cache" / "r54_phase3_track_pop.json"
MAPS_CACHE = REPO / "cache" / "r54_phase3_payload_maps.pkl"

OUT = REPO / "exp" / "eval" / "expR55_changed_rows_analysis.json"


def lr_score_for_tid(rec, variant_pairs, ranker, valid_catalog, track_album,
                    ta, tt, ttl, tat, tmt, als_factors, als_to_idx,
                    track_pop, max_pop, target_tid):
    """Apply LR to {variant} feature config and return the score for target_tid."""
    variant_list = [t for t, _ in variant_pairs[:300]]
    variant_score_map = {t: float(s) for t, s in variant_pairs}
    variant_rank_map = {tid: r + 1 for r, tid in enumerate(variant_list)}
    als_vec = np.array(rec["als_vec"], dtype=np.float32) if rec["als_vec"] is not None else None
    src_lists = {
        "A": rec["src_a"], "B": rec["src_b"], "C": rec["src_c"],
        "D": rec["src_d"], "F": rec["src_f"],
        "ALS": rec["als_tracks"], "R21": rec["r21_list"], "R54": variant_list,
    }
    pool = weighted_rrf(src_lists, SW, topk=POOL_K, k=RRF_K)
    feats = _featurize_row(
        pool, src_lists, rec["r21_rank_map"], variant_rank_map, variant_score_map,
        rec["user_query"], rec["history"], rec["music_turns"], set(rec["music_turns"]),
        ta, tt, ttl, tat, tmt,
        als_factors, als_to_idx, als_vec, track_pop, max_pop, track_album,
    )
    scores = ranker.predict(feats)
    if target_tid not in pool:
        return None, None
    pool_rank = pool.index(target_tid) + 1
    target_score = float(scores[pool.index(target_tid)])
    return target_score, pool_rank


def main():
    import lightgbm as lgb  # type: ignore[reportMissingImports]

    with open(SRC_CACHE, "rb") as f:
        src_cache = pickle.load(f)
    with open(R54_BLIND) as f:
        r54_blind = json.load(f)["lists"]
    with open(R55_BLIND) as f:
        r55_blind = json.load(f)["lists"]
    with open(R54C_SUB) as f:
        r54c = {r["session_id"]: r for r in json.load(f)}
    with open(R55_SUB) as f:
        r55_sub = {r["session_id"]: r for r in json.load(f)}

    catalog = load_catalog_dict()
    track_album = load_track_albums()
    valid_catalog = set(track_album.keys())
    with open(POP_CACHE) as f:
        track_pop = json.load(f)
    als_data = np.load(ALS_CACHE, allow_pickle=True)
    als_factors = als_data["factors"]
    als_ids = als_data["track_ids"].tolist()
    als_to_idx = {tid: i for i, tid in enumerate(als_ids)}
    with open(MAPS_CACHE, "rb") as f:
        maps = pickle.load(f)
    ta = maps["track_artist"]
    tt = maps["track_tags"]
    ttl = maps["track_title_toks"]
    tat = maps["track_artist_toks"]
    tmt = maps["track_meta_toks"]
    max_pop = max(track_pop.values()) if track_pop else 1
    lr_model = lgb.Booster(model_file=str(LR_MODEL))

    changed_sids = [sid for sid in r55_sub
                     if r55_sub[sid]["predicted_track_ids"][0]
                     != r54c[sid]["predicted_track_ids"][0]]
    print(f"R55 vs R54c top-1 changes: {len(changed_sids)}/80\n")

    rows = []
    for sid in changed_sids:
        rec = src_cache[sid]
        r54c_top = r54c[sid]["predicted_track_ids"][0]
        r55_top = r55_sub[sid]["predicted_track_ids"][0]
        r54c_resp = r54c[sid]["predicted_response"]
        r55_resp = r55_sub[sid]["predicted_response"]
        response_changed = r54c_resp != r55_resp

        r54c_meta = catalog.get(r54c_top, {})
        r55_meta = catalog.get(r55_top, {})

        # LR scores: feed R54 ensemble config and R55 config separately, ask each
        # for the OTHER's top-1 score so we can see the disagreement margin
        r54_pairs = r54_blind.get(sid, [])
        r55_pairs = r55_blind.get(sid, [])
        r54c_top_in_r54 = lr_score_for_tid(rec, r54_pairs, lr_model, valid_catalog,
                                             track_album, ta, tt, ttl, tat, tmt,
                                             als_factors, als_to_idx,
                                             track_pop, max_pop, r54c_top)
        r55_top_in_r54 = lr_score_for_tid(rec, r54_pairs, lr_model, valid_catalog,
                                            track_album, ta, tt, ttl, tat, tmt,
                                            als_factors, als_to_idx,
                                            track_pop, max_pop, r55_top)
        r54c_top_in_r55 = lr_score_for_tid(rec, r55_pairs, lr_model, valid_catalog,
                                             track_album, ta, tt, ttl, tat, tmt,
                                             als_factors, als_to_idx,
                                             track_pop, max_pop, r54c_top)
        r55_top_in_r55 = lr_score_for_tid(rec, r55_pairs, lr_model, valid_catalog,
                                            track_album, ta, tt, ttl, tat, tmt,
                                            als_factors, als_to_idx,
                                            track_pop, max_pop, r55_top)

        # Heuristic flags
        r54c_resp_low = r54c_resp.lower()
        r54c_name = (r54c_meta.get("name") or "").lower()
        r55_name = (r55_meta.get("name") or "").lower()
        r54c_artist = (r54c_meta.get("artist") or "").lower()
        r55_artist = (r55_meta.get("artist") or "").lower()

        def quoted(low):
            return f'"{low}"' in r54c_resp_low or f"'{low}'" in r54c_resp_low \
                   or f"{low} by " in r54c_resp_low

        r54c_resp_describes_r54c = bool(r54c_name) and quoted(r54c_name)
        r54c_resp_describes_r55 = bool(r55_name) and quoted(r55_name)

        same_artist = r54c_artist and r55_artist and \
                       (set(r54c_artist.split(", ")) & set(r55_artist.split(", ")))

        verdict = []
        if r54c_resp_describes_r55 and not r54c_resp_describes_r54c:
            verdict.append("R55_FIXES_R54c_ALIGNMENT")
        elif r54c_resp_describes_r54c and not r54c_resp_describes_r55:
            verdict.append("R54c_RESPONSE_WAS_CORRECT_FOR_R54c_TRACK")
        if same_artist:
            verdict.append("SAME_ARTIST_SWAP")
        else:
            verdict.append("DIFFERENT_ARTIST")

        row = {
            "sid": sid,
            "user_query": rec["user_query"],
            "history_tail": [
                {"role": h["role"],
                 "content": str(h.get("content", ""))[:100]}
                for h in rec["history"][-4:]
            ],
            "r54c_top": {
                "tid": r54c_top,
                "name": r54c_meta.get("name") or "?",
                "artist": r54c_meta.get("artist") or "?",
                "tags": r54c_meta.get("tags", [])[:5],
                "lr_score_in_r54_config": r54c_top_in_r54[0],
                "lr_rank_in_r54_pool": r54c_top_in_r54[1],
                "lr_score_in_r55_config": r54c_top_in_r55[0],
                "lr_rank_in_r55_pool": r54c_top_in_r55[1],
            },
            "r55_top": {
                "tid": r55_top,
                "name": r55_meta.get("name") or "?",
                "artist": r55_meta.get("artist") or "?",
                "tags": r55_meta.get("tags", [])[:5],
                "lr_score_in_r54_config": r55_top_in_r54[0],
                "lr_rank_in_r54_pool": r55_top_in_r54[1],
                "lr_score_in_r55_config": r55_top_in_r55[0],
                "lr_rank_in_r55_pool": r55_top_in_r55[1],
            },
            "responses": {
                "r54c": r54c_resp,
                "r55": r55_resp,
                "response_changed": response_changed,
            },
            "heuristics": {
                "r54c_response_describes_r54c_track": r54c_resp_describes_r54c,
                "r54c_response_describes_r55_track": r54c_resp_describes_r55,
                "same_artist": bool(same_artist),
                "verdict_tags": verdict,
            },
        }
        rows.append(row)

        # Pretty print
        print("=" * 78)
        print(f"sid={sid[:8]}   tags={verdict}")
        print(f"  user_query: {rec['user_query'][:140]}")
        if rec["history"]:
            last_user = next((h for h in reversed(rec["history"]) if h["role"] == "user"), None)
            if last_user and last_user.get("content"):
                print(f"  prior user: {str(last_user['content'])[:140]}")
        print(f"  R54c top-1: {r54c_meta.get('name')!r} by {r54c_meta.get('artist')!r}")
        print(f"              tags={r54c_meta.get('tags', [])[:5]}")
        print(f"              LR rank in R54-config={r54c_top_in_r54[1]} score={r54c_top_in_r54[0]}")
        print(f"              LR rank in R55-config={r54c_top_in_r55[1]} score={r54c_top_in_r55[0]}")
        print(f"  R55  top-1: {r55_meta.get('name')!r} by {r55_meta.get('artist')!r}")
        print(f"              tags={r55_meta.get('tags', [])[:5]}")
        print(f"              LR rank in R54-config={r55_top_in_r54[1]} score={r55_top_in_r54[0]}")
        print(f"              LR rank in R55-config={r55_top_in_r55[1]} score={r55_top_in_r55[0]}")
        print(f"  R54c response (first 180): {r54c_resp[:180]}")
        if response_changed:
            print(f"  R55  response (first 180): {r55_resp[:180]}")
        else:
            print(f"  R55  response: [kept R54c]")
        print()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "n_changed": len(changed_sids),
            "rows": rows,
            "created_at": datetime.now().isoformat(),
        }, f, indent=2)
    print(f"\nSaved diagnostic: {OUT}")


if __name__ == "__main__":
    main()
