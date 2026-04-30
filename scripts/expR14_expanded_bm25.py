#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R14: Expanded BM25 content retrieval.

Goal: Recover cold/pop=0/different-artist GTs that behavioral sources cannot reach.
Hypothesis: Tag-heavy BM25 with field weighting and query expansion beats flat BM25.

Stage 1 only — no fusion, no LambdaRank, no blind.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import bm25s
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank_grouped import als_session_vector, build_als, grouped_session_folds
from scripts.expS2_lr_v2 import build_popularity_stats

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20

# Mood/genre/activity terms for deterministic query expansion
MOOD_TERMS = {
    "chill": "chillout lounge ambient relaxing mellow",
    "relaxing": "chillout lounge ambient mellow calm",
    "calm": "ambient mellow soft acoustic peaceful",
    "upbeat": "energetic dance party uplifting happy",
    "energetic": "dance party power upbeat workout",
    "sad": "melancholy emotional ballad heartbreak",
    "melancholy": "sad emotional introspective dark",
    "happy": "upbeat feel-good joyful positive cheerful",
    "romantic": "love ballad smooth soft intimate",
    "dark": "gothic noir brooding intense atmospheric",
    "intense": "heavy aggressive powerful epic driving",
    "dreamy": "ethereal ambient shoegaze atmospheric",
    "funky": "funk groove bass danceable soul",
    "groovy": "funk groove soul rhythm danceable",
    "workout": "energetic power intense driving motivational",
    "study": "ambient instrumental focus calm lo-fi",
    "sleep": "ambient calm quiet soft peaceful",
    "party": "dance upbeat energetic club electronic",
    "road trip": "rock driving upbeat classic anthemic",
    "morning": "acoustic soft upbeat fresh positive",
}

GENRE_TERMS = {
    "rock": "rock alternative indie guitar",
    "pop": "pop mainstream catchy",
    "hip hop": "hip-hop rap hiphop",
    "hip-hop": "hip-hop rap hiphop",
    "rap": "hip-hop rap hiphop",
    "jazz": "jazz swing bebop smooth",
    "classical": "classical orchestral symphony",
    "electronic": "electronic edm synth dance",
    "edm": "electronic edm dance club",
    "r&b": "rnb r&b soul rhythm",
    "soul": "soul rnb motown rhythm",
    "country": "country americana folk",
    "folk": "folk acoustic singer-songwriter",
    "metal": "metal heavy thrash",
    "punk": "punk hardcore rock",
    "blues": "blues guitar rhythm",
    "reggae": "reggae dub ska",
    "indie": "indie alternative lo-fi",
    "grunge": "grunge alternative 90s rock",
    "latin": "latin salsa reggaeton",
}


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def load_full_metadata():
    """Load all track metadata from HuggingFace cache."""
    from datasets import Dataset, concatenate_datasets
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    splits = []
    for split in ["all_tracks", "test_tracks"]:
        matches = sorted(hf_cache.glob(
            f"talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            f"talk_play_data-challenge-track-metadata-{split}.arrow"))
        if matches:
            splits.append(Dataset.from_file(str(matches[-1])))
    if not splits:
        raise RuntimeError(f"No arrow files found in {hf_cache}")
    combined = concatenate_datasets(splits)
    cols = combined.to_dict()
    track_ids = [str(tid) for tid in cols["track_id"]]
    meta = {}
    for i, tid in enumerate(track_ids):
        meta[tid] = {k: cols[k][i] for k in cols}
    return meta, track_ids


def build_index_variant(meta, track_ids, variant):
    """Build a BM25 index with a specific document construction strategy."""
    corpus_texts = []
    for tid in track_ids:
        m = meta[tid]
        name = " ".join(m.get("track_name", []) if isinstance(m.get("track_name"), list) else [str(m.get("track_name", ""))])
        artist = " ".join(m.get("artist_name", []) if isinstance(m.get("artist_name"), list) else [str(m.get("artist_name", ""))])
        album = " ".join(m.get("album_name", []) if isinstance(m.get("album_name"), list) else [str(m.get("album_name", ""))])
        tags = m.get("tag_list", [])
        if isinstance(tags, list):
            tags_str = " ".join(str(t) for t in tags)
        else:
            tags_str = str(tags)

        if variant == "base":
            doc = f"{name} {artist} {album} {tags_str}"
        elif variant == "tags_3x":
            doc = f"{name} {artist} {album} {tags_str} {tags_str} {tags_str}"
        elif variant == "tags_5x":
            doc = f"{name} {artist} {album} {tags_str} {tags_str} {tags_str} {tags_str} {tags_str}"
        elif variant == "artist_tags_3x":
            doc = f"{name} {artist} {artist} {album} {tags_str} {tags_str} {tags_str}"
        elif variant == "tags_only":
            doc = f"{tags_str} {tags_str} {tags_str}"
        else:
            raise ValueError(f"Unknown variant: {variant}")
        corpus_texts.append(doc)

    corpus_tokens = bm25s.tokenize(corpus_texts)
    model = bm25s.BM25()
    model.index(corpus_tokens)
    return model, track_ids


def expand_query(query_text):
    """Deterministic query expansion with mood/genre terms."""
    ql = query_text.lower()
    expansions = []
    for term, expansion in MOOD_TERMS.items():
        if term in ql:
            expansions.append(expansion)
    for term, expansion in GENRE_TERMS.items():
        if term in ql:
            expansions.append(expansion)
    if expansions:
        return query_text + " " + " ".join(expansions)
    return query_text


def build_query_variants(cases, ta, tt):
    """Build multiple query variants for each case."""
    variants = {}

    # V1: current user query only (baseline — same as source B/C query)
    variants["q_current"] = [c["user_query"] for c in cases]

    # V2: current query + expanded mood/genre
    variants["q_expanded"] = [expand_query(c["user_query"]) for c in cases]

    # V3: current query + last played track tags (if any)
    q_with_tags = []
    for c in cases:
        q = c["user_query"]
        played = c["music_turns"]
        if played:
            last_tags = tt.get(played[-1], set())
            if last_tags:
                q = q + " " + " ".join(last_tags)
        q_with_tags.append(q)
    variants["q_plus_tags"] = q_with_tags

    # V4: current query + all played artist names
    q_with_artists = []
    for c in cases:
        q = c["user_query"]
        played = c["music_turns"]
        artists = set()
        for tid in played:
            a = ta.get(tid, "")
            if isinstance(a, list):
                artists.update(a)
            elif a:
                artists.add(a)
        if artists:
            q = q + " " + " ".join(artists)
        q_with_artists.append(q)
    variants["q_plus_artists"] = q_with_artists

    # V5: expanded + tags + artists (kitchen sink)
    q_full = []
    for i, c in enumerate(cases):
        q = expand_query(c["user_query"])
        played = c["music_turns"]
        parts = [q]
        if played:
            last_tags = tt.get(played[-1], set())
            if last_tags:
                parts.append(" ".join(last_tags))
        artists = set()
        for tid in played:
            a = ta.get(tid, "")
            if isinstance(a, list):
                artists.update(a)
            elif a:
                artists.add(a)
        if artists:
            parts.append(" ".join(artists))
        q_full.append(" ".join(parts))
    variants["q_kitchen_sink"] = q_full

    return variants


def retrieve_batch(model, track_ids, queries, topk=200):
    tokens = bm25s.tokenize([q.lower() for q in queries])
    results, scores = model.retrieve(tokens, k=topk)
    out = []
    for i in range(len(queries)):
        row = results[i]
        out.append([track_ids[int(idx)] for idx in row if int(idx) >= 0])
    return out


def evaluate_source(name, x_results, cases, abcdf_pools, payload, als_source,
                    ta, tt, track_pop, n):
    """Evaluate a candidate source X against baselines."""
    hit20 = hit50 = hit100 = hit200 = 0
    unique_vs_pool = 0
    unique_unreachable = 0
    pop0_recovery = 0
    diff_artist_recovery = 0
    hist0_recovery = 0

    overlap_B = []
    overlap_C = []

    hist_buckets = defaultdict(lambda: {"n": 0, "hit200": 0, "unique": 0, "unreachable": 0})

    diff_artist_total = 0
    diff_artist_fused_hit = 0
    hist0_total = 0
    hist0_fused_hit = 0
    pop0_total = 0
    pop0_fused_hit = 0

    for i, c in enumerate(cases):
        gt = c["gt"]
        played = c["music_turns"]
        n_hist = len(played)
        bk = f"hist_{min(n_hist, 7)}"
        hist_buckets[bk]["n"] += 1

        x_set = set(x_results[i][:200])
        fused_pool = abcdf_pools[i] | x_set

        if gt in x_results[i][:20]: hit20 += 1
        if gt in x_results[i][:50]: hit50 += 1
        if gt in x_results[i][:100]: hit100 += 1
        gt_hit = gt in x_set
        if gt_hit:
            hit200 += 1
            hist_buckets[bk]["hit200"] += 1

            if gt not in abcdf_pools[i]:
                unique_vs_pool += 1
                hist_buckets[bk]["unique"] += 1

                in_any_source = False
                for sname in ["src_a", "src_b", "src_c", "src_d", "src_f"]:
                    if gt in payload[sname][i][:500]:
                        in_any_source = True
                        break
                if not in_any_source and gt not in als_source[i][:500]:
                    unique_unreachable += 1
                    hist_buckets[bk]["unreachable"] += 1

            if track_pop.get(gt, 0) == 0:
                pop0_recovery += 1

            if played:
                last_artist = ta.get(played[-1], "")
                gt_artist = ta.get(gt, "")
                if isinstance(last_artist, list): last_artist = last_artist[0] if last_artist else ""
                if isinstance(gt_artist, list): gt_artist = gt_artist[0] if gt_artist else ""
                if gt_artist and last_artist and gt_artist != last_artist:
                    diff_artist_recovery += 1

            if n_hist == 0:
                hist0_recovery += 1

        # Fused pool stats
        if played:
            last_artist = ta.get(played[-1], "")
            gt_artist = ta.get(gt, "")
            if isinstance(last_artist, list): last_artist = last_artist[0] if last_artist else ""
            if isinstance(gt_artist, list): gt_artist = gt_artist[0] if gt_artist else ""
            if gt_artist and last_artist and gt_artist != last_artist:
                diff_artist_total += 1
                if gt in fused_pool:
                    diff_artist_fused_hit += 1
        if n_hist == 0:
            hist0_total += 1
            if gt in fused_pool:
                hist0_fused_hit += 1
        if track_pop.get(gt, 0) == 0:
            pop0_total += 1
            if gt in fused_pool:
                pop0_fused_hit += 1

        b_set = set(payload["src_b"][i][:200])
        c_set = set(payload["src_c"][i][:200])
        if x_set:
            overlap_B.append(len(x_set & b_set) / len(x_set))
            overlap_C.append(len(x_set & c_set) / len(x_set))

    return {
        "hit20": hit20, "hit50": hit50, "hit100": hit100, "hit200": hit200,
        "unique_vs_pool": unique_vs_pool, "unique_unreachable": unique_unreachable,
        "pop0_recovery": pop0_recovery, "diff_artist_recovery": diff_artist_recovery,
        "hist0_recovery": hist0_recovery,
        "overlap_B": float(np.mean(overlap_B)) if overlap_B else 0,
        "overlap_C": float(np.mean(overlap_C)) if overlap_C else 0,
        "diff_artist_total": diff_artist_total,
        "diff_artist_fused_hit": diff_artist_fused_hit,
        "hist0_total": hist0_total, "hist0_fused_hit": hist0_fused_hit,
        "pop0_total": pop0_total, "pop0_fused_hit": pop0_fused_hit,
        "hist_buckets": {k: dict(v) for k, v in hist_buckets.items()},
    }


def main():
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)

    ta = payload["track_artist"]
    tt = payload["track_tags"]
    track_pop = build_popularity_stats()

    print(f"{ts()} Loading metadata & building indices...", flush=True)
    meta, all_track_ids = load_full_metadata()
    print(f"  {len(all_track_ids)} tracks in metadata", flush=True)

    # Build ALS for baseline pool
    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            played_idx = [als_track_to_idx[t] for t in played if t in als_track_to_idx]
            for idx in played_idx:
                scores[idx] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    print(f"{ts()} Building ABCDF+ALS@200 pools...", flush=True)
    base_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}
    abcdf_pools = []
    for i in range(n):
        src_lists = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
        }
        pool = weighted_rrf(src_lists, base_weights, topk=200, k=RRF_K)
        abcdf_pools.append(set(pool))

    # Compute baselines
    base_diff_artist_hit = 0
    diff_artist_total = 0
    base_hist0_hit = 0
    hist0_total = 0
    base_pop0_hit = 0
    pop0_total = 0
    for i, c in enumerate(cases):
        gt = c["gt"]
        played = c["music_turns"]
        if played:
            last_artist = ta.get(played[-1], "")
            gt_artist = ta.get(gt, "")
            if isinstance(last_artist, list): last_artist = last_artist[0] if last_artist else ""
            if isinstance(gt_artist, list): gt_artist = gt_artist[0] if gt_artist else ""
            if gt_artist and last_artist and gt_artist != last_artist:
                diff_artist_total += 1
                if gt in abcdf_pools[i]:
                    base_diff_artist_hit += 1
        if len(played) == 0:
            hist0_total += 1
            if gt in abcdf_pools[i]:
                base_hist0_hit += 1
        if track_pop.get(gt, 0) == 0:
            pop0_total += 1
            if gt in abcdf_pools[i]:
                base_pop0_hit += 1

    base_diff_rate = base_diff_artist_hit / diff_artist_total if diff_artist_total else 0
    base_hist0_rate = base_hist0_hit / hist0_total if hist0_total else 0
    base_pop0_rate = base_pop0_hit / pop0_total if pop0_total else 0
    print(f"\n{ts()} Baselines (ABCDF+ALS@200):")
    print(f"  diff-artist pool_hit: {base_diff_artist_hit}/{diff_artist_total} ({base_diff_rate:.1%})")
    print(f"  hist_0 pool_hit:      {base_hist0_hit}/{hist0_total} ({base_hist0_rate:.1%})")
    print(f"  pop=0 pool_hit:       {base_pop0_hit}/{pop0_total} ({base_pop0_rate:.1%})")

    # Build query variants
    print(f"\n{ts()} Building query variants...", flush=True)
    query_variants = build_query_variants(cases, ta, tt)

    # Build index variants
    index_variants = ["base", "tags_3x", "tags_5x", "artist_tags_3x"]
    print(f"{ts()} Building {len(index_variants)} index variants...", flush=True)

    all_results = {}
    for idx_var in index_variants:
        t_idx = time.time()
        model, tids = build_index_variant(meta, all_track_ids, idx_var)
        print(f"  {idx_var} index built in {time.time()-t_idx:.1f}s", flush=True)

        for q_var_name, queries in query_variants.items():
            config_name = f"{idx_var}__{q_var_name}"
            t_ret = time.time()
            results = retrieve_batch(model, tids, queries, topk=200)
            ret_time = time.time() - t_ret
            ev = evaluate_source(config_name, results, cases, abcdf_pools, payload,
                                als_source, ta, tt, track_pop, n)
            ev["retrieval_time"] = ret_time

            fused_diff_rate = ev["diff_artist_fused_hit"] / ev["diff_artist_total"] if ev["diff_artist_total"] else 0
            diff_lift = fused_diff_rate - base_diff_rate
            fused_hist0_rate = ev["hist0_fused_hit"] / ev["hist0_total"] if ev["hist0_total"] else 0
            fused_pop0_rate = ev["pop0_fused_hit"] / ev["pop0_total"] if ev["pop0_total"] else 0

            print(f"\n  {config_name}:")
            print(f"    hit@20={ev['hit20']}/{n} ({ev['hit20']/n:.1%})  "
                  f"hit@50={ev['hit50']}/{n}  hit@100={ev['hit100']}/{n}  "
                  f"hit@200={ev['hit200']}/{n} ({ev['hit200']/n:.1%})")
            print(f"    unique_vs_pool={ev['unique_vs_pool']}  "
                  f"unique_unreachable={ev['unique_unreachable']}  "
                  f"pop0={ev['pop0_recovery']}  "
                  f"diff_artist={ev['diff_artist_recovery']}  "
                  f"hist0={ev['hist0_recovery']}")
            print(f"    overlap B={ev['overlap_B']:.1%}  overlap C={ev['overlap_C']:.1%}")
            print(f"    fused diff-artist lift={diff_lift:+.1%}  "
                  f"fused hist0={fused_hist0_rate:.1%} ({fused_hist0_rate-base_hist0_rate:+.1%})  "
                  f"fused pop0={fused_pop0_rate:.1%} ({fused_pop0_rate-base_pop0_rate:+.1%})")

            gate_unreach = ev["unique_unreachable"] >= 150
            gate_diff = diff_lift >= 0.03
            mean_overlap = (ev["overlap_B"] + ev["overlap_C"]) / 2
            gate_not_redundant = mean_overlap < 0.5
            print(f"    GATES: unreach={'PASS' if gate_unreach else 'FAIL'}({ev['unique_unreachable']})  "
                  f"diff_lift={'PASS' if gate_diff else 'FAIL'}({diff_lift:+.1%})  "
                  f"not_redundant={'PASS' if gate_not_redundant else 'FAIL'}({mean_overlap:.1%})")

            ev["gates"] = {"unreachable": gate_unreach, "diff_artist": gate_diff,
                           "not_redundant": gate_not_redundant}
            ev["diff_artist_lift"] = diff_lift
            all_results[config_name] = ev

        del model

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"{ts()} R14 Stage 1 complete. Elapsed: {elapsed:.1f}s")

    best = max(all_results.items(), key=lambda kv: kv[1]["unique_unreachable"])
    print(f"\nBest by unique_unreachable: {best[0]} = {best[1]['unique_unreachable']}")
    any_pass = any(all(r["gates"].values()) for r in all_results.values())
    print(f"Any config passes all gates: {'YES → proceed to Stage 2' if any_pass else 'NO → stop here'}")

    # Top 5 by unique_unreachable
    print(f"\nTop 5 configs by unique_unreachable:")
    top5 = sorted(all_results.items(), key=lambda kv: kv[1]["unique_unreachable"], reverse=True)[:5]
    for name, ev in top5:
        print(f"  {name}: unreach={ev['unique_unreachable']}  hit@200={ev['hit200']}  "
              f"diff_lift={ev['diff_artist_lift']:+.1%}  overlap_BC={((ev['overlap_B']+ev['overlap_C'])/2):.1%}")

    out_path = REPO_ROOT / "exp" / "eval" / "expR14_expanded_bm25.json"
    with open(out_path, "w") as f:
        json.dump({"stage": "stage1", "results": all_results,
                   "baselines": {"diff_artist_pool_hit": base_diff_rate,
                                 "hist0_pool_hit": base_hist0_rate,
                                 "pop0_pool_hit": base_pop0_rate},
                   "elapsed_s": elapsed}, f, indent=2, default=str)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
