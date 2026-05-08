#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R32: Last-turn error taxonomy + oracle analysis on hist_7.

Quantify where the nDCG gap comes from:
- candidate coverage per source
- oracle ceilings (perfect rerank, perfect union)
- error categories (missing, in-pool-but-missed, top20-but-wrong-order)
- 20 concrete failure examples with source ranks
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    print(f"{ts()} R32: Last-Turn Error Taxonomy + Oracle")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    ta = payload["track_artist"]

    with open(R21_OOF) as f:
        r21_lists = json.load(f)

    # Load additional source lists if available
    extra_sources: dict[str, list] = {}
    als_path = REPO / "cache" / "r26" / "q3_dense_results.json"
    if als_path.exists():
        with open(als_path) as f:
            extra_sources["Q3_intent"] = json.load(f)
    r31_path = REPO / "cache" / "r31" / "fold0_r31_lists.json"
    if r31_path.exists():
        with open(r31_path) as f:
            extra_sources["R31_V0"] = json.load(f)
    r31v1_path = REPO / "cache" / "r31v1" / "fold0_r31v1_lists.json"
    if r31v1_path.exists():
        with open(r31v1_path) as f:
            extra_sources["R31_V1"] = json.load(f)

    # Train track set
    train_tracks = set()
    from datasets import DownloadConfig, load_dataset
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))["train"]
    for item in ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                train_tracks.add(str(c["content"]).strip())
    del ds

    h7_idx = [i for i in range(len(cases)) if cases[i]["n_prior_music"] == 7]
    print(f"\n{ts()} hist_7 cases: {len(h7_idx)}")

    # ---------------------------------------------------------------
    # 1. Source coverage analysis
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("1. SOURCE COVERAGE (hist_7)")
    print(f"{'='*70}")

    sources = {
        "A": [payload["src_a"][i] for i in h7_idx],
        "B": [payload["src_b"][i] for i in h7_idx],
        "C": [payload["src_c"][i] for i in h7_idx],
        "D": [payload["src_d"][i] for i in h7_idx],
        "F": [payload["src_f"][i] for i in h7_idx],
        "R21": [r21_lists[i] for i in h7_idx],
    }

    # ALS from payload
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_lists = []
    for i in h7_idx:
        played = cases[i]["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_lists.append([als_track_ids[j] for j in top_idx])
        else:
            als_lists.append([])
    sources["ALS"] = als_lists
    del als_factors, als_track_ids, als_track_to_idx

    h7_cases = [cases[i] for i in h7_idx]
    h7_gts = [c["gt"] for c in h7_cases]

    print(f"\n  {'Source':<15} {'@20':>8} {'@50':>8} {'@100':>8} {'@200':>8} {'@300':>8}")
    print(f"  {'-'*55}")
    for src_name, src_lists in sources.items():
        row = ""
        for k in [20, 50, 100, 200, 300]:
            hit = sum(1 for j in range(len(h7_cases))
                     if h7_gts[j] in set(src_lists[j][:k]))
            if k == 20:
                row = f"  {src_name:<15} {hit:>8}"
            else:
                row += f" {hit:>8}"
        print(row)

    # Union coverage
    print(f"\n  {'Union':<15} {'@20':>8} {'@50':>8} {'@100':>8} {'@200':>8} {'@300':>8}")
    print(f"  {'-'*55}")
    row = ""
    for k in [20, 50, 100, 200, 300]:
        union_hit = 0
        for j in range(len(h7_cases)):
            all_cands = set()
            for src_lists_v in sources.values():
                all_cands.update(src_lists_v[j][:k])
            if h7_gts[j] in all_cands:
                union_hit += 1
        if k == 20:
            row = f"  {'ALL_UNION':<15} {union_hit:>8}"
        else:
            row += f" {union_hit:>8}"
    print(row)

    # ---------------------------------------------------------------
    # 2. Error categories
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("2. ERROR CATEGORIES (hist_7)")
    print(f"{'='*70}")

    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}

    cats = Counter()
    examples: list[dict] = []

    for j in range(len(h7_cases)):
        c = h7_cases[j]
        gt = c["gt"]
        i_global = h7_idx[j]

        src_lists_j = {
            "A": payload["src_a"][i_global], "B": payload["src_b"][i_global],
            "C": payload["src_c"][i_global], "D": payload["src_d"][i_global],
            "F": payload["src_f"][i_global], "ALS": sources["ALS"][j],
            "R21": r21_lists[i_global],
        }
        pool = weighted_rrf(src_lists_j, sw, topk=300, k=20)

        gt_in_sources = {}
        for sname, slist in src_lists_j.items():
            if gt in slist:
                gt_in_sources[sname] = slist.index(gt) + 1

        in_pool = gt in pool
        pool_rank = pool.index(gt) + 1 if in_pool else -1
        in_any_source = len(gt_in_sources) > 0
        gt_seen = gt in train_tracks
        gt_artist = ta.get(gt, "")
        played_artists = {ta.get(t, "") for t in c["music_turns"]} - {""}
        same_artist = gt_artist in played_artists if gt_artist else False

        # Query mention check
        query_lower = c["user_query"].lower()
        artist_mentioned = gt_artist.lower() in query_lower if gt_artist else False

        if in_pool and pool_rank <= 20:
            cats["in_top20"] += 1
        elif in_pool and pool_rank <= 100:
            cats["in_pool_21_100"] += 1
        elif in_pool:
            cats["in_pool_101_300"] += 1
        elif in_any_source:
            cats["in_source_not_pool"] += 1
        elif gt_seen:
            cats["seen_not_sourced"] += 1
        else:
            cats["unseen_unreachable"] += 1

        if not in_pool or pool_rank > 20:
            examples.append({
                "case_idx": i_global,
                "session_id": c["session_id"],
                "gt": gt,
                "gt_artist": gt_artist,
                "same_artist": same_artist,
                "artist_mentioned": artist_mentioned,
                "gt_seen": gt_seen,
                "pool_rank": pool_rank,
                "source_ranks": gt_in_sources,
                "user_query": c["user_query"][:100],
                "n_sources_with_gt": len(gt_in_sources),
                "category": ("in_pool_21_100" if in_pool and pool_rank <= 100
                            else "in_pool_101_300" if in_pool
                            else "in_source_not_pool" if in_any_source
                            else "seen_not_sourced" if gt_seen
                            else "unseen_unreachable"),
            })

    print(f"\n  {'Category':<25} {'Count':>8} {'%':>8}")
    print(f"  {'-'*41}")
    total = len(h7_cases)
    for cat in ["in_top20", "in_pool_21_100", "in_pool_101_300",
                "in_source_not_pool", "seen_not_sourced", "unseen_unreachable"]:
        n = cats.get(cat, 0)
        print(f"  {cat:<25} {n:>8} {n/total*100:>7.1f}%")
    print(f"  {'TOTAL':<25} {total:>8}")

    # ---------------------------------------------------------------
    # 3. Oracle ceilings
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("3. ORACLE CEILINGS (hist_7 nDCG@20)")
    print(f"{'='*70}")

    def oracle_ndcg(cases_list, candidate_fn):
        ndcg_sum = 0
        for j, c in enumerate(cases_list):
            gt = c["gt"]
            cands = candidate_fn(j)
            if gt in cands[:20]:
                pos = cands.index(gt)
                ndcg_sum += 1.0 / np.log2(pos + 2)
        return ndcg_sum / len(cases_list)

    # Precompute pools for oracle analysis
    h7_pools: list[list[str]] = []
    for j in range(len(h7_cases)):
        ig = h7_idx[j]
        sl = {"A": payload["src_a"][ig], "B": payload["src_b"][ig],
              "C": payload["src_c"][ig], "D": payload["src_d"][ig],
              "F": payload["src_f"][ig], "ALS": sources["ALS"][j],
              "R21": r21_lists[ig]}
        h7_pools.append(weighted_rrf(sl, sw, topk=300, k=20))

    def oracle_list(j, cand_set):
        gt = h7_cases[j]["gt"]
        if gt in cand_set:
            return [gt] + [t for t in list(cand_set) if t != gt][:19]
        return []

    current_ndcg = oracle_ndcg(h7_cases, lambda j: r21_lists[h7_idx[j]][:20])
    print(f"\n  Current R21 top-20 (as-is):     {current_ndcg:.4f}")

    perfect_pool300 = oracle_ndcg(
        h7_cases, lambda j: oracle_list(j, set(h7_pools[j])))
    print(f"  Perfect rerank pool@300:        {perfect_pool300:.4f}")

    perfect_r21_300 = oracle_ndcg(
        h7_cases, lambda j: oracle_list(j, set(r21_lists[h7_idx[j]][:300])))
    print(f"  Perfect rerank R21@300:         {perfect_r21_300:.4f}")

    def all_union_set(j):
        u = set()
        for s in sources.values():
            u.update(s[j][:300])
        return u

    perfect_all = oracle_ndcg(
        h7_cases, lambda j: oracle_list(j, all_union_set(j)))
    print(f"  Perfect rerank ALL sources:     {perfect_all:.4f}")

    # ---------------------------------------------------------------
    # 4. Attribute analysis
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("4. ATTRIBUTE ANALYSIS (hist_7)")
    print(f"{'='*70}")

    seen_h7 = sum(1 for c in h7_cases if c["gt"] in train_tracks)
    unseen_h7 = len(h7_cases) - seen_h7
    same_art = sum(1 for c in h7_cases
                  if ta.get(c["gt"], "") and ta.get(c["gt"], "") in
                  {ta.get(t, "") for t in c["music_turns"]})
    diff_art = len(h7_cases) - same_art

    print("\n  Seen vs unseen GT:")
    print(f"    seen: {seen_h7} ({seen_h7/len(h7_cases)*100:.1f}%)")
    print(f"    unseen: {unseen_h7} ({unseen_h7/len(h7_cases)*100:.1f}%)")

    print("\n  Same-artist vs different-artist GT:")
    print(f"    same: {same_art} ({same_art/len(h7_cases)*100:.1f}%)")
    print(f"    diff: {diff_art} ({diff_art/len(h7_cases)*100:.1f}%)")

    # Per-attribute hit rates
    for attr_name, attr_fn in [
        ("seen", lambda c: c["gt"] in train_tracks),
        ("unseen", lambda c: c["gt"] not in train_tracks),
        ("same_artist", lambda c: ta.get(c["gt"], "") and ta.get(c["gt"], "") in
         {ta.get(t, "") for t in c["music_turns"]}),
        ("diff_artist", lambda c: not (ta.get(c["gt"], "") and ta.get(c["gt"], "") in
         {ta.get(t, "") for t in c["music_turns"]})),
    ]:
        subset = [j for j, c in enumerate(h7_cases) if attr_fn(c)]
        if not subset:
            continue
        r21_hit = sum(1 for j in subset if h7_gts[j] in set(r21_lists[h7_idx[j]][:20]))
        pool_hit = sum(1 for j in subset if h7_gts[j] in set(
            weighted_rrf({"A": payload["src_a"][h7_idx[j]], "B": payload["src_b"][h7_idx[j]],
                         "C": payload["src_c"][h7_idx[j]], "D": payload["src_d"][h7_idx[j]],
                         "F": payload["src_f"][h7_idx[j]], "ALS": sources["ALS"][j],
                         "R21": r21_lists[h7_idx[j]]}, sw, topk=300, k=20)))
        print(f"\n  {attr_name} ({len(subset)}):")
        print(f"    R21@20: {r21_hit}/{len(subset)} ({r21_hit/len(subset)*100:.1f}%)")
        print(f"    pool@300: {pool_hit}/{len(subset)} ({pool_hit/len(subset)*100:.1f}%)")

    # ---------------------------------------------------------------
    # 5. Concrete failure examples
    # ---------------------------------------------------------------
    print(f"\n{'='*70}")
    print("5. CONCRETE FAILURE EXAMPLES (20 cases)")
    print(f"{'='*70}")

    examples.sort(key=lambda x: x["n_sources_with_gt"], reverse=True)
    for ex in examples[:20]:
        print(f"\n  Case {ex['case_idx']} [{ex['category']}]")
        print(f"    GT: {ex['gt']} artist={ex['gt_artist']}")
        print(f"    same_artist={ex['same_artist']} mentioned={ex['artist_mentioned']} "
              f"seen={ex['gt_seen']}")
        print(f"    pool_rank={ex['pool_rank']} sources={ex['source_ranks']}")
        print(f"    query: {ex['user_query']}")

    # Save
    out_path = REPO / "exp" / "eval" / "expR32_error_taxonomy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "categories": dict(cats),
        "oracles": {
            "current_r21_top20": current_ndcg,
            "perfect_pool300": perfect_pool300,
            "perfect_r21_300": perfect_r21_300,
            "perfect_all_sources": perfect_all,
        },
        "attributes": {
            "seen": seen_h7, "unseen": unseen_h7,
            "same_artist": same_art, "diff_artist": diff_art,
        },
        "n_examples": len(examples),
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")


if __name__ == "__main__":
    main()
