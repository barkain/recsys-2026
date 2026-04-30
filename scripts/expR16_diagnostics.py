#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R16 diagnostics: graph sanity, per-variant eval, popularity debiasing.

Run after expR16_graph_retrieval.py to validate results aren't just
popularity hubs and to isolate which edge types contribute signal.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.expF1_cfbpr_retrieval import weighted_rrf
from scripts.expS2_lambdarank_grouped import als_session_vector, build_als
from scripts.expS2_lr_v2 import build_popularity_stats

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_graph_variant(session_tracks, track_artists, track_tags_map,
                        session_users, track_list, track_idx, session_idx,
                        user_idx, artist_idx, tag_idx, N, variant="full"):
    """Build graph with specific edge types enabled."""
    rows, cols, weights = [], [], []

    include_cooccur = variant in ("full", "cooccur_only", "behavioral")
    include_transition = variant in ("full", "transition_only", "behavioral")
    include_user = variant in ("full", "user_only", "behavioral")
    include_artist = variant in ("full", "hetero_only")
    include_tag = variant in ("full", "hetero_only")
    include_session = variant in ("full", "behavioral")

    if include_session:
        for sid, tracks in session_tracks.items():
            if sid not in session_idx:
                continue
            si = session_idx[sid]
            for pos, tid in enumerate(tracks):
                if tid not in track_idx:
                    continue
                ti = track_idx[tid]
                w = 1.0 / (pos + 1)
                rows.extend([si, ti])
                cols.extend([ti, si])
                weights.extend([w, w])

    if include_cooccur:
        cooccur_counts = Counter()
        for sid, tracks in session_tracks.items():
            unique = list(dict.fromkeys(t for t in tracks if t in track_idx))
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    cooccur_counts[(unique[i], unique[j])] += 1
        for (t1, t2), cnt in cooccur_counts.items():
            i1, i2 = track_idx[t1], track_idx[t2]
            w = float(cnt)
            rows.extend([i1, i2])
            cols.extend([i2, i1])
            weights.extend([w, w])

    if include_transition:
        transition_counts = Counter()
        for sid, tracks in session_tracks.items():
            for i in range(len(tracks) - 1):
                if tracks[i] in track_idx and tracks[i + 1] in track_idx:
                    transition_counts[(tracks[i], tracks[i + 1])] += 1
        for (t1, t2), cnt in transition_counts.items():
            i1, i2 = track_idx[t1], track_idx[t2]
            w = float(cnt) * 2.0
            rows.append(i1)
            cols.append(i2)
            weights.append(w)

    if include_user:
        user_track_counts = Counter()
        for sid, tracks in session_tracks.items():
            uid = session_users.get(sid)
            if not uid or uid not in user_idx:
                continue
            for tid in tracks:
                if tid in track_idx:
                    user_track_counts[(uid, tid)] += 1
        for (uid, tid), cnt in user_track_counts.items():
            ui, ti = user_idx[uid], track_idx[tid]
            w = float(cnt)
            rows.extend([ui, ti])
            cols.extend([ti, ui])
            weights.extend([w, w * 0.5])

    if include_artist:
        for tid in track_list:
            if tid not in track_artists or tid not in track_idx:
                continue
            ti = track_idx[tid]
            for aid in track_artists[tid]:
                if aid in artist_idx:
                    ai = artist_idx[aid]
                    rows.extend([ti, ai])
                    cols.extend([ai, ti])
                    weights.extend([1.0, 1.0])

    if include_tag:
        for tid in track_list:
            if tid not in track_tags_map or tid not in track_idx:
                continue
            ti = track_idx[tid]
            for tag in track_tags_map[tid]:
                if tag in tag_idx:
                    tgi = tag_idx[tag]
                    rows.extend([ti, tgi])
                    cols.extend([tgi, ti])
                    weights.extend([0.3, 0.3])

    adj = sparse.csr_matrix((weights, (rows, cols)), shape=(N, N), dtype=np.float32)
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv = sparse.diags(1.0 / row_sums, format="csr")
    return inv @ adj


def ppr(adj_norm, seed_indices, seed_weights, n_tracks, alpha=0.15,
        max_iter=20, tol=1e-6):
    N = adj_norm.shape[0]
    p = np.zeros(N, dtype=np.float32)
    for idx, w in zip(seed_indices, seed_weights):
        p[idx] = w
    if p.sum() > 0:
        p /= p.sum()
    r = p.copy()
    for _ in range(max_iter):
        r_new = (1 - alpha) * (adj_norm.T @ r) + alpha * p
        if np.abs(r_new - r).sum() < tol:
            break
        r = r_new
    return r[:n_tracks]


def evaluate_source(name, results_list, cases, abcdf_pools, payload, als_source,
                    ta, track_pop, track_degree, n):
    """Evaluate with full diagnostics including median rank and pop bias."""
    hit50 = hit100 = hit200 = hit500 = 0
    unique_vs_pool = unique_unreachable = 0
    unique_beyond_all_behavioral = 0
    pop0_rec = diff_rec = hist0_rec = same_artist_rec = 0
    gt_ranks = []
    hit_pops = []

    diff_total = h0_total = p0_total = 0
    fused_diff_hit = fused_h0_hit = fused_p0_hit = 0
    overlap_als = []
    overlap_cfbpr = []
    overlap_v3pool = []

    for i, c in enumerate(cases):
        gt = c["gt"]
        played = c["music_turns"]
        pr = results_list[i]
        pr_set = set(pr[:200])
        fused = abcdf_pools[i] | pr_set

        if gt in pr[:50]: hit50 += 1
        if gt in pr[:100]: hit100 += 1
        if gt in pr[:200]: hit200 += 1
        if gt in pr[:500]: hit500 += 1

        if gt in pr:
            gt_rank = pr.index(gt) + 1
            gt_ranks.append(gt_rank)
            hit_pops.append(track_pop.get(gt, 0))

        if gt in pr_set:
            if gt not in abcdf_pools[i]:
                unique_vs_pool += 1
                in_any = False
                for sn in ["src_a", "src_b", "src_c", "src_d", "src_f"]:
                    if gt in payload[sn][i][:500]:
                        in_any = True; break
                if not in_any and gt not in als_source[i][:500]:
                    unique_unreachable += 1
                # Also check: unique beyond V3+ALS+CF-BPR
                in_behavioral = (gt in als_source[i][:500] or
                                 gt in payload["src_f"][i][:500])
                if not in_any and not in_behavioral:
                    unique_beyond_all_behavioral += 1
            if track_pop.get(gt, 0) == 0: pop0_rec += 1
            if played:
                la = ta.get(played[-1], "")
                ga = ta.get(gt, "")
                if isinstance(la, list): la = la[0] if la else ""
                if isinstance(ga, list): ga = ga[0] if ga else ""
                if ga and la:
                    if ga == la: same_artist_rec += 1
                    else: diff_rec += 1
            if len(played) == 0: hist0_rec += 1

        # Fused pool slices
        if played:
            la = ta.get(played[-1], "")
            ga = ta.get(gt, "")
            if isinstance(la, list): la = la[0] if la else ""
            if isinstance(ga, list): ga = ga[0] if ga else ""
            if ga and la and ga != la:
                diff_total += 1
                if gt in fused: fused_diff_hit += 1
        if len(played) == 0:
            h0_total += 1
            if gt in fused: fused_h0_hit += 1
        if track_pop.get(gt, 0) == 0:
            p0_total += 1
            if gt in fused: fused_p0_hit += 1

        # Overlap with ALS, CF-BPR, V3 pool
        als_set = set(als_source[i][:200])
        cfbpr_set = set(payload["src_f"][i][:200])
        if pr_set:
            overlap_als.append(len(pr_set & als_set) / len(pr_set))
            overlap_cfbpr.append(len(pr_set & cfbpr_set) / len(pr_set))
            overlap_v3pool.append(len(pr_set & abcdf_pools[i]) / len(pr_set))

    # Popularity bias: what pop percentile do hits come from?
    pop_buckets = Counter()
    for p_val in hit_pops:
        if p_val == 0: pop_buckets["pop=0"] += 1
        elif p_val < 10: pop_buckets["pop=1-9"] += 1
        elif p_val < 50: pop_buckets["pop=10-49"] += 1
        else: pop_buckets["pop=50+"] += 1

    # Top-200 candidate popularity distribution
    all_cand_pops = []
    for pr in results_list[:500]:  # sample 500 cases
        for tid in pr[:200]:
            all_cand_pops.append(track_pop.get(tid, 0))

    return {
        "hit50": hit50, "hit100": hit100, "hit200": hit200, "hit500": hit500,
        "unique_vs_pool": unique_vs_pool, "unique_unreachable": unique_unreachable,
        "unique_beyond_all_behavioral": unique_beyond_all_behavioral,
        "pop0_rec": pop0_rec, "diff_rec": diff_rec, "hist0_rec": hist0_rec,
        "same_artist_rec": same_artist_rec,
        "median_gt_rank": float(np.median(gt_ranks)) if gt_ranks else 0,
        "mean_gt_rank": float(np.mean(gt_ranks)) if gt_ranks else 0,
        "overlap_als": float(np.mean(overlap_als)) if overlap_als else 0,
        "overlap_cfbpr": float(np.mean(overlap_cfbpr)) if overlap_cfbpr else 0,
        "overlap_v3pool": float(np.mean(overlap_v3pool)) if overlap_v3pool else 0,
        "pop_buckets": dict(pop_buckets),
        "cand_median_pop": float(np.median(all_cand_pops)) if all_cand_pops else 0,
        "cand_mean_pop": float(np.mean(all_cand_pops)) if all_cand_pops else 0,
        "fused_diff_hit": fused_diff_hit, "diff_total": diff_total,
        "fused_h0_hit": fused_h0_hit, "h0_total": h0_total,
        "fused_p0_hit": fused_p0_hit, "p0_total": p0_total,
    }


def main():
    t0 = time.time()
    from datasets import DownloadConfig, load_dataset, Dataset, concatenate_datasets

    # ---- Load training data ----
    print(f"{ts()} Loading training data...", flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    train = ds["train"]

    all_tracks = set()
    all_sessions = set()
    all_users = set()
    session_tracks = {}
    session_users = {}

    for item in train:
        sid = str(item["session_id"])
        uid = str(item.get("user_id", ""))
        all_sessions.add(sid)
        if uid:
            all_users.add(uid)
            session_users[sid] = uid
        tracks = [str(c["content"]).strip() for c in item["conversations"] if c["role"] == "music"]
        session_tracks[sid] = tracks
        all_tracks.update(tracks)

    # Load metadata
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    splits = []
    for split in ["all_tracks", "test_tracks"]:
        matches = sorted(hf_cache.glob(
            f"talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            f"talk_play_data-challenge-track-metadata-{split}.arrow"))
        if matches:
            splits.append(Dataset.from_file(str(matches[-1])))
    meta_ds = concatenate_datasets(splits)
    mcols = meta_ds.to_dict()

    all_artists = set()
    all_tags_set = set()
    track_artists = {}
    track_tags_map = {}
    for i in range(len(mcols["track_id"])):
        tid = str(mcols["track_id"][i])
        artists = [str(a) for a in (mcols["artist_id"][i] or [])]
        tags = [str(t) for t in (mcols["tag_list"][i] or [])[:10]]
        track_artists[tid] = artists
        track_tags_map[tid] = tags
        all_artists.update(artists)
        all_tags_set.update(tags)

    track_list = sorted(all_tracks)
    session_list = sorted(all_sessions)
    user_list = sorted(all_users)
    artist_list = sorted(all_artists)
    tag_list_sorted = sorted(all_tags_set)

    track_idx = {t: i for i, t in enumerate(track_list)}
    n_tracks = len(track_list)
    session_idx = {s: n_tracks + i for i, s in enumerate(session_list)}
    user_idx = {u: n_tracks + len(session_list) + i for i, u in enumerate(user_list)}
    artist_idx = {a: n_tracks + len(session_list) + len(user_list) + i for i, a in enumerate(artist_list)}
    tag_idx = {t: n_tracks + len(session_list) + len(user_list) + len(artist_list) + i
               for i, t in enumerate(tag_list_sorted)}
    N = n_tracks + len(session_list) + len(user_list) + len(artist_list) + len(tag_list_sorted)

    print(f"  Nodes: {N} (tracks={n_tracks})")

    # ---- Graph sanity checks ----
    print(f"\n{ts()} === GRAPH SANITY CHECKS ===")

    # Build full graph for degree analysis
    adj_full = build_graph_variant(
        session_tracks, track_artists, track_tags_map, session_users,
        track_list, track_idx, session_idx, user_idx, artist_idx, tag_idx, N, "full")

    # Track degree distribution
    track_degrees = np.array(adj_full[:n_tracks].sum(axis=1)).flatten()
    print(f"  Track degree: min={track_degrees.min():.0f} median={np.median(track_degrees):.0f} "
          f"mean={track_degrees.mean():.1f} max={track_degrees.max():.0f}")
    print(f"  Tracks with degree=0: {(track_degrees == 0).sum()}")

    # Top-20 highest degree tracks
    top20_idx = np.argsort(-track_degrees)[:20]
    track_pop = build_popularity_stats()
    print(f"\n  Top-20 highest-degree tracks:")
    for rank, ti in enumerate(top20_idx):
        tid = track_list[ti]
        meta_name = mcols["track_name"][0] if ti < len(mcols["track_name"]) else "?"
        pop = track_pop.get(tid, 0)
        names = track_artists.get(tid, [])
        print(f"    {rank+1}. degree={track_degrees[ti]:.0f} pop={pop} tid={tid[:12]}...")

    # GT leakage check
    print(f"\n  GT leakage check:")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    dev_gts = set(c["gt"] for c in cases)
    dev_gt_in_graph = sum(1 for gt in dev_gts if gt in track_idx)
    print(f"    Dev GTs in graph (as train tracks): {dev_gt_in_graph}/{len(dev_gts)}")
    print(f"    (These appear because GTs also appear as played tracks in other sessions)")
    print(f"    Graph is built from TRAIN data only — no blind/dev targets as edges ✓")

    # ---- Load dev data and baselines ----
    n = len(cases)
    ta = payload["track_artist"]
    tt = payload["track_tags"]

    print(f"\n{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx: scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids[j] for j in top_idx])
        else:
            als_source.append([])

    base_weights = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}
    abcdf_pools = []
    for i in range(n):
        sl = {"A": payload["src_a"][i], "B": payload["src_b"][i],
              "C": payload["src_c"][i], "D": payload["src_d"][i],
              "F": payload["src_f"][i], "ALS": als_source[i]}
        pool = weighted_rrf(sl, base_weights, topk=200, k=RRF_K)
        abcdf_pools.append(set(pool))

    base_pool_hit = np.mean([c["gt"] in abcdf_pools[i] for i, c in enumerate(cases)])

    # ---- Evaluate graph variants ----
    variants = [
        ("cooccur_only", 0.15),
        ("transition_only", 0.15),
        ("user_only", 0.15),
        ("behavioral", 0.15),
        ("hetero_only", 0.15),
        ("full", 0.15),
        ("full_debiased", 0.15),
    ]

    all_results = {}

    for variant_name, alpha in variants:
        is_debiased = variant_name.endswith("_debiased")
        graph_variant = variant_name.replace("_debiased", "")

        print(f"\n{ts()} === {variant_name} (alpha={alpha}) ===", flush=True)
        t_var = time.time()

        adj = build_graph_variant(
            session_tracks, track_artists, track_tags_map, session_users,
            track_list, track_idx, session_idx, user_idx, artist_idx, tag_idx,
            N, graph_variant)

        # Track degrees for debiasing
        var_degrees = np.array(adj[:n_tracks].sum(axis=1)).flatten()

        ppr_results = []
        for i, c in enumerate(cases):
            played = c["music_turns"]
            seed_indices = []
            seed_weights = []
            for j, tid in enumerate(reversed(played)):
                if tid in track_idx:
                    seed_indices.append(track_idx[tid])
                    seed_weights.append(1.0 / (j + 1))

            if not seed_indices:
                ppr_results.append([])
                continue

            scores = ppr(adj, seed_indices, seed_weights, n_tracks, alpha=alpha)

            if is_debiased:
                # Divide by sqrt(degree) to penalize hubs
                deg_penalty = np.sqrt(var_degrees + 1)
                scores = scores / deg_penalty

            for tid in played:
                if tid in track_idx:
                    scores[track_idx[tid]] = 0.0

            top_k = min(500, max(1, (scores > 0).sum()))
            if top_k > 0:
                top_idx = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
                top_idx = top_idx[np.argsort(-scores[top_idx])]
                ppr_results.append([track_list[j] for j in top_idx if scores[j] > 0])
            else:
                ppr_results.append([])

            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{n} done", flush=True)

        var_time = time.time() - t_var
        print(f"  {n} cases in {var_time:.1f}s ({n/var_time:.0f}/s)")

        ev = evaluate_source(variant_name, ppr_results, cases, abcdf_pools,
                             payload, als_source, ta, track_pop, var_degrees, n)

        fused_pool_hit = np.mean([
            cases[i]["gt"] in (abcdf_pools[i] | set(ppr_results[i][:200]))
            for i in range(n)])
        pool_lift = fused_pool_hit - base_pool_hit

        base_diff = ev["fused_diff_hit"] / ev["diff_total"] if ev["diff_total"] else 0
        # Need base diff rate
        base_diff_hit = sum(1 for i, c in enumerate(cases)
                            if c["music_turns"] and c["gt"] in abcdf_pools[i]
                            and (lambda la, ga: ga and la and ga != la)(
                                (ta.get(c["music_turns"][-1], "") if not isinstance(ta.get(c["music_turns"][-1], ""), list) else (ta.get(c["music_turns"][-1], [""])[0] if ta.get(c["music_turns"][-1], [""]) else "")),
                                (ta.get(c["gt"], "") if not isinstance(ta.get(c["gt"], ""), list) else (ta.get(c["gt"], [""])[0] if ta.get(c["gt"], [""]) else ""))))
        diff_lift = (ev["fused_diff_hit"] / ev["diff_total"] - base_diff_hit / ev["diff_total"]) if ev["diff_total"] else 0

        print(f"  hit@50={ev['hit50']}  hit@100={ev['hit100']}  hit@200={ev['hit200']}  hit@500={ev['hit500']}")
        print(f"  unique_vs_V3pool={ev['unique_vs_pool']}  unique_unreachable={ev['unique_unreachable']}  "
              f"unique_beyond_all_behavioral={ev['unique_beyond_all_behavioral']}")
        print(f"  pop0={ev['pop0_rec']}  diff_artist={ev['diff_rec']}  same_artist={ev['same_artist_rec']}  hist0={ev['hist0_rec']}")
        if ev["hist0_rec"] > 0 and variant_name not in ("hetero_only",):
            print(f"  ⚠ SANITY: hist_0 hits in track-seeded variant — check for leakage or user/tag seeds")
        print(f"  median_gt_rank={ev['median_gt_rank']:.0f}  mean_gt_rank={ev['mean_gt_rank']:.0f}")
        print(f"  overlap: ALS={ev['overlap_als']:.3f}  CF-BPR={ev['overlap_cfbpr']:.3f}  V3pool={ev['overlap_v3pool']:.3f}")
        print(f"  candidate median_pop={ev['cand_median_pop']:.0f}  mean_pop={ev['cand_mean_pop']:.0f}")
        print(f"  hit pop distribution: {ev['pop_buckets']}")
        print(f"  fused pool_hit={fused_pool_hit:.4f} (Δ={pool_lift:+.4f})")

        gate_pool = pool_lift >= 0.03
        gate_unreach = ev["unique_unreachable"] >= 150
        gate_diff = diff_lift >= 0.03
        print(f"  GATES: pool_lift={'PASS' if gate_pool else 'FAIL'}({pool_lift:+.4f})  "
              f"unreach={'PASS' if gate_unreach else 'FAIL'}({ev['unique_unreachable']})  "
              f"diff_lift={'PASS' if gate_diff else 'FAIL'}({diff_lift:+.3f})")

        all_results[variant_name] = ev
        all_results[variant_name]["fused_pool_hit"] = fused_pool_hit

    elapsed = time.time() - t0
    print(f"\n{ts()} R16 diagnostics complete. Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expR16_diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
