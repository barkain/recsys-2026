#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R16: Pixie-style heterogeneous graph retrieval via Personalized PageRank.

Build a heterogeneous graph (track/session/user/artist/tag) from training data
and run PPR seeded by each dev case's played tracks. Evaluate as candidate source.

Stage 1 only — no fusion, no blind.
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


def build_graph():
    """Build heterogeneous graph from training data.

    Node types: track, session, user, artist, tag
    Edge types: session-track, track-track co-occurrence, track->next transition,
                user-track, track-artist, track-tag
    """
    from datasets import DownloadConfig, load_dataset

    print(f"{ts()} Loading training data...", flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    train = ds["train"]

    # Collect all entities
    all_tracks = set()
    all_sessions = set()
    all_users = set()
    all_artists = set()
    all_tags = set()

    session_tracks = {}  # sid → ordered list of track_ids
    session_users = {}   # sid → user_id

    for item in train:
        sid = str(item["session_id"])
        uid = str(item.get("user_id", ""))
        all_sessions.add(sid)
        if uid:
            all_users.add(uid)
            session_users[sid] = uid

        tracks = []
        for c in item["conversations"]:
            if c["role"] == "music":
                tid = str(c["content"]).strip()
                tracks.append(tid)
                all_tracks.add(tid)
        session_tracks[sid] = tracks

    # Load track metadata for artist/tag edges
    from datasets import Dataset, concatenate_datasets
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    splits = []
    for split in ["all_tracks", "test_tracks"]:
        matches = sorted(hf_cache.glob(
            f"talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
            f"talk_play_data-challenge-track-metadata-{split}.arrow"))
        if matches:
            splits.append(Dataset.from_file(str(matches[-1])))
    meta_ds = concatenate_datasets(splits)
    cols = meta_ds.to_dict()

    track_artists = {}  # tid → list of artist_ids
    track_tags = {}     # tid → list of tags

    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        artists = cols["artist_id"][i] or []
        tags = cols["tag_list"][i] or []
        track_artists[tid] = [str(a) for a in artists]
        track_tags[tid] = [str(t) for t in tags[:10]]  # limit tags per track
        for a in artists:
            all_artists.add(str(a))
        for t in tags[:10]:
            all_tags.add(str(t))

    # Assign node IDs
    # Layout: [tracks | sessions | users | artists | tags]
    track_list = sorted(all_tracks)
    session_list = sorted(all_sessions)
    user_list = sorted(all_users)
    artist_list = sorted(all_artists)
    tag_list = sorted(all_tags)

    track_idx = {t: i for i, t in enumerate(track_list)}
    n_tracks = len(track_list)
    session_idx = {s: n_tracks + i for i, s in enumerate(session_list)}
    n_sessions = len(session_list)
    user_idx = {u: n_tracks + n_sessions + i for i, u in enumerate(user_list)}
    n_users = len(user_list)
    artist_idx = {a: n_tracks + n_sessions + n_users + i for i, a in enumerate(artist_list)}
    n_artists = len(artist_list)
    tag_idx = {t: n_tracks + n_sessions + n_users + n_artists + i for i, t in enumerate(tag_list)}
    n_tags = len(tag_list)

    N = n_tracks + n_sessions + n_users + n_artists + n_tags
    print(f"  Nodes: {N} (tracks={n_tracks}, sessions={n_sessions}, "
          f"users={n_users}, artists={n_artists}, tags={n_tags})")

    # Build edges
    rows, col_list, weights = [], [], []

    # 1. Session-track edges (bidirectional, weighted by position)
    for sid, tracks in session_tracks.items():
        si = session_idx[sid]
        for pos, tid in enumerate(tracks):
            if tid not in track_idx:
                continue
            ti = track_idx[tid]
            w = 1.0 / (pos + 1)  # recency/position decay
            rows.extend([si, ti])
            col_list.extend([ti, si])
            weights.extend([w, w])

    # 2. Track-track co-occurrence (same session, weighted)
    cooccur_counts = Counter()
    for sid, tracks in session_tracks.items():
        unique_tracks = list(dict.fromkeys(t for t in tracks if t in track_idx))
        for i in range(len(unique_tracks)):
            for j in range(i + 1, len(unique_tracks)):
                pair = (unique_tracks[i], unique_tracks[j])
                cooccur_counts[pair] += 1

    for (t1, t2), cnt in cooccur_counts.items():
        i1, i2 = track_idx[t1], track_idx[t2]
        w = float(cnt)
        rows.extend([i1, i2])
        col_list.extend([i2, i1])
        weights.extend([w, w])
    print(f"  Co-occurrence edges: {len(cooccur_counts)}")

    # 3. Directed transitions (track → next track)
    transition_counts = Counter()
    for sid, tracks in session_tracks.items():
        for i in range(len(tracks) - 1):
            if tracks[i] in track_idx and tracks[i + 1] in track_idx:
                transition_counts[(tracks[i], tracks[i + 1])] += 1

    for (t1, t2), cnt in transition_counts.items():
        i1, i2 = track_idx[t1], track_idx[t2]
        w = float(cnt) * 2.0  # boost transitions
        rows.append(i1)
        col_list.append(i2)
        weights.append(w)
    print(f"  Transition edges: {len(transition_counts)}")

    # 4. User-track edges
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
        col_list.extend([ti, ui])
        weights.extend([w, w * 0.5])  # asymmetric: user→track stronger
    print(f"  User-track edges: {len(user_track_counts)}")

    # 5. Track-artist edges
    ta_edges = 0
    for tid in track_list:
        if tid not in track_artists:
            continue
        ti = track_idx[tid]
        for aid in track_artists[tid]:
            if aid in artist_idx:
                ai = artist_idx[aid]
                rows.extend([ti, ai])
                col_list.extend([ai, ti])
                weights.extend([1.0, 1.0])
                ta_edges += 1
    print(f"  Track-artist edges: {ta_edges}")

    # 6. Track-tag edges (lower weight — tags are noisy)
    tt_edges = 0
    for tid in track_list:
        if tid not in track_tags:
            continue
        ti = track_idx[tid]
        for tag in track_tags[tid]:
            if tag in tag_idx:
                tgi = tag_idx[tag]
                rows.extend([ti, tgi])
                col_list.extend([tgi, ti])
                weights.extend([0.3, 0.3])  # low weight for tag edges
                tt_edges += 1
    print(f"  Track-tag edges: {tt_edges}")

    # Build sparse adjacency matrix
    adj = sparse.csr_matrix(
        (weights, (rows, col_list)),
        shape=(N, N), dtype=np.float32,
    )
    print(f"  Total edges: {adj.nnz}")

    # Row-normalize for PPR
    row_sums = np.array(adj.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0
    inv = sparse.diags(1.0 / row_sums, format="csr")
    adj_norm = inv @ adj

    return (adj_norm, track_list, track_idx, session_idx, user_idx,
            artist_idx, tag_idx, n_tracks, track_artists, track_tags)


def ppr(adj_norm, seed_indices, seed_weights, n_tracks, alpha=0.15,
        max_iter=20, tol=1e-6):
    """Personalized PageRank via power iteration.

    Returns scores for track nodes only (first n_tracks).
    """
    N = adj_norm.shape[0]
    # Build personalization vector
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


def main():
    t0 = time.time()

    # Build graph
    (adj_norm, track_list, track_idx, session_idx, user_idx,
     artist_idx, tag_idx, n_tracks, track_artists, track_tags) = build_graph()

    # Load dev data
    print(f"\n{ts()} Loading dev data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    ta = payload["track_artist"]
    tt = payload["track_tags"]
    track_pop = build_popularity_stats()

    # Build ALS + baseline pools
    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids, als_track_to_idx = build_als()
    als_source = []
    for c in cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
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

    # Baselines
    base_diff_hit = diff_total = 0
    base_h0_hit = h0_total = 0
    base_p0_hit = p0_total = 0
    for i, c in enumerate(cases):
        gt = c["gt"]
        played = c["music_turns"]
        if played:
            la = ta.get(played[-1], "")
            ga = ta.get(gt, "")
            if isinstance(la, list): la = la[0] if la else ""
            if isinstance(ga, list): ga = ga[0] if ga else ""
            if ga and la and ga != la:
                diff_total += 1
                if gt in abcdf_pools[i]:
                    base_diff_hit += 1
        if len(played) == 0:
            h0_total += 1
            if gt in abcdf_pools[i]:
                base_h0_hit += 1
        if track_pop.get(gt, 0) == 0:
            p0_total += 1
            if gt in abcdf_pools[i]:
                base_p0_hit += 1
    base_diff_rate = base_diff_hit / diff_total if diff_total else 0
    base_h0_rate = base_h0_hit / h0_total if h0_total else 0
    base_p0_rate = base_p0_hit / p0_total if p0_total else 0
    base_pool_hit = np.mean([c["gt"] in abcdf_pools[i] for i, c in enumerate(cases)])

    print(f"\n{ts()} Baselines (ABCDF+ALS@200):")
    print(f"  pool_hit: {base_pool_hit:.4f}")
    print(f"  diff-artist: {base_diff_hit}/{diff_total} ({base_diff_rate:.1%})")
    print(f"  hist_0: {base_h0_hit}/{h0_total} ({base_h0_rate:.1%})")
    print(f"  pop=0: {base_p0_hit}/{p0_total} ({base_p0_rate:.1%})")

    # Run PPR for each dev case
    print(f"\n{ts()} Running PPR for {n} cases...", flush=True)

    # Test multiple alpha values
    for alpha in [0.10, 0.15, 0.25]:
        print(f"\n{ts()} === PPR alpha={alpha} ===")
        t_ppr = time.time()

        ppr_results = []
        for i, c in enumerate(cases):
            played = c["music_turns"]
            user_query = c["user_query"]

            # Build seed set
            seed_indices = []
            seed_weights = []

            # Seed from played tracks (recency-weighted)
            for j, tid in enumerate(reversed(played)):
                if tid in track_idx:
                    seed_indices.append(track_idx[tid])
                    seed_weights.append(1.0 / (j + 1))  # recency decay

            # Seed from query-mentioned artists/tags (if parseable)
            ql = user_query.lower()
            for aid, aidx in artist_idx.items():
                # Check if any artist name matches query
                pass  # Too expensive to match all artists; skip for now

            if not seed_indices:
                ppr_results.append([])
                continue

            scores = ppr(adj_norm, seed_indices, seed_weights, n_tracks,
                         alpha=alpha)

            # Zero out played tracks
            for tid in played:
                if tid in track_idx:
                    scores[track_idx[tid]] = 0.0

            # Get top-500
            top_idx = np.argpartition(-scores, min(500, len(scores) - 1))[:500]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            ppr_results.append([track_list[j] for j in top_idx if scores[j] > 0])

            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{n} done ({time.time()-t_ppr:.1f}s)", flush=True)

        ppr_time = time.time() - t_ppr
        print(f"  PPR done in {ppr_time:.1f}s ({n/ppr_time:.0f} cases/s)")

        # Evaluate
        hit50 = hit100 = hit200 = hit500 = 0
        unique_vs_pool = 0
        unique_unreachable = 0
        pop0_rec = diff_rec = hist0_rec = 0
        fused_diff_hit = fused_h0_hit = fused_p0_hit = 0
        overlap_als = []

        for i, c in enumerate(cases):
            gt = c["gt"]
            played = c["music_turns"]
            pr = ppr_results[i]
            pr_set = set(pr[:200])
            fused_pool = abcdf_pools[i] | pr_set

            if gt in pr[:50]: hit50 += 1
            if gt in pr[:100]: hit100 += 1
            if gt in pr[:200]: hit200 += 1
            if gt in pr[:500]: hit500 += 1

            if gt in pr_set:
                if gt not in abcdf_pools[i]:
                    unique_vs_pool += 1
                    in_any = False
                    for sn in ["src_a", "src_b", "src_c", "src_d", "src_f"]:
                        if gt in payload[sn][i][:500]:
                            in_any = True
                            break
                    if not in_any and gt not in als_source[i][:500]:
                        unique_unreachable += 1
                if track_pop.get(gt, 0) == 0:
                    pop0_rec += 1
                if played:
                    la = ta.get(played[-1], "")
                    ga = ta.get(gt, "")
                    if isinstance(la, list): la = la[0] if la else ""
                    if isinstance(ga, list): ga = ga[0] if ga else ""
                    if ga and la and ga != la:
                        diff_rec += 1
                if len(played) == 0:
                    hist0_rec += 1

            # Fused pool stats
            if played:
                la = ta.get(played[-1], "")
                ga = ta.get(gt, "")
                if isinstance(la, list): la = la[0] if la else ""
                if isinstance(ga, list): ga = ga[0] if ga else ""
                if ga and la and ga != la:
                    if gt in fused_pool:
                        fused_diff_hit += 1
            if len(played) == 0:
                if gt in fused_pool:
                    fused_h0_hit += 1
            if track_pop.get(gt, 0) == 0:
                if gt in fused_pool:
                    fused_p0_hit += 1

            als_set = set(als_source[i][:200])
            if pr_set:
                overlap_als.append(len(pr_set & als_set) / len(pr_set))

        fused_diff_rate = fused_diff_hit / diff_total if diff_total else 0
        fused_h0_rate = fused_h0_hit / h0_total if h0_total else 0
        fused_p0_rate = fused_p0_hit / p0_total if p0_total else 0
        diff_lift = fused_diff_rate - base_diff_rate
        fused_pool_hit = np.mean([
            cases[i]["gt"] in (abcdf_pools[i] | set(ppr_results[i][:200]))
            for i in range(n)])

        print(f"  hit@50:  {hit50}/{n} ({hit50/n:.1%})")
        print(f"  hit@100: {hit100}/{n} ({hit100/n:.1%})")
        print(f"  hit@200: {hit200}/{n} ({hit200/n:.1%})")
        print(f"  hit@500: {hit500}/{n} ({hit500/n:.1%})")
        print(f"  Unique vs ABCDF+ALS@200: {unique_vs_pool}")
        print(f"  Unique unreachable: {unique_unreachable}")
        print(f"  Pop=0 recovery: {pop0_rec}")
        print(f"  Different-artist recovery: {diff_rec}")
        print(f"  hist_0 recovery: {hist0_rec}")
        print(f"  Overlap with ALS: {np.mean(overlap_als):.3f}")
        print(f"\n  Fused pool_hit (V3+PPR@200): {fused_pool_hit:.4f} (Δ={fused_pool_hit-base_pool_hit:+.4f})")
        print(f"  Fused diff-artist: {fused_diff_hit}/{diff_total} ({fused_diff_rate:.1%}, lift={diff_lift:+.1%})")
        print(f"  Fused hist_0: {fused_h0_hit}/{h0_total} ({fused_h0_rate:.1%}, lift={fused_h0_rate-base_h0_rate:+.1%})")
        print(f"  Fused pop=0: {fused_p0_hit}/{p0_total} ({fused_p0_rate:.1%}, lift={fused_p0_rate-base_p0_rate:+.1%})")

        gate_pool = fused_pool_hit - base_pool_hit >= 0.03
        gate_unreach = unique_unreachable >= 150
        gate_diff = diff_lift >= 0.03
        print(f"\n  GATES:")
        print(f"    pool_hit lift >=+3%:    {'PASS' if gate_pool else 'FAIL'} ({fused_pool_hit-base_pool_hit:+.4f})")
        print(f"    unique unreach >=150:   {'PASS' if gate_unreach else 'FAIL'} ({unique_unreachable})")
        print(f"    diff-artist lift >=+3%: {'PASS' if gate_diff else 'FAIL'} ({diff_lift:+.1%})")
        print(f"    ANY GATE: {'PASS' if any([gate_pool, gate_unreach, gate_diff]) else 'FAIL'}")

    elapsed = time.time() - t0
    print(f"\n{ts()} R16 Stage 1 complete. Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
