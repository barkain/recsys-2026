#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R8: Data leakage and nearest-neighbor sequence audit.

Can nDCG 0.57 be explained by session-sequence matching / data structure?
Audits train/dev/blind for exploitable patterns.

No API. No blind submission. Read-only audit.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import DownloadConfig, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval_inference import build_ground_truth, cached_test_arrow_path, lookup_ground_truth
from datasets import Dataset


def ndcg_at_k(predicted, gt_id, k=20):
    for i, tid in enumerate(predicted[:k]):
        if tid == gt_id:
            return 1.0 / math.log2(i + 2)
    return 0.0


def jaccard(a, b):
    if not a and not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def edit_distance(a, b):
    """Simple Levenshtein for short sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[m]


def main():
    t0 = time.time()

    # =====================================================================
    # Load all datasets
    # =====================================================================
    print("Loading datasets...", flush=True)
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                      download_config=DownloadConfig(local_files_only=True))
    train = ds["train"]
    test = ds["test"]

    blind = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A",
                         download_config=DownloadConfig(local_files_only=True), split="test")

    print(f"  Train: {len(train)} sessions")
    print(f"  Dev/test: {len(test)} sessions")
    print(f"  Blind-A: {len(blind)} sessions")

    # Extract sequences
    def extract_sequences(dataset):
        """Extract (session_id, user_id, ordered_track_ids, per_turn_tracks) from dataset."""
        results = []
        for item in dataset:
            sid = str(item["session_id"])
            uid = str(item["user_id"])
            convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
            tracks = []
            per_turn = {}
            for c in convs:
                if c["role"] == "music":
                    tid = str(c["content"]).strip()
                    turn = int(c["turn_number"])
                    tracks.append(tid)
                    per_turn[turn] = tid
            results.append({
                "session_id": sid,
                "user_id": uid,
                "tracks": tracks,
                "per_turn": per_turn,
                "conversations": convs,
            })
        return results

    train_sessions = extract_sequences(train)
    test_sessions = extract_sequences(test)
    blind_sessions = extract_sequences(blind)

    print(f"  Train track sequences: {len(train_sessions)}")
    print(f"  Train tracks per session: {set(len(s['tracks']) for s in train_sessions)}")

    # =====================================================================
    # 1. USER OVERLAP
    # =====================================================================
    print(f"\n{'='*70}")
    print("1. USER OVERLAP")
    print(f"{'='*70}")

    train_users = set(s["user_id"] for s in train_sessions)
    test_users = set(s["user_id"] for s in test_sessions)
    blind_users = set(s["user_id"] for s in blind_sessions)

    print(f"  Train users: {len(train_users)}")
    print(f"  Test users: {len(test_users)}")
    print(f"  Blind-A users: {len(blind_users)}")
    print(f"  Test ∩ Train: {len(test_users & train_users)} ({len(test_users & train_users)/len(test_users):.1%})")
    print(f"  Blind ∩ Train: {len(blind_users & train_users)} ({len(blind_users & train_users)/len(blind_users):.1%})")

    # For overlapping users, how many train sessions do they have?
    train_user_sessions = defaultdict(list)
    for s in train_sessions:
        train_user_sessions[s["user_id"]].append(s)

    blind_overlap_users = blind_users & train_users
    if blind_overlap_users:
        sessions_per_user = [len(train_user_sessions[u]) for u in blind_overlap_users]
        print(f"  Blind overlapping users' train sessions: min={min(sessions_per_user)} "
              f"max={max(sessions_per_user)} mean={np.mean(sessions_per_user):.1f}")

    # =====================================================================
    # 2. EXACT SESSION/SEQUENCE MATCHING
    # =====================================================================
    print(f"\n{'='*70}")
    print("2. EXACT SEQUENCE MATCHING")
    print(f"{'='*70}")

    # Build train sequence index
    train_by_tracks = defaultdict(list)  # tuple(tracks) -> [session]
    train_by_prefix = defaultdict(list)  # tuple(tracks[:k]) -> [session]
    for s in train_sessions:
        key = tuple(s["tracks"])
        train_by_tracks[key].append(s)
        for k in range(1, len(s["tracks"]) + 1):
            prefix = tuple(s["tracks"][:k])
            train_by_prefix[prefix].append(s)

    # Check test/blind for exact matches
    for label, sessions in [("Test", test_sessions), ("Blind-A", blind_sessions)]:
        exact = 0
        prefix_matches = {k: 0 for k in range(1, 9)}
        for s in sessions:
            if tuple(s["tracks"]) in train_by_tracks:
                exact += 1
            for k in range(1, min(len(s["tracks"]) + 1, 9)):
                if tuple(s["tracks"][:k]) in train_by_prefix:
                    prefix_matches[k] += 1
        print(f"  {label} exact full-sequence match: {exact}/{len(sessions)}")
        print(f"  {label} prefix matches: {dict(prefix_matches)}")

    # =====================================================================
    # 3. NEAREST-NEIGHBOR SEQUENCE AUDIT (DEV)
    # =====================================================================
    print(f"\n{'='*70}")
    print("3. NEAREST-NEIGHBOR SEQUENCE CONTINUATION (DEV)")
    print(f"{'='*70}")

    # Build GT map for dev
    arrow_path = cached_test_arrow_path()
    dev_ds = Dataset.from_file(arrow_path) if arrow_path else None
    gt_map = build_ground_truth(dev_ds) if dev_ds else {}

    # For each dev session, at each turn:
    # - look at played tracks so far (prefix)
    # - find nearest train session by prefix match / Jaccard
    # - predict next track from that train session
    # - measure hit rate

    # Build train track-set index for fast Jaccard lookup
    train_track_sets = [(set(s["tracks"]), s["tracks"], s) for s in train_sessions]
    # Build train last-k index for exact prefix matching
    train_last_k_index = defaultdict(list)  # tuple(last_k_tracks) -> [(session, next_track)]
    for s in train_sessions:
        tracks = s["tracks"]
        for i in range(len(tracks)):
            played_so_far = tracks[:i]
            next_track = tracks[i]
            for k in range(1, min(len(played_so_far) + 1, 6)):
                suffix = tuple(played_so_far[-k:])
                train_last_k_index[suffix].append((s, next_track))

    # Baselines on dev
    print("  Building dev baselines...")

    baselines = {
        "exact_prefix_continuation": {"ndcg": [], "hit20": [], "hit50": []},
        "last1_continuation": {"ndcg": [], "hit20": [], "hit50": []},
        "last2_continuation": {"ndcg": [], "hit20": [], "hit50": []},
        "last3_continuation": {"ndcg": [], "hit20": [], "hit50": []},
        "user_history_tracks": {"ndcg": [], "hit20": [], "hit50": []},
        "jaccard_nn_continuation": {"ndcg": [], "hit20": [], "hit50": []},
    }

    n_cases = 0
    for s in test_sessions:
        uid = s["user_id"]
        tracks = s["tracks"]
        convs = s["conversations"]
        user_turns = [c for c in convs if c["role"] == "user"]

        for ut in user_turns:
            turn = int(ut["turn_number"])
            gt = lookup_ground_truth(gt_map, s["session_id"], uid, turn)
            if not gt:
                continue

            played = [str(c["content"]).strip() for c in convs
                      if c["role"] == "music" and int(c["turn_number"]) < turn]
            n_cases += 1

            # 1. Exact prefix continuation
            prefix = tuple(played)
            matches = train_by_prefix.get(prefix, [])
            preds = []
            for m in matches:
                idx = len(played)
                if idx < len(m["tracks"]):
                    preds.append(m["tracks"][idx])
            # Count-based ranking
            if preds:
                counted = Counter(preds)
                ranked = [tid for tid, _ in counted.most_common(50)]
            else:
                ranked = []
            baselines["exact_prefix_continuation"]["ndcg"].append(ndcg_at_k(ranked, gt))
            baselines["exact_prefix_continuation"]["hit20"].append(gt in ranked[:20])
            baselines["exact_prefix_continuation"]["hit50"].append(gt in ranked[:50])

            # 2-4. Last-k continuation
            for k_name, k_val in [("last1", 1), ("last2", 2), ("last3", 3)]:
                suffix = tuple(played[-k_val:]) if len(played) >= k_val else tuple(played)
                matches_k = train_last_k_index.get(suffix, [])
                preds_k = [nt for _, nt in matches_k if nt not in set(played)]
                if preds_k:
                    counted_k = Counter(preds_k)
                    ranked_k = [tid for tid, _ in counted_k.most_common(50)]
                else:
                    ranked_k = []
                baselines[f"{k_name}_continuation"]["ndcg"].append(ndcg_at_k(ranked_k, gt))
                baselines[f"{k_name}_continuation"]["hit20"].append(gt in ranked_k[:20])
                baselines[f"{k_name}_continuation"]["hit50"].append(gt in ranked_k[:50])

            # 5. User history tracks (if user is in train)
            user_train = train_user_sessions.get(uid, [])
            user_tracks_all = []
            for us in user_train:
                user_tracks_all.extend(us["tracks"])
            # Remove already played, count-rank
            user_preds = [t for t in user_tracks_all if t not in set(played)]
            if user_preds:
                counted_u = Counter(user_preds)
                ranked_u = [tid for tid, _ in counted_u.most_common(50)]
            else:
                ranked_u = []
            baselines["user_history_tracks"]["ndcg"].append(ndcg_at_k(ranked_u, gt))
            baselines["user_history_tracks"]["hit20"].append(gt in ranked_u[:20])
            baselines["user_history_tracks"]["hit50"].append(gt in ranked_u[:50])

            # 6. Jaccard nearest-neighbor continuation
            if played:
                played_set = set(played)
                best_jacc = -1
                best_sessions = []
                for ts, tl, tsess in train_track_sets:
                    j = len(played_set & ts) / len(played_set | ts) if (played_set | ts) else 0
                    if j > best_jacc:
                        best_jacc = j
                        best_sessions = [(tsess, j)]
                    elif j == best_jacc:
                        best_sessions.append((tsess, j))

                # Predict from top Jaccard matches
                jacc_preds = []
                # Actually, search top-20 by Jaccard
                scored = []
                for ts, tl, tsess in train_track_sets:
                    j = len(played_set & ts) / len(played_set | ts) if (played_set | ts) else 0
                    if j > 0:
                        scored.append((j, tsess))
                scored.sort(key=lambda x: -x[0])
                for _, tsess in scored[:20]:
                    for t in tsess["tracks"]:
                        if t not in played_set and t not in set(jacc_preds):
                            jacc_preds.append(t)
                            if len(jacc_preds) >= 50:
                                break
                    if len(jacc_preds) >= 50:
                        break
            else:
                jacc_preds = []
            baselines["jaccard_nn_continuation"]["ndcg"].append(ndcg_at_k(jacc_preds, gt))
            baselines["jaccard_nn_continuation"]["hit20"].append(gt in jacc_preds[:20])
            baselines["jaccard_nn_continuation"]["hit50"].append(gt in jacc_preds[:50])

            if n_cases % 500 == 0:
                print(f"    processed {n_cases} cases...", flush=True)

    print(f"\n  Total dev cases evaluated: {n_cases}")
    print(f"\n  {'Baseline':30s} {'nDCG@20':>8s} {'hit@20':>7s} {'hit@50':>7s}")
    for bname, bdata in baselines.items():
        n = len(bdata["ndcg"])
        ndcg_mean = sum(bdata["ndcg"]) / n if n else 0
        hit20 = sum(bdata["hit20"]) / n if n else 0
        hit50 = sum(bdata["hit50"]) / n if n else 0
        print(f"  {bname:30s} {ndcg_mean:8.4f} {hit20:7.4f} {hit50:7.4f}")

    # =====================================================================
    # 4. BLIND-A NEAREST-NEIGHBOR COVERAGE
    # =====================================================================
    print(f"\n{'='*70}")
    print("4. BLIND-A NEAREST-NEIGHBOR COVERAGE")
    print(f"{'='*70}")

    # For each blind session, find best train match
    blind_nn_stats = []
    for s in blind_sessions:
        convs = s["conversations"]
        user_turns = [c for c in convs if c["role"] == "user"]
        last_turn = max(int(c["turn_number"]) for c in user_turns)
        played = [str(c["content"]).strip() for c in convs
                  if c["role"] == "music" and int(c["turn_number"]) < last_turn]

        # Exact prefix match
        prefix = tuple(played)
        exact_matches = len(train_by_prefix.get(prefix, []))

        # Last-3 suffix match
        suffix3 = tuple(played[-3:]) if len(played) >= 3 else tuple(played)
        suffix_matches = len(train_last_k_index.get(suffix3, []))

        # Jaccard best
        if played:
            played_set = set(played)
            best_jacc = 0
            for ts, _, _ in train_track_sets:
                j = len(played_set & ts) / len(played_set | ts) if (played_set | ts) else 0
                if j > best_jacc:
                    best_jacc = j
        else:
            best_jacc = 0

        blind_nn_stats.append({
            "session_id": s["session_id"],
            "n_played": len(played),
            "exact_prefix_matches": exact_matches,
            "last3_suffix_matches": suffix_matches,
            "best_jaccard": best_jacc,
        })

    print(f"  Blind-A sessions: {len(blind_nn_stats)}")
    exact_any = sum(1 for s in blind_nn_stats if s["exact_prefix_matches"] > 0)
    suffix_any = sum(1 for s in blind_nn_stats if s["last3_suffix_matches"] > 0)
    jacc_high = sum(1 for s in blind_nn_stats if s["best_jaccard"] >= 0.5)
    jacc_very_high = sum(1 for s in blind_nn_stats if s["best_jaccard"] >= 0.8)
    jaccards = [s["best_jaccard"] for s in blind_nn_stats]
    print(f"  Exact prefix match in train: {exact_any}/{len(blind_nn_stats)}")
    print(f"  Last-3 suffix match: {suffix_any}/{len(blind_nn_stats)}")
    print(f"  Jaccard >= 0.5: {jacc_high}/{len(blind_nn_stats)}")
    print(f"  Jaccard >= 0.8: {jacc_very_high}/{len(blind_nn_stats)}")
    print(f"  Jaccard: min={min(jaccards):.3f} max={max(jaccards):.3f} "
          f"mean={np.mean(jaccards):.3f} median={np.median(jaccards):.3f}")

    # =====================================================================
    # 5. CAN SEQUENCE CONTINUATION EXPLAIN nDCG 0.57?
    # =====================================================================
    print(f"\n{'='*70}")
    print("5. FEASIBILITY: CAN SEQUENCE METHODS REACH nDCG 0.57?")
    print(f"{'='*70}")

    best_baseline = max(baselines, key=lambda b: sum(baselines[b]["ndcg"]) / len(baselines[b]["ndcg"]))
    best_ndcg = sum(baselines[best_baseline]["ndcg"]) / len(baselines[best_baseline]["ndcg"])
    print(f"  Best dev baseline: {best_baseline} → nDCG {best_ndcg:.4f}")
    print(f"  Our F1 CF-BPR dev (last-turn 400): ~0.170")
    print(f"  Our F1 CF-BPR blind: 0.3971")
    print(f"  Leader blind: 0.57")
    if best_ndcg > 0.20:
        print(f"  → Sequence continuation is STRONG. Could partially explain leader score.")
    elif best_ndcg > 0.10:
        print(f"  → Sequence continuation is MODERATE. Supplements but doesn't fully explain leader.")
    else:
        print(f"  → Sequence continuation is WEAK. Leader likely uses a different method.")

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expR8_leakage_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "meta": {"elapsed_sec": elapsed, "n_train": len(train_sessions),
                 "n_test": len(test_sessions), "n_blind": len(blind_sessions)},
        "user_overlap": {
            "train_users": len(train_users),
            "test_in_train": len(test_users & train_users),
            "blind_in_train": len(blind_users & train_users),
        },
        "baselines": {
            name: {
                "ndcg": sum(d["ndcg"]) / len(d["ndcg"]) if d["ndcg"] else 0,
                "hit20": sum(d["hit20"]) / len(d["hit20"]) if d["hit20"] else 0,
                "hit50": sum(d["hit50"]) / len(d["hit50"]) if d["hit50"] else 0,
                "n": len(d["ndcg"]),
            }
            for name, d in baselines.items()
        },
        "blind_nn_coverage": {
            "exact_prefix": exact_any,
            "last3_suffix": suffix_any,
            "jaccard_50": jacc_high,
            "jaccard_80": jacc_very_high,
            "jaccard_mean": float(np.mean(jaccards)),
            "jaccard_median": float(np.median(jaccards)),
        },
    }
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
