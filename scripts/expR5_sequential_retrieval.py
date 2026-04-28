#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R5: Session-sequential retrieval — transition graph candidate source.

Stage 1: Data audit — extract ordered track sequences from train conversations.
Stage 2: Build cheap sequential sources G (transition, co-occurrence, artist backoff).
Stage 3: Evaluate on dev400 + synthetic low-history slices.

No API. No blind submission.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, DownloadConfig, load_dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.r3_confirm_400_deterministic import (
    build_or_load_payload,
    cv_folds,
    fit_weights,
    vec_ndcg,
)
from scripts.tune_postrank_v23 import FEATURE_NAMES, INIT_WEIGHTS, tokens
from scripts.expF1_cfbpr_retrieval import (
    build_cfbpr_index,
    cfbpr_max_recent,
    weighted_rrf,
)
from offline_retrieval_sweep import load_track_metadata
from mcrs.retrieval_modules.track_sim import TrackSimilarityRetriever


# =====================================================================
# STAGE 1: Data Audit
# =====================================================================

def stage1_audit():
    """Extract and analyze track sequences from train conversations."""
    print("=" * 70)
    print("STAGE 1: DATA AUDIT — Train conversation track sequences")
    print("=" * 70)

    ds = load_dataset(
        "talkpl-ai/TalkPlayData-Challenge-Dataset",
        download_config=DownloadConfig(local_files_only=True),
    )
    train = ds["train"]

    # Extract ordered track sequences per session
    session_sequences = {}  # sid -> list of track_ids in turn order
    user_sequences = defaultdict(list)  # uid -> list of (session_date, [track_ids])
    all_tracks = set()
    total_events = 0

    for item in train:
        sid = str(item["session_id"])
        uid = str(item["user_id"])
        date = item.get("session_date", "")
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        tracks = []
        for c in convs:
            if c["role"] == "music":
                tid = str(c["content"]).strip()
                tracks.append(tid)
                all_tracks.add(tid)
                total_events += 1
        session_sequences[sid] = tracks
        user_sequences[uid].append((date, tracks))

    # Sequence length distribution
    lengths = [len(v) for v in session_sequences.values()]
    length_dist = Counter(lengths)

    print(f"  Sessions: {len(session_sequences)}")
    print(f"  Total track events: {total_events}")
    print(f"  Unique tracks: {len(all_tracks)}")
    print(f"  Unique users: {len(user_sequences)}")
    print(f"  Sequence length: min={min(lengths)} max={max(lengths)} "
          f"mean={sum(lengths)/len(lengths):.1f} median={sorted(lengths)[len(lengths)//2]}")
    print(f"  Length distribution:")
    for l in sorted(length_dist.keys()):
        print(f"    len={l}: {length_dist[l]} sessions")

    # Verify turn_number ordering is reliable
    order_issues = 0
    for item in train:
        convs = sorted(item["conversations"], key=lambda c: int(c["turn_number"]))
        music_turns = [(int(c["turn_number"]), c["content"]) for c in convs if c["role"] == "music"]
        for i in range(1, len(music_turns)):
            if music_turns[i][0] <= music_turns[i-1][0]:
                order_issues += 1
    print(f"  Turn order issues: {order_issues}")

    return session_sequences, user_sequences, all_tracks


# =====================================================================
# STAGE 2: Build Sequential Sources
# =====================================================================

class SessionTransitionGraph:
    """Cheap transition/co-occurrence graph from train session sequences."""

    def __init__(self, session_sequences: dict[str, list[str]], metadata: dict):
        self.metadata = metadata
        t0 = time.time()

        # 1. Item-level transitions: last_track -> next_track counts
        self.item_transitions: dict[str, Counter] = defaultdict(Counter)
        # 2. Session co-occurrence: track -> set of co-occurring tracks
        self.cooccurrence: dict[str, Counter] = defaultdict(Counter)
        # 3. Artist-level transitions: last_artist -> tracks by next_artist
        self.artist_transitions: dict[str, Counter] = defaultdict(Counter)
        # 4. Artist -> tracks mapping for backoff
        self.artist_to_tracks: dict[str, list[str]] = defaultdict(list)

        for sid, tracks in session_sequences.items():
            # Item transitions (adjacent pairs)
            for i in range(len(tracks) - 1):
                self.item_transitions[tracks[i]][tracks[i+1]] += 1
            # Co-occurrence (all pairs in session)
            track_set = set(tracks)
            for t in tracks:
                for other in track_set:
                    if other != t:
                        self.cooccurrence[t][other] += 1
            # Artist transitions
            for i in range(len(tracks) - 1):
                a1 = self._artist(tracks[i])
                a2 = self._artist(tracks[i+1])
                if a1 and a2:
                    self.artist_transitions[a1][tracks[i+1]] += 1

        # Build artist -> tracks
        for tid in set().union(*[set(seq) for seq in session_sequences.values()]):
            a = self._artist(tid)
            if a:
                self.artist_to_tracks[a].append(tid)

        elapsed = time.time() - t0
        print(f"  Transition graph built in {elapsed:.1f}s")
        print(f"  Item transitions: {len(self.item_transitions)} source tracks")
        print(f"  Co-occurrence: {len(self.cooccurrence)} source tracks")
        print(f"  Artist transitions: {len(self.artist_transitions)} source artists")
        # Sparsity stats
        trans_counts = [sum(v.values()) for v in self.item_transitions.values()]
        print(f"  Item transition counts: mean={np.mean(trans_counts):.1f} "
              f"median={np.median(trans_counts):.0f} max={max(trans_counts)}")
        cooc_sizes = [len(v) for v in self.cooccurrence.values()]
        print(f"  Co-occurrence neighbors: mean={np.mean(cooc_sizes):.1f} "
              f"median={np.median(cooc_sizes):.0f} max={max(cooc_sizes)}")

    def _artist(self, tid: str) -> str:
        meta = self.metadata.get(str(tid), {})
        return str(meta.get("artist_name", "")).lower().strip()

    def g_last_transition(self, played: list[str], topn: int) -> list[str]:
        """Candidates from next-track transition counts after last played."""
        if not played:
            return []
        last = played[-1]
        exclude = set(played)
        counts = self.item_transitions.get(last, Counter())
        return [tid for tid, _ in counts.most_common(topn + len(exclude))
                if tid not in exclude][:topn]

    def g_recent3_transition(self, played: list[str], topn: int) -> list[str]:
        """Union of next-track transitions from last 3 played, weighted by recency."""
        if not played:
            return []
        exclude = set(played)
        scores: dict[str, float] = {}
        for i, tid in enumerate(reversed(played[-3:])):
            weight = 1.0 / (i + 1)
            for next_tid, count in self.item_transitions.get(tid, Counter()).items():
                if next_tid not in exclude:
                    scores[next_tid] = scores.get(next_tid, 0.0) + weight * count
        return sorted(scores, key=scores.__getitem__, reverse=True)[:topn]

    def g_session_cooccur(self, played: list[str], topn: int) -> list[str]:
        """Tracks co-occurring in same sessions, weighted by recency of anchor."""
        if not played:
            return []
        exclude = set(played)
        scores: dict[str, float] = {}
        for i, tid in enumerate(reversed(played[-5:])):
            weight = 1.0 / (i + 1)
            for co_tid, count in self.cooccurrence.get(tid, Counter()).items():
                if co_tid not in exclude:
                    scores[co_tid] = scores.get(co_tid, 0.0) + weight * count
        return sorted(scores, key=scores.__getitem__, reverse=True)[:topn]

    def g_artist_transition_backoff(self, played: list[str], topn: int) -> list[str]:
        """Artist-level transition backoff: last artist -> tracks by next-artist."""
        if not played:
            return []
        exclude = set(played)
        last_artist = self._artist(played[-1])
        if not last_artist:
            return []
        counts = self.artist_transitions.get(last_artist, Counter())
        return [tid for tid, _ in counts.most_common(topn + len(exclude))
                if tid not in exclude][:topn]


# =====================================================================
# STAGE 3: Evaluation
# =====================================================================

def build_8f_features(pool, case, track_artist, track_tags,
                      track_title_toks, track_artist_toks, track_meta_toks,
                      pool_k=50):
    """Build 8-feature row for a single case."""
    F = len(FEATURE_NAMES)
    X = np.zeros((pool_k, F), dtype=np.float64)
    user_msgs = [str(r["content"]) for r in case["history"] if r["role"] == "user"] + [case["user_query"]]
    played = case["music_turns"]
    now_tok = tokens(user_msgs[-1]) if user_msgs else set()
    all_tok = tokens(" ".join(user_msgs))
    played_set = set(played)
    l_artist = track_artist.get(played[-1], "") if played else ""
    l_tags = track_tags.get(played[-1], set()) if played else set()
    prior = [(1.0/(j+1), track_artist.get(t,""), track_tags.get(t,set()))
             for j,t in enumerate(reversed(played))]
    for rank, tid in enumerate(pool[:pool_k], start=1):
        ca = track_artist.get(tid, "")
        ct = track_tags.get(tid, set())
        row = X[rank-1]
        row[0] = 1.0/rank
        row[1] = 1.0 if ca and ca == l_artist else 0.0
        if ct or l_tags: row[2] = len(ct & l_tags) / len(ct | l_tags)
        row[3] = float(len(track_artist_toks.get(tid, set()) & now_tok))
        row[4] = float(len(track_title_toks.get(tid, set()) & now_tok))
        row[5] = float(len(track_meta_toks.get(tid, set()) & all_tok))
        row[6] = 1.0 if tid in played_set else 0.0
        rec = 0.0
        for wd, pa, pt in prior:
            am = 1.0 if ca and ca == pa else 0.0
            tm = len(ct & pt) / len(ct | pt) if (ct or pt) else 0.0
            rec += wd * (am + tm)
        row[7] = rec
    return X


def main():
    t0 = time.time()

    # =====================================================================
    # STAGE 1
    # =====================================================================
    session_sequences, user_sequences, all_train_tracks = stage1_audit()

    # =====================================================================
    # STAGE 2
    # =====================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: BUILD SEQUENTIAL SOURCES")
    print("=" * 70)

    metadata = load_track_metadata()
    graph = SessionTransitionGraph(session_sequences, metadata)

    # =====================================================================
    # STAGE 3: Evaluation
    # =====================================================================
    print("\n" + "=" * 70)
    print("STAGE 3: EVALUATION")
    print("=" * 70)

    # Load dev400 payload
    payload = build_or_load_payload()
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    track_artist = payload["track_artist"]
    track_tags = payload["track_tags"]
    track_title_toks = payload["track_title_toks"]
    track_artist_toks = payload["track_artist_toks"]
    track_meta_toks = payload["track_meta_toks"]

    def ensure_meta(tids):
        for tid in tids:
            if tid in track_artist:
                continue
            meta = metadata.get(str(tid), {})
            artist = str(meta.get("artist_name", "")).lower().strip()
            raw_tags = meta.get("tag_list") or []
            tags = {str(t).lower().strip() for t in raw_tags if str(t).strip()} if isinstance(raw_tags, list) else set()
            title = str(meta.get("track_name", ""))
            meta_parts = [title, str(meta.get("artist_name", "")), str(meta.get("album_name", ""))]
            if isinstance(raw_tags, list):
                meta_parts.extend(str(t) for t in raw_tags[:12])
            track_artist[tid] = artist
            track_tags[tid] = tags
            track_title_toks[tid] = tokens(title)
            track_artist_toks[tid] = tokens(meta.get("artist_name", ""))
            track_meta_toks[tid] = tokens(" ".join(meta_parts))

    # Build A' and CF-BPR sources for comparison
    print("\nComputing A' (qwen3) for overlap analysis...", flush=True)
    qwen_sim = TrackSimilarityRetriever(cache_dir="./cache")
    src_a_prime = []
    for c in cases:
        played = c["music_turns"]
        a_idxs = [qwen_sim._id_to_idx.get(str(t)) for t in played[-5:]]
        a_idxs = [i for i in a_idxs if i is not None]
        if a_idxs:
            anchor_vecs = qwen_sim.vectors[a_idxs]
            sims = qwen_sim.vectors @ anchor_vecs.T
            scores_a = sims.max(axis=1)
            exclude_a = {qwen_sim._id_to_idx[t] for t in played if t in qwen_sim._id_to_idx}
            cap_a = min(len(scores_a), 200 + len(exclude_a))
            cand_a = np.argpartition(-scores_a, cap_a - 1)[:cap_a]
            cand_a = cand_a[np.argsort(-scores_a[cand_a])]
            out_a = []
            for ii in cand_a:
                if int(ii) in exclude_a:
                    continue
                out_a.append(qwen_sim.track_ids[int(ii)])
                if len(out_a) >= 200:
                    break
            src_a_prime.append(out_a)
        else:
            src_a_prime.append([])

    print("Computing CF-BPR source...", flush=True)
    track_ids_cf, vectors_cf, id_to_idx_cf = build_cfbpr_index()
    cf_sources = []
    for c in cases:
        played = c["music_turns"]
        if played:
            result = cfbpr_max_recent(played, vectors_cf, id_to_idx_cf, track_ids_cf, 5, 200)
            ensure_meta(result)
        else:
            result = []
        cf_sources.append(result)

    # cfg0209 ABCD baseline pools
    cfg0209_pools = []
    for i, c in enumerate(cases):
        sources_dict = {"A": src_a_prime[i], "B": payload["src_b"][i],
                        "C": payload["src_c"][i], "D": payload["src_d"][i]}
        w = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5}
        pool = weighted_rrf(sources_dict, w, topk=50, k=20)
        cfg0209_pools.append(set(pool))

    # cfg0209+F1 pools
    cfg0209f1_pools = []
    for i, c in enumerate(cases):
        sources_dict = {"A": src_a_prime[i], "B": payload["src_b"][i],
                        "C": payload["src_c"][i], "D": payload["src_d"][i],
                        "F": cf_sources[i]}
        w = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
        pool = weighted_rrf(sources_dict, w, topk=50, k=20)
        cfg0209f1_pools.append(set(pool))

    # ---- Evaluate G variants on full dev400 ----
    G_DEPTH = 200
    g_variants = {
        "G_last_transition": lambda played: graph.g_last_transition(played, G_DEPTH),
        "G_recent3_transition": lambda played: graph.g_recent3_transition(played, G_DEPTH),
        "G_session_cooccur": lambda played: graph.g_session_cooccur(played, G_DEPTH),
        "G_artist_backoff": lambda played: graph.g_artist_transition_backoff(played, G_DEPTH),
    }

    def eval_source(label, src_lists, history_label="full"):
        """Evaluate a source's standalone recall and unique hits."""
        gt_hits = {k: 0 for k in [50, 100, 200]}
        unique_vs_cfg0209 = 0
        unique_vs_cfg0209f1 = 0
        overlap_cf = []
        overlap_qwen = []
        n_nonempty = 0
        n_total = len(src_lists)

        for i, (g_list, c) in enumerate(zip(src_lists, cases)):
            gt = c["gt"]
            if g_list:
                n_nonempty += 1
            for k in gt_hits:
                if gt in g_list[:k]:
                    gt_hits[k] += 1
            if gt in g_list[:200] and gt not in cfg0209_pools[i]:
                unique_vs_cfg0209 += 1
            if gt in g_list[:200] and gt not in cfg0209f1_pools[i]:
                unique_vs_cfg0209f1 += 1
            g_set = set(g_list[:200])
            cf_set = set(cf_sources[i][:200])
            a_set = set(src_a_prime[i][:200])
            if g_set:
                overlap_cf.append(len(g_set & cf_set) / len(g_set) if cf_set else 0)
                overlap_qwen.append(len(g_set & a_set) / len(g_set) if a_set else 0)

        recall = {f"@{k}": gt_hits[k] / n_total for k in gt_hits}
        result = {
            "history": history_label,
            "n_nonempty": n_nonempty,
            "n_total": n_total,
            "recall": recall,
            "unique_gt_vs_cfg0209": unique_vs_cfg0209,
            "unique_gt_vs_cfg0209f1": unique_vs_cfg0209f1,
            "overlap_cf_mean": float(np.mean(overlap_cf)) if overlap_cf else None,
            "overlap_qwen_mean": float(np.mean(overlap_qwen)) if overlap_qwen else None,
        }
        print(f"  {label:30s} [{history_label}]  nonempty={n_nonempty}/{n_total}  "
              f"recall@50={recall['@50']:.4f} @200={recall['@200']:.4f}  "
              f"uniq_vs_cfg0209={unique_vs_cfg0209}  uniq_vs_cfg0209+F1={unique_vs_cfg0209f1}  "
              f"overlap_cf={result['overlap_cf_mean']:.3f}" if result['overlap_cf_mean'] else "")
        return result

    print("\n--- Full dev400 evaluation ---")
    full_results = {}
    for vname, retriever_fn in g_variants.items():
        src_lists = []
        for c in cases:
            played = c["music_turns"]
            result = retriever_fn(played) if played else []
            ensure_meta(result)
            src_lists.append(result)
        full_results[vname] = eval_source(vname, src_lists, "full_dev400")

    # ---- Synthetic low-history evaluation ----
    print("\n--- Synthetic low-history evaluation ---")
    low_history_results = {}
    for max_history in [1, 2, 3]:
        label = f"trunc_{max_history}"
        print(f"\n  Truncated history: last {max_history} music events")
        for vname, retriever_fn in g_variants.items():
            src_lists = []
            for c in cases:
                played = c["music_turns"][-max_history:]  # truncate
                result = retriever_fn(played) if played else []
                ensure_meta(result)
                src_lists.append(result)
            key = f"{vname}_{label}"
            low_history_results[key] = eval_source(vname, src_lists, label)

    # ---- Fusion evaluation for promising variants ----
    # Pick best G variant by unique_gt_vs_cfg0209f1 on full dev400
    best_g = max(full_results, key=lambda v: full_results[v]["unique_gt_vs_cfg0209f1"])
    print(f"\n--- Best G variant: {best_g} ---")
    print(f"  unique GT vs cfg0209+F1: {full_results[best_g]['unique_gt_vs_cfg0209f1']}")

    # Fusion: cfg0209 + F1 + G
    print("\n--- Fusion: cfg0209 + F1 + G ---")
    best_retriever = g_variants[best_g]
    g_weights_grid = [0.25, 0.5, 1.0, 2.0]
    POOL_K = 50
    seeds = [0, 1, 2]

    # Build G source for all cases
    g_full = []
    for c in cases:
        played = c["music_turns"]
        result = best_retriever(played) if played else []
        ensure_meta(result)
        g_full.append(result)

    # Baseline: ABCD+F (F1)
    print("  Building F1 baseline (ABCD+F)...")
    bl_X = np.zeros((n, POOL_K, len(FEATURE_NAMES)), dtype=np.float64)
    bl_gt = np.full(n, -1, dtype=np.int64)
    bl_sz = np.zeros(n, dtype=np.int64)
    for i, c in enumerate(cases):
        sources_dict = {"A": src_a_prime[i], "B": payload["src_b"][i],
                        "C": payload["src_c"][i], "D": payload["src_d"][i],
                        "F": cf_sources[i]}
        w = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
        pool = weighted_rrf(sources_dict, w, topk=POOL_K, k=20)
        bl_sz[i] = len(pool)
        if c["gt"] in pool:
            bl_gt[i] = pool.index(c["gt"])
        bl_X[i] = build_8f_features(pool, c, track_artist, track_tags,
                                     track_title_toks, track_artist_toks, track_meta_toks)

    bl_hit = float(np.mean(bl_gt >= 0))
    bl_cv5_seeds = []
    for seed in seeds:
        folds = cv_folds(sessions, seed)
        fold_sc = []
        for fold in folds:
            held = set(fold.tolist())
            train_idx = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
            w, _ = fit_weights(bl_X, bl_gt, bl_sz, train_idx)
            fold_sc.append(vec_ndcg(bl_X, bl_gt, bl_sz, w, fold))
        bl_cv5_seeds.append(float(np.mean(fold_sc)))
    bl_cv5 = float(np.mean(bl_cv5_seeds))
    print(f"  F1 baseline: pool_hit={bl_hit:.4f}  CV5={bl_cv5:.4f}")

    # Fusion grid
    fusion_results = []
    for g_weight in g_weights_grid:
        X = np.zeros((n, POOL_K, len(FEATURE_NAMES)), dtype=np.float64)
        gt_idx = np.full(n, -1, dtype=np.int64)
        sizes = np.zeros(n, dtype=np.int64)
        for i, c in enumerate(cases):
            sources_dict = {"A": src_a_prime[i], "B": payload["src_b"][i],
                            "C": payload["src_c"][i], "D": payload["src_d"][i],
                            "F": cf_sources[i], "G": g_full[i]}
            w = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "G": g_weight}
            pool = weighted_rrf(sources_dict, w, topk=POOL_K, k=20)
            sizes[i] = len(pool)
            if c["gt"] in pool:
                gt_idx[i] = pool.index(c["gt"])
            X[i] = build_8f_features(pool, c, track_artist, track_tags,
                                      track_title_toks, track_artist_toks, track_meta_toks)
        hit = float(np.mean(gt_idx >= 0))
        cv5_seeds = []
        for seed in seeds:
            folds = cv_folds(sessions, seed)
            fold_sc = []
            for fold in folds:
                held = set(fold.tolist())
                train_idx = np.asarray([j for j in range(n) if j not in held], dtype=np.int64)
                w, _ = fit_weights(X, gt_idx, sizes, train_idx)
                fold_sc.append(vec_ndcg(X, gt_idx, sizes, w, fold))
            cv5_seeds.append(float(np.mean(fold_sc)))
        cv5 = float(np.mean(cv5_seeds))
        fusion_results.append({
            "g_weight": g_weight, "pool_hit": hit, "cv5": cv5,
            "cv5_delta": cv5 - bl_cv5,
        })
        print(f"  w_G={g_weight:.2f}  pool_hit={hit:.4f} (Δ={hit-bl_hit:+.4f})  "
              f"CV5={cv5:.4f} (Δ={cv5-bl_cv5:+.4f})")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    elapsed = time.time() - t0
    best_fusion = max(fusion_results, key=lambda r: r["cv5"])

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best G variant: {best_g}")
    print(f"  full dev400 unique GT vs cfg0209+F1: {full_results[best_g]['unique_gt_vs_cfg0209f1']}")
    print(f"F1 baseline (ABCD+F):  CV5={bl_cv5:.4f}  pool_hit={bl_hit:.4f}")
    print(f"Best fusion (ABCD+F+G): CV5={best_fusion['cv5']:.4f} (Δ={best_fusion['cv5_delta']:+.4f})  "
          f"w_G={best_fusion['g_weight']}")

    # Gate
    unique_rate = full_results[best_g]["unique_gt_vs_cfg0209f1"] / n
    if best_fusion["cv5"] >= 0.178:
        verdict = "STRONG"
    elif best_fusion["cv5_delta"] >= 0.008 or unique_rate >= 0.03:
        verdict = "PROMISING"
    elif unique_rate < 0.01:
        verdict = "FAIL"
    else:
        verdict = "WEAK"

    print(f"\nGATE VERDICT: {verdict}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Save
    out_path = REPO_ROOT / "exp" / "eval" / "expR5_sequential_retrieval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "meta": {"n_cases": n, "elapsed_sec": elapsed},
        "stage1": {
            "n_sessions": len(session_sequences),
            "n_events": sum(len(v) for v in session_sequences.values()),
            "n_unique_tracks": len(all_train_tracks),
        },
        "full_dev400": {k: {kk: vv for kk, vv in v.items()} for k, v in full_results.items()},
        "low_history": {k: v for k, v in low_history_results.items()},
        "best_g": best_g,
        "f1_baseline": {"cv5": bl_cv5, "pool_hit": bl_hit},
        "fusion": fusion_results,
        "best_fusion": best_fusion,
        "gate_verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nArtifact: {out_path}")


if __name__ == "__main__":
    main()
