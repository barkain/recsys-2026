#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R23a: Hard-negative supervised retriever.

Same as R21 except the contrastive loss uses explicit hard negatives:
- 4 V3 top-ranked wrong candidates (RRF pool high-rank misses)
- 4 BM25 B/C wrong candidates (lexical near-misses)
- 4 same-artist wrong tracks
- 4 popular/tag-matched wrong tracks

Fast mode: train fold 0 only, compare to R21 fold 0 baseline (780/1600).
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL_NAME = "BAAI/bge-base-en-v1.5"
CACHE_DIR = REPO / "cache" / "r23a"
R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF_DIR = REPO / "cache" / "r21_production" / "oof"

EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-5
TOPK = 300
TAU = 0.07
N_HARD_NEG = 8  # total per query: 2 V3 + 2 BM25 + 2 same-artist + 2 popular


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_track_text(tid, meta):
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    album = m.get("album_name", [])
    tags = m.get("tag_list", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = ", ".join(artists) if isinstance(artists, list) else str(artists)
    alb = album[0] if isinstance(album, list) and album else str(album)
    tag_str = ", ".join(str(t) for t in tags[:10]) if isinstance(tags, list) else str(tags)
    return f"{name} by {artist}. Album: {alb}. Tags: {tag_str}"


def build_query_text(case):
    parts = []
    for h in case["history"]:
        if h["role"] == "user":
            parts.append(str(h["content"]))
    parts.append(case["user_query"])
    return " ".join(parts[-3:])


def load_catalog():
    from datasets import Dataset
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    if not matches:
        raise FileNotFoundError("all_tracks arrow not found")
    ds = Dataset.from_file(str(matches[-1]))
    cols = ds.to_dict()
    meta = {}
    track_ids = []
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        track_ids.append(tid)
        meta[tid] = {k: cols[k][i] for k in cols}
    assert len(track_ids) == len(set(track_ids))
    return meta, track_ids


def build_artist_index(meta):
    """Build artist → [track_id] index for same-artist negative mining."""
    artist_tracks = defaultdict(list)
    for tid, m in meta.items():
        artists = m.get("artist_name", [])
        if isinstance(artists, list) and artists:
            artist_tracks[artists[0]].append(tid)
    return artist_tracks


def build_popularity_index(meta):
    """Return track_ids sorted by popularity proxy (tag count)."""
    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    track_counts = Counter()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                track_counts[str(c["content"]).strip()] += 1
    popular = sorted(track_counts.keys(), key=lambda t: track_counts[t], reverse=True)
    return popular[:2000]


def mine_hard_negatives(case, gt, payload_idx, payload, meta, artist_tracks,
                        popular_tracks, rng, sources=None):
    """Mine hard negatives for a training case. sources: set of {v3,bm25,artist,tag}."""
    if sources is None:
        sources = {"v3", "bm25", "artist", "tag"}
    played_set = set(case["music_turns"]) | {gt}
    negatives = []
    active = [s for s in ["v3", "bm25", "artist", "tag"] if s in sources]
    n_per_source = max(N_HARD_NEG // max(len(active), 1), 1) if active else 0

    if "v3" in sources:
        from scripts.expF1_cfbpr_retrieval import weighted_rrf
        src = {
            "A": payload["src_a"][payload_idx],
            "B": payload["src_b"][payload_idx],
            "C": payload["src_c"][payload_idx],
            "D": payload["src_d"][payload_idx],
            "F": payload["src_f"][payload_idx],
        }
        sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
        pool = weighted_rrf(src, sw, topk=50, k=20)
        v3_negs = [t for t in pool if t not in played_set and t in meta][:n_per_source]
        negatives.extend(v3_negs)

    if "bm25" in sources:
        bm25_cands = list(payload["src_b"][payload_idx]) + list(payload["src_c"][payload_idx])
        bm25_negs = [t for t in bm25_cands if t not in played_set and t not in negatives and t in meta][:n_per_source]
        negatives.extend(bm25_negs)

    if "artist" in sources:
        gt_artists = meta.get(gt, {}).get("artist_name", [])
        if isinstance(gt_artists, list) and gt_artists:
            same_artist_pool = [t for t in artist_tracks.get(gt_artists[0], [])
                               if t not in played_set and t not in negatives]
            if len(same_artist_pool) > n_per_source:
                same_artist_pool = list(rng.choice(same_artist_pool, n_per_source, replace=False))
            negatives.extend(same_artist_pool[:n_per_source])

    if "tag" in sources:
        gt_tags = set(meta.get(gt, {}).get("tag_list", [])[:5])
        if gt_tags:
            tag_matched = [t for t in popular_tracks
                          if t not in played_set and t not in negatives and t in meta
                          and set(meta[t].get("tag_list", [])[:5]) & gt_tags][:n_per_source]
            negatives.extend(tag_matched)

    # Fill remaining with random popular if needed (only if we have active sources)
    if active:
        while len(negatives) < N_HARD_NEG:
            t = popular_tracks[rng.randint(len(popular_tracks))]
            if t not in played_set and t not in negatives and t in meta:
                negatives.append(t)

    return negatives[:N_HARD_NEG]


def train_fold_hard_neg(train_cases, train_indices, payload, meta, artist_tracks,
                        popular_tracks, model_dir, epochs=EPOCHS, batch_size=BATCH_SIZE,
                        lr=LR, neg_sources=None):
    """Train BGE with hard-negative contrastive loss."""
    import torch
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer

    rng = np.random.RandomState(42)

    # Pre-mine negatives for all training cases
    print(f"  {ts()} Mining hard negatives for {len(train_cases)} cases...", flush=True)
    all_queries = []
    all_positives = []
    all_neg_tids = []  # list of lists of track IDs
    all_unique_neg_tids = set()

    for local_i, (case, global_i) in enumerate(zip(train_cases, train_indices)):
        gt = case["gt"]
        if gt not in meta:
            continue
        all_queries.append(build_query_text(case))
        all_positives.append(build_track_text(gt, meta))
        neg_tids = mine_hard_negatives(case, gt, global_i, payload, meta,
                                        artist_tracks, popular_tracks, rng,
                                        sources=neg_sources)
        all_neg_tids.append(neg_tids)
        all_unique_neg_tids.update(neg_tids)

    n_examples = len(all_queries)
    avg_negs = np.mean([len(n) for n in all_neg_tids])
    print(f"  {n_examples} examples, avg {avg_negs:.1f} hard negatives each", flush=True)
    print(f"  {len(all_unique_neg_tids)} unique negative tracks", flush=True)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256,
                            return_tensors="pt")
        out = model.forward(encoded)
        return F_t.normalize(out["sentence_embedding"], dim=-1)

    # Pre-encode all unique negative tracks (no grad, done once per epoch)
    def precompute_neg_embeddings():
        neg_tid_list = sorted(all_unique_neg_tids)
        if not neg_tid_list:
            print(f"  No hard negatives (in-batch only mode)", flush=True)
            return torch.zeros(0, 768, dtype=torch.float32), {}
        neg_tid_to_idx = {t: i for i, t in enumerate(neg_tid_list)}
        neg_texts = [build_track_text(t, meta) for t in neg_tid_list]
        print(f"  Pre-encoding {len(neg_texts)} unique negative tracks...", flush=True)
        neg_embs = model.encode(neg_texts, batch_size=128, show_progress_bar=False,
                                 normalize_embeddings=True)
        return torch.tensor(neg_embs, dtype=torch.float32), neg_tid_to_idx

    # Use R21's exact training loop with optional hard-negative extension.
    # In-batch loss: sim = q @ p.T / tau, labels = arange(B) — identical to R21.
    # Hard negs: append extra columns to sim matrix, labels unchanged (positive still on diagonal).
    model.train()
    t_start = time.time()
    total_batches = (n_examples + batch_size - 1) // batch_size

    for epoch in range(epochs):
        # Pre-encode negatives at start of each epoch
        if all_unique_neg_tids:
            model.eval()
            neg_emb_bank, neg_tid_to_idx = precompute_neg_embeddings()
            model.train()
        else:
            neg_emb_bank = torch.zeros(0, 768, dtype=torch.float32)
            neg_tid_to_idx = {}

        # Shuffle examples — use global np.random like R21 (no fixed seed)
        perm = np.random.permutation(n_examples)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, n_examples, batch_size):
            batch_idx = perm[start:start + batch_size]
            bs = len(batch_idx)

            queries = [all_queries[i] for i in batch_idx]
            positives = [all_positives[i] for i in batch_idx]

            q_emb = encode_with_grad(queries)
            p_emb = encode_with_grad(positives)

            # R21-identical: full sim matrix, positive on diagonal
            sim = q_emb @ p_emb.T / TAU  # [B, B]

            # Append hard negative columns if available
            has_hard_negs = len(neg_emb_bank) > 0 and any(len(all_neg_tids[i]) > 0 for i in batch_idx)
            if has_hard_negs:
                D = q_emb.shape[1]
                K = max(len(all_neg_tids[i]) for i in batch_idx)
                neg_emb_batch = torch.zeros(bs, K, D)
                for bi, i in enumerate(batch_idx):
                    tids = all_neg_tids[i]
                    idxs = [neg_tid_to_idx[t] for t in tids if t in neg_tid_to_idx]
                    for ki, idx in enumerate(idxs[:K]):
                        neg_emb_batch[bi, ki] = neg_emb_bank[idx]
                hard_scores = torch.bmm(neg_emb_batch, q_emb.unsqueeze(2)).squeeze(2) / TAU  # [B, K]
                sim = torch.cat([sim, hard_scores], dim=1)  # [B, B+K]

            labels = torch.arange(bs, device=sim.device)  # positive at position i
            loss = F_t.cross_entropy(sim, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 20 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / n_batches * (total_batches * epochs - (epoch * total_batches + n_batches))
                print(f"    [{ts()}] e{epoch} batch {n_batches}/{total_batches} "
                      f"loss={loss.item():.4f} avg={epoch_loss/n_batches:.4f} "
                      f"ETA {eta/60:.0f}m", flush=True)

        print(f"  [{ts()}] Epoch {epoch}: avg_loss={epoch_loss/max(n_batches,1):.4f} "
              f"({n_batches} batches)", flush=True)

    model.save(str(model_dir))
    return model


def encode_and_retrieve(model, track_texts, all_track_ids, val_cases, topk=TOPK):
    print(f"  Encoding {len(all_track_ids)} tracks...", flush=True)
    track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    val_queries = [build_query_text(c) for c in val_cases]
    print(f"  Encoding {len(val_queries)} queries...", flush=True)
    query_embs = model.encode(val_queries, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    results = []
    for i in range(len(val_cases)):
        scores = track_embs @ query_embs[i]
        played_set = set(val_cases[i]["music_turns"])
        for idx, tid in enumerate(all_track_ids):
            if tid in played_set:
                scores[idx] = -np.inf
        top_idx = np.argpartition(-scores, topk)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results.append([all_track_ids[j] for j in top_idx])
    return results, track_embs


def evaluate_fold(val_cases, r23_lists, r21_lists, v3_pools, train_tracks):
    n = len(val_cases)
    r23_hit = sum(1 for i in range(n) if val_cases[i]["gt"] in r23_lists[i][:200])
    r21_hit = sum(1 for i in range(n) if val_cases[i]["gt"] in r21_lists[i][:200])

    unique_vs_r21 = sum(1 for i in range(n)
                        if val_cases[i]["gt"] in r23_lists[i][:200]
                        and val_cases[i]["gt"] not in r21_lists[i][:200])
    lost_vs_r21 = sum(1 for i in range(n)
                      if val_cases[i]["gt"] not in r23_lists[i][:200]
                      and val_cases[i]["gt"] in r21_lists[i][:200])

    unique_vs_v3 = sum(1 for i in range(n)
                       if val_cases[i]["gt"] in r23_lists[i][:200]
                       and val_cases[i]["gt"] not in v3_pools[i])

    # Median GT rank when hit
    gt_ranks = []
    for i in range(n):
        gt = val_cases[i]["gt"]
        if gt in r23_lists[i][:200]:
            gt_ranks.append(r23_lists[i].index(gt) + 1)
    r21_gt_ranks = []
    for i in range(n):
        gt = val_cases[i]["gt"]
        if gt in r21_lists[i][:200]:
            r21_gt_ranks.append(r21_lists[i].index(gt) + 1)

    # Unseen
    unseen_total = sum(1 for i in range(n) if val_cases[i]["gt"] not in train_tracks)
    unseen_hit = sum(1 for i in range(n)
                     if val_cases[i]["gt"] not in train_tracks
                     and val_cases[i]["gt"] in r23_lists[i][:200])

    return {
        "r23_hit200": r23_hit, "r21_hit200": r21_hit,
        "unique_vs_r21": unique_vs_r21, "lost_vs_r21": lost_vs_r21,
        "net": unique_vs_r21 - lost_vs_r21,
        "unique_vs_v3": unique_vs_v3,
        "r23_median_rank": float(np.median(gt_ranks)) if gt_ranks else 999,
        "r21_median_rank": float(np.median(r21_gt_ranks)) if r21_gt_ranks else 999,
        "unseen_hit": unseen_hit, "unseen_total": unseen_total,
        "n": n,
    }


def main():
    import argparse
    global TAU, N_HARD_NEG, EPOCHS
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, default=TAU)
    parser.add_argument("--n-neg", type=int, default=N_HARD_NEG)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--neg-sources", type=str, default="v3,bm25,artist,tag",
                        help="Comma-separated: v3,bm25,artist,tag. Use 'none' for R21-style baseline.")
    parser.add_argument("--tag", type=str, default="", help="Tag for output directory")
    parser.add_argument("--force", action="store_true", help="Delete existing run dir before training")
    args = parser.parse_args()
    TAU = args.tau
    N_HARD_NEG = args.n_neg
    EPOCHS = args.epochs
    neg_sources = set(args.neg_sources.split(",")) if args.neg_sources != "none" else set()
    run_tag = args.tag or f"tau{TAU}_neg{N_HARD_NEG}_{'_'.join(sorted(neg_sources)) or 'inbatch'}"

    t0 = time.time()
    run_dir = CACHE_DIR / run_tag
    if args.force and run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
        print(f"  Deleted existing run dir: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"{ts()} R23a: Hard-negative supervised retriever (fold 0 smoke)")
    print(f"  tau={TAU}, n_neg={N_HARD_NEG}, epochs={EPOCHS}, sources={neg_sources or 'in-batch only'}")
    print(f"  tag={run_tag}")
    print(f"{'='*60}")

    # Load data
    print(f"\n{ts()} Loading data...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    meta, all_track_ids = load_catalog()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    print(f"  Catalog: {len(all_track_ids)} tracks")

    artist_tracks = build_artist_index(meta)
    popular_tracks = build_popularity_index(meta)
    print(f"  Artist index: {len(artist_tracks)} artists")
    print(f"  Popular tracks: {len(popular_tracks)}")

    # Train tracks for unseen classification
    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_tracks = set()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                train_tracks.add(str(c["content"]).strip())

    # Load V3 pools
    v3_path = REPO / "cache" / "r21_production" / "v3_pools.json"
    with open(v3_path) as f:
        v3_pools_all = [set(p) for p in json.load(f)]

    # Fold 0
    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)
    held = folds[0].tolist()
    train_idx = [j for j in range(n) if j not in set(held)]

    print(f"\n  Fold 0: train={len(train_idx)} val={len(held)}")

    # Load R21 fold 0 lists for comparison
    r21_fold_path = R21_OOF_DIR / "fold_0_r21_lists.json"
    with open(r21_fold_path) as f:
        r21_fold_data = json.load(f)
    r21_fold_lists = r21_fold_data["lists"]

    # Train
    train_cases = [cases[j] for j in train_idx]
    model_dir = run_dir / "model_fold_0"

    if model_dir.exists():
        print(f"\n{ts()} Model exists, loading...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model_dir), device="cpu")
    else:
        print(f"\n{ts()} Training R23a fold 0...")
        model = train_fold_hard_neg(train_cases, train_idx, payload, meta,
                                     artist_tracks, popular_tracks, model_dir,
                                     neg_sources=neg_sources)

    # Retrieve
    val_cases = [cases[j] for j in held]
    val_v3 = [v3_pools_all[j] for j in held]
    print(f"\n{ts()} Encoding and retrieving...")
    r23_lists, _ = encode_and_retrieve(model, track_texts, all_track_ids, val_cases)

    # Evaluate
    print(f"\n{ts()} Evaluating fold 0...")
    results = evaluate_fold(val_cases, r23_lists, r21_fold_lists, val_v3, train_tracks)

    # Report
    print(f"\n{'='*60}")
    print(f"R23a FOLD 0 RESULTS")
    print(f"{'='*60}")
    print(f"\n  R23a hit@200: {results['r23_hit200']}/{results['n']} ({results['r23_hit200']/results['n']:.1%})")
    print(f"  R21  hit@200: {results['r21_hit200']}/{results['n']} ({results['r21_hit200']/results['n']:.1%})")
    print(f"  Δ: {results['r23_hit200'] - results['r21_hit200']:+d}")
    print(f"\n  Unique vs R21: {results['unique_vs_r21']} gained, {results['lost_vs_r21']} lost, net {results['net']:+d}")
    print(f"  Unique vs V3:  {results['unique_vs_v3']}")
    print(f"\n  Median GT rank: R23a={results['r23_median_rank']:.0f}  R21={results['r21_median_rank']:.0f}")
    print(f"  Unseen: {results['unseen_hit']}/{results['unseen_total']}")

    # Gates
    gate_hit = results["r23_hit200"] >= 780
    gate_unique = results["unique_vs_r21"] > results["lost_vs_r21"]
    gate_rank = results["r23_median_rank"] <= results["r21_median_rank"]
    print(f"\n  Gates:")
    print(f"    hit@200 >= 780:       {'PASS' if gate_hit else 'FAIL'} ({results['r23_hit200']})")
    print(f"    unique > lost:        {'PASS' if gate_unique else 'FAIL'} ({results['unique_vs_r21']} vs {results['lost_vs_r21']})")
    print(f"    median rank improves: {'PASS' if gate_rank else 'FAIL'} ({results['r23_median_rank']:.0f} vs {results['r21_median_rank']:.0f})")

    if gate_hit and gate_unique and gate_rank:
        print(f"\n>>> ALL FOLD 0 GATES PASS — proceed to 5-fold OOF <<<")
    elif gate_hit:
        print(f"\n>>> HIT GATE PASSES — review other metrics before 5-fold <<<")
    else:
        print(f"\n>>> FOLD 0 FAIL — diagnose before continuing <<<")

    # Save
    results["config"] = {"tau": TAU, "n_neg": N_HARD_NEG, "epochs": EPOCHS,
                         "batch_size": BATCH_SIZE, "lr": LR, "model": MODEL_NAME,
                         "neg_sources": sorted(neg_sources) if neg_sources else [],
                         "run_tag": run_tag}
    results["gates"] = {"hit": gate_hit, "unique": gate_unique, "rank": gate_rank}
    results["created_at"] = datetime.now().isoformat()
    out = run_dir / "fold_0_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    with open(run_dir / "fold_0_r23_lists.json", "w") as f:
        json.dump(r23_lists, f)

    elapsed = time.time() - t0
    print(f"\n{ts()} Complete. {elapsed:.0f}s ({elapsed/60:.0f}m)")


if __name__ == "__main__":
    main()
