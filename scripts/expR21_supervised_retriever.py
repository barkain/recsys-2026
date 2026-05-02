#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R21: Supervised bi-encoder content retriever.

Fine-tune BGE-base on (conversation → next track) pairs with hard negatives.
The key difference from R13: R13 used frozen embeddings. R21 learns the
mapping from conversation text to track metadata on THIS dataset.

Stage 1: retrieval-only diagnostic.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
RRF_K = 20
MODEL_NAME = "BAAI/bge-base-en-v1.5"
CACHE_DIR = REPO_ROOT / "cache" / "r21_supervised"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def build_track_text(tid, meta):
    """Build a text description of a track from metadata."""
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
    """Build query text from conversation context."""
    parts = []
    for h in case["history"]:
        if h["role"] == "user":
            parts.append(str(h["content"]))
    parts.append(case["user_query"])
    return " ".join(parts[-3:])  # last 3 user utterances for focus


def load_catalog_metadata():
    """Load full track metadata."""
    from datasets import Dataset, concatenate_datasets
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
    return meta, track_ids


def train_and_encode(train_cases, meta, all_track_ids, model_dir, epochs=3,
                     batch_size=32, lr=2e-5):
    """Fine-tune BGE on (query, positive_track) pairs and encode all tracks."""
    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    print(f"{ts()} Building training pairs...", flush=True)

    # Build positive pairs
    examples = []
    for c in train_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        query_text = build_query_text(c)
        track_text = build_track_text(gt, meta)
        examples.append(InputExample(texts=[query_text, track_text]))

    print(f"  {len(examples)} positive pairs")

    # Add hard negatives: for each positive, add 2-3 hard negatives
    # from the played tracks' same artist (wrong track, same artist)
    hard_examples = []
    for c in train_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        query_text = build_query_text(c)
        played = c["music_turns"]

        # Same-artist hard negative: pick a track by same artist that is NOT the GT
        gt_artists = meta.get(gt, {}).get("artist_name", [])
        if gt_artists and isinstance(gt_artists, list):
            gt_artist = gt_artists[0]
            for tid in all_track_ids:
                if tid == gt:
                    continue
                t_artists = meta.get(tid, {}).get("artist_name", [])
                if isinstance(t_artists, list) and t_artists and t_artists[0] == gt_artist:
                    neg_text = build_track_text(tid, meta)
                    hard_examples.append(InputExample(texts=[query_text, track_text, neg_text]))
                    break

    print(f"  {len(hard_examples)} hard negative triplets")

    # Use MultipleNegativesRankingLoss (in-batch negatives)
    import torch
    import torch.nn.functional as F_t

    device = "cpu"
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print(f"{ts()} Manual fine-tune on {device} ({len(examples)} pairs, {epochs} epochs, bs={batch_size})...", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    tokenizer = model.tokenizer

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256,
                            return_tensors="pt").to(device)
        out = model.forward(encoded)
        emb = out["sentence_embedding"]
        return F_t.normalize(emb, dim=-1)

    for epoch in range(epochs):
        np.random.shuffle(examples)
        epoch_loss = 0
        n_batches = 0
        for start in range(0, len(examples), batch_size):
            batch = examples[start:start + batch_size]
            queries = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]

            q_emb = encode_with_grad(queries)
            p_emb = encode_with_grad(positives)

            sim = q_emb @ p_emb.T / 0.05
            labels = torch.arange(len(batch), device=sim.device)
            loss = F_t.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                print(f"      batch {n_batches}: loss={loss.item():.4f}", flush=True)

        print(f"    Epoch {epoch}: loss={epoch_loss/max(n_batches,1):.4f}", flush=True)

    model.save(str(model_dir))

    # Encode all tracks
    print(f"{ts()} Encoding {len(all_track_ids)} tracks...", flush=True)
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True)

    return model, track_embs.astype(np.float32)


def retrieve_top_k(model, query_texts, track_embs, track_ids, played_lists,
                   topk=200, batch_size=64):
    """Retrieve top-k tracks for each query."""
    print(f"{ts()} Encoding {len(query_texts)} queries...", flush=True)
    query_embs = model.encode(query_texts, batch_size=batch_size,
                               show_progress_bar=True, normalize_embeddings=True)
    query_embs = query_embs.astype(np.float32)

    print(f"{ts()} Retrieving top-{topk}...", flush=True)
    results = []
    for i in range(len(query_texts)):
        scores = track_embs @ query_embs[i]
        played_set = set(played_lists[i])
        for idx, tid in enumerate(track_ids):
            if tid in played_set:
                scores[idx] = -np.inf
        top_idx = np.argpartition(-scores, topk)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results.append([track_ids[j] for j in top_idx])
    return results


def evaluate(name, results, cases, v3_pools, train_tracks_set, ta, track_pop, n):
    """Evaluate retrieval results."""
    hit20 = hit50 = hit100 = hit200 = 0
    unique_vs_v3 = 0
    unique_unreachable = 0
    pop0_rec = diff_rec = hist0_rec = 0
    unseen_hit200 = unseen_total = 0
    seen_hit200 = seen_total = 0
    gt_ranks = []

    for i, c in enumerate(cases):
        gt = c["gt"]
        played = c["music_turns"]
        r = results[i]
        r_set = set(r[:200])
        is_unseen = gt not in train_tracks_set

        if is_unseen: unseen_total += 1
        else: seen_total += 1

        if gt in r[:20]: hit20 += 1
        if gt in r[:50]: hit50 += 1
        if gt in r[:100]: hit100 += 1
        if gt in r_set:
            hit200 += 1
            if is_unseen: unseen_hit200 += 1
            else: seen_hit200 += 1

            gt_rank = r.index(gt)
            gt_ranks.append(gt_rank)

            if gt not in v3_pools[i]:
                unique_vs_v3 += 1
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

    return {
        "hit20": hit20, "hit50": hit50, "hit100": hit100, "hit200": hit200,
        "unique_vs_v3": unique_vs_v3,
        "pop0_rec": pop0_rec, "diff_rec": diff_rec, "hist0_rec": hist0_rec,
        "unseen_hit200": unseen_hit200, "unseen_total": unseen_total,
        "seen_hit200": seen_hit200, "seen_total": seen_total,
        "median_gt_rank": float(np.median(gt_ranks)) if gt_ranks else -1,
    }


def main():
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    ta = payload["track_artist"]

    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_tracks = set()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                train_tracks.add(str(c["content"]).strip())

    from scripts.expS2_lr_v2 import build_popularity_stats
    track_pop = build_popularity_stats()

    print(f"{ts()} Loading catalog metadata...", flush=True)
    meta, all_track_ids = load_catalog_metadata()
    print(f"  {len(all_track_ids)} tracks in catalog")

    # Build V3 pools for comparison (without ALS to avoid implicit conflict)
    # Use payload sources directly
    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0}
    v3_pools = []
    for i in range(n):
        sl = {"A": payload["src_a"][i], "B": payload["src_b"][i],
              "C": payload["src_c"][i], "D": payload["src_d"][i],
              "F": payload["src_f"][i]}
        v3_pools.append(set(weighted_rrf(sl, sw, topk=200, k=RRF_K)))

    # Grouped-session CV: train on 4 folds, evaluate on 1
    print(f"\n{ts()} === GROUPED-SESSION CV (seed=0) ===")
    from scripts.expS2_lambdarank_grouped import grouped_session_folds

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    folds = grouped_session_folds(sessions, seed=0)
    all_results = []

    for fold_i in range(min(2, len(folds))):
        held = set(folds[fold_i].tolist())
        train_idx = [j for j in range(n) if j not in held]
        val_idx = folds[fold_i].tolist()

        train_cases = [cases[j] for j in train_idx]
        val_cases = [cases[j] for j in val_idx]

        print(f"\n{ts()} Fold {fold_i}: train={len(train_idx)} val={len(val_idx)}")

        model_dir = CACHE_DIR / f"fold_{fold_i}"

        # Train and encode
        model, track_embs = train_and_encode(
            train_cases, meta, all_track_ids, model_dir,
            epochs=2, batch_size=32, lr=2e-5)

        # Retrieve for val cases
        val_queries = [build_query_text(c) for c in val_cases]
        val_played = [c["music_turns"] for c in val_cases]
        val_results = retrieve_top_k(model, val_queries, track_embs, all_track_ids,
                                      val_played, topk=200)

        # Also get BM25 baseline for same val cases
        # (BM25 results are already in payload)
        bm25_results = []
        for j in val_idx:
            bc_set = set(payload["src_b"][j][:200]) | set(payload["src_c"][j][:200])
            bm25_results.append(list(bc_set)[:200])

        # Evaluate
        v3_pools_val = [v3_pools[j] for j in val_idx]
        ev_r21 = evaluate("R21", val_results, val_cases, v3_pools_val,
                          train_tracks, ta, track_pop, len(val_cases))
        ev_bm25 = evaluate("BM25_BC", bm25_results, val_cases, v3_pools_val,
                           train_tracks, ta, track_pop, len(val_cases))

        nv = len(val_cases)
        print(f"  R21:  hit@200={ev_r21['hit200']}/{nv} ({ev_r21['hit200']/nv:.1%})  "
              f"unseen={ev_r21['unseen_hit200']}/{ev_r21['unseen_total']}  "
              f"unique_vs_v3={ev_r21['unique_vs_v3']}  "
              f"median_rank={ev_r21['median_gt_rank']:.0f}")
        print(f"  BM25: hit@200={ev_bm25['hit200']}/{nv} ({ev_bm25['hit200']/nv:.1%})  "
              f"unseen={ev_bm25['unseen_hit200']}/{ev_bm25['unseen_total']}")

        all_results.append({"fold": fold_i, "r21": ev_r21, "bm25": ev_bm25})
        del model, track_embs

    # Aggregate
    print(f"\n{ts()} === AGGREGATE RESULTS ===")
    total_r21 = Counter()
    total_bm25 = Counter()
    total_n = 0
    for r in all_results:
        for k, v in r["r21"].items():
            if isinstance(v, (int, float)):
                total_r21[k] += v
        for k, v in r["bm25"].items():
            if isinstance(v, (int, float)):
                total_bm25[k] += v
        total_n += r["r21"]["unseen_total"] + r["r21"]["seen_total"]

    print(f"  R21 aggregate (n={total_n}):")
    print(f"    hit@200: {total_r21['hit200']}/{total_n} ({total_r21['hit200']/total_n:.1%})")
    print(f"    unseen hit@200: {total_r21['unseen_hit200']}/{total_r21['unseen_total']} "
          f"({total_r21['unseen_hit200']/max(total_r21['unseen_total'],1):.1%})")
    print(f"    seen hit@200: {total_r21['seen_hit200']}/{total_r21['seen_total']} "
          f"({total_r21['seen_hit200']/max(total_r21['seen_total'],1):.1%})")
    print(f"    unique_vs_v3: {total_r21['unique_vs_v3']}")
    print(f"    pop0: {total_r21['pop0_rec']}  diff_artist: {total_r21['diff_rec']}  "
          f"hist0: {total_r21['hist0_rec']}")

    print(f"\n  BM25 B∪C aggregate:")
    print(f"    hit@200: {total_bm25['hit200']}/{total_n} ({total_bm25['hit200']/total_n:.1%})")
    print(f"    unseen hit@200: {total_bm25['unseen_hit200']}/{total_bm25['unseen_total']} "
          f"({total_bm25['unseen_hit200']/max(total_bm25['unseen_total'],1):.1%})")

    print(f"\n  GATE: R21 unseen hit@200 > BM25 unseen hit@200?")
    r21_unseen_rate = total_r21['unseen_hit200'] / max(total_r21['unseen_total'], 1)
    bm25_unseen_rate = total_bm25['unseen_hit200'] / max(total_bm25['unseen_total'], 1)
    print(f"    R21: {r21_unseen_rate:.1%}  BM25: {bm25_unseen_rate:.1%}  "
          f"{'PASS' if r21_unseen_rate > bm25_unseen_rate else 'FAIL'}")

    elapsed = time.time() - t0
    print(f"\n{ts()} R21 complete. Elapsed: {elapsed:.1f}s")

    out_path = REPO_ROOT / "exp" / "eval" / "expR21_supervised_retriever.json"
    with open(out_path, "w") as f:
        json.dump({"results": all_results, "elapsed_s": elapsed}, f, indent=2, default=str)
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
