#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R54 Phase 1: Fold-0 R21 reproduction.

Train BGE-base on dev folds 1-4 with R21-exact query/positive text.
Evaluate on fold-0: standalone hit@200/300, h7 retrieval hit@300.
Compare to saved R21 OOF fold-0 (hit@200=780, rate=0.4875).

No train-split data, no enriched metadata, no hard negatives.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np  # type: ignore[reportMissingImports]

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF_DIR = REPO / "cache" / "r21_production" / "oof"
R21_FOLD_INDICES = REPO / "cache" / "r21_production" / "fold_indices.json"
CACHE_DIR = REPO / "cache" / "r54" / "phase1"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

EPOCHS = 2
BATCH_SIZE = 32
LR = 2e-5
TAU = 0.05
MAX_SEQ_LEN = 256
TOPK = 300


def ts():
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


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
    from datasets import Dataset  # type: ignore[reportMissingImports]
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


def train_and_encode(train_cases, meta, all_track_ids, model_dir):
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    examples = []
    for c in train_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        query_text = build_query_text(c)
        track_text = build_track_text(gt, meta)
        examples.append((query_text, track_text))

    print(f"  {len(examples)} positive pairs", flush=True)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    model.train()
    tokenizer = model.tokenizer

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN,
                            return_tensors="pt")
        out = model.forward(encoded)
        emb = out["sentence_embedding"]
        return F_t.normalize(emb, dim=-1)

    for epoch in range(EPOCHS):
        perm = np.random.permutation(len(examples))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(examples), BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            queries = [examples[i][0] for i in batch_idx]
            positives = [examples[i][1] for i in batch_idx]

            q_emb = encode_with_grad(queries)
            p_emb = encode_with_grad(positives)

            sim = q_emb @ p_emb.T / TAU
            labels = torch.arange(len(queries), device=sim.device)
            loss = F_t.cross_entropy(sim, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                print(f"      batch {n_batches}: loss={loss.item():.4f}", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"    Epoch {epoch}: loss={avg_loss:.4f} ({n_batches} batches)", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    print(f"  Model saved to {model_dir}", flush=True)

    print(f"{ts()} Encoding {len(all_track_ids)} tracks...", flush=True)
    model.eval()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    return model, track_embs


def retrieve_top_k(model, queries, track_embs, track_ids, played_lists, topk=TOPK):
    print(f"{ts()} Encoding {len(queries)} queries...", flush=True)
    query_embs = model.encode(queries, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    print(f"{ts()} Retrieving top-{topk}...", flush=True)
    results = []
    for i in range(len(queries)):
        played_set = set(str(t) for t in played_lists[i]) if played_lists[i] else set()
        sims = query_embs[i] @ track_embs.T
        ranked = np.argsort(-sims)
        top = []
        for j in ranked:
            tid = track_ids[j]
            if tid not in played_set:
                top.append(tid)
                if len(top) >= topk:
                    break
        results.append(top)
    return results


def evaluate(name, retrieval_lists, val_cases, meta):
    n = len(val_cases)
    hit20 = hit50 = hit100 = hit200 = hit300 = 0
    h7_total = 0
    h7_hit200 = 0
    h7_hit300 = 0
    gt_ranks = []

    for i, c in enumerate(val_cases):
        gt = c["gt"]
        r = retrieval_lists[i]
        r_set = set(r[:300])

        if gt in set(r[:20]):
            hit20 += 1
        if gt in set(r[:50]):
            hit50 += 1
        if gt in set(r[:100]):
            hit100 += 1
        if gt in set(r[:200]):
            hit200 += 1
        if gt in r_set:
            hit300 += 1
            gt_ranks.append(r.index(gt))

        if c.get("n_prior_music") == 7:
            h7_total += 1
            if gt in set(r[:200]):
                h7_hit200 += 1
            if gt in r_set:
                h7_hit300 += 1

    print(f"  {name}: hit@20={hit20}/{n} ({hit20/n:.3f})  hit@200={hit200}/{n} ({hit200/n:.3f})  "
          f"hit@300={hit300}/{n} ({hit300/n:.3f})")
    if h7_total:
        print(f"    h7: hit@200={h7_hit200}/{h7_total} ({h7_hit200/h7_total:.3f})  "
              f"hit@300={h7_hit300}/{h7_total} ({h7_hit300/h7_total:.3f})")
    if gt_ranks:
        print(f"    median GT rank: {np.median(gt_ranks):.0f}")

    return {
        "hit20": hit20, "hit50": hit50, "hit100": hit100,
        "hit200": hit200, "hit300": hit300,
        "hit20_rate": hit20 / n, "hit200_rate": hit200 / n, "hit300_rate": hit300 / n,
        "h7_total": h7_total, "h7_hit200": h7_hit200, "h7_hit300": h7_hit300,
        "h7_hit200_rate": h7_hit200 / max(h7_total, 1),
        "h7_hit300_rate": h7_hit300 / max(h7_total, 1),
        "median_gt_rank": float(np.median(gt_ranks)) if gt_ranks else -1,
        "n": n,
    }


def compare_to_r21_oof(r54_lists, val_cases, r21_lists):
    n = len(val_cases)
    r54_only = 0
    r21_only = 0
    both = 0
    neither = 0

    for i, c in enumerate(val_cases):
        gt = c["gt"]
        in_r54 = gt in set(r54_lists[i][:300])
        in_r21 = gt in set(r21_lists[i][:300])
        if in_r54 and in_r21:
            both += 1
        elif in_r54:
            r54_only += 1
        elif in_r21:
            r21_only += 1
        else:
            neither += 1

    print(f"  Overlap vs R21 OOF fold-0:")
    print(f"    both={both}  R54-only={r54_only}  R21-only={r21_only}  neither={neither}")
    return {
        "both": both, "r54_only": r54_only, "r21_only": r21_only, "neither": neither,
    }


def main():
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)
    print(f"  {n} cases", flush=True)

    print(f"{ts()} Loading catalog metadata...", flush=True)
    meta, all_track_ids = load_catalog()
    print(f"  {len(all_track_ids)} tracks", flush=True)

    print(f"{ts()} Building fold-0 split...", flush=True)
    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)

    # Verify fold indices match saved R21
    if R21_FOLD_INDICES.exists():
        saved_fi = json.load(open(R21_FOLD_INDICES))
        saved_fold0 = saved_fi.get("0", saved_fi.get(0, []))
        computed_fold0 = folds[0].tolist()
        if saved_fold0 == computed_fold0:
            print(f"  Fold indices match saved R21 ✓", flush=True)
        else:
            print(f"  WARNING: Fold indices differ from saved R21!", flush=True)
            print(f"    Saved: {saved_fold0[:5]}...  Computed: {computed_fold0[:5]}...")

    val_idx = folds[0].tolist()
    train_idx = []
    for fi in range(1, 5):
        train_idx.extend(folds[fi].tolist())

    val_cases = [cases[j] for j in val_idx]
    train_cases = [cases[j] for j in train_idx]
    print(f"  Fold 0: train={len(train_cases)} val={len(val_cases)}", flush=True)

    # Train
    print(f"\n{ts()} Training R21-exact retriever (fold-0 held out)...", flush=True)
    model_dir = CACHE_DIR / "model_fold0"
    model, track_embs = train_and_encode(train_cases, meta, all_track_ids, model_dir)

    # Save embeddings
    emb_path = CACHE_DIR / "track_embs_fold0.npy"
    np.save(emb_path, track_embs)
    ids_path = CACHE_DIR / "track_ids_fold0.json"
    with open(ids_path, "w") as f:
        json.dump(all_track_ids, f)
    print(f"  Embeddings saved: {emb_path}", flush=True)

    # Retrieve
    val_queries = [build_query_text(c) for c in val_cases]
    val_played = [c["music_turns"] for c in val_cases]
    r54_lists = retrieve_top_k(model, val_queries, track_embs, all_track_ids, val_played)

    # Save retrieval lists
    lists_path = CACHE_DIR / "fold0_r54_lists.json"
    with open(lists_path, "w") as f:
        json.dump({"lists": r54_lists, "manifest": {
            "fold": 0,
            "train_cases": len(train_cases),
            "val_cases": len(val_cases),
            "model_dir": str(model_dir),
            "query_format": "r21_exact",
            "positive_format": "r21_exact",
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "tau": TAU,
            "created_at": datetime.now().isoformat(),
        }}, f)
    print(f"  Lists saved: {lists_path}", flush=True)

    # Evaluate
    print(f"\n{ts()} === EVALUATION ===", flush=True)
    r54_eval = evaluate("R54-Phase1", r54_lists, val_cases, meta)

    # Load and evaluate R21 OOF fold-0 for comparison
    r21_f0_path = R21_OOF_DIR / "fold_0_r21_lists.json"
    comparison = None
    r21_eval = None
    if r21_f0_path.exists():
        r21_data = json.load(open(r21_f0_path))
        r21_lists = r21_data["lists"]
        r21_eval = evaluate("R21-OOF-f0", r21_lists, val_cases, meta)
        comparison = compare_to_r21_oof(r54_lists, val_cases, r21_lists)

    # Hard check
    print(f"\n{ts()} === HARD CHECK ===", flush=True)
    r21_ref_hit200 = 780
    r21_ref_rate = 0.4875
    delta_hit200 = r54_eval["hit200"] - r21_ref_hit200
    delta_rate = r54_eval["hit200_rate"] - r21_ref_rate
    print(f"  R21 ref: hit@200={r21_ref_hit200} ({r21_ref_rate:.4f})")
    print(f"  R54:     hit@200={r54_eval['hit200']} ({r54_eval['hit200_rate']:.4f})")
    print(f"  Delta:   {delta_hit200:+d} ({delta_rate:+.4f})")

    tolerance = 30
    reproduced = abs(delta_hit200) <= tolerance
    status = "PASS" if reproduced else "FAIL"
    print(f"  Reproduction (±{tolerance}): {status}")

    if not reproduced:
        print(f"\n  *** REPRODUCTION FAILED — DO NOT PROCEED TO PHASE 2 ***")

    # Save results
    results = {
        "r54_phase1": r54_eval,
        "r21_oof_fold0": r21_eval,
        "comparison": comparison,
        "hard_check": {
            "r21_ref_hit200": r21_ref_hit200,
            "r54_hit200": r54_eval["hit200"],
            "delta": delta_hit200,
            "tolerance": tolerance,
            "status": status,
        },
        "config": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "tau": TAU, "max_seq_len": MAX_SEQ_LEN,
            "model": MODEL_NAME, "topk": TOPK,
            "query_format": "r21_exact (last 3 user utterances)",
            "positive_format": "r21_exact (name by artist. Album. Tags[:10])",
        },
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }

    out_path = REPO / "exp" / "eval" / "expR54_phase1_fold0_repro.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{ts()} Phase 1 complete. Elapsed: {time.time() - t0:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
