#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""Train 5-fold OOF R21 retrieval lists + production model.

Writes per-fold artifacts for recoverability. Resumes from existing folds.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME = "BAAI/bge-base-en-v1.5"
OOF_DIR = REPO_ROOT / "cache" / "r21_production" / "oof"
PROD_DIR = REPO_ROOT / "cache" / "r21_production"
R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"


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
    assert len(track_ids) == len(set(track_ids)), f"Duplicate track IDs! {len(track_ids)} vs {len(set(track_ids))}"
    return meta, track_ids


def train_fold_model(train_cases, meta, model_dir, epochs=2, batch_size=32, lr=2e-5):
    import torch
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer, InputExample

    examples = []
    for c in train_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        query_text = build_query_text(c)
        track_text = build_track_text(gt, meta)
        examples.append(InputExample(texts=[query_text, track_text]))

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256,
                            return_tensors="pt")
        out = model.forward(encoded)
        return F_t.normalize(out["sentence_embedding"], dim=-1)

    model.train()
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
    return model


def encode_and_retrieve(model, track_texts, all_track_ids, val_cases, topk=300):
    print(f"    Encoding {len(all_track_ids)} tracks...", flush=True)
    track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    val_queries = [build_query_text(c) for c in val_cases]
    print(f"    Encoding {len(val_queries)} queries...", flush=True)
    query_embs = model.encode(val_queries, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    print(f"    Retrieving top-{topk}...", flush=True)
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


def main():
    t0 = time.time()
    OOF_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]

    # Train track set for unseen classification
    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_tracks = set()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                train_tracks.add(str(c["content"]).strip())

    meta, all_track_ids = load_catalog()
    print(f"  Catalog: {len(all_track_ids)} tracks, {len(set(all_track_ids))} unique")
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]

    # Build V3 pools for comparison
    from scripts.expF1_cfbpr_retrieval import weighted_rrf
    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector, grouped_session_folds

    print(f"{ts()} Training ALS...", flush=True)
    als_factors, als_track_ids_als, als_track_to_idx = build_als()
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
            als_source.append([als_track_ids_als[j] for j in top_idx])
        else:
            als_source.append([])

    sw = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0}
    v3_pools = []
    for i in range(n):
        sl = {"A": payload["src_a"][i], "B": payload["src_b"][i],
              "C": payload["src_c"][i], "D": payload["src_d"][i],
              "F": payload["src_f"][i], "ALS": als_source[i]}
        v3_pools.append(set(weighted_rrf(sl, sw, topk=200, k=20)))

    # ===== 5-FOLD OOF =====
    folds = grouped_session_folds(sessions, seed=0)
    all_oof_lists = [None] * n
    manifest = {"catalog_size": len(all_track_ids), "unique_track_ids": len(set(all_track_ids)),
                "folds": {}}

    for fold_i in range(5):
        fold_file = OOF_DIR / f"fold_{fold_i}_r21_lists.json"
        manifest_key = str(fold_i)

        # Resume check
        if fold_file.exists():
            print(f"\n{ts()} Fold {fold_i}: FOUND existing artifact, loading...", flush=True)
            with open(fold_file) as f:
                fold_data = json.load(f)
            held = folds[fold_i].tolist()
            for j_local, j_global in enumerate(held):
                all_oof_lists[j_global] = fold_data["lists"][j_local]
            manifest["folds"][manifest_key] = fold_data["manifest"]
            hit200 = fold_data["manifest"]["hit@200"]
            print(f"  Loaded {len(held)} lists, hit@200={hit200}")
            continue

        held = folds[fold_i].tolist()
        train_idx = [j for j in range(n) if j not in set(held)]

        print(f"\n{ts()} Fold {fold_i}: train={len(train_idx)} val={len(held)}", flush=True)

        # Train model
        model_dir = OOF_DIR / f"model_fold_{fold_i}"
        train_cases = [cases[j] for j in train_idx]
        print(f"  Training R21 fold model...", flush=True)
        model = train_fold_model(train_cases, meta, model_dir)

        # Retrieve
        val_cases = [cases[j] for j in held]
        fold_lists, _ = encode_and_retrieve(model, track_texts, all_track_ids, val_cases)

        # Store
        for j_local, j_global in enumerate(held):
            all_oof_lists[j_global] = fold_lists[j_local]

        # Compute metrics
        hit200 = sum(1 for j_local, j_global in enumerate(held)
                     if cases[j_global]["gt"] in fold_lists[j_local][:200])
        unseen_hit = sum(1 for j_local, j_global in enumerate(held)
                         if cases[j_global]["gt"] not in train_tracks
                         and cases[j_global]["gt"] in fold_lists[j_local][:200])
        unseen_total = sum(1 for j_global in held
                           if cases[j_global]["gt"] not in train_tracks)
        unique_vs_v3 = sum(1 for j_local, j_global in enumerate(held)
                           if cases[j_global]["gt"] in fold_lists[j_local][:200]
                           and cases[j_global]["gt"] not in v3_pools[j_global])

        fold_manifest = {
            "fold": fold_i,
            "train_cases": len(train_idx),
            "val_cases": len(held),
            "model_path": str(model_dir),
            "hit@200": hit200,
            "hit@200_rate": hit200 / len(held),
            "unseen_hit@200": unseen_hit,
            "unseen_total": unseen_total,
            "unseen_hit_rate": unseen_hit / max(unseen_total, 1),
            "unique_vs_v3": unique_vs_v3,
            "catalog_size": len(all_track_ids),
            "created_at": datetime.now().isoformat(),
        }

        # Save per-fold artifact
        with open(fold_file, "w") as f:
            json.dump({"lists": fold_lists, "manifest": fold_manifest}, f)
        manifest["folds"][manifest_key] = fold_manifest

        print(f"  Fold {fold_i}: hit@200={hit200}/{len(held)} ({hit200/len(held):.1%})  "
              f"unseen={unseen_hit}/{unseen_total}  unique_vs_v3={unique_vs_v3}")
        print(f"  Saved: {fold_file}")

        del model

    # Combine all folds
    assert all(x is not None for x in all_oof_lists), "Missing OOF lists!"
    with open(PROD_DIR / "dev_r21_oof_lists.json", "w") as f:
        json.dump(all_oof_lists, f)

    # Save manifest
    manifest["total_hit@200"] = sum(1 for i in range(n)
                                     if cases[i]["gt"] in all_oof_lists[i][:200])
    manifest["total_hit@200_rate"] = manifest["total_hit@200"] / n
    manifest["created_at"] = datetime.now().isoformat()
    with open(PROD_DIR / "oof_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{ts()} ===== ALL 5 FOLDS COMPLETE =====")
    print(f"  Total OOF hit@200: {manifest['total_hit@200']}/{n} ({manifest['total_hit@200_rate']:.1%})")
    for fi in range(5):
        fm = manifest["folds"][str(fi)]
        print(f"  Fold {fi}: hit@200={fm['hit@200']}/{fm['val_cases']} ({fm['hit@200_rate']:.1%})  "
              f"unseen={fm['unseen_hit@200']}/{fm['unseen_total']}  unique_vs_v3={fm['unique_vs_v3']}")

    # ===== PRODUCTION MODEL =====
    prod_model_dir = PROD_DIR / "model"
    prod_embs_path = PROD_DIR / "track_embeddings.npy"
    prod_ids_path = PROD_DIR / "track_ids.json"

    if prod_model_dir.exists() and prod_embs_path.exists():
        print(f"\n{ts()} Production model already exists, skipping.")
    else:
        print(f"\n{ts()} Training production R21 model (all {n} cases)...", flush=True)
        prod_model = train_fold_model(cases, meta, prod_model_dir)

        print(f"{ts()} Encoding catalog for production...", flush=True)
        prod_model.eval()
        track_embs = prod_model.encode(track_texts, batch_size=128, show_progress_bar=True,
                                        normalize_embeddings=True).astype(np.float32)
        np.save(prod_embs_path, track_embs)
        with open(prod_ids_path, "w") as f:
            json.dump(all_track_ids, f)
        print(f"  Saved production embeddings: {track_embs.shape}")
        del prod_model

    elapsed = time.time() - t0
    print(f"\n{ts()} Complete. Elapsed: {elapsed:.1f}s ({elapsed/3600:.1f}h)")


if __name__ == "__main__":
    main()
