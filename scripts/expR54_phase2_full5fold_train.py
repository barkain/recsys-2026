#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R54 Phase 2 full 5-fold OOF retriever training.

Trains a structured-query BGE retriever for each fold (held-out = val fold).
Reuses Phase 2 fold-0 artifacts if present.

For each fold f in 0..4:
- Train on cases in folds {0..4} \ {f}
- Encode full catalog
- Retrieve top-300 for cases in fold f
- Save: model dir, track embeddings, OOF lists with cosine scores, fold manifest

Aggregated output: cache/r54/phase2_full/oof_r54_lists.json
  { case_idx -> [{tid, score, rank}, ...] }  for all 8000 cases
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
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
PHASE2_FOLD0 = REPO / "cache" / "r54" / "phase2"
CACHE_DIR = REPO / "cache" / "r54" / "phase2_full"
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


def build_short_track_ref(tid, meta):
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = artists[0] if isinstance(artists, list) and artists else str(artists)
    return f"{name} by {artist}"


def build_query_structured(case, meta):
    user_utterances = []
    played_tracks = []
    for h in case["history"]:
        role = h.get("role", "")
        content = str(h.get("content", ""))
        if role == "user":
            user_utterances.append(content)
        elif role == "music":
            tid = content.strip()
            if tid in meta:
                played_tracks.append(build_short_track_ref(tid, meta))

    current_query = case["user_query"]
    history = user_utterances[-3:]
    context_tracks = played_tracks[-5:]

    parts = [f"[QUERY] {current_query}"]
    if history:
        parts.append(f"[HISTORY] {' '.join(history)}")
    if context_tracks:
        parts.append(f"[CONTEXT] {'; '.join(context_tracks)}")
    return " ".join(parts)


def load_catalog():
    from datasets import Dataset  # type: ignore[reportMissingImports]
    hf_cache = Path.home() / ".cache" / "huggingface" / "datasets"
    matches = sorted(hf_cache.glob(
        "talkpl-ai___talk_play_data-challenge-track-metadata/default/*/*/"
        "talk_play_data-challenge-track-metadata-all_tracks.arrow"))
    ds = Dataset.from_file(str(matches[-1]))
    cols = ds.to_dict()
    meta = {}
    track_ids = []
    for i in range(len(cols["track_id"])):
        tid = str(cols["track_id"][i])
        track_ids.append(tid)
        meta[tid] = {k: cols[k][i] for k in cols}
    return meta, track_ids


def train_model(train_cases, meta, model_dir):
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    examples = []
    for c in train_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        examples.append((build_query_structured(c, meta), build_track_text(gt, meta)))

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

        print(f"    Epoch {epoch}: loss={epoch_loss/max(n_batches,1):.4f}", flush=True)

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(model_dir))
    return model


def encode_catalog(model, meta, all_track_ids):
    print(f"{ts()} Encoding {len(all_track_ids)} tracks...", flush=True)
    model.eval()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                        normalize_embeddings=True).astype(np.float32)
    return embs


def retrieve(model, queries, track_embs, track_ids, played_lists, topk=TOPK):
    """Return list of [(tid, score)] per query, played tracks excluded."""
    print(f"{ts()} Encoding {len(queries)} queries...", flush=True)
    q_embs = model.encode(queries, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True).astype(np.float32)
    print(f"{ts()} Retrieving top-{topk}...", flush=True)
    results = []
    for i in range(len(queries)):
        played_set = set(str(t) for t in played_lists[i]) if played_lists[i] else set()
        sims = q_embs[i] @ track_embs.T  # cosine since both normalized
        ranked = np.argsort(-sims)
        top = []
        for j in ranked:
            tid = track_ids[j]
            if tid not in played_set:
                top.append((tid, float(sims[j])))
                if len(top) >= topk:
                    break
        results.append(top)
    return results


def main():
    t0 = time.time()
    print(f"{ts()} R54 Phase 2 full 5-fold OOF training")

    print(f"{ts()} Loading payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    n = len(cases)
    print(f"  {n} cases")

    print(f"{ts()} Loading catalog...")
    meta, all_track_ids = load_catalog()
    print(f"  {len(all_track_ids)} tracks")

    print(f"{ts()} Building folds...")
    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)
    for fi, f in enumerate(folds):
        print(f"  Fold {fi}: {len(f)} cases")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    oof_results = [None] * n  # case_idx -> list of (tid, score)
    fold_manifests = {}

    for fold_i in range(5):
        fold_t0 = time.time()
        val_idx = folds[fold_i].tolist()
        train_idx = []
        for fj in range(5):
            if fj != fold_i:
                train_idx.extend(folds[fj].tolist())

        val_cases = [cases[j] for j in val_idx]
        train_cases = [cases[j] for j in train_idx]
        print(f"\n{ts()} === FOLD {fold_i} === train={len(train_cases)} val={len(val_cases)}")

        fold_dir = CACHE_DIR / f"fold_{fold_i}"
        model_dir = fold_dir / "model"
        embs_path = fold_dir / "track_embs.npy"
        lists_path = fold_dir / "oof_lists.json"

        # Reuse Phase 2 fold-0 artifacts if available
        reused = False
        if fold_i == 0 and (PHASE2_FOLD0 / "model_fold0").exists():
            print(f"  Reusing Phase 2 fold-0 artifacts")
            if not model_dir.exists():
                fold_dir.mkdir(parents=True, exist_ok=True)
                shutil.copytree(str(PHASE2_FOLD0 / "model_fold0"), str(model_dir))
            if not embs_path.exists():
                src_embs = PHASE2_FOLD0 / "track_embs_fold0.npy"
                shutil.copy(str(src_embs), str(embs_path))
            reused = True

        if lists_path.exists():
            print(f"  Fold {fold_i} lists already exist — loading")
            fold_data = json.load(open(lists_path))
            for k, v in enumerate(val_idx):
                oof_results[v] = fold_data["lists"][k]
            fold_manifests[fold_i] = fold_data["manifest"]
            continue

        from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]
        if reused or model_dir.exists():
            print(f"  Loading existing model from {model_dir}")
            model = SentenceTransformer(str(model_dir), device="cpu")
        else:
            print(f"{ts()} Training model for fold {fold_i}...")
            model = train_model(train_cases, meta, model_dir)

        if embs_path.exists():
            print(f"  Loading existing embeddings")
            track_embs = np.load(embs_path)
        else:
            track_embs = encode_catalog(model, meta, all_track_ids)
            np.save(embs_path, track_embs)
            print(f"  Saved embeddings to {embs_path}")

        val_queries = [build_query_structured(c, meta) for c in val_cases]
        val_played = [c["music_turns"] for c in val_cases]
        fold_lists = retrieve(model, val_queries, track_embs, all_track_ids, val_played)

        for k, v in enumerate(val_idx):
            oof_results[v] = fold_lists[k]

        manifest = {
            "fold": fold_i,
            "train_cases": len(train_cases),
            "val_cases": len(val_cases),
            "model_dir": str(model_dir),
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "tau": TAU,
            "topk": TOPK, "query_format": "structured",
            "positive_format": "r21_exact",
            "reused_phase2": reused,
            "elapsed_s": time.time() - fold_t0,
            "created_at": datetime.now().isoformat(),
        }
        fold_manifests[fold_i] = manifest

        with open(lists_path, "w") as f:
            json.dump({"lists": fold_lists, "manifest": manifest, "val_idx": val_idx}, f)
        print(f"  Fold {fold_i} saved. Elapsed: {time.time() - fold_t0:.0f}s")

        # Quick sanity: standalone hit@200 on val
        hits200 = sum(
            1 for k, c in enumerate(val_cases)
            if c["gt"] in [t for t, _ in fold_lists[k][:200]]
        )
        print(f"  Fold {fold_i} val hit@200: {hits200}/{len(val_cases)} ({hits200/len(val_cases):.4f})")

        # Free memory
        del model, track_embs

    # Aggregate OOF lists
    print(f"\n{ts()} Saving aggregated OOF lists...")
    agg_path = CACHE_DIR / "oof_r54_lists.json"
    with open(agg_path, "w") as f:
        json.dump({
            "lists": oof_results,
            "fold_manifests": fold_manifests,
            "n_cases": n,
            "created_at": datetime.now().isoformat(),
        }, f)
    print(f"  Aggregated lists saved: {agg_path}")

    print(f"\n{ts()} Phase 2 full 5-fold training complete. Total elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
