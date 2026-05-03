#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R22b: Scale R21 supervised retriever to ALL training sessions.

R21 trained BGE on 8K dev cases (~6.4K per OOF fold).
R22b trains on 15,199 non-dev training sessions (~121K pairs).
Dev sessions (1,000 sessions / 8K cases) are fully disjoint — no OOF needed.

Evaluates against R21 baseline on:
  - Standalone hit@200 on dev 8K
  - Unique GTs vs R21 pool and V3 pool
  - pool_hit@300 (primary gate: must reach >= 0.620)
  - LambdaRank last-turn nDCG (secondary gate: must not drop)

Run in subprocess to avoid implicit/torch conflicts.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
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
CACHE_DIR = REPO_ROOT / "cache" / "r22b"
R12_CACHE = REPO_ROOT / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO_ROOT / "cache" / "r21_production" / "dev_r21_oof_lists.json"
V3_POOLS = REPO_ROOT / "cache" / "r21_production" / "v3_pools.json"

EPOCHS = 1
BATCH_SIZE = 16
GRAD_ACCUM = 2  # effective batch size = 16 * 2 = 32
LR = 2e-5
TOPK = 300
DEVICE = "cpu"
CHECKPOINT_EVERY = 1000
PROBE_ENABLED = False  # disable to save memory on resume


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


def build_query_text_from_history(history, user_query):
    parts = []
    for h in history:
        if h["role"] == "user":
            parts.append(str(h["content"]))
    parts.append(user_query)
    return " ".join(parts[-3:])


def build_query_text(case):
    return build_query_text_from_history(case["history"], case["user_query"])


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
    assert len(track_ids) == len(set(track_ids)), "Duplicate track IDs"
    return meta, track_ids


def build_all_train_pairs(meta, dev_session_ids):
    """Build supervised (query_text, track_text) pairs from ALL non-dev training sessions."""
    from datasets import DownloadConfig, load_dataset

    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]

    pairs = []
    skipped_no_meta = 0
    skipped_dev = 0
    n_sessions_used = 0

    for item in train_ds:
        sid = item["session_id"]
        if sid in dev_session_ids:
            skipped_dev += 1
            continue

        convs = item["conversations"]
        n_sessions_used += 1

        history = []
        for conv in convs:
            role = conv["role"]
            content = str(conv["content"])

            if role == "music":
                track_id = content.strip()
                if track_id not in meta:
                    skipped_no_meta += 1
                    history.append({"role": "music", "content": content})
                    continue

                query_text = build_query_text_from_history(history, "")
                if not query_text.strip():
                    # First turn, use prior user message if any
                    user_msgs = [h["content"] for h in history if h["role"] == "user"]
                    if user_msgs:
                        query_text = " ".join(user_msgs[-3:])
                    else:
                        history.append({"role": "music", "content": content})
                        continue

                track_text = build_track_text(track_id, meta)
                pairs.append((query_text, track_text))

            history.append({"role": role, "content": content})

    print(f"  Sessions used: {n_sessions_used}")
    print(f"  Sessions skipped (dev): {skipped_dev}")
    print(f"  Pairs: {len(pairs)}")
    print(f"  Skipped (no metadata): {skipped_no_meta}")
    return pairs


def train_model(pairs, model_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR,
                device=DEVICE, probe_cases=None, probe_meta=None, probe_track_ids=None):
    import torch
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer, InputExample
    import gc

    examples = [InputExample(texts=[q, t]) for q, t in pairs]

    # Build probe data (only if enabled and data provided)
    probe_ready = False
    if PROBE_ENABLED and probe_cases is not None and probe_meta is not None and probe_track_ids is not None:
        rng = np.random.RandomState(42)
        probe_n = min(500, len(probe_cases))
        probe_idx = rng.choice(len(probe_cases), probe_n, replace=False)
        probe_sample = [probe_cases[i] for i in probe_idx]
        probe_gts = {probe_cases[i]["gt"] for i in probe_idx}
        gt_tids = list(probe_gts & set(probe_track_ids))
        other_tids = [t for t in probe_track_ids if t not in probe_gts]
        sample_other = list(rng.choice(other_tids, min(5000, len(other_tids)), replace=False))
        probe_tids = gt_tids + sample_other
        probe_tid_set = set(probe_tids)
        probe_texts = [build_track_text(t, probe_meta) for t in probe_tids]
        probe_queries = [build_query_text(c) for c in probe_sample]
        probe_ready = True
        print(f"  Probe ready: {probe_n} queries, {len(probe_tids)} tracks "
              f"({len(gt_tids)} GTs included)", flush=True)
    elif not PROBE_ENABLED:
        print(f"  Probe disabled (PROBE_ENABLED=False)", flush=True)

    # Resume from checkpoint if available
    ckpt_dir = CACHE_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    resume_batch = 0
    resume_epoch = 0

    latest_ckpt = sorted(ckpt_dir.glob("ckpt_e*_b*.pt"))
    if latest_ckpt:
        ckpt_path = latest_ckpt[-1]
        parts = ckpt_path.stem.split("_")
        resume_epoch = int(parts[1][1:])
        ckpt_batch = int(parts[2][1:])
        # Checkpoint may have been saved with different batch_size.
        # Estimate samples seen and convert to current batch count.
        ckpt_bs = 32  # original batch size used during checkpointing
        samples_seen = ckpt_batch * ckpt_bs
        resume_batch = samples_seen // batch_size
        print(f"  Resuming from checkpoint: {ckpt_path.name} "
              f"(~{samples_seen} samples seen → skip {resume_batch} batches of {batch_size})",
              flush=True)
        model = SentenceTransformer(MODEL_NAME, device=device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
    else:
        model = SentenceTransformer(MODEL_NAME, device=device)

    tokenizer = model.tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if latest_ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        del ckpt
        gc.collect()

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=256,
                            return_tensors="pt").to(device)
        out = model.forward(encoded)
        return F_t.normalize(out["sentence_embedding"], dim=-1)

    total_batches = (len(examples) + batch_size - 1) // batch_size
    model.train()
    t_train_start = time.time()
    optimizer.zero_grad()
    for epoch in range(resume_epoch, epochs):
        np.random.shuffle(examples)
        epoch_loss = 0
        n_batches = 0
        accum_count = 0
        for start in range(0, len(examples), batch_size):
            n_batches += 1
            if epoch == resume_epoch and n_batches <= resume_batch:
                continue

            batch = examples[start:start + batch_size]
            queries = [ex.texts[0] for ex in batch]
            positives = [ex.texts[1] for ex in batch]
            q_emb = encode_with_grad(queries)
            p_emb = encode_with_grad(positives)
            sim = q_emb @ p_emb.T / 0.05
            labels = torch.arange(len(batch), device=sim.device)
            loss = F_t.cross_entropy(sim, labels) / GRAD_ACCUM
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM
            accum_count += 1

            if accum_count >= GRAD_ACCUM:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

            if n_batches % 200 == 0:
                elapsed = time.time() - t_train_start
                done = n_batches - resume_batch if epoch == resume_epoch else n_batches
                remaining = total_batches - n_batches
                eta_sec = (elapsed / max(done, 1)) * remaining
                print(f"    [{ts()}] batch {n_batches}/{total_batches} "
                      f"loss={loss.item()*GRAD_ACCUM:.4f} avg={epoch_loss/max(done,1):.4f} "
                      f"ETA {eta_sec/3600:.1f}h", flush=True)

            if n_batches % CHECKPOINT_EVERY == 0:
                # Flush any remaining gradients
                if accum_count > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_count = 0

                ckpt_path = ckpt_dir / f"ckpt_e{epoch}_b{n_batches}.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch, "batch": n_batches,
                    "loss": epoch_loss / max(n_batches - (resume_batch if epoch == resume_epoch else 0), 1),
                }, ckpt_path)
                # Delete older checkpoints to save disk
                for old in sorted(ckpt_dir.glob("ckpt_e*_b*.pt"))[:-1]:
                    old.unlink()
                print(f"    [{ts()}] Checkpoint saved: {ckpt_path.name}", flush=True)

                if probe_ready:
                    model.eval()
                    with torch.no_grad():
                        t_emb = model.encode(probe_texts, batch_size=128,
                                             normalize_embeddings=True).astype(np.float32)
                        q_emb_p = model.encode(probe_queries, batch_size=64,
                                               normalize_embeddings=True).astype(np.float32)
                    hits = 0
                    for pi in range(len(probe_sample)):
                        gt = probe_sample[pi]["gt"]
                        if gt not in probe_tid_set:
                            continue
                        scores = t_emb @ q_emb_p[pi]
                        played = set(probe_sample[pi]["music_turns"])
                        for ti, tid in enumerate(probe_tids):
                            if tid in played:
                                scores[ti] = -np.inf
                        top200 = np.argpartition(-scores, min(200, len(scores)-1))[:200]
                        if gt in {probe_tids[j] for j in top200}:
                            hits += 1
                    gt_in_catalog = sum(1 for c in probe_sample if c["gt"] in probe_tid_set)
                    print(f"    [{ts()}] PROBE hit@200: {hits}/{gt_in_catalog} "
                          f"({hits/max(gt_in_catalog,1):.1%}) on {len(probe_sample)} sample",
                          flush=True)
                    del t_emb, q_emb_p
                    gc.collect()
                    model.train()

        actual_trained = n_batches - (resume_batch if epoch == resume_epoch else 0)
        print(f"    [{ts()}] Epoch {epoch} complete: avg_loss={epoch_loss/max(actual_trained,1):.4f} "
              f"({actual_trained} batches trained, {time.time()-t_train_start:.0f}s elapsed)", flush=True)

    # Save final model (move to CPU for encoding compatibility)
    model.to("cpu")
    model.save(str(model_dir))
    return model


def encode_and_retrieve(model, track_texts, all_track_ids, dev_cases, topk=TOPK):
    print(f"  Encoding {len(all_track_ids)} tracks...", flush=True)
    track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    dev_queries = [build_query_text(c) for c in dev_cases]
    print(f"  Encoding {len(dev_queries)} dev queries...", flush=True)
    query_embs = model.encode(dev_queries, batch_size=64, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    print(f"  Retrieving top-{topk}...", flush=True)
    results = []
    for i in range(len(dev_cases)):
        scores = track_embs @ query_embs[i]
        played_set = set(dev_cases[i]["music_turns"])
        for idx, tid in enumerate(all_track_ids):
            if tid in played_set:
                scores[idx] = -np.inf
        top_idx = np.argpartition(-scores, topk)[:topk]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        results.append([all_track_ids[j] for j in top_idx])
    return results, track_embs, query_embs


def evaluate(dev_cases, r22b_lists, r21_oof_lists, v3_pools, train_tracks):
    """Comprehensive evaluation against R21 baseline and gates."""
    n = len(dev_cases)

    # Integrity checks
    assert len(r22b_lists) == n, f"R22b lists length mismatch: {len(r22b_lists)} vs {n}"
    n_dup_lists = sum(1 for lst in r22b_lists if len(lst) != len(set(lst)))
    print(f"  Duplicate-ID check: {n_dup_lists}/{n} lists have duplicates", flush=True)

    # Standalone R22b metrics at multiple depths
    r22b_hit = {}
    for depth in [20, 50, 100, 200, 300]:
        r22b_hit[depth] = sum(1 for i in range(n)
                              if dev_cases[i]["gt"] in r22b_lists[i][:depth])

    # R21 OOF baseline at same depths
    r21_hit = {}
    for depth in [20, 50, 100, 200, 300]:
        r21_hit[depth] = sum(1 for i in range(n)
                             if dev_cases[i]["gt"] in r21_oof_lists[i][:depth])

    # Unique vs R21
    unique_vs_r21 = sum(1 for i in range(n)
                        if dev_cases[i]["gt"] in r22b_lists[i][:200]
                        and dev_cases[i]["gt"] not in r21_oof_lists[i][:200])
    lost_vs_r21 = sum(1 for i in range(n)
                      if dev_cases[i]["gt"] not in r22b_lists[i][:200]
                      and dev_cases[i]["gt"] in r21_oof_lists[i][:200])

    # Unique vs V3
    unique_vs_v3 = sum(1 for i in range(n)
                       if dev_cases[i]["gt"] in r22b_lists[i][:200]
                       and dev_cases[i]["gt"] not in v3_pools[i])

    # Unseen analysis
    unseen_total = sum(1 for i in range(n)
                       if dev_cases[i]["gt"] not in train_tracks)
    unseen_hit = sum(1 for i in range(n)
                     if dev_cases[i]["gt"] not in train_tracks
                     and dev_cases[i]["gt"] in r22b_lists[i][:200])
    seen_hit = sum(1 for i in range(n)
                   if dev_cases[i]["gt"] in train_tracks
                   and dev_cases[i]["gt"] in r22b_lists[i][:200])
    seen_total = n - unseen_total

    # Pool hit simulation: V3 + R22b at pool_k=300
    from scripts.expF1_cfbpr_retrieval import weighted_rrf

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)

    sw_r22b = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R22b": 1.0}

    from scripts.expS2_lambdarank import build_als
    from scripts.expS2_lambdarank_grouped import als_session_vector

    print(f"  Building ALS for pool simulation...", flush=True)
    als_factors, als_track_ids_als, als_track_to_idx = build_als()
    als_source = []
    for c in dev_cases:
        played = c["music_turns"]
        sv = als_session_vector(played, als_track_to_idx, als_factors)
        if sv is not None:
            scores = als_factors @ sv
            for t in played:
                if t in als_track_to_idx:
                    scores[als_track_to_idx[t]] = -np.inf
            top_idx = np.argpartition(-scores, 200)[:200]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            als_source.append([als_track_ids_als[j] for j in top_idx])
        else:
            als_source.append([])

    pool_hit_r22b = 0
    pool_hit_r21 = 0
    for i in range(n):
        gt = dev_cases[i]["gt"]
        src = {
            "A": payload["src_a"][i], "B": payload["src_b"][i],
            "C": payload["src_c"][i], "D": payload["src_d"][i],
            "F": payload["src_f"][i], "ALS": als_source[i],
            "R22b": r22b_lists[i][:200],
        }
        pool_r22b = set(weighted_rrf(src, sw_r22b, topk=300, k=20))
        if gt in pool_r22b:
            pool_hit_r22b += 1

        src_r21 = dict(src)
        src_r21["R21"] = r21_oof_lists[i][:200]
        del src_r21["R22b"]
        sw_r21 = {"A": 1.0, "B": 1.0, "C": 1.0, "D": 0.5, "F": 1.0, "ALS": 1.0, "R21": 1.0}
        pool_r21 = set(weighted_rrf(src_r21, sw_r21, topk=300, k=20))
        if gt in pool_r21:
            pool_hit_r21 += 1

    # Catalog and retrieval list stats
    all_retrieved = set()
    for lst in r22b_lists:
        all_retrieved.update(lst)

    results = {
        "catalog": {
            "catalog_size": len(set(r22b_lists[0])) if r22b_lists else 0,
            "unique_ids_retrieved": len(all_retrieved),
            "duplicate_lists": n_dup_lists,
        },
        "r22b_standalone": {
            f"hit@{d}": r22b_hit[d] for d in [20, 50, 100, 200, 300]
        },
        "r21_baseline": {
            f"hit@{d}": r21_hit[d] for d in [20, 50, 100, 200, 300]
        },
        "comparison": {
            "unique_vs_r21": unique_vs_r21, "lost_vs_r21": lost_vs_r21,
            "net_vs_r21": unique_vs_r21 - lost_vs_r21,
            "unique_vs_v3": unique_vs_v3,
        },
        "unseen": {
            "unseen_total": unseen_total, "unseen_hit": unseen_hit,
            "unseen_hit_rate": unseen_hit / max(unseen_total, 1),
            "seen_total": seen_total, "seen_hit": seen_hit,
            "seen_hit_rate": seen_hit / max(seen_total, 1),
        },
        "pool_hit": {
            "r22b_pool_hit@300": pool_hit_r22b / n,
            "r21_pool_hit@300": pool_hit_r21 / n,
            "delta": (pool_hit_r22b - pool_hit_r21) / n,
        },
        "gates": {
            "pool_hit_primary": pool_hit_r22b / n >= 0.620,
            "pool_hit_strong": pool_hit_r22b / n >= 0.650,
            "pool_hit_value": pool_hit_r22b / n,
        },
    }

    # Add rates to standalone
    for d in [20, 50, 100, 200, 300]:
        results["r22b_standalone"][f"hit@{d}_rate"] = r22b_hit[d] / n
        results["r21_baseline"][f"hit@{d}_rate"] = r21_hit[d] / n

    return results


def phase_train():
    """Phase 1: Train BGE, encode catalog, retrieve dev lists. Torch only — no implicit."""
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{ts()} R22b Phase 1: Train + encode + retrieve")
    print(f"{'='*60}")

    # Load dev data
    print(f"\n{ts()} Loading dev data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    dev_cases = payload["cases"]
    dev_sessions = set(c["session_id"] for c in dev_cases)
    print(f"  Dev: {len(dev_cases)} cases from {len(dev_sessions)} sessions")

    # Load catalog
    print(f"\n{ts()} Loading catalog...", flush=True)
    meta, all_track_ids = load_catalog()
    print(f"  Catalog: {len(all_track_ids)} tracks")
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]

    # Build all-train supervised pairs
    print(f"\n{ts()} Building supervised pairs from all training sessions...", flush=True)
    pairs = build_all_train_pairs(meta, dev_sessions)
    print(f"  Total pairs: {len(pairs)}")

    # Train model
    model_dir = CACHE_DIR / "model"
    if model_dir.exists():
        print(f"\n{ts()} Model already exists at {model_dir}, loading...", flush=True)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(str(model_dir), device="cpu")
    else:
        print(f"\n{ts()} Training BGE on {len(pairs)} pairs ({EPOCHS} epochs)...", flush=True)
        model = train_model(pairs, model_dir,
                            probe_cases=dev_cases, probe_meta=meta,
                            probe_track_ids=all_track_ids)

    # Encode and retrieve
    print(f"\n{ts()} Encoding and retrieving...", flush=True)
    r22b_lists, track_embs, query_embs = encode_and_retrieve(
        model, track_texts, all_track_ids, dev_cases)

    # Save artifacts
    print(f"\n{ts()} Saving artifacts...", flush=True)
    np.save(CACHE_DIR / "track_embeddings.npy", track_embs)
    np.save(CACHE_DIR / "query_embeddings.npy", query_embs)
    with open(CACHE_DIR / "track_ids.json", "w") as f:
        json.dump(all_track_ids, f)
    with open(CACHE_DIR / "dev_r22b_lists.json", "w") as f:
        json.dump(r22b_lists, f)

    elapsed = time.time() - t0
    print(f"\n{ts()} Phase 1 complete. Elapsed: {elapsed:.1f}s ({elapsed/3600:.1f}h)")


def phase_eval():
    """Phase 2: Evaluate against gates. Loads ALS (implicit) — run in separate process."""
    t0 = time.time()

    print(f"{ts()} R22b Phase 2: Evaluate against gates")
    print(f"{'='*60}")

    # Load dev data
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    dev_cases = payload["cases"]

    # Load R22b lists
    with open(CACHE_DIR / "dev_r22b_lists.json") as f:
        r22b_lists = json.load(f)

    # Load baselines
    with open(R21_OOF) as f:
        r21_oof_lists = json.load(f)
    with open(V3_POOLS) as f:
        v3_pools_raw = json.load(f)
    v3_pools = [set(p) for p in v3_pools_raw]

    # Load train tracks for unseen classification
    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_tracks = set()
    for item in train_ds:
        for c in item["conversations"]:
            if c["role"] == "music":
                train_tracks.add(str(c["content"]).strip())

    # Evaluate
    print(f"\n{ts()} Evaluating against gates...", flush=True)
    results = evaluate(dev_cases, r22b_lists, r21_oof_lists, v3_pools, train_tracks)

    # Print report
    print(f"\n{'='*60}")
    print(f"R22b RESULTS")
    print(f"{'='*60}")

    cat = results["catalog"]
    print(f"\nCatalog / integrity:")
    print(f"  Unique IDs retrieved: {cat['unique_ids_retrieved']}")
    print(f"  Lists with duplicate IDs: {cat['duplicate_lists']}")

    s = results["r22b_standalone"]
    r = results["r21_baseline"]
    print(f"\nStandalone retrieval (R22b vs R21 OOF):")
    print(f"  {'Depth':<10} {'R22b':>10} {'R22b%':>8} {'R21':>10} {'R21%':>8} {'Δ':>8}")
    print(f"  {'-'*55}")
    for d in [20, 50, 100, 200, 300]:
        r22_v = s[f"hit@{d}"]
        r21_v = r[f"hit@{d}"]
        print(f"  hit@{d:<5} {r22_v:>10} {s[f'hit@{d}_rate']:>7.1%} "
              f"{r21_v:>10} {r[f'hit@{d}_rate']:>7.1%} {r22_v-r21_v:>+8d}")

    c = results["comparison"]
    print(f"\nComparison:")
    print(f"  Unique GTs vs R21: {c['unique_vs_r21']} (new hits R22b finds that R21 misses)")
    print(f"  Lost GTs vs R21:   {c['lost_vs_r21']} (R21 finds but R22b misses)")
    print(f"  Net:               {c['net_vs_r21']:+d}")
    print(f"  Unique vs V3 pool: {c['unique_vs_v3']}")

    u = results["unseen"]
    print(f"\nUnseen analysis:")
    print(f"  Unseen: {u['unseen_hit']}/{u['unseen_total']} ({u['unseen_hit_rate']:.1%})")
    print(f"  Seen:   {u['seen_hit']}/{u['seen_total']} ({u['seen_hit_rate']:.1%})")

    p = results["pool_hit"]
    print(f"\nPool hit@300 (PRIMARY GATE):")
    print(f"  R22b: {p['r22b_pool_hit@300']:.4f}")
    print(f"  R21:  {p['r21_pool_hit@300']:.4f}")
    print(f"  Δ:    {p['delta']:+.4f}")

    g = results["gates"]
    print(f"\nGates:")
    print(f"  Primary (pool_hit >= 0.620): {'PASS' if g['pool_hit_primary'] else 'FAIL'} ({g['pool_hit_value']:.4f})")
    print(f"  Strong  (pool_hit >= 0.650): {'PASS' if g['pool_hit_strong'] else 'FAIL'}")

    predicted_blind = 0.4628 * g["pool_hit_value"] + 0.1894
    print(f"\nProxy prediction: blind_ndcg ~ {predicted_blind:.4f}")
    print(f"  (R21 baseline: 0.4734)")

    if g["pool_hit_strong"]:
        print(f"\n>>> STRONG PASS — high-priority submission candidate <<<")
    elif g["pool_hit_primary"]:
        print(f"\n>>> PASS — proceed to LambdaRank / blind build <<<")
    else:
        print(f"\n>>> FAIL — pool_hit below 0.620, do not submit <<<")

    # Save results
    results["proxy_prediction"] = predicted_blind
    results["created_at"] = datetime.now().isoformat()

    with open(CACHE_DIR / "r22b_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(REPO_ROOT / "exp" / "eval" / "expR22b_all_train_retriever.json", "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n{ts()} Phase 2 complete. Elapsed: {elapsed:.1f}s ({elapsed/3600:.1f}h)")
    print(f"Saved: {CACHE_DIR / 'r22b_results.json'}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["train", "eval", "all"], default="all")
    args = parser.parse_args()

    if args.phase in ("train", "all"):
        phase_train()
    if args.phase in ("eval", "all"):
        if args.phase == "all":
            print(f"\n{ts()} WARNING: Running eval in same process as train.")
            print(f"  If this crashes (implicit/torch conflict), run with --phase eval separately.")
        phase_eval()


if __name__ == "__main__":
    main()
