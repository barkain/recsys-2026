#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R54 Phase 3 fold-0: structured query + train-split data — RESUMABLE.

Phase 3 full run intentionally accepted despite ~20h ETA because it is the first
clean test of structured query + all-train data. Defensible only because it is
checkpointed:

- Checkpoint every 500 batches (model + optimizer + RNG + batch_idx + sample_perm)
- Keep only latest 2 checkpoints
- Heartbeat file every 100 batches
- Log batch/loss/elapsed/ETA every 100 batches
- Resume from latest checkpoint on restart (exact sample order preserved)
- No mid-run hyperparam changes, no MPS

Same hypothesis as before: does train-split data + structured query improve h7
admissions/conversion vs Phase 2?
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
PHASE2_FOLD0_LISTS = REPO / "cache" / "r54" / "phase2_full" / "fold_0" / "oof_lists.json"
CACHE_DIR = REPO / "cache" / "r54" / "phase3" / "fold_0"
CKPT_DIR = CACHE_DIR / "checkpoints"
HEARTBEAT = CACHE_DIR / "heartbeat.json"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

EPOCHS = 1
BATCH_SIZE = 32
LR = 2e-5
TAU = 0.05
MAX_SEQ_LEN = 256
TOPK = 300

CKPT_EVERY = 500
LOG_EVERY = 100
KEEP_CKPTS = 2
RNG_SEED = 0


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


def build_query_structured_from_dev(case, meta):
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


def build_query_structured_from_session(user_msgs_so_far, played_so_far, current_user_msg, meta):
    history = user_msgs_so_far[-3:] if len(user_msgs_so_far) > 3 else user_msgs_so_far
    context_tracks = []
    for tid in played_so_far[-5:]:
        if tid in meta:
            context_tracks.append(build_short_track_ref(tid, meta))
    parts = [f"[QUERY] {current_user_msg}"]
    older_history = history[:-1] if history else []
    if older_history:
        parts.append(f"[HISTORY] {' '.join(older_history)}")
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


def build_train_split_pairs(meta, dev_session_ids):
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    pairs = []
    n_sessions = 0
    n_dev_overlap = 0
    for item in train_ds:
        sid = item["session_id"]
        if sid in dev_session_ids:
            n_dev_overlap += 1
            continue
        n_sessions += 1
        convs = item["conversations"]
        user_msgs_so_far = []
        played_so_far = []
        most_recent_user_msg = ""
        for conv in convs:
            role = conv["role"]
            content = str(conv["content"])
            if role == "user":
                user_msgs_so_far.append(content)
                most_recent_user_msg = content
            elif role == "music":
                tid = content.strip()
                if tid not in meta:
                    played_so_far.append(tid)
                    continue
                if most_recent_user_msg:
                    query = build_query_structured_from_session(
                        user_msgs_so_far, played_so_far, most_recent_user_msg, meta)
                    track_text = build_track_text(tid, meta)
                    pairs.append((query, track_text))
                played_so_far.append(tid)
    print(f"  train_split: {n_sessions} sessions, {n_dev_overlap} dev overlap")
    print(f"  train_split: {len(pairs)} pairs")
    return pairs


def list_ckpts():
    if not CKPT_DIR.exists():
        return []
    return sorted(CKPT_DIR.glob("ckpt_batch_*.pkl"),
                  key=lambda p: int(p.stem.split("_")[-1]))


def save_checkpoint(model, optimizer, batch_idx, perm, total_batches, epoch_loss, n_batches_seen):
    import torch  # type: ignore[reportMissingImports]
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"ckpt_batch_{batch_idx:06d}.pkl"
    ckpt = {
        "batch_idx": batch_idx,
        "total_batches": total_batches,
        "perm": perm.tolist(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "epoch_loss": epoch_loss,
        "n_batches_seen": n_batches_seen,
        "created_at": datetime.now().isoformat(),
    }
    tmp_path = ckpt_path.with_suffix(".pkl.tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(ckpt, f)  # noqa: S301
    tmp_path.rename(ckpt_path)
    # Cleanup old
    all_ckpts = list_ckpts()
    for old in all_ckpts[:-KEEP_CKPTS]:
        old.unlink()


def write_heartbeat(batch_idx, total, loss, elapsed, eta):
    HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
    hb = {
        "batch_idx": batch_idx,
        "total_batches": total,
        "loss": loss,
        "elapsed_s": elapsed,
        "eta_s": eta,
        "pct": batch_idx / max(total, 1) * 100,
        "updated_at": datetime.now().isoformat(),
    }
    tmp = HEARTBEAT.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(hb, f, indent=2)
    tmp.rename(HEARTBEAT)


def train_with_checkpoints(all_pairs, final_model_dir):
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer  # type: ignore[reportMissingImports]

    total_batches = (len(all_pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  {len(all_pairs)} total pairs  total_batches={total_batches}", flush=True)

    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    # Resume detection
    ckpts = list_ckpts()
    start_batch = 0
    perm = None
    epoch_loss = 0.0
    n_batches_seen = 0
    optimizer_state = None
    model_state = None

    if ckpts:
        latest = ckpts[-1]
        print(f"{ts()} Resuming from {latest}", flush=True)
        with open(latest, "rb") as f:
            ckpt = pickle.load(f)  # noqa: S301
        if ckpt["total_batches"] != total_batches:
            print(f"  WARNING: total_batches mismatch (ckpt={ckpt['total_batches']} now={total_batches}). "
                  f"Discarding checkpoint.", flush=True)
        else:
            start_batch = ckpt["batch_idx"]
            perm = np.array(ckpt["perm"], dtype=np.int64)
            epoch_loss = ckpt["epoch_loss"]
            n_batches_seen = ckpt["n_batches_seen"]
            model_state = ckpt["model_state_dict"]
            optimizer_state = ckpt["optimizer_state_dict"]
            np.random.set_state(ckpt["numpy_rng_state"])
            torch.set_rng_state(ckpt["torch_rng_state"])
            print(f"  resumed at batch {start_batch}/{total_batches}", flush=True)

    if perm is None:
        perm = np.random.permutation(len(all_pairs)).astype(np.int64)

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    if model_state is not None:
        model.load_state_dict(model_state)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    model.train()
    tokenizer = model.tokenizer

    def encode_with_grad(texts):
        encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_SEQ_LEN,
                            return_tensors="pt")
        out = model.forward(encoded)
        emb = out["sentence_embedding"]
        return F_t.normalize(emb, dim=-1)

    t_epoch = time.time()
    for batch_idx in range(start_batch, total_batches):
        start = batch_idx * BATCH_SIZE
        batch_indices = perm[start:start + BATCH_SIZE]
        queries = [all_pairs[int(i)][0] for i in batch_indices]
        positives = [all_pairs[int(i)][1] for i in batch_indices]
        if len(queries) == 0:
            continue

        q_emb = encode_with_grad(queries)
        p_emb = encode_with_grad(positives)
        sim = q_emb @ p_emb.T / TAU
        labels = torch.arange(len(queries), device=sim.device)
        loss = F_t.cross_entropy(sim, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = float(loss.item())
        epoch_loss += loss_val
        n_batches_seen += 1

        completed = batch_idx + 1
        if completed % LOG_EVERY == 0:
            elapsed = time.time() - t_epoch
            seen_this_session = completed - start_batch
            sec_per_batch = elapsed / max(seen_this_session, 1)
            remaining = total_batches - completed
            eta = remaining * sec_per_batch
            avg_loss = epoch_loss / max(n_batches_seen, 1)
            print(f"      batch {completed}/{total_batches}  loss={loss_val:.4f}  "
                  f"avg_loss={avg_loss:.4f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s "
                  f"({eta/3600:.1f}h)", flush=True)
            write_heartbeat(completed, total_batches, loss_val, elapsed, eta)

        if completed % CKPT_EVERY == 0:
            save_checkpoint(model, optimizer, completed, perm, total_batches,
                            epoch_loss, n_batches_seen)
            print(f"      checkpoint saved at batch {completed}", flush=True)

    # Final save
    save_checkpoint(model, optimizer, total_batches, perm, total_batches,
                    epoch_loss, n_batches_seen)
    final_model_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(final_model_dir))
    print(f"  Final model saved to {final_model_dir}", flush=True)
    return model


def encode_catalog(model, meta, all_track_ids):
    print(f"{ts()} Encoding {len(all_track_ids)} tracks...", flush=True)
    model.eval()
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                        normalize_embeddings=True).astype(np.float32)
    return embs


def retrieve(model, queries, track_embs, track_ids, played_lists, topk=TOPK):
    print(f"{ts()} Encoding {len(queries)} queries...", flush=True)
    q_embs = model.encode(queries, batch_size=64, show_progress_bar=True,
                          normalize_embeddings=True).astype(np.float32)
    print(f"{ts()} Retrieving top-{topk}...", flush=True)
    results = []
    for i in range(len(queries)):
        played_set = set(str(t) for t in played_lists[i]) if played_lists[i] else set()
        sims = q_embs[i] @ track_embs.T
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


def evaluate(name, retrieval_lists, val_cases):
    n = len(val_cases)
    hit20 = hit50 = hit100 = hit200 = hit300 = 0
    h7_total = h7_hit200 = h7_hit300 = 0
    gt_ranks = []
    for i, c in enumerate(val_cases):
        gt = c["gt"]
        r = [t for t, _ in retrieval_lists[i]]
        if gt in set(r[:20]):
            hit20 += 1
        if gt in set(r[:50]):
            hit50 += 1
        if gt in set(r[:100]):
            hit100 += 1
        if gt in set(r[:200]):
            hit200 += 1
        if gt in set(r[:300]):
            hit300 += 1
            gt_ranks.append(r.index(gt))
        if c.get("n_prior_music") == 7:
            h7_total += 1
            if gt in set(r[:200]):
                h7_hit200 += 1
            if gt in set(r[:300]):
                h7_hit300 += 1
    print(f"  {name}: hit@20={hit20}/{n} ({hit20/n:.3f})  hit@200={hit200}/{n} ({hit200/n:.3f})  "
          f"hit@300={hit300}/{n} ({hit300/n:.3f})")
    if h7_total:
        print(f"    h7: hit@200={h7_hit200}/{h7_total} ({h7_hit200/h7_total:.3f})  "
              f"hit@300={h7_hit300}/{h7_total} ({h7_hit300/h7_total:.3f})")
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


def main():
    t0 = time.time()
    print(f"{ts()} R54 Phase 3 fold-0 (checkpointed)")
    print(f"  Intentionally accepted ~20h ETA. Resumable from disk checkpoints.")

    print(f"{ts()} Loading dev payload...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    dev_session_ids = set(sessions)

    print(f"{ts()} Loading catalog...")
    meta, all_track_ids = load_catalog()

    print(f"{ts()} Building fold-0 split...")
    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)
    val_idx = folds[0].tolist()
    train_idx = []
    for fi in range(1, 5):
        train_idx.extend(folds[fi].tolist())
    val_cases = [cases[j] for j in val_idx]
    train_dev_cases = [cases[j] for j in train_idx]
    print(f"  Fold 0: train_dev={len(train_dev_cases)} val={len(val_cases)}")

    # Dev pairs
    print(f"\n{ts()} Building dev training pairs (structured query)...")
    dev_pairs = []
    for c in train_dev_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        dev_pairs.append((build_query_structured_from_dev(c, meta), build_track_text(gt, meta)))
    print(f"  Dev pairs: {len(dev_pairs)}")

    # Train-split pairs
    print(f"\n{ts()} Building train-split pairs (structured query)...")
    train_split_pairs = build_train_split_pairs(meta, dev_session_ids)

    all_pairs = dev_pairs + train_split_pairs
    print(f"  TOTAL: {len(all_pairs)} pairs ({len(dev_pairs)} dev + {len(train_split_pairs)} train-split)")

    # Train with checkpoints
    print(f"\n{ts()} Training Phase 3 retriever (checkpointed)...")
    model_dir = CACHE_DIR / "model"
    model = train_with_checkpoints(all_pairs, model_dir)
    print(f"  Trained. Total elapsed: {time.time() - t0:.0f}s")

    # Encode + retrieve
    embs_path = CACHE_DIR / "track_embs.npy"
    track_embs = encode_catalog(model, meta, all_track_ids)
    np.save(embs_path, track_embs)

    val_queries = [build_query_structured_from_dev(c, meta) for c in val_cases]
    val_played = [c["music_turns"] for c in val_cases]
    p3_lists = retrieve(model, val_queries, track_embs, all_track_ids, val_played)

    lists_path = CACHE_DIR / "oof_lists.json"
    with open(lists_path, "w") as f:
        json.dump({"lists": p3_lists, "manifest": {
            "fold": 0,
            "train_dev_pairs": len(dev_pairs),
            "train_split_pairs": len(train_split_pairs),
            "total_pairs": len(all_pairs),
            "val_cases": len(val_cases),
            "model_dir": str(model_dir),
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "tau": TAU,
            "max_seq_len": MAX_SEQ_LEN, "rng_seed": RNG_SEED,
            "query_format": "structured",
            "positive_format": "r21_exact",
            "data_source": "dev_folds_1-4 + train-split",
            "elapsed_s": time.time() - t0,
            "created_at": datetime.now().isoformat(),
        }}, f)

    print(f"\n{ts()} === EVALUATION ===")
    p3_eval = evaluate("R54-Phase3", p3_lists, val_cases)

    p2_data = json.load(open(PHASE2_FOLD0_LISTS))
    p2_lists = p2_data["lists"]
    p2_eval = evaluate("R54-Phase2", p2_lists, val_cases)

    out = {
        "p3": p3_eval, "p2": p2_eval,
        "pair_exposures": {
            "phase2": "6,400 pairs × 2 epochs = 12,800",
            "phase3": f"{len(all_pairs)} pairs × {EPOCHS} epoch = {len(all_pairs) * EPOCHS}",
        },
        "config": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE, "tau": TAU,
            "dev_pairs": len(dev_pairs), "train_split_pairs": len(train_split_pairs),
            "total_pairs": len(all_pairs),
            "checkpointed": True, "ckpt_every": CKPT_EVERY,
        },
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }
    out_path = REPO / "exp" / "eval" / "expR54_phase3_fold0_train_split.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n{ts()} Phase 3 fold-0 complete. Elapsed: {time.time() - t0:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
