#!/usr/bin/env python3
# ruff: noqa: E402,T201
"""R54 Phase 2: Fold-0 structured query retriever.

Single-variable change from Phase 1: structured query text.
Everything else identical: same fold split, same train data (dev folds 1-4),
same model/hyperparams, same R21 track text, no train split, no enriched metadata,
no hard negatives, no same-session positives.

Query format:
[QUERY] current user utterance
[HISTORY] last 2-3 user utterances
[CONTEXT] last 3-5 played tracks as "track by artist"
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
PHASE1_LISTS = REPO / "cache" / "r54" / "phase1" / "fold0_r54_lists.json"
CACHE_DIR = REPO / "cache" / "r54" / "phase2"
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
    """R21-exact track text — unchanged from Phase 1."""
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
    """Short 'track by artist' for [CONTEXT] section."""
    m = meta.get(tid, {})
    names = m.get("track_name", [])
    artists = m.get("artist_name", [])
    name = names[0] if isinstance(names, list) and names else str(names)
    artist = artists[0] if isinstance(artists, list) and artists else str(artists)
    return f"{name} by {artist}"


def build_query_r21(case):
    """R21-exact query: last 3 user utterances concatenated."""
    parts = []
    for h in case["history"]:
        if h["role"] == "user":
            parts.append(str(h["content"]))
    parts.append(case["user_query"])
    return " ".join(parts[-3:])


def build_query_structured(case, meta):
    """Structured query with [QUERY], [HISTORY], [CONTEXT]."""
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
        query_text = build_query_structured(c, meta)
        track_text = build_track_text(gt, meta)
        examples.append((query_text, track_text))

    print(f"  {len(examples)} positive pairs", flush=True)

    # Print a few example queries for verification
    print(f"\n  Sample structured queries:", flush=True)
    for i in range(min(3, len(examples))):
        q = examples[i][0]
        print(f"    [{i}] {q[:200]}{'...' if len(q) > 200 else ''}", flush=True)
    print(flush=True)

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


def evaluate(name, retrieval_lists, val_cases):
    n = len(val_cases)
    hit20 = hit50 = hit100 = hit200 = hit300 = 0
    h7_total = h7_hit200 = h7_hit300 = 0
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


def compare_lists(name_a, lists_a, name_b, lists_b, val_cases):
    """Compare two retrieval list sets, report unique/lost/both/neither."""
    n = len(val_cases)
    both = a_only = b_only = neither = 0

    for i, c in enumerate(val_cases):
        gt = c["gt"]
        in_a = gt in set(lists_a[i][:300])
        in_b = gt in set(lists_b[i][:300])
        if in_a and in_b:
            both += 1
        elif in_a:
            a_only += 1
        elif in_b:
            b_only += 1
        else:
            neither += 1

    print(f"  {name_a} vs {name_b}: both={both}  {name_a}-only={a_only}  "
          f"{name_b}-only={b_only}  neither={neither}")
    return {"both": both, f"{name_a}_only": a_only, f"{name_b}_only": b_only, "neither": neither}


def main():
    t0 = time.time()

    print(f"{ts()} Loading data...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    print(f"  {len(cases)} cases", flush=True)

    print(f"{ts()} Loading catalog metadata...", flush=True)
    meta, all_track_ids = load_catalog()
    print(f"  {len(all_track_ids)} tracks", flush=True)

    print(f"{ts()} Building fold-0 split...", flush=True)
    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)

    val_idx = folds[0].tolist()
    train_idx = []
    for fi in range(1, 5):
        train_idx.extend(folds[fi].tolist())

    val_cases = [cases[j] for j in val_idx]
    train_cases = [cases[j] for j in train_idx]
    print(f"  Fold 0: train={len(train_cases)} val={len(val_cases)}", flush=True)

    # Train
    print(f"\n{ts()} Training structured-query retriever (fold-0 held out)...", flush=True)
    model_dir = CACHE_DIR / "model_fold0"
    model, track_embs = train_and_encode(train_cases, meta, all_track_ids, model_dir)

    # Save embeddings
    emb_path = CACHE_DIR / "track_embs_fold0.npy"
    np.save(emb_path, track_embs)
    print(f"  Embeddings saved: {emb_path}", flush=True)

    # Retrieve — use structured query for val cases too
    val_queries = [build_query_structured(c, meta) for c in val_cases]
    val_played = [c["music_turns"] for c in val_cases]
    p2_lists = retrieve_top_k(model, val_queries, track_embs, all_track_ids, val_played)

    # Save retrieval lists
    lists_path = CACHE_DIR / "fold0_r54p2_lists.json"
    with open(lists_path, "w") as f:
        json.dump({"lists": p2_lists, "manifest": {
            "fold": 0,
            "train_cases": len(train_cases),
            "val_cases": len(val_cases),
            "model_dir": str(model_dir),
            "query_format": "structured [QUERY]+[HISTORY]+[CONTEXT]",
            "positive_format": "r21_exact",
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "tau": TAU, "created_at": datetime.now().isoformat(),
        }}, f)
    print(f"  Lists saved: {lists_path}", flush=True)

    # Evaluate
    print(f"\n{ts()} === EVALUATION ===", flush=True)
    p2_eval = evaluate("R54-Phase2-Structured", p2_lists, val_cases)

    # Load R21 OOF fold-0
    r21_f0_path = R21_OOF_DIR / "fold_0_r21_lists.json"
    r21_lists = json.load(open(r21_f0_path))["lists"]
    r21_eval = evaluate("R21-OOF-f0", r21_lists, val_cases)

    # Load Phase 1 lists
    p1_lists = json.load(open(PHASE1_LISTS))["lists"]
    p1_eval = evaluate("R54-Phase1-R21exact", p1_lists, val_cases)

    # Comparisons
    print(f"\n{ts()} === COMPARISONS ===", flush=True)
    cmp_vs_r21 = compare_lists("P2", p2_lists, "R21", r21_lists, val_cases)
    cmp_vs_p1 = compare_lists("P2", p2_lists, "P1", p1_lists, val_cases)

    # Delta summary
    print(f"\n{ts()} === DELTA SUMMARY ===", flush=True)
    print(f"  {'Metric':<20} {'R21-OOF':>10} {'Phase1':>10} {'Phase2':>10} {'Δ vs R21':>10} {'Δ vs P1':>10}")
    for metric in ["hit20", "hit200", "hit300", "h7_hit200", "h7_hit300", "median_gt_rank"]:
        r21_v = r21_eval[metric]
        p1_v = p1_eval[metric]
        p2_v = p2_eval[metric]
        d_r21 = p2_v - r21_v
        d_p1 = p2_v - p1_v
        print(f"  {metric:<20} {r21_v:>10} {p1_v:>10} {p2_v:>10} {d_r21:>+10} {d_p1:>+10}")

    # Save results
    results = {
        "r54_phase2": p2_eval,
        "r54_phase1": p1_eval,
        "r21_oof_fold0": r21_eval,
        "comparison_vs_r21": cmp_vs_r21,
        "comparison_vs_p1": cmp_vs_p1,
        "config": {
            "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
            "tau": TAU, "max_seq_len": MAX_SEQ_LEN,
            "model": MODEL_NAME, "topk": TOPK,
            "query_format": "structured [QUERY]+[HISTORY]+[CONTEXT]",
            "positive_format": "r21_exact (name by artist. Album. Tags[:10])",
            "change_vs_phase1": "query text only",
        },
        "elapsed_s": time.time() - t0,
        "created_at": datetime.now().isoformat(),
    }

    out_path = REPO / "exp" / "eval" / "expR54_phase2_structured_query.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{ts()} Phase 2 complete. Elapsed: {time.time() - t0:.0f}s")
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    main()
