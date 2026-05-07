#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R30: Deep-history R21 specialist — fold-0 diagnostic.

Train a BGE retriever only on late-session cases (hist>=5), initialize from
R21 production model, and evaluate as standalone + fused source on hist_7.

Different from failed experiments:
- R28 only changed LambdaRank weighting/features, not representation
- R26 intent helped hist_0 but hurt hist_7 — wrong objective
- R29 zero-shot CE was generic relevance, not music-session relevance
- R30 trains the retriever itself on the target slice

Primary gate: hist_7 >= R21 +0.005 as standalone retriever.
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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))




R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_MODEL = REPO / "cache" / "r21_production" / "model"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R30_DIR = REPO / "cache" / "r30"

HIST_MIN_TRAIN = 5
TOPK = 300


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


def build_query_text_deep(case):
    """Build query text emphasizing recent session context for deep-history cases."""
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
    return meta, track_ids


def train_r30_model(train_cases, meta, model_dir, init_from=None,
                    epochs=2, batch_size=32, lr=2e-5):
    """Train contrastive BGE retriever on late-turn cases."""
    import torch
    import torch.nn.functional as F_t
    from sentence_transformers import SentenceTransformer, InputExample

    examples = []
    for c in train_cases:
        gt = c["gt"]
        if gt not in meta:
            continue
        query_text = build_query_text_deep(c)
        track_text = build_track_text(gt, meta)
        examples.append(InputExample(texts=[query_text, track_text]))

    print(f"    {len(examples)} training pairs from {len(train_cases)} cases")

    init_path = str(init_from) if init_from and init_from.exists() else "BAAI/bge-base-en-v1.5"
    print(f"    Initializing from: {init_path}")
    model = SentenceTransformer(init_path, device="cpu")
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
    model.training = False  # noqa: S301 — PyTorch inference mode, not exec
    track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    val_queries = [build_query_text_deep(c) for c in val_cases]
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


def eval_retriever_hist7(cases, retrieval_lists, label=""):
    """Evaluate hit@k on hist_7 and hist_5..7 slices."""
    metrics = {}
    for depth in range(8):
        idx = [i for i in range(len(cases)) if cases[i]["n_prior_music"] == depth]
        if not idx:
            continue
        for k in [20, 50, 100, 200]:
            hit = sum(1 for i in idx if cases[i]["gt"] in set(retrieval_lists[i][:k]))
            metrics[f"hist_{depth}_hit@{k}"] = hit / len(idx)

    hist57 = [i for i in range(len(cases)) if cases[i]["n_prior_music"] >= 5]
    for k in [20, 50, 100, 200]:
        hit = sum(1 for i in hist57 if cases[i]["gt"] in set(retrieval_lists[i][:k]))
        metrics[f"hist57_hit@{k}"] = hit / len(hist57)

    all_hit200 = sum(1 for i in range(len(cases)) if cases[i]["gt"] in set(retrieval_lists[i][:200]))
    metrics["all_hit@200"] = all_hit200 / len(cases)

    if label:
        h7_20 = metrics.get("hist_7_hit@20", 0)
        h7_200 = metrics.get("hist_7_hit@200", 0)
        h57_200 = metrics.get("hist57_hit@200", 0)
        all_200 = metrics["all_hit@200"]
        print(f"  {label}: h7@20={h7_20:.4f}  h7@200={h7_200:.4f}  "
              f"h57@200={h57_200:.4f}  all@200={all_200:.4f}")
    return metrics


def save_retrieval_artifact(name, val_cases, retrieval_lists, metrics):
    """Persist fold-0 retrieval lists so fusion diagnostics can reuse them."""
    out_path = R30_DIR / f"fold0_{name}_lists.json"
    payload = {
        "config": name,
        "metrics": metrics,
        "case_keys": [
            {
                "session_id": c["session_id"],
                "turn_number": c["turn_number"],
                "n_prior_music": c["n_prior_music"],
                "gt": c["gt"],
            }
            for c in val_cases
        ],
        "retrieval_lists": retrieval_lists,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"  Saved {name} retrieval lists to {out_path}", flush=True)
    return out_path


def main():
    t0 = time.time()
    print(f"{ts()} R30: Deep-History R21 Specialist — Fold-0 Diagnostic")
    print("=" * 70)

    R30_DIR.mkdir(parents=True, exist_ok=True)

    import gc

    # Phase 1: Build training pairs and save to disk (no model loaded)
    pairs_path = R30_DIR / "fold0_pairs.json"
    val_path = R30_DIR / "fold0_val.json"

    if not pairs_path.exists() or not val_path.exists():
        print("  Phase 1: Building training pairs...")
        with open(R12_CACHE, "rb") as f:
            payload = pickle.load(f)
        raw_cases = payload["cases"]
        sessions = [c["session_id"] for c in raw_cases]

        cases = []
        for c in raw_cases:
            cases.append({
                "session_id": c["session_id"],
                "turn_number": c["turn_number"],
                "user_query": c["user_query"],
                "gt": c["gt"],
                "history": c["history"],
                "music_turns": c["music_turns"],
                "n_prior_music": c["n_prior_music"],
            })
        del payload, raw_cases
        gc.collect()

        with open(R21_OOF) as f:
            r21_oof_lists = json.load(f)

        from scripts.expS2_lambdarank_grouped import grouped_session_folds
        folds = grouped_session_folds(sessions, seed=0)
        fold0_val = set(folds[0].tolist())
        train_all = [j for j in range(len(cases)) if j not in fold0_val]
        val_indices = sorted(fold0_val)

        val_data = {
            "val_cases": [cases[j] for j in val_indices],
            "val_r21_lists": [r21_oof_lists[j] for j in val_indices],
        }
        with open(val_path, "w") as f:
            json.dump(val_data, f)

        train_deep_cases = [cases[j] for j in train_all
                           if cases[j]["n_prior_music"] >= HIST_MIN_TRAIN]
        train_all_cases = [cases[j] for j in train_all]

        with open(pairs_path, "w") as f:
            json.dump({
                "train_deep": train_deep_cases,
                "train_all": train_all_cases,
            }, f)

        del cases, r21_oof_lists, val_data
        gc.collect()
        print(f"  Saved {len(train_deep_cases)} deep + {len(train_all_cases)} all pairs")
        print("  Phase 1 complete. Restarting for Phase 2 (clean memory)...")
        os.execv(sys.executable, [sys.executable] + sys.argv)  # noqa: S606
    else:
        print("  Phase 1: Pairs cached, skipping to Phase 2...")

    with open(val_path) as f:
        val_data = json.load(f)
    val_cases = val_data["val_cases"]
    val_r21_lists = val_data["val_r21_lists"]
    del val_data
    gc.collect()

    # Baseline: R21 OOF on fold-0 val
    print(f"\n{ts()} Baseline: R21 OOF on fold-0 val ({len(val_cases)} cases)")
    r21_metrics = eval_retriever_hist7(val_cases, val_r21_lists, "R21_OOF")

    # Phase 2: Load training pairs and catalog for model training
    print(f"\n{ts()} Loading training pairs and catalog...")
    with open(pairs_path) as f:
        pairs_data = json.load(f)
    train_deep = pairs_data["train_deep"]
    train_all_cases = pairs_data["train_all"]
    del pairs_data
    gc.collect()

    meta, all_track_ids = load_catalog()
    print(f"  Catalog: {len(all_track_ids)} tracks")
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]

    # ---------------------------------------------------------------
    # Config A: R30 trained on hist>=5 only, init from R21 production
    # ---------------------------------------------------------------
    print(f"\n{ts()} Config A: R30 deep-history retriever")
    print(f"  Training: {len(train_deep)} cases (hist>={HIST_MIN_TRAIN} from fold-0 train)")
    print("  Init: R21 production model")

    model_dir_a = R30_DIR / "fold0_deep_from_r21"
    if model_dir_a.exists():
        print("  Model exists, loading...")
        from sentence_transformers import SentenceTransformer
        model_a = SentenceTransformer(str(model_dir_a), device="cpu")
    else:
        model_a = train_r30_model(train_deep, meta, model_dir_a, init_from=R21_MODEL)

    r30a_lists, _ = encode_and_retrieve(model_a, track_texts, all_track_ids, val_cases)
    r30a_metrics = eval_retriever_hist7(val_cases, r30a_lists, "R30_deep_r21init")
    save_retrieval_artifact("deep_from_r21", val_cases, r30a_lists, r30a_metrics)
    del model_a
    gc.collect()

    # ---------------------------------------------------------------
    # Config B: R30 trained on hist>=5, init from raw BGE (no R21)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Config B: R30 deep-history retriever (raw BGE init)")
    print(f"  Training: {len(train_deep)} cases (hist>={HIST_MIN_TRAIN})")
    print("  Init: raw BAAI/bge-base-en-v1.5")

    model_dir_b = R30_DIR / "fold0_deep_from_bge"
    if model_dir_b.exists():
        print("  Model exists, loading...")
        from sentence_transformers import SentenceTransformer
        model_b = SentenceTransformer(str(model_dir_b), device="cpu")
    else:
        model_b = train_r30_model(train_deep, meta, model_dir_b, init_from=None)

    r30b_lists, _ = encode_and_retrieve(model_b, track_texts, all_track_ids, val_cases)
    r30b_metrics = eval_retriever_hist7(val_cases, r30b_lists, "R30_deep_bge")
    save_retrieval_artifact("deep_from_bge", val_cases, r30b_lists, r30b_metrics)
    del model_b
    gc.collect()

    # ---------------------------------------------------------------
    # Config C: R21-style trained on ALL data, init from R21 (control)
    # ---------------------------------------------------------------
    print(f"\n{ts()} Config C: R21 retrain control (all {len(train_all_cases)} cases)")
    print("  Init: R21 production model")

    model_dir_c = R30_DIR / "fold0_all_from_r21"
    if model_dir_c.exists():
        print("  Model exists, loading...")
        from sentence_transformers import SentenceTransformer
        model_c = SentenceTransformer(str(model_dir_c), device="cpu")
    else:
        model_c = train_r30_model(train_all_cases, meta, model_dir_c, init_from=R21_MODEL)

    r30c_lists, _ = encode_and_retrieve(model_c, track_texts, all_track_ids, val_cases)
    r30c_metrics = eval_retriever_hist7(val_cases, r30c_lists, "R21_retrain_ctrl")
    save_retrieval_artifact("all_from_r21", val_cases, r30c_lists, r30c_metrics)
    del model_c
    gc.collect()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("R30 FOLD-0 DIAGNOSTIC — STANDALONE RETRIEVER")
    print(sep)
    configs = {
        "R21_OOF": r21_metrics,
        "R30_deep_r21": r30a_metrics,
        "R30_deep_bge": r30b_metrics,
        "R21_retrain": r30c_metrics,
    }
    print(f"  {'Config':<20} {'h7@20':>8} {'h7@200':>8} {'h57@200':>8} {'all@200':>8} "
          f"{'Dh7@200':>10}")
    print(f"  {'-'*64}")
    base_h7 = r21_metrics.get("hist_7_hit@200", 0)
    for name, m in configs.items():
        h7_20 = m.get("hist_7_hit@20", 0)
        h7_200 = m.get("hist_7_hit@200", 0)
        h57_200 = m.get("hist57_hit@200", 0)
        all_200 = m.get("all_hit@200", 0)
        dh7 = h7_200 - base_h7
        print(f"  {name:<20} {h7_20:>8.4f} {h7_200:>8.4f} {h57_200:>8.4f} {all_200:>8.4f} "
              f"{dh7:>+10.4f}")

    # Gate check
    print(f"\n{sep}")
    print("GATE CHECK (hist_7 hit@200 improvement)")
    for name, m in configs.items():
        if name == "R21_OOF":
            continue
        dh7 = m.get("hist_7_hit@200", 0) - base_h7
        status = "PASS" if dh7 >= 0.005 else "FAIL"
        print(f"  {name:<20} Δ hist_7@200={dh7:+.4f} → {status}")

    out_path = REPO / "exp" / "eval" / "expR30_fold0_diagnostic.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"configs": configs}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
