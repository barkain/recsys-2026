#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R30 clean fold-0: retrain R21 on folds 1-4, then fine-tune deep-history.

Phase 1 (subprocess): build fold split + pairs, save to disk, exec restart
Phase 2: retrain fold-0 R21 model (clean, folds 1-4 only)
Phase 3: fine-tune R30 deep-history from clean R21 fold-0 checkpoint
Phase 4: evaluate + persist lists
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
R21_OOF = REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json"
R30_DIR = REPO / "cache" / "r30"
CLEAN_DIR = R30_DIR / "clean_fold0"

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
    return meta, track_ids


def train_model(train_cases, meta, model_dir, init_from,
                epochs=2, batch_size=32, lr=2e-5):
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

    print(f"    {len(examples)} training pairs")

    init_path = str(init_from) if init_from and Path(init_from).exists() else "BAAI/bge-base-en-v1.5"
    print(f"    Init: {init_path}")
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
    model.training = False
    track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                               normalize_embeddings=True).astype(np.float32)

    val_queries = [build_query_text(c) for c in val_cases]
    print(f"    Encoding {len(val_queries)} queries...", flush=True)
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
    return results


def eval_retriever(cases, lists, label=""):
    metrics = {}
    for depth in range(8):
        idx = [i for i in range(len(cases)) if cases[i]["n_prior_music"] == depth]
        if not idx:
            continue
        for k in [20, 50, 100, 200]:
            hit = sum(1 for i in idx if cases[i]["gt"] in set(lists[i][:k]))
            metrics[f"hist_{depth}_hit@{k}"] = hit / len(idx)

    hist57 = [i for i in range(len(cases)) if cases[i]["n_prior_music"] >= 5]
    for k in [20, 50, 100, 200]:
        hit = sum(1 for i in hist57 if cases[i]["gt"] in set(lists[i][:k]))
        metrics[f"hist57_hit@{k}"] = hit / len(hist57)

    all_hit200 = sum(1 for i in range(len(cases)) if cases[i]["gt"] in set(lists[i][:200]))
    metrics["all_hit@200"] = all_hit200 / len(cases)

    if label:
        h7_20 = metrics.get("hist_7_hit@20", 0)
        h7_200 = metrics.get("hist_7_hit@200", 0)
        h57_200 = metrics.get("hist57_hit@200", 0)
        all_200 = metrics["all_hit@200"]
        print(f"  {label}: h7@20={h7_20:.4f}  h7@200={h7_200:.4f}  "
              f"h57@200={h57_200:.4f}  all@200={all_200:.4f}")
    return metrics


def main():
    t0 = time.time()
    print(f"{ts()} R30 Clean Fold-0: Retrain R21 + Deep-History Specialist")
    print("=" * 70)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Build fold split (uses heavy imports, then exec restart)
    pairs_path = CLEAN_DIR / "pairs.json"
    val_path = CLEAN_DIR / "val.json"

    if not pairs_path.exists() or not val_path.exists():
        print(f"\n{ts()} Phase 1: Building fold-0 split...")
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
        train_idx = [j for j in range(len(cases)) if j not in fold0_val]
        val_indices = sorted(fold0_val)

        val_data = {
            "val_cases": [cases[j] for j in val_indices],
            "val_r21_lists": [r21_oof_lists[j] for j in val_indices],
        }
        with open(val_path, "w") as f:
            json.dump(val_data, f)

        train_all = [cases[j] for j in train_idx]
        train_deep = [c for c in train_all if c["n_prior_music"] >= HIST_MIN_TRAIN]

        with open(pairs_path, "w") as f:
            json.dump({"train_all": train_all, "train_deep": train_deep}, f)

        print(f"  train_all={len(train_all)}, train_deep={len(train_deep)}, val={len(val_indices)}")
        print("  Phase 1 done. Restarting for Phase 2 (clean memory)...")
        os.execv(sys.executable, [sys.executable] + sys.argv)  # noqa: S606

    # Phase 2+: Clean memory, no heavy imports from Phase 1
    print(f"\n{ts()} Phase 2: Loading cached data...")
    with open(val_path) as f:
        val_data = json.load(f)
    val_cases = val_data["val_cases"]
    val_r21_lists = val_data["val_r21_lists"]
    del val_data
    gc.collect()

    with open(pairs_path) as f:
        pairs_data = json.load(f)
    train_all = pairs_data["train_all"]
    train_deep = pairs_data["train_deep"]
    del pairs_data
    gc.collect()

    meta, all_track_ids = load_catalog()
    print(f"  Catalog: {len(all_track_ids)} tracks")
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]

    # Baseline
    print(f"\n{ts()} Baseline: R21 OOF (honest)")
    r21_m = eval_retriever(val_cases, val_r21_lists, "R21_OOF")

    # Step 1: Retrain clean fold-0 R21 (folds 1-4 train)
    r21_fold0_dir = CLEAN_DIR / "r21_fold0_model"
    r21_lists_path = CLEAN_DIR / "r21_fold0_lists.json"

    if r21_lists_path.exists():
        print(f"\n{ts()} Step 1: Clean R21 fold-0 lists cached")
        with open(r21_lists_path) as f:
            r21_clean_lists = json.load(f)
    else:
        print(f"\n{ts()} Step 1: Training clean R21 fold-0 ({len(train_all)} cases)")
        if r21_fold0_dir.exists():
            print("  Model exists, loading...")
            from sentence_transformers import SentenceTransformer
            model_r21 = SentenceTransformer(str(r21_fold0_dir), device="cpu")
        else:
            model_r21 = train_model(train_all, meta, r21_fold0_dir,
                                     init_from="BAAI/bge-base-en-v1.5")
        r21_clean_lists = encode_and_retrieve(model_r21, track_texts, all_track_ids, val_cases)
        with open(r21_lists_path, "w") as f:
            json.dump(r21_clean_lists, f)
        del model_r21
        gc.collect()

    r21_clean_m = eval_retriever(val_cases, r21_clean_lists, "R21_clean_fold0")

    # Verify clean R21 matches original OOF
    orig_h200 = r21_m.get("all_hit@200", 0)
    clean_h200 = r21_clean_m.get("all_hit@200", 0)
    print(f"  Sanity: orig OOF all@200={orig_h200:.4f}, clean retrain={clean_h200:.4f}, "
          f"delta={clean_h200-orig_h200:+.4f}")

    # Step 2: Fine-tune R30 deep-history from clean R21 fold-0
    r30_dir = CLEAN_DIR / "r30_deep_model"
    r30_lists_path = CLEAN_DIR / "r30_deep_lists.json"

    if r30_lists_path.exists():
        print(f"\n{ts()} Step 2: R30 deep-history lists cached")
        with open(r30_lists_path) as f:
            r30_lists = json.load(f)
    else:
        print(f"\n{ts()} Step 2: Fine-tuning R30 deep-history ({len(train_deep)} cases)")
        print("  Init: clean R21 fold-0 checkpoint")
        if r30_dir.exists():
            print("  Model exists, loading...")
            from sentence_transformers import SentenceTransformer
            model_r30 = SentenceTransformer(str(r30_dir), device="cpu")
        else:
            model_r30 = train_model(train_deep, meta, r30_dir,
                                     init_from=str(r21_fold0_dir))
        r30_lists = encode_and_retrieve(model_r30, track_texts, all_track_ids, val_cases)
        with open(r30_lists_path, "w") as f:
            json.dump(r30_lists, f)
        del model_r30
        gc.collect()

    r30_m = eval_retriever(val_cases, r30_lists, "R30_clean")

    # Summary
    sep = "=" * 70
    print(f"\n{sep}")
    print("R30 CLEAN FOLD-0 DIAGNOSTIC")
    print(sep)
    configs = {"R21_OOF": r21_m, "R21_clean": r21_clean_m, "R30_clean": r30_m}
    print(f"  {'Config':<20} {'h7@20':>8} {'h7@200':>8} {'h57@200':>8} {'all@200':>8} {'Dh7@200':>10}")
    print(f"  {'-'*64}")
    base_h7 = r21_m.get("hist_7_hit@200", 0)
    for name, m in configs.items():
        h7_20 = m.get("hist_7_hit@20", 0)
        h7_200 = m.get("hist_7_hit@200", 0)
        h57 = m.get("hist57_hit@200", 0)
        a200 = m.get("all_hit@200", 0)
        dh7 = h7_200 - base_h7
        print(f"  {name:<20} {h7_20:>8.4f} {h7_200:>8.4f} {h57:>8.4f} {a200:>8.4f} {dh7:>+10.4f}")

    # Gate
    print(f"\n{sep}")
    print("GATE CHECK")
    r30_dh7 = r30_m.get("hist_7_hit@200", 0) - base_h7
    r30_dh7_20 = r30_m.get("hist_7_hit@20", 0) - r21_m.get("hist_7_hit@20", 0)
    g_h200 = r30_dh7 >= 0.030
    g_h20 = r30_dh7_20 >= 0.0
    status = "PASS" if (g_h200 and g_h20) else "FAIL"
    print(f"  R30 clean Δh7@200={r30_dh7:+.4f} (need +0.030): {'PASS' if g_h200 else 'FAIL'}")
    print(f"  R30 clean Δh7@20={r30_dh7_20:+.4f} (need >=0):   {'PASS' if g_h20 else 'FAIL'}")
    print(f"  Overall: {status}")

    if status == "PASS":
        print("\n  R30 deep-history specialist CONFIRMED with clean fold-0.")
        print("  → Next: fold-0 fusion/LambdaRank test")
    else:
        print("\n  R30 does not pass clean fold-0 gate.")
        print("  Contaminated result was misleading.")

    out_path = REPO / "exp" / "eval" / "expR30_clean_fold0.json"
    with open(out_path, "w") as f:
        json.dump({"configs": {n: m for n, m in configs.items()},
                    "gate_h200": r30_dh7, "gate_h20": r30_dh7_20,
                    "status": status}, f, indent=2)
    print(f"\n{ts()} Saved to {out_path}")
    print(f"Elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
