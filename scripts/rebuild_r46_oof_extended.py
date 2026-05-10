#!/usr/bin/env python3
# ruff: noqa: E402,T201,S301,S101
"""Generate OOF-clean R21 extended lists (top-2000) for R46 rebuild.

For each fold (0-4):
  - Load fold-specific model checkpoint (trained on 4 folds, held out 1)
  - Encode all 47071 catalog tracks
  - Encode held-out queries
  - Retrieve top-2000 per query (excluding played tracks)
  - Save per-fold and combined results

Fold models:
  - Folds 2,3,4: cache/r21_production/oof/model_fold_{2,3,4}/
  - Folds 0,1:   cache/r21_supervised/fold_{0,1}/ (same fold split)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import torch BEFORE datasets/lightgbm to avoid loky/torch segfault
import torch  # noqa: F401  # type: ignore[import-untyped]
import numpy as np  # type: ignore[import-untyped]

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
TRACK_IDS_PATH = REPO / "cache" / "r21_production" / "track_ids.json"
FOLD_INDICES_PATH = REPO / "cache" / "r21_production" / "fold_indices.json"
OUTPUT_PATH = REPO / "cache" / "r46_r21_oof_extended.json"

# Model paths per fold
MODEL_DIRS = {
    0: REPO / "cache" / "r21_supervised" / "fold_0",
    1: REPO / "cache" / "r21_supervised" / "fold_1",
    2: REPO / "cache" / "r21_production" / "oof" / "model_fold_2",
    3: REPO / "cache" / "r21_production" / "oof" / "model_fold_3",
    4: REPO / "cache" / "r21_production" / "oof" / "model_fold_4",
}

TOP_K = 2000


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
    from datasets import Dataset  # type: ignore[import-untyped]
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


def main():
    t0 = time.time()
    print(f"{ts()} OOF-clean R21 extended list generation (top-{TOP_K})")
    print("=" * 70)

    # Load dev payload
    print(f"{ts()} Loading dev payload...")
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    print(f"  {n} cases")

    # Load fold indices
    print(f"{ts()} Loading fold indices...")
    with open(FOLD_INDICES_PATH) as f:
        fold_indices_raw = json.load(f)
    fold_indices = {int(k): v for k, v in fold_indices_raw.items()}

    # Verify fold indices match grouped_session_folds
    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    computed_folds = grouped_session_folds(sessions, seed=0)
    for fi in range(5):
        assert list(computed_folds[fi]) == fold_indices[fi], \
            f"Fold {fi} mismatch between computed and cached indices!"
    print("  Fold indices verified: match grouped_session_folds(seed=0)")

    # Load catalog
    print(f"{ts()} Loading catalog metadata...")
    meta, all_track_ids = load_catalog()
    print(f"  {len(all_track_ids)} tracks")

    # Verify track IDs match cached
    with open(TRACK_IDS_PATH) as f:
        cached_track_ids = json.load(f)
    assert all_track_ids == cached_track_ids, "Track ID mismatch!"
    print("  Track IDs verified: match cached track_ids.json")

    # Pre-build track texts (shared across folds)
    print(f"{ts()} Building track texts...")
    track_texts = [build_track_text(tid, meta) for tid in all_track_ids]
    print(f"  {len(track_texts)} track texts built")

    # Process each fold
    all_oof_extended = [None] * n
    fold_stats = {}

    for fold_i in range(5):
        fold_t0 = time.time()
        model_dir = MODEL_DIRS[fold_i]
        held = fold_indices[fold_i]
        val_cases = [cases[j] for j in held]

        print(f"\n{ts()} === Fold {fold_i} === ({len(held)} val cases)")
        print(f"  Model: {model_dir}")

        # Load model
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        model = SentenceTransformer(str(model_dir), device="cpu")
        # Set model to inference mode
        set_eval = getattr(model, "eval")
        set_eval()

        # Encode all tracks
        print(f"  Encoding {len(all_track_ids)} tracks...")
        track_embs = model.encode(track_texts, batch_size=128, show_progress_bar=True,
                                   normalize_embeddings=True).astype(np.float32)

        # Encode queries
        val_queries = [build_query_text(c) for c in val_cases]
        print(f"  Encoding {len(val_queries)} queries...")
        query_embs = model.encode(val_queries, batch_size=64, show_progress_bar=True,
                                   normalize_embeddings=True).astype(np.float32)

        # Retrieve top-2000
        print(f"  Retrieving top-{TOP_K}...")
        fold_lists = []
        for j_local in range(len(val_cases)):
            scores = track_embs @ query_embs[j_local]
            played_set = set(val_cases[j_local]["music_turns"])
            for idx, tid in enumerate(all_track_ids):
                if tid in played_set:
                    scores[idx] = -np.inf
            top_idx = np.argpartition(-scores, TOP_K)[:TOP_K]
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            fold_lists.append([all_track_ids[k] for k in top_idx])

            if (j_local + 1) % 400 == 0:
                print(f"    {j_local + 1}/{len(val_cases)} done")

        # Place into global array
        for j_local, j_global in enumerate(held):
            all_oof_extended[j_global] = fold_lists[j_local]

        # Stats
        gt_found = 0
        gt_in_300 = 0
        for j_local, j_global in enumerate(held):
            gt = cases[j_global]["gt"]
            if gt in fold_lists[j_local]:
                gt_found += 1
                rank = fold_lists[j_local].index(gt) + 1
                if rank <= 300:
                    gt_in_300 += 1

        elapsed = time.time() - fold_t0
        fold_stats[fold_i] = {
            "gt_in_top2000": gt_found,
            "gt_in_top300": gt_in_300,
            "val_cases": len(held),
            "elapsed_s": elapsed,
        }
        print(f"  GT in top-{TOP_K}: {gt_found}/{len(held)} ({gt_found/len(held):.4f})")
        print(f"  GT in top-300: {gt_in_300}/{len(held)} ({gt_in_300/len(held):.4f})")
        print(f"  Elapsed: {elapsed:.1f}s")

        del model, track_embs, query_embs

    # Verify completeness
    assert all(x is not None for x in all_oof_extended), "Missing OOF lists!"
    assert len(all_oof_extended) == n

    # Cross-check: top-300 should match existing OOF lists
    print(f"\n{ts()} Cross-checking top-300 vs existing OOF lists...")
    with open(REPO / "cache" / "r21_production" / "dev_r21_oof_lists.json") as f:
        existing_oof = json.load(f)

    # The existing OOF was generated from fold models 0-4 in the production dir.
    # Folds 2,3,4 use the SAME model, so top-300 should match exactly.
    # Folds 0,1 use r21_supervised models, which may differ slightly in training.
    match_count = 0
    mismatch_folds = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for fi in range(5):
        for j_local, j_global in enumerate(fold_indices[fi]):
            new_top300 = all_oof_extended[j_global][:300]
            old_top300 = existing_oof[j_global]
            if new_top300 == old_top300:
                match_count += 1
            else:
                mismatch_folds[fi] += 1

    print(f"  Top-300 exact match: {match_count}/{n}")
    for fi in range(5):
        if mismatch_folds[fi] > 0:
            print(f"  Fold {fi}: {mismatch_folds[fi]} mismatches (expected for folds 0,1 with different model)")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_oof_extended, f)
    print(f"\n{ts()} Saved {n} extended OOF lists to {OUTPUT_PATH}")

    # Summary
    total_gt = sum(s["gt_in_top2000"] for s in fold_stats.values())
    total_gt300 = sum(s["gt_in_top300"] for s in fold_stats.values())
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"  Total GT in top-{TOP_K}: {total_gt}/{n} ({total_gt/n:.4f})")
    print(f"  Total GT in top-300: {total_gt300}/{n} ({total_gt300/n:.4f})")
    for fi in range(5):
        s = fold_stats[fi]
        print(f"  Fold {fi}: gt@{TOP_K}={s['gt_in_top2000']}/{s['val_cases']}  "
              f"gt@300={s['gt_in_top300']}/{s['val_cases']}  {s['elapsed_s']:.0f}s")

    elapsed_total = time.time() - t0
    print(f"\n{ts()} Total elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")


if __name__ == "__main__":
    main()
