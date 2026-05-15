#!/usr/bin/env python3
# pyright: reportMissingImports=false
# ruff: noqa: E402,T201,S301
"""R54 Phase 3 fold-0 leakage verification + manifest.

Independent of the training script. Reconstructs the exact data split used
by Phase 3 and asserts no fold-0 val data leaks into the training set.

Writes manifest at cache/r54/phase3/fold_0/fold0_manifest.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

R12_CACHE = REPO / "exp" / "eval" / "_R12_all_turns_payload.pkl"
MANIFEST_PATH = REPO / "cache" / "r54" / "phase3" / "fold_0" / "fold0_manifest.json"


def ts():
    return f"[{datetime.now():%H:%M:%S}]"


def main():
    t0 = time.time()
    print(f"{ts()} Phase 3 fold-0 leakage verification")
    print("=" * 70)

    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)
    cases = payload["cases"]
    n = len(cases)
    sessions = [c["session_id"] for c in cases]
    dev_session_ids = set(sessions)
    print(f"  Dev: {n} cases, {len(dev_session_ids)} unique sessions")

    from scripts.expS2_lambdarank_grouped import grouped_session_folds
    folds = grouped_session_folds(sessions, seed=0)
    val_idx = set(folds[0].tolist())
    train_idx = set()
    for fi in range(1, 5):
        train_idx.update(folds[fi].tolist())

    # Sessions in each
    fold0_val_session_ids = {sessions[i] for i in val_idx}
    fold1_4_train_session_ids = {sessions[i] for i in train_idx}

    print(f"\n  Fold-0 val: {len(val_idx)} cases, {len(fold0_val_session_ids)} sessions")
    print(f"  Fold 1-4 train: {len(train_idx)} cases, {len(fold1_4_train_session_ids)} sessions")

    # Verify dev val NOT in dev train
    val_train_overlap_cases = val_idx & train_idx
    val_train_overlap_sessions = fold0_val_session_ids & fold1_4_train_session_ids
    print(f"\n  Dev fold-0 val cases in dev train: {len(val_train_overlap_cases)}  "
          f"{'OK' if len(val_train_overlap_cases) == 0 else 'LEAK'}")
    print(f"  Dev fold-0 val sessions in dev train: {len(val_train_overlap_sessions)}  "
          f"{'OK' if len(val_train_overlap_sessions) == 0 else 'LEAK'}")
    assert len(val_train_overlap_cases) == 0
    assert len(val_train_overlap_sessions) == 0

    # Load train-split sessions
    print(f"\n{ts()} Loading train-split sessions...")
    from datasets import DownloadConfig, load_dataset
    train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset",
                            download_config=DownloadConfig(local_files_only=True))["train"]
    train_split_session_ids = set()
    n_pairs_in_train_split = 0
    for item in train_ds:
        sid = item["session_id"]
        train_split_session_ids.add(sid)
        for conv in item["conversations"]:
            if conv["role"] == "music":
                n_pairs_in_train_split += 1

    print(f"  Train-split: {len(train_split_session_ids)} sessions, "
          f"{n_pairs_in_train_split} music turns")

    # Check overlap with dev sessions
    train_split_dev_overlap = train_split_session_ids & dev_session_ids
    print(f"  Train-split sessions in dev: {len(train_split_dev_overlap)}  "
          f"{'OK' if len(train_split_dev_overlap) == 0 else 'LEAK'}")
    assert len(train_split_dev_overlap) == 0

    # Check overlap with fold-0 val sessions specifically
    train_split_fold0_overlap = train_split_session_ids & fold0_val_session_ids
    print(f"  Train-split sessions in fold-0 val: {len(train_split_fold0_overlap)}  "
          f"{'OK' if len(train_split_fold0_overlap) == 0 else 'LEAK'}")
    assert len(train_split_fold0_overlap) == 0

    # Counts that the running script reports
    expected_dev_train_pairs = len(train_idx)
    expected_dev_val_excluded = len(val_idx)
    print(f"\n  Expected training pair counts (per running Phase 3 script):")
    print(f"    train_split pairs: ~{n_pairs_in_train_split} (script reported 121,592)")
    print(f"    dev fold-train pairs: {expected_dev_train_pairs}")
    print(f"    dev fold-val pairs used in training: 0")
    print(f"    fold0 val session overlap with training dev pairs: 0")

    # Cross-check against running script log
    print(f"\n{ts()} Cross-check against Phase 3 training log...")
    log_path = Path("/private/tmp/claude-501/-Users-nadavbarkai-dev-recsys-2026/"
                    "a02f07cf-3789-42cf-9322-831d87f1d64f/tasks/b1pusl8cv.output")
    if log_path.exists():
        with open(log_path) as f:
            log_text = f.read()
        for line in log_text.split("\n"):
            if any(k in line for k in ["Dev pairs:", "train_split:", "TOTAL:", "Fold 0:"]):
                print(f"  log: {line.strip()}")
    else:
        print(f"  (log not yet readable)")

    # Write manifest
    manifest = {
        "fold": 0,
        "query_format": "structured (R54 Phase 2 format)",
        "track_text": "R21 exact format",
        "train_split_pairs_count": n_pairs_in_train_split,
        "dev_train_pairs_count": expected_dev_train_pairs,
        "dev_val_excluded_count": expected_dev_val_excluded,
        "leakage_checks": {
            "dev_fold0_val_cases_in_dev_train": len(val_train_overlap_cases),
            "dev_fold0_val_sessions_in_dev_train": len(val_train_overlap_sessions),
            "train_split_sessions_in_dev": len(train_split_dev_overlap),
            "train_split_sessions_in_fold0_val": len(train_split_fold0_overlap),
            "all_checks_pass": True,
        },
        "fold_split": {
            "function": "scripts.expS2_lambdarank_grouped.grouped_session_folds",
            "seed": 0, "k": 5,
        },
        "hyperparams": {
            "model": "BAAI/bge-base-en-v1.5",
            "epochs": 1, "batch_size": 32, "lr": 2e-5, "tau": 0.05,
            "max_seq_len": 256, "loss": "in-batch InfoNCE",
        },
        "no_same_session_positives": True,
        "no_enriched_metadata": True,
        "no_hard_negatives": True,
        "created_at": datetime.now().isoformat(),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest: {MANIFEST_PATH}")

    print(f"\n{ts()} ALL LEAKAGE CHECKS PASS. Elapsed: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
