"""R84 Phase 0A — full-corpus pair census + authoritative manifest.

Walks the TalkPlayData train split + dev folds with R54's query format
(`build_query_structured_from_session` / `build_query_structured_from_dev`),
but **without** R54's 20K-pair cap or 2-per-session cap. Dev sessions are
excluded globally from train-split pairs (not per-fold) to match R54's
exclusion behavior.

Writes `cache/r84/phase0a/pair_manifest.parquet` as the authoritative pair
set, plus `build_config.json` (parameters that affect manifest contents) and
`pair_manifest.sha256` (tamper detection). Phase 0B training MUST consume the
manifest via `load_pair_manifest()` and call `assert_manifest_matches_build_config()`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Strict reuse of R54 query helpers — never reimplement
from scripts.expR54_phase3_full5fold_train import (  # noqa: E402
    R12_CACHE,
    build_query_structured_from_dev,
    build_query_structured_from_session,
    load_catalog,
)
from scripts.expS2_lambdarank_grouped import grouped_session_folds  # noqa: E402

PHASE0A_DIR = REPO / "cache" / "r84" / "phase0a"
MANIFEST_PATH = PHASE0A_DIR / "pair_manifest.parquet"
SHA256_PATH = PHASE0A_DIR / "pair_manifest.sha256"
BUILD_CONFIG_PATH = PHASE0A_DIR / "build_config.json"
CENSUS_JSON_PATH = REPO / "exp" / "eval" / "expR84_phase0a_census.json"

# R54 baseline (for the gate)
R54_TRAIN_SPLIT_SAMPLE_CAP = 20000
R54_PAIRS_PER_FOLD = 26400  # 6400 dev + 20000 train-split

# Gate
GATE_PAIRS_PER_FOLD_MIN_MULTIPLIER = 1.5  # > 1.5x R54 baseline


def ts() -> str:
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def census_train_split(meta: dict, dev_session_ids: set[str]) -> list[dict]:
    """Walk full train split, build all (query, gt) pairs, exclude dev sessions globally.

    Mirrors `expR54_phase3_full5fold_train.build_train_split_sample` lines 161-216,
    minus the 2-per-session cap (line 202) and 20K global cap (line 208).
    """
    from datasets import DownloadConfig, load_dataset  # type: ignore[reportMissingImports]

    try:
        train_ds = load_dataset(
            "talkpl-ai/TalkPlayData-Challenge-Dataset",
            download_config=DownloadConfig(local_files_only=True),
        )["train"]
    except Exception:
        train_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")["train"]

    records: list[dict] = []
    n_sessions_seen = 0
    n_dev_overlap = 0
    n_unknown_tid_skipped = 0
    n_no_user_msg_skipped = 0

    for item in train_ds:
        n_sessions_seen += 1
        sid = str(item["session_id"])
        if sid in dev_session_ids:
            n_dev_overlap += 1
            continue
        convs = item["conversations"]
        user_msgs_so_far: list[str] = []
        played_so_far: list[str] = []
        most_recent_user_msg = ""
        for conv in convs:
            role = conv["role"]
            content = str(conv["content"])
            turn_number = int(conv["turn_number"])
            if role == "user":
                user_msgs_so_far.append(content)
                most_recent_user_msg = content
            elif role == "music":
                tid = content.strip()
                if tid not in meta:
                    n_unknown_tid_skipped += 1
                    played_so_far.append(tid)
                    continue
                if not most_recent_user_msg:
                    n_no_user_msg_skipped += 1
                    played_so_far.append(tid)
                    continue
                q = build_query_structured_from_session(
                    user_msgs_so_far, played_so_far, most_recent_user_msg, meta
                )
                records.append({
                    "fold_idx": -1,
                    "source": "train_split",
                    "session_id": sid,
                    "turn_number": turn_number,
                    "query_structured": q,
                    "gt_track_id": tid,
                })
                played_so_far.append(tid)

    print(
        f"  train_split: sessions_seen={n_sessions_seen}, dev_overlap={n_dev_overlap}, "
        f"pairs={len(records)}, unknown_tid={n_unknown_tid_skipped}, "
        f"no_user_msg={n_no_user_msg_skipped}",
        flush=True,
    )
    return records


def census_dev_folds(cases: list[dict], folds: list, meta: dict) -> list[dict]:
    """For each fold 0..4, build records for cases in the OTHER 4 folds.

    Each dev case yields one (query_structured_from_dev, gt) pair per fold it
    is *not* in. fold_idx in the resulting record = the held-out fold for which
    this pair is a training pair (i.e., this case is in one of the other 4 folds).
    """
    records: list[dict] = []
    n_no_gt_in_meta = 0
    case_to_fold = {}
    for fi, idx_array in enumerate(folds):
        for idx in idx_array.tolist():
            case_to_fold[idx] = fi

    for case_idx, c in enumerate(cases):
        gt = c.get("gt")
        if gt not in meta:
            n_no_gt_in_meta += 1
            continue
        owning_fold = case_to_fold[case_idx]
        q = build_query_structured_from_dev(c, meta)
        sid = c["session_id"]
        # The case lives in owning_fold; it serves as a *training pair* for each
        # of the 4 held-out folds != owning_fold. Encode that with fold_idx =
        # the held-out fold for which this is a training pair.
        for held_out_fold in range(5):
            if held_out_fold == owning_fold:
                continue
            records.append({
                "fold_idx": held_out_fold,
                "source": "dev_fold",
                "session_id": sid,
                "turn_number": int(c.get("turn_number", 0)),
                "query_structured": q,
                "gt_track_id": gt,
            })

    print(f"  dev_folds: cases={len(cases)}, no_gt_in_meta={n_no_gt_in_meta}, "
          f"records={len(records)}", flush=True)
    return records


def write_manifest_parquet(records: list[dict], path: Path) -> None:
    """Write deterministic parquet. Sort order = (fold_idx, source, session_id,
    turn_number, gt_track_id) for stable pair_idx and reproducible sha256."""
    import pyarrow as pa  # type: ignore[reportMissingImports]
    import pyarrow.parquet as pq  # type: ignore[reportMissingImports]

    records_sorted = sorted(records, key=lambda r: (
        r["fold_idx"], r["source"], r["session_id"], r["turn_number"], r["gt_track_id"]
    ))
    for i, r in enumerate(records_sorted):
        r["pair_idx"] = i

    schema = pa.schema([
        ("pair_idx", pa.int64()),
        ("fold_idx", pa.int32()),
        ("source", pa.string()),
        ("session_id", pa.string()),
        ("turn_number", pa.int32()),
        ("query_structured", pa.string()),
        ("gt_track_id", pa.string()),
    ])
    arrays = {col.name: [] for col in schema}
    for r in records_sorted:
        arrays["pair_idx"].append(r["pair_idx"])
        arrays["fold_idx"].append(r["fold_idx"])
        arrays["source"].append(r["source"])
        arrays["session_id"].append(r["session_id"])
        arrays["turn_number"].append(r["turn_number"])
        arrays["query_structured"].append(r["query_structured"])
        arrays["gt_track_id"].append(r["gt_track_id"])
    table = pa.table(arrays, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table, path,
        compression="zstd", compression_level=3,
        use_dictionary=True,
        write_statistics=False,  # statistics include random_id-ish fields; turn off for repro
    )


def load_pair_manifest(path: Path = MANIFEST_PATH) -> Any:
    """Phase 0B loader. Returns a pyarrow Table."""
    import pyarrow.parquet as pq  # type: ignore[reportMissingImports]
    return pq.read_table(path)


def assert_manifest_matches_build_config(
    manifest_path: Path = MANIFEST_PATH,
    sha256_path: Path = SHA256_PATH,
    build_config_path: Path = BUILD_CONFIG_PATH,
    expected_config_fingerprint: str | None = None,
) -> None:
    """Phase 0B guard. Fail-fast if manifest hash or config drifts."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest missing: {manifest_path}")
    if not sha256_path.exists():
        raise FileNotFoundError(f"sha256 missing: {sha256_path}")
    if not build_config_path.exists():
        raise FileNotFoundError(f"build_config missing: {build_config_path}")
    recorded_sha = sha256_path.read_text().strip().split()[0]
    actual_sha = sha256_file(manifest_path)
    if recorded_sha != actual_sha:
        raise RuntimeError(
            f"manifest sha256 drift: recorded={recorded_sha[:16]} actual={actual_sha[:16]}"
        )
    with open(build_config_path) as f:
        cfg = json.load(f)
    if expected_config_fingerprint and cfg.get("fingerprint") != expected_config_fingerprint:
        raise RuntimeError(
            f"build_config fingerprint drift: expected={expected_config_fingerprint} "
            f"actual={cfg.get('fingerprint')}"
        )
    return cfg


def compute_build_config(
    n_catalog: int,
    dev_payload_sha: str,
) -> dict:
    cfg = {
        "experiment": "R84 Phase 0A",
        "created_at": datetime.now().isoformat(),
        "query_format_template": "R54_phase3 (line refs: expR54_phase3_full5fold_train.py:102-137)",
        "query_format_template_version": "v1",
        "dev_exclusion_mode": "global",  # ALL 8000 dev sids excluded from train-split pairs
        "train_split_per_session_cap_disabled": True,
        "train_split_global_cap_disabled": True,
        "dev_pair_count_per_case": 1,
        "fold_assignment_seed": 0,
        "fold_assignment_source": "scripts.expS2_lambdarank_grouped.grouped_session_folds(sessions, seed=0)",
        "train_dataset_uri": "talkpl-ai/TalkPlayData-Challenge-Dataset[train]",
        "catalog_dataset_uri": "talkpl-ai/TalkPlayData-Challenge-Track-Metadata[all_tracks]",
        "catalog_n_tracks": n_catalog,
        "dev_payload_path": str(R12_CACHE.relative_to(REPO)),
        "dev_payload_sha256": dev_payload_sha,
    }
    # Fingerprint = hash of all fields that affect manifest contents
    fp_fields = [
        "query_format_template_version",
        "dev_exclusion_mode",
        "train_split_per_session_cap_disabled",
        "train_split_global_cap_disabled",
        "dev_pair_count_per_case",
        "fold_assignment_seed",
        "train_dataset_uri",
        "catalog_dataset_uri",
        "catalog_n_tracks",
        "dev_payload_sha256",
    ]
    fp_blob = json.dumps({k: cfg[k] for k in fp_fields}, sort_keys=True).encode()
    cfg["fingerprint"] = hashlib.sha256(fp_blob).hexdigest()
    return cfg


def query_length_stats(records: list[dict]) -> dict:
    import numpy as np  # type: ignore[reportMissingImports]
    lens = np.array([len(r["query_structured"]) for r in records], dtype=np.int64)
    return {
        "n": int(len(lens)),
        "char_p50": int(np.percentile(lens, 50)),
        "char_p90": int(np.percentile(lens, 90)),
        "char_p99": int(np.percentile(lens, 99)),
        "char_max": int(lens.max()) if len(lens) else 0,
        # Rough token estimate: ~4 chars/token for English subword tokenizers
        "token_p90_est": int(np.percentile(lens, 90) / 4) if len(lens) else 0,
    }


def per_fold_pair_counts(records: list[dict]) -> dict:
    """For each fold f, count pairs available for training (fold_idx == -1 OR fold_idx == f)."""
    by_fold = defaultdict(int)
    n_train_split = 0
    for r in records:
        if r["fold_idx"] == -1:
            n_train_split += 1
        else:
            by_fold[r["fold_idx"]] += 1
    out = {"train_split_global": n_train_split}
    for f in range(5):
        out[f"fold_{f}_training_pairs"] = n_train_split + sum(
            v for k, v in by_fold.items() if k != f
        )
        out[f"fold_{f}_dev_only_pairs"] = sum(v for k, v in by_fold.items() if k != f)
    return out


def same_diff_artist_split(records: list[dict], meta: dict) -> dict:
    """In train-split pairs, what fraction has same-artist as a played track in the query?
    Cheap proxy — checks if the gt artist appears literally in the [CONTEXT] section.
    """
    n_same = 0
    n_diff = 0
    n_no_context = 0
    for r in records:
        if r["source"] != "train_split":
            continue
        gt = r["gt_track_id"]
        gt_artists = meta.get(gt, {}).get("artist_name", [])
        if isinstance(gt_artists, list):
            gt_artist = gt_artists[0] if gt_artists else None
        else:
            gt_artist = str(gt_artists)
        if not gt_artist:
            n_diff += 1
            continue
        ctx_start = r["query_structured"].find("[CONTEXT]")
        if ctx_start == -1:
            n_no_context += 1
            continue
        ctx_blob = r["query_structured"][ctx_start:]
        if gt_artist in ctx_blob:
            n_same += 1
        else:
            n_diff += 1
    total = n_same + n_diff + n_no_context
    return {
        "n_total_train_split": total,
        "same_artist": n_same,
        "diff_artist": n_diff,
        "no_context": n_no_context,
        "same_artist_rate": (n_same / max(1, n_same + n_diff)) if total else 0.0,
    }


def evaluate_gate(per_fold: dict) -> dict:
    min_pairs_required = int(R54_PAIRS_PER_FOLD * GATE_PAIRS_PER_FOLD_MIN_MULTIPLIER)
    counts = [per_fold[f"fold_{f}_training_pairs"] for f in range(5)]
    min_observed = min(counts)
    ratio_to_r54 = min_observed / R54_PAIRS_PER_FOLD
    proceed = min_observed > min_pairs_required
    return {
        "rule": (
            f"min(per_fold_training_pairs) > {GATE_PAIRS_PER_FOLD_MIN_MULTIPLIER}x R54 baseline "
            f"({R54_PAIRS_PER_FOLD}) -> >{min_pairs_required}"
        ),
        "min_observed_per_fold": min_observed,
        "max_observed_per_fold": max(counts),
        "ratio_to_r54_baseline": ratio_to_r54,
        "verdict": "PROCEED_TO_PHASE_0B" if proceed else "STOP_DATA_SCALE_INSUFFICIENT",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=PHASE0A_DIR,
                   help="Phase 0A output dir (default cache/r84/phase0a/)")
    args = p.parse_args()

    PHASE0A_DIR.mkdir(parents=True, exist_ok=True)
    CENSUS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"{ts()} Loading dev payload (R12 cache)...", flush=True)
    with open(R12_CACHE, "rb") as f:
        payload = pickle.load(f)  # noqa: S301
    cases = payload["cases"]
    sessions = [c["session_id"] for c in cases]
    dev_session_ids = set(sessions)
    dev_payload_sha = sha256_file(R12_CACHE)[:16]
    print(f"  {len(cases)} cases, {len(dev_session_ids)} unique sessions, "
          f"R12 sha={dev_payload_sha}", flush=True)

    print(f"\n{ts()} Loading catalog...", flush=True)
    meta, all_track_ids = load_catalog()
    print(f"  {len(meta)} tracks in catalog", flush=True)

    print(f"\n{ts()} Building 5-fold session-grouped folds (seed=0)...", flush=True)
    folds = grouped_session_folds(sessions, seed=0)
    for fi, idx in enumerate(folds):
        print(f"  fold {fi}: {len(idx)} val cases", flush=True)

    print(f"\n{ts()} Censusing dev folds...", flush=True)
    dev_records = census_dev_folds(cases, folds, meta)

    print(f"\n{ts()} Censusing full train split (NO caps, global dev exclusion)...", flush=True)
    train_records = census_train_split(meta, dev_session_ids)

    all_records = dev_records + train_records
    print(f"\n{ts()} Total records: dev={len(dev_records)} + train_split={len(train_records)} "
          f"= {len(all_records)}", flush=True)

    print(f"\n{ts()} Writing manifest parquet -> {MANIFEST_PATH}...", flush=True)
    write_manifest_parquet(all_records, MANIFEST_PATH)
    manifest_sha = sha256_file(MANIFEST_PATH)
    SHA256_PATH.write_text(f"{manifest_sha}  pair_manifest.parquet\n")
    print(f"  manifest sha256 = {manifest_sha[:16]}", flush=True)
    print(f"  manifest size = {MANIFEST_PATH.stat().st_size / 1e6:.1f} MB", flush=True)

    print(f"\n{ts()} Writing build_config...", flush=True)
    cfg = compute_build_config(n_catalog=len(meta), dev_payload_sha=dev_payload_sha)
    cfg["manifest_path"] = str(MANIFEST_PATH.relative_to(REPO))
    cfg["manifest_sha256"] = manifest_sha
    cfg["manifest_n_rows"] = len(all_records)
    with open(BUILD_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=False)
    print(f"  fingerprint = {cfg['fingerprint'][:16]}", flush=True)

    print(f"\n{ts()} Computing distributions...", flush=True)
    per_fold = per_fold_pair_counts(all_records)
    train_split_lens = query_length_stats([r for r in all_records if r["source"] == "train_split"])
    dev_lens = query_length_stats([r for r in all_records if r["source"] == "dev_fold"])
    same_diff = same_diff_artist_split(all_records, meta)
    gate = evaluate_gate(per_fold)

    census = {
        "experiment": "R84 Phase 0A — full-corpus pair census",
        "created_at": datetime.now().isoformat(),
        "n_records_total": len(all_records),
        "n_records_train_split": len(train_records),
        "n_records_dev": len(dev_records),
        "per_fold_pair_counts": per_fold,
        "r54_baseline_per_fold": R54_PAIRS_PER_FOLD,
        "r54_train_split_cap": R54_TRAIN_SPLIT_SAMPLE_CAP,
        "train_split_query_length_stats": train_split_lens,
        "dev_query_length_stats": dev_lens,
        "train_split_same_diff_artist": same_diff,
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO)),
        "manifest_sha256": manifest_sha,
        "manifest_size_mb": round(MANIFEST_PATH.stat().st_size / 1e6, 2),
        "build_config_path": str(BUILD_CONFIG_PATH.relative_to(REPO)),
        "build_config_fingerprint": cfg["fingerprint"],
        "gate": gate,
    }
    with open(CENSUS_JSON_PATH, "w") as f:
        json.dump(census, f, indent=2)
    print(f"\n{ts()} Wrote census -> {CENSUS_JSON_PATH}", flush=True)
    print(f"\n{ts()} GATE: {gate['verdict']} "
          f"(min_per_fold={gate['min_observed_per_fold']}, "
          f"ratio_to_r54={gate['ratio_to_r54_baseline']:.2f}x)", flush=True)


if __name__ == "__main__":
    main()
