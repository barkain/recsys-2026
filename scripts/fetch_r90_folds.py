"""Fetch + extract R90 Phase 1 fold tarballs from ~/Downloads, then run 5-fold compare.

Default: extract folds 1-4 from ~/Downloads/r90_phase1_fold{N}_varA_eval.tar.gz
(fold-0 is already in the repo from the prior fetch). Verifies sha256 if --sha
provided for each fold. After successful extraction of all folds, optionally
invokes the multi-fold compare automatically.

Usage:
  uv run python scripts/fetch_r90_folds.py
  uv run python scripts/fetch_r90_folds.py --sha 1:<sha>,2:<sha>,3:<sha>,4:<sha>
  uv run python scripts/fetch_r90_folds.py --no-compare   # extract only
  uv run python scripts/fetch_r90_folds.py --folds 2,3    # subset

This is Mac-side only; does NOT touch Colab.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOWNLOADS = Path.home() / "Downloads"
DEFAULT_FOLDS = [1, 2, 3, 4]

# Required files inside each fold's tarball (relative to REPO)
REQUIRED_FILES_TEMPLATE = [
    "cache/r90/phase1_fold{fold}_varA/oof_r84_lists.json",
    "cache/r90/phase1_fold{fold}_varA/eval_summary.json",
    "cache/r90/phase1_fold{fold}_varA/training_summary.json",
    "cache/r90/phase1_fold{fold}_varA/training_log.txt",
    "cache/r90/phase1_fold{fold}_varA/eval_log.txt",
]


def ts() -> str:
    return f"[{datetime.now():%H:%M:%S}]"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def parse_sha_arg(s: str | None) -> dict[int, str]:
    if not s:
        return {}
    out = {}
    for piece in s.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if ":" not in piece:
            raise ValueError(f"--sha entry must be 'fold:hex': {piece!r}")
        f_str, hexsha = piece.split(":", 1)
        out[int(f_str)] = hexsha.strip()
    return out


def fetch_one_fold(fold: int, expected_sha: str | None) -> bool:
    tar_name = f"r90_phase1_fold{fold}_varA_eval.tar.gz"
    tar_path = DOWNLOADS / tar_name
    print(f"\n=== fold {fold} ===")
    if not tar_path.exists():
        print(f"  FAIL: tarball missing at {tar_path}")
        return False
    size_mb = tar_path.stat().st_size / 1e6
    print(f"  tarball: {tar_path} ({size_mb:.1f} MB)")

    actual_sha = sha256_of(tar_path)
    print(f"  sha256:   {actual_sha[:16]}...")
    if expected_sha:
        if not actual_sha.startswith(expected_sha) and actual_sha != expected_sha:
            print(f"  FAIL: sha mismatch — expected {expected_sha[:16]}..., "
                  f"got {actual_sha[:16]}...")
            return False
        print(f"  sha verified")

    # Extract
    with tarfile.open(tar_path, "r:gz") as tf:
        members = tf.getnames()
        print(f"  members ({len(members)}):")
        for m in members:
            print(f"    {m}")
        tf.extractall(REPO)
    print(f"  extracted into {REPO}")

    # Verify required files
    missing = []
    for rel in REQUIRED_FILES_TEMPLATE:
        p = REPO / rel.format(fold=fold)
        if not p.exists():
            missing.append(str(p.relative_to(REPO)))
    if missing:
        print(f"  FAIL: missing after extract: {missing}")
        return False
    print(f"  all {len(REQUIRED_FILES_TEMPLATE)} required files present")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default=",".join(str(f) for f in DEFAULT_FOLDS),
                    help="Comma-separated fold list (default: 1,2,3,4)")
    ap.add_argument("--sha", default=None,
                    help="Per-fold sha256 expectations: e.g. "
                         "'1:abc...,2:def...'. Optional. "
                         "Prefix match accepted (provide >=12 hex chars).")
    ap.add_argument("--no-compare", action="store_true",
                    help="Extract only; do NOT invoke 5-fold compare.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Don't fail-stop if a fold tarball missing; "
                         "report and continue with the others.")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    shas = parse_sha_arg(args.sha)

    print(f"{ts()} R90 fold fetch — folds={folds} downloads_dir={DOWNLOADS}")
    if shas:
        print(f"  sha verification: {sorted(shas)}")

    fetched = []
    failed = []
    for f in folds:
        ok = fetch_one_fold(f, shas.get(f))
        if ok:
            fetched.append(f)
        else:
            failed.append(f)
            if not args.allow_partial:
                print(f"\nFATAL: fold {f} fetch failed; stopping. "
                      f"Use --allow-partial to continue past failures.")
                sys.exit(1)

    print(f"\n{ts()} Summary: fetched {fetched}, failed {failed}")

    if args.no_compare:
        print("--no-compare set; skipping 5-fold compare invocation.")
        return

    # Only invoke compare if all 5 R90 folds present (0 + the requested 4)
    r90_fold_dirs = [REPO / f"cache/r90/phase1_fold{f}_varA/oof_r84_lists.json"
                      for f in range(5)]
    missing_r90 = [str(p.relative_to(REPO)) for p in r90_fold_dirs if not p.exists()]
    if missing_r90:
        print(f"\nNot running 5-fold compare yet; still missing: {missing_r90}")
        return

    print(f"\n{ts()} Invoking 5-fold compare...")
    cmd = ["uv", "run", "python", "scripts/expR90_phase1_compare.py", "--multi-fold"]
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        print(f"compare exited with {r.returncode}")
        sys.exit(r.returncode)


if __name__ == "__main__":
    main()
