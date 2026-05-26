"""R84c blind dry-run — replays R84c pipeline on Blind-A and verifies the
output track hash matches the original r84c_selective_submission.zip.

Used as a smoke test before Blind-B drops. If this passes, the harness is
known-deterministic on the production stack.

Usage:
  uv run python scripts/expR84c_blind_dryrun.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def ts(): return f"[{datetime.now():%H:%M:%S}]"


def track_hash_from_items(items):
    payload = sorted(
        (i["session_id"], int(i["turn_number"]), tuple(i["predicted_track_ids"]))
        for i in items
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def main():
    print(f"{ts()} R84c blind DRY-RUN — Blind-A reproducibility test")
    print("=" * 70)

    # 1. Compute hash of the original R84c production submission
    orig_zip = REPO / "exp" / "inference" / "blind_a" / "r84c_selective_submission.zip"
    print(f"\n{ts()} Loading original R84c submission: {orig_zip.name}")
    with zipfile.ZipFile(orig_zip) as z:
        orig_items = json.loads(z.read("prediction.json"))
    orig_hash = track_hash_from_items(orig_items)
    print(f"  original n_cases: {len(orig_items)}")
    print(f"  original track hash: {orig_hash[:16]}...")

    # 2. Run preflight
    print(f"\n{ts()} Running preflight...")
    r = subprocess.run(
        [sys.executable, "scripts/expR84c_blind_preflight.py", "--blind-name", "blind_a"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(f"PREFLIGHT FAILED with rc={r.returncode}")
        sys.exit(1)

    # 3. Run replay (tracks-only)
    print(f"\n{ts()} Running replay (tracks-only)...")
    r = subprocess.run(
        [sys.executable, "scripts/expR84c_blind_replay.py",
         "--blind-name", "blind_a", "--tracks-only"],
        cwd=str(REPO), capture_output=True, text=True,
    )
    print(r.stdout[-2000:])
    if r.returncode != 0:
        print(f"REPLAY FAILED with rc={r.returncode}")
        print(r.stderr[-1500:])
        sys.exit(1)

    # 4. Read replay output + verify hash
    replay_lists = REPO / "exp" / "inference" / "blind_a" / "r84c_replay_blind_a_track_lists.json"
    print(f"\n{ts()} Verifying replay track hash...")
    with open(replay_lists) as f:
        replay_items = json.load(f)
    replay_hash = track_hash_from_items(replay_items)
    print(f"  replay n_cases: {len(replay_items)}")
    print(f"  replay track hash:   {replay_hash[:16]}...")

    print(f"\n  Original hash: {orig_hash}")
    print(f"  Replay hash:   {replay_hash}")

    if orig_hash == replay_hash:
        print(f"\n  ✓ HASH MATCH — replay is deterministic and reproduces R84c production")
        print(f"\nDRY-RUN PASS. Harness is ready for Blind-B.")
        sys.exit(0)
    else:
        print(f"\n  ✗ HASH MISMATCH — replay diverged from original R84c")
        # Diff per-case
        orig_by_key = {(i["session_id"], int(i["turn_number"])): i["predicted_track_ids"]
                        for i in orig_items}
        replay_by_key = {(i["session_id"], int(i["turn_number"])): i["predicted_track_ids"]
                          for i in replay_items}
        all_keys = sorted(set(orig_by_key) | set(replay_by_key))
        n_diff = 0
        for k in all_keys:
            if orig_by_key.get(k) != replay_by_key.get(k):
                n_diff += 1
                if n_diff <= 5:
                    print(f"    diff {k}: orig top-3={orig_by_key.get(k, [])[:3]} "
                          f"replay top-3={replay_by_key.get(k, [])[:3]}")
        print(f"  Total diff cases: {n_diff}/{len(all_keys)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
