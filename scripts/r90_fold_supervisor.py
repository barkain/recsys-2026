"""R90 Phase 1 Variant A — resilient fold supervisor (detached, Drive-anchored).

Designed to be launched on Colab via `nohup ... &` so it survives browser
disconnects and MCP visibility loss. Drive is the source of truth:

- Logs stream to Drive (per fold + supervisor log)
- Status JSON updates on every state transition
- Eval tarballs land in Drive immediately after each fold's eval
- Idempotent resume: skips folds whose eval tarball already exists in Drive

States per fold: queued / running_train / train_done / running_eval / eval_done / failed
Plus supervisor-level: supervisor_started / supervisor_done / supervisor_failed

Usage (inside Colab, where Drive is already mounted at /content/drive):
  nohup python scripts/r90_fold_supervisor.py --folds 1,2,3,4 \\
      --drive-dir /content/drive/MyDrive/r90_phase1_varA_artifacts \\
      --repo /content/recsys-2026 \\
      > /content/drive/MyDrive/r90_phase1_varA_artifacts/supervisor_stdout.log 2>&1 &
  echo $! > /content/drive/MyDrive/r90_phase1_varA_artifacts/supervisor.pid

To poll progress (from any environment with Drive access):
  cat <drive>/status.json
  tail -50 <drive>/fold_{N}_log.txt
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path


def ts() -> str:
    return f"[{datetime.now():%Y-%m-%d %H:%M:%S}]"


def utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"


def write_status(drive_dir: Path, payload: dict) -> None:
    """Atomic-ish status write — tmp + rename."""
    target = drive_dir / "status.json"
    tmp = drive_dir / "status.json.tmp"
    payload = dict(payload)
    payload["updated_at"] = utcnow()
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, target)


def sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def supervisor_log(drive_dir: Path, msg: str) -> None:
    line = f"{ts()} {msg}\n"
    with open(drive_dir / "supervisor.log", "a") as f:
        f.write(line)


def stream_subprocess(cmd: list[str], cwd: Path, env: dict,
                       drive_log_path: Path) -> int:
    """Run subprocess, tee stdout+stderr to a Drive log file in real-time."""
    with open(drive_log_path, "a") as logf:
        logf.write(f"\n{ts()} CMD: {' '.join(cmd)}\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
        rc = proc.wait()
        logf.write(f"{ts()} EXIT: {rc}\n")
        return rc


def pack_eval_artifacts(repo: Path, fold: int, drive_dir: Path) -> tuple[Path, str]:
    """Tar eval-only artifacts for one fold to Drive. Returns (tar_path, sha)."""
    out_dir = repo / f"cache/r90/phase1_fold{fold}_varA"
    tar_path = drive_dir / f"r90_phase1_fold{fold}_varA_eval.tar.gz"
    files = [
        out_dir / "oof_r84_lists.json",
        out_dir / "r84_features.npy",
        out_dir / "eval_summary.json",
        out_dir / "training_summary.json",
        out_dir / "training_log.txt",
        out_dir / "eval_log.txt",
    ]
    with tarfile.open(tar_path, "w:gz") as tf:
        for p in files:
            if p.exists():
                tf.add(str(p), arcname=str(p.relative_to(repo)))
    return tar_path, sha256_of(tar_path)


def run_fold(repo: Path, fold: int, drive_dir: Path, batch_size: int,
              folds_state: dict, env_overrides: dict) -> bool:
    """Run train + eval + backup for one fold. Returns True on success."""
    out_dir = repo / f"cache/r90/phase1_fold{fold}_varA"
    out_dir.mkdir(parents=True, exist_ok=True)
    drive_log = drive_dir / f"fold_{fold}_log.txt"

    env = {**os.environ, **env_overrides, "PYTHONPATH": str(repo)}

    # Idempotent resume: skip if tarball already in Drive
    expected_tar = drive_dir / f"r90_phase1_fold{fold}_varA_eval.tar.gz"
    if expected_tar.exists():
        supervisor_log(drive_dir, f"fold {fold}: SKIP (tarball already in Drive: "
                                   f"{expected_tar.stat().st_size/1e6:.1f} MB)")
        folds_state[str(fold)] = {
            "state": "eval_done", "resumed": True,
            "tarball": expected_tar.name,
            "tarball_sha256": sha256_of(expected_tar),
        }
        return True

    # ---- TRAIN ----
    folds_state[str(fold)] = {"state": "running_train", "started_at": utcnow()}
    write_status(drive_dir, {"current_fold": fold, "folds": folds_state})
    supervisor_log(drive_dir, f"fold {fold}: TRAIN start")
    t0 = time.time()
    rc = stream_subprocess(
        [sys.executable, "-u", "scripts/expR90_phase1_train.py",
         "--fold", str(fold),
         "--output-dir", str(out_dir),
         "--batch-size", str(batch_size)],
        cwd=repo, env=env, drive_log_path=drive_log,
    )
    train_min = (time.time() - t0) / 60
    if rc != 0:
        folds_state[str(fold)].update({
            "state": "failed", "phase": "train", "exit_code": rc,
            "train_min": round(train_min, 1),
        })
        write_status(drive_dir, {"current_fold": fold, "folds": folds_state})
        supervisor_log(drive_dir, f"fold {fold}: TRAIN FAIL rc={rc} after {train_min:.1f} min")
        return False

    # Loss sanity from training_summary.json
    ts_path = out_dir / "training_summary.json"
    init_loss = final_loss = None
    if ts_path.exists():
        s = json.loads(ts_path.read_text())
        init_loss = s.get("initial_loss_first50")
        final_loss = s.get("final_loss_avg_last50")
    bad_loss = (final_loss is None or
                (isinstance(final_loss, float) and (final_loss != final_loss or final_loss > 5.0)))
    folds_state[str(fold)].update({
        "state": "train_done" if not bad_loss else "failed",
        "train_min": round(train_min, 1),
        "initial_loss": init_loss, "final_loss": final_loss,
    })
    if bad_loss:
        folds_state[str(fold)]["phase"] = "train_bad_loss"
        write_status(drive_dir, {"current_fold": fold, "folds": folds_state})
        supervisor_log(drive_dir, f"fold {fold}: BAD LOSS init={init_loss} final={final_loss}")
        return False
    write_status(drive_dir, {"current_fold": fold, "folds": folds_state})
    supervisor_log(drive_dir, f"fold {fold}: TRAIN done {train_min:.1f} min "
                              f"loss {init_loss} -> {final_loss}")

    # ---- EVAL ----
    folds_state[str(fold)].update({
        "state": "running_eval", "eval_started_at": utcnow(),
    })
    write_status(drive_dir, {"current_fold": fold, "folds": folds_state})
    supervisor_log(drive_dir, f"fold {fold}: EVAL start")
    t0 = time.time()
    rc = stream_subprocess(
        [sys.executable, "-u", "scripts/expR90_phase1_eval.py",
         "--fold", str(fold),
         "--model-dir", str(out_dir / "model"),
         "--output-dir", str(out_dir),
         "--batch-size", "128"],
        cwd=repo, env=env, drive_log_path=drive_log,
    )
    eval_min = (time.time() - t0) / 60
    if rc != 0:
        folds_state[str(fold)].update({
            "state": "failed", "phase": "eval", "exit_code": rc,
            "eval_min": round(eval_min, 1),
        })
        write_status(drive_dir, {"current_fold": fold, "folds": folds_state})
        supervisor_log(drive_dir, f"fold {fold}: EVAL FAIL rc={rc}")
        return False
    folds_state[str(fold)].update({
        "state": "eval_done", "eval_min": round(eval_min, 1),
    })
    write_status(drive_dir, {"current_fold": fold, "folds": folds_state})
    supervisor_log(drive_dir, f"fold {fold}: EVAL done {eval_min:.1f} min")

    # Read eval metrics for summary
    es_path = out_dir / "eval_summary.json"
    if es_path.exists():
        es = json.loads(es_path.read_text())
        folds_state[str(fold)]["metrics"] = es.get("source_alone_metrics", {})

    # ---- BACKUP ----
    supervisor_log(drive_dir, f"fold {fold}: BACKUP -> Drive")
    t0 = time.time()
    tar_path, sha = pack_eval_artifacts(repo, fold, drive_dir)
    folds_state[str(fold)].update({
        "tarball": tar_path.name,
        "tarball_sha256": sha,
        "tarball_mb": round(tar_path.stat().st_size / 1e6, 2),
        "backup_s": round(time.time() - t0, 1),
        "completed_at": utcnow(),
    })
    write_status(drive_dir, {"current_fold": fold, "folds": folds_state})
    supervisor_log(drive_dir, f"fold {fold}: BACKUP done -> {tar_path.name} "
                              f"({tar_path.stat().st_size/1e6:.1f} MB, sha {sha[:12]}...)")

    # Free GPU between folds
    try:
        import torch  # type: ignore[reportMissingImports]
        torch.cuda.empty_cache()
    except Exception:
        pass
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="1,2,3,4")
    ap.add_argument("--drive-dir", required=True, type=Path,
                    help="Drive dir for status + logs + tarballs.")
    ap.add_argument("--repo", required=True, type=Path,
                    help="Repo root (e.g. /content/recsys-2026).")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip()]
    drive_dir = args.drive_dir
    repo = args.repo
    drive_dir.mkdir(parents=True, exist_ok=True)
    assert repo.exists(), f"Repo not at {repo}"

    # Set common env
    env_overrides = {
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    }

    # Initial status
    state = {
        "supervisor": "supervisor_started",
        "started_at": utcnow(),
        "folds": {str(f): {"state": "queued"} for f in folds},
        "args": {"folds": folds, "drive_dir": str(drive_dir),
                 "repo": str(repo), "batch_size": args.batch_size},
    }
    folds_state = state["folds"]
    write_status(drive_dir, state)
    supervisor_log(drive_dir, f"SUPERVISOR START folds={folds} pid={os.getpid()}")

    t_total = time.time()
    for fold in folds:
        ok = run_fold(repo, fold, drive_dir, args.batch_size, folds_state, env_overrides)
        if not ok:
            state.update({
                "supervisor": "supervisor_failed",
                "failed_fold": fold,
                "total_wall_min": round((time.time() - t_total) / 60, 1),
                "completed_at": utcnow(),
                "folds": folds_state,
            })
            write_status(drive_dir, state)
            supervisor_log(drive_dir, f"SUPERVISOR FAIL on fold {fold}")
            sys.exit(2)

    state.update({
        "supervisor": "supervisor_done",
        "total_wall_min": round((time.time() - t_total) / 60, 1),
        "completed_at": utcnow(),
        "folds": folds_state,
    })
    write_status(drive_dir, state)
    supervisor_log(drive_dir, f"SUPERVISOR DONE total {(time.time() - t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
