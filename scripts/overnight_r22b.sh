#!/usr/bin/env bash
# R22b overnight pipeline: all-train supervised retriever
# Estimated: ~7 hours on CPU
#
# Phase 1: Train BGE on 121K pairs + encode catalog + retrieve dev lists (torch only)
# Phase 2: Evaluate against gates (loads ALS via implicit — separate subprocess)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO/cache/r22b/overnight.log"
mkdir -p "$REPO/cache/r22b"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "R22b overnight pipeline starting"

# Phase 1: Train + encode (torch only, no implicit)
log "Phase 1: Train + encode + retrieve"
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  uv run python3 "$REPO/scripts/expR22b_all_train_retriever.py" --phase train 2>&1 | tee -a "$LOG"

log "Phase 1 complete. Starting Phase 2."

# Phase 2: Evaluate (loads ALS/implicit — separate process)
log "Phase 2: Evaluate against gates"
OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
  uv run python3 "$REPO/scripts/expR22b_all_train_retriever.py" --phase eval 2>&1 | tee -a "$LOG"

log "R22b pipeline complete"
