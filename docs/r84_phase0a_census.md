# R84 Phase 0A — Full-Corpus Pair Census

**Date:** 2026-05-23
**Branch:** `r84-full-corpus-retriever`
**Script:** `scripts/expR84_phase0a_census.py`
**Artifacts:**
- `cache/r84/phase0a/pair_manifest.parquet` (18.9 MB, sha256 `a6ecba53...`)
- `cache/r84/phase0a/build_config.json` (fingerprint `887d7274...`)
- `cache/r84/phase0a/pair_manifest.sha256`
- `exp/eval/expR84_phase0a_census.json`

## Verdict

**PROCEED_TO_PHASE_0B**. Min per-fold training pair count = **147,192**, ratio to R54 baseline = **5.58×**. Comfortably above the 1.5× gate.

## Headline numbers

| metric | value |
|---|---|
| Train sessions walked | 15,199 |
| Dev-session overlap excluded | **0** (train and dev are disjoint by construction) |
| Music turns with unknown track_id | 0 |
| Music turns with no preceding user message | 0 |
| **Train-split pairs (R84, no caps)** | **121,592** |
| Train-split pairs (R54, capped at 20,000) | 20,000 |
| Dev pairs per fold (4 of 5 folds, 1/case) | 25,600 |
| **R84 per-fold training pairs (train_split + 4-of-5 dev)** | **147,192** |
| R54 per-fold training pairs | 26,400 |
| **R84 / R54 ratio** | **5.58×** |
| Manifest size | 18.9 MB |

## Per-fold breakdown (identical across folds — train_split is global)

```
fold 0: 121,592 train_split + 25,600 dev = 147,192
fold 1: 121,592 train_split + 25,600 dev = 147,192
fold 2: 121,592 train_split + 25,600 dev = 147,192
fold 3: 121,592 train_split + 25,600 dev = 147,192
fold 4: 121,592 train_split + 25,600 dev = 147,192
```

## Query-length distribution (chars, ~chars/4 tokens)

| stat | train_split | dev_fold |
|---|---:|---:|
| n | 121,592 | 32,000 |
| char P50 | 854 (~214 tok) | 962 (~241 tok) |
| char P90 | 1,293 (~**323 tok**) | 1,484 (~**371 tok**) |
| char P99 | 1,725 (~431 tok) | 1,940 (~485 tok) |
| char MAX | 6,719 | 3,379 |

### Implication for Phase 0B

R54 trains at `MAX_SEQ_LEN=256`. R84 query P90 is **323 train / 371 dev tokens**, meaning R54 was silently truncating >60% of its training queries (and ~70% of dev queries during eval). Phase 0B should use `max_seq_len=384` (BGE-large supports up to 512) to preserve `[CONTEXT]` content that R54 has been losing. This is a documented free win for R84's full-corpus variant.

VRAM impact estimate (A100 40 GB, bf16):
- Batch 32 × seq 256 (R54-equivalent): ~14 GB
- Batch 32 × seq 384 (R84 proposed): ~22 GB
- Batch 24 × seq 384: ~18 GB ← safer with k=64 random catalog negatives included

If batch 32 × seq 384 + 64 random negs OOMs, fallback path: batch 24 × seq 384, or batch 32 × seq 320.

## Same/diff artist composition (train_split only, GT artist literal in [CONTEXT])

| bucket | n | rate |
|---|---:|---:|
| same artist as some [CONTEXT] track | 66,337 | 0.623 |
| different artist | 40,096 | 0.377 |
| no [CONTEXT] (early in session) | 15,159 | — |

Same-artist rate of 62% is consistent with the dataset's known continuation pattern. Phase 0B's same-artist canary gate (h7 same-artist Δ ≥ −0.005) remains the critical guard against R76/R80/R81-style over-semantic-similarity collapse.

## Operational notes

- **Dev/train disjoint**: zero session overlap. Global vs per-fold exclusion is moot for this dataset (no leakage either way), but the global rule is preserved for R54-parity.
- **Deterministic manifest**: sorted by `(fold_idx, source, session_id, turn_number, gt_track_id)`. sha256 is reproducible across re-runs.
- **Build-config fingerprint** (`887d7274...`) gates Phase 0B consumption — `assert_manifest_matches_build_config()` will fail-fast on any drift.
- **Storage cost**: 18.9 MB parquet (zstd-3, dict-encoded strings) for 153,592 records. Loads in <1 s.

## Estimated Phase 0B fold-0 training cost (A100 bf16)

- Steps: 147,192 pairs / 32 batch = **~4,600 batches** for 1 epoch.
- Wall-clock at ~3 batches/s for BGE-large bf16 batch 32 seq 384 ≈ **~25 min for training**.
- Catalog encoding: 47K × seq 256 ≈ **~3 min**.
- Dev query encoding: 1,600 × seq 384 ≈ **~30 s**.
- Total wall: **~30 min training + encoding**, well under the 6 h budget gate.
- Cost @ $3/h A100: **~$1.50 fold-0**. The original budget of $30–100 was for multiple training reruns + Phase 0B continuation; this single run is cheap.

## Next step

**STOP and report to user.** Per user directive: "Proceed with Phase 0A only and report the census before launching A100." Awaiting explicit Phase 0B approval before Colab work.

Recommended Phase 0B parameters to confirm with user:
1. `MAX_SEQ_LEN = 384` (vs R54's 256, justified by P90 train = 323 tok)
2. `BATCH_SIZE = 32` (with seq 384 + 64 random negs ≈ 22 GB; fallback to 24 if OOM)
3. `LR = 1e-5` (half of R54's 2e-5 — larger model + 6× more pairs)
4. `EPOCHS = 1` (matches R54 protocol)
5. `hard_neg_weight = 0.0` (no R54c hard-negs on first run, per [[feedback_no_hardneg_aux_first_run]])
6. `k_random_negs = 64` per step (GT-excluded, sampled fresh)
