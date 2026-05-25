# R85 Phase 1 — Multimodal Integration (Mac-only OOF)

**Date:** 2026-05-25
**Verdict:** INVESTIGATE_OR_ARCHIVE — multimodal source signal does not convert through sibling LR

## Headline

Phase 0 inventory found genuine multimodal headroom:
- image_siglip: **86 unique h7 top-30 recoveries** vs R84 5-fold OOF
- attributes_qwen: 59 unique
- audio_clap: 29 unique
- lyrics_qwen: 22 unique

Phase 1 tested three integration approaches against R84c sibling-R84 LR baseline. **All failed the +0.005 h7 gate.**

| variant | h7 Δ | all Δ | same Δ | diff Δ | rec/lost | overlap | gate |
|---|---:|---:|---:|---:|---|---:|---|
| **R85a** (10-source pool, sibling-R84 LR retrained) | **+0.0002** | +0.0010 | +0.0021 | +0.0003 | 29/20 | 16.96 | h7 sub-gate |
| **R85b** (43-col LR: R39 + R84 + IMG + META rank/pres/cosine) | **−0.0075** | −0.0012 | −0.0044 | +0.0005 | 23/30 | 17.08 | regress |
| **R85c** (selective: R54 margin × IMG top-1) | −0.0017 to −0.0025 | +0.0003-0.0008 | 0.0000-0.0005 | +0.0005-0.0010 | 17-25 / 11-19 | 17.7-18.6 | fail (worse than R85a) |

## Diagnosis

The 86 source-alone recoveries do not translate to LR-ranked nDCG gains. Three failure modes:

1. **R85b (LR feature addition)**: LightGBM with 6 raw multimodal features regressed h7 by 0.0075. The features correlate with same-artist (which R54-stack already covers), so they don't add discriminative signal but do add noise. LR over-weights them on the train fold, then they hurt on eval fold.
2. **R85a (RRF pool extension)**: Adding image_siglip + attributes_qwen as sources 9 and 10 (weight 0.5) brings more candidates into the pool. Sibling-R84 LR can rank them slightly better than R84c (+0.0002 h7, +9 net recoveries), but the lift is sub-statistical.
3. **R85c (selective routing)**: IMG top-1 cosine is saturated at 1.0 across the distribution (p25=p50=p75=max=1.0) because album art similarity is intrinsically high. The IMG_THR routing signal is degenerate — sweep collapses to R54-margin-only routing, which actively hurts vs R85a-only.

## Structural pattern (not specific to R85)

Source-alone retrieval signal frequently fails to convert through sibling LR:
- R59 C3 (matched pool admission): +0.0596 pool_hit, frozen LR buried 91.7%
- R66 (8-profile RRF menu): oracle headroom not extractable
- R68 (BGE-large): +0.035 h7 pool_hit, conversion failed
- R76, R80, R81 (neural over top-300): same-artist canary fires
- R85 (multimodal): 86 unique recoveries, h7 Δ +0.0002

R84 was the exception (h7 +0.0042 dev → +0.0144 blind) because text retrieval upgraded the SAME signal R54 LR was already calibrated for. Multimodal adds *different* signal (image, attributes) the LR isn't tuned for, and sibling LR retraining doesn't bridge the gap.

## Cost

R85 Phase 0 + 1: $0 (Mac), ~600 MB HF download (one-time cache at `.hf_cache/`).

## Next moves

Per user direction, one bounded R88 sprint (CPU-only) testing constrained multimodal interfaces:
- Monotone LightGBM (shallow trees, monotone constraints, L2)
- Guarded additive boost (R84c score + small beta * multimodal indicator)
- Quota injection (preserve R84c top-15, allow ≤2 multimodal swaps)

If R88 fails, multimodal conversion is genuinely closed under this pipeline.

## Files

- `scripts/expR85_phase0_inventory.py` (Phase 0 inventory, committed)
- `scripts/expR85_phase1_sweep.py` (R85a + R85b sweep)
- `scripts/expR85c_selective_routing.py` (R85c routing)
- `exp/eval/expR85_phase0_inventory.json`
- `exp/eval/expR85_phase1_sweep.json`
- `exp/eval/expR85c_selective_routing.json`
- `cache/r85/multimodal_lists/{image_siglip,attributes_qwen}_top300.json` (cached, gitignored)
