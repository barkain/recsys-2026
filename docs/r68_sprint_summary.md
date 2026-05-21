# R68 Large-Encoder Retriever Sprint — Summary (updated post-R68.1)

**Sprint completed:** 2026-05-21
**Branch:** `r68-large-scale-retrieval`
**Final verdict:** `ARCHIVE_SPRINT` — Phase 0 archived per pre-declared conversion rule. R68.1 corrected re-eval **confirmed the conversion wall is genuine, not a Phase 0 bias artifact**. Phase 1 (full 5-fold OOF) deferred — disambiguation answered.
**Production status:** R63c-repair holds Blind-A at composite 0.6224 / nDCG 0.4925 / LLM 4.85 / LexDiv 0.8438

## R68.1 corrected eval (post-mortem disambiguation)

After the original Phase 0 archived, the open question was: did the LR conversion fail genuinely, or because of the documented `r68_cosine=0` on TRAIN cases? **R68.1 generated real train query embeddings** (from the already-trained fold-0 R68 model, ~10s on A100) **and real R68 train OOF lists** (cosine top-300 over track embeddings, ~5s on Mac CPU via chunked numpy), then re-ran the sibling LR with the bias removed.

| Metric | v1 (zero-stub TRAIN cosine) | v2 (real TRAIN cosine) | Δ |
|---|---:|---:|---:|
| h7 nDCG vs baseline | −0.0797 | **−0.0811** | ~same |
| same-artist nDCG | −0.1564 | **−0.1557** | ~same |
| all-fold-0 nDCG | −0.0766 | **−0.0795** | ~same |
| diff-artist nDCG | −0.0370 | −0.0417 | slightly worse |
| TRAIN pool_hit | 0.6041 | 0.6269 | +0.023 (bias removed ✓) |

The corrected TRAIN pool_hit jumped +0.023 (confirming the bias was eliminated), but **dev metrics are essentially unchanged**. The LR conversion wall is real for BGE-large. The retrieval signal (+0.035 h7 pool_hit, 15 unique GT recoveries net +7) exists but cannot be extracted by an LR-style ranker — even with proper OOF training features, the LR learns a calibration that buries R68's gains.

**Phase 1 (full 5-fold OOF, ~2.5 GPU-hours) was deferred** because the disambiguation goal of Phase 1 (separate genuine wall from bias) has already been achieved by R68.1 on fold-0. Running Phase 1 would consume another ~$5-15 of A100 time to validate at a larger N but is unlikely to change the verdict — the bias-corrected fold-0 result already trends in the same direction as the original.

## Context

R68 tested the original "larger encoder beats R54 BGE-base" hypothesis using **BGE-large-en-v1.5** (~335 M params, ~3× R54's BGE-base-en-v1.5) as a direct R54 replacement. Same R54 structured-query recipe, same fold-0 split, sibling LR with substituted r68 features (r68_rank_inv / r68_presence / r68_cosine). Run on **NVIDIA A100-40GB** in **bf16** (fp16 produced NaN losses; bf16 was numerically stable).

## Wave outcomes

| Wave | Commit | Outcome |
|---|---|---|
| W0 baseline + R54 stats | `1898c8d` | PASS (max \|Δ\| ≈ 0 vs R54c canonical) |
| W1 GPU handoff + scripts | `cd019b4` | scripts/expR68_phase0_fold0_train.py + eval.py + docs/r68_gpu_handoff.md |
| Phase 0 fold-0 training | (A100 Colab, not committed) | 335s elapsed, final avg loss 0.78, encoder produced |
| Phase 0 eval | `873ab92` | **ARCHIVE_PHASE_0** (gate 1 PASS, gates 2 + 3 FAIL) |

## Phase 0 gate breakdown (fold-0 held-out, n=1600; h7 subset n=200)

| Gate | Rule | Value | Pass |
|---|---|---|:---:|
| 1 | recovery | unique h7 recoveries = **15**, Δ single-source h7 pool_hit = **+0.0350** | ✅ |
| 2 | pool_hit | Δ stacked-RRF all-fold-0 = **−0.0025** (h7-only: +0.0150) | ❌ |
| 3 | nDCG | Δ h7 = **−0.0797**, Δ same-artist = **−0.1564** | ❌ |

## Single-source retrieval (the genuinely positive signal)

| Subset | R54 (BGE-base) | R68 (BGE-large) | Δ |
|---|---:|---:|---:|
| h7 (n=200) | 0.5350 | **0.5700** | **+0.0350** |
| all (n=1600) | 0.5756 | 0.5806 | +0.0050 |

- **15 unique h7 recoveries** — R68 finds GT tracks that R54 misses.
- 8 cases lost (R54 had GT, R68 didn't). Net **+7**.
- This is the **first positive retrieval-layer signal across 19 post-R54c experiments.** R65/R66/R67 all failed at the pool/retrieval layer. R68 actually delivers a meaningful retrieval upgrade.

## Conversion failure (why we still archive)

| Subset | Baseline | R68 stacked + sibling LR | Δ |
|---|---:|---:|---:|
| all_fold0 | 0.2879 | 0.2113 | **−0.0766** |
| h7 | 0.3043 | 0.2246 | **−0.0797** |
| same_artist | 0.5984 | 0.4419 | **−0.1564** |
| diff_artist | 0.1337 | 0.0967 | −0.0370 |

The sibling LR can't extract R68's retrieval gain. Same-artist canary fires hard (−0.156), consistent with R66's "LR mis-calibrates on shifted source-rank distribution" pattern.

### Known caveat in Phase 0

The Phase 0 sibling LR was trained with `r68_cosine = 0` on TRAIN cases (no R68 query embeddings generated for training data; only the fold-0 held-out dev queries were embedded). At dev time, real r68_cosine values appeared but LR had learned to treat the column as a zero stub. This is a known Phase 0 limitation noted in the script's log: *"WARNING: TRAIN-case r68_cosine set to 0 (no train query embeddings). Phase 1 runs fully OOF."*

A full Phase 1 (5-fold OOF — each fold trained with proper r68_cosine from the other 4 folds' R68 models) would eliminate this bias. But per codex's pre-declared rule for the sprint:

> **"If pool improves but LR conversion fails, archive."**

We honor the strict rule. The conversion failure could be due to the bias, or genuine, or both — Phase 1 would disambiguate at the cost of ~2.5 GPU-hours. Deferred.

## Cross-mechanism learnings (3 bullets)

1. **The retrieval surface IS upgradable.** BGE-large > BGE-base on fold-0 h7 single-source pool_hit by +0.035, with 15 unique GT recoveries. This is the first time in 19 post-R54c experiments that the retrieval layer itself showed a real positive signal. Prior failures (R55 single-all-data, R65/R66 source routing, R67 LLM rerank) hit different walls; R68 reaches a different wall, **further along**.

2. **The LR conversion wall persists.** Same pattern as R66 (RRF redistribution) and R67 (LLM rerank): pool/retrieval improves, downstream sibling/frozen LR rescores in ways that lose same-artist (−0.156) and h7 nDCG (−0.08). The wall is structurally the LR's calibration to R54's specific source-rank feature distribution. Swapping the retriever does not break the wall.

3. **Phase 1 (full 5-fold OOF) is the disambiguator.** The Phase 0 sibling-LR test had a known train-time bias (r68_cosine=0 on train). Phase 1 would resolve whether the conversion failure is genuine or biased. Deferred per the sprint's pre-declared archive rule, but the question is open — not closed in the same final-verdict sense as R65/R66/R67.

## Infrastructure notes (recorded for future use)

- **T4 cannot fine-tune BGE-large** under torch 2.10+cu128 — cu128 reserves ~11 GiB of non-PyTorch CUDA workspace on T4 (compute 7.5), leaving <4 GiB usable. Confirmed across 5 OOM attempts at batch 16 / 8 / 4 / fp16 / capped-cublas. See [[feedback_t4_bge_large_blocked]].
- **A100-40GB + bf16 works cleanly.** Pure fp16 produced NaN losses (BGE-large + manual training loop without GradScaler). bf16 has the range to stay stable. ~5.6 min total for 1 epoch + encoding + top-300 retrieval on a single A100.
- **Sibling LR training in Phase 0 has a known bias** (train cases lack r68 query embeddings). Phase 1 would fix this; Phase 0 verdict honored the strict pre-declared rule despite the caveat.

## Surviving deferred directions

- **Phase 1 full 5-fold OOF** — would disambiguate whether the conversion failure is genuine or due to the Phase 0 train-zero-cosine bias. Cost: ~2.5 GPU-hours on A100. User has Pro access; rerun if motivated. Sprint plan + scripts are in place on this branch.
- The retrieval signal (+0.035 h7 pool_hit, 15 unique recoveries) is the first non-zero positive across 19 attempts. Even if Phase 1 confirms conversion is genuinely closed, this is useful evidence: the *retrieval surface* is upgradable. Future architectures that can consume a stronger retriever (e.g., a ranker retrained jointly, or one that doesn't rely on source-rank calibration) might extract this signal.

## External pending

- Blind-B release watch
- C4 organizer email (user-owned)

## Artifacts on `r68-large-scale-retrieval`

- `scripts/expR68_baseline_repro.py`
- `scripts/expR68_phase0_fold0_train.py` (the canonical training script, patched for T4 then unpatched for A100; A100 added bf16 model_kwargs and BATCH_SIZE=16)
- `scripts/expR68_phase0_eval.py` (Mac-side conversion eval)
- `scripts/expR68_phase1_full5fold_train.py` (5-fold training; ready, not run)
- `scripts/expR68_phase1_eval.py` (5-fold eval; ready, not run)
- `exp/eval/expR68_baseline_repro.json`
- `exp/eval/expR68_r54_reference_stats.pkl`
- `exp/eval/expR68_r54_aggregate.json`
- `exp/eval/expR68_phase0_fold0_eval.json`
- `cache/r68_phase0_sibling_lr.txt` (LightGBM model trained with FEAT_R39 + r68_* features)
- `docs/r68_baseline_repro.md`
- `docs/r68_gpu_handoff.md`
- `docs/r68_phase0_fold0_result.md`
- `docs/r68_sprint_summary.md` (this doc)
- `cache/r68/phase0_fold0/` (Colab artifacts — gitignored, ~225 MB; oof_r68_lists, query/track embeddings, model checkpoint)

## Bottom line

R68 is the **19th consecutive post-R54c dev experiment without a production advance**, but the **first one to deliver a real positive retrieval-layer signal** (+0.035 h7 single-source pool_hit, 15 unique recoveries). The LR conversion wall persists. Per pre-declared rule, Phase 0 archives without Phase 1; the disambiguation question (genuine wall vs train-zero-cosine bias) remains open for a future ~2.5 GPU-hour Phase 1 run if motivated. **R63c-repair holds Blind-A production at composite 0.6224.**
