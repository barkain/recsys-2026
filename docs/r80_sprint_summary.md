# R80 Sprint — Listwise Transformer over Top-300 — ARCHIVE_SPRINT

**Date:** 2026-05-23

## Verdict

ARCHIVE at Phase 0B per predeclared rule. All five gates fail. ~$3 spent
(under $15 budget). No Phase 1 / Phase 2 spend.

## What we built

Phase 0A (Mac, $0): 
- 1600 fold-0 cases × 300 candidates
- 47-dim numeric features per candidate (37 LR + R54c score/rank + 3 R68 + 5 semantic scalars)
- 1024-dim BGE-large track + query embeddings (referenced via catalog, not duplicated)
- Compact 36 MB pkl.gz + 96 MB fp16 catalog (fits in git, no Drive needed)

Phase 0B (A100, $3):
- ListwiseRanker: 4-layer Transformer (d=256, heads=8, ff=512)
- ~2.65M params, listwise softmax CE on cases with GT in top-300 pool (60%)
- 5-way inner CV within fold-0, bf16 autocast, 20 epochs
- Training: ~6 min on A100, loss 3.95 → 2.45 (healthy convergence)

## Result

| metric | OOF R54c baseline | R80 listwise | Δ |
|---|---:|---:|---:|
| all_fold0 | 0.2110 | 0.1808 | **−0.0302** |
| h7 | 0.2213 | 0.1953 | **−0.0260** |
| same_artist | 0.4447 | 0.4150 | **−0.0296** |
| diff_artist | 0.0949 | 0.0644 | **−0.0305** |

- h7 recovered = **6** (real signal — GT pulled into top-20 from R54c's miss)
- h7 lost = 17 (model demoted GT R54c had right)
- net = **−11**
- top-1 churn /80 = 50.6 (>>25 cap)
- top-20 overlap = 11.72/20 (<14 floor)

All five gates fail. Verdict: ARCHIVE.

## Diagnosis

R80 trained cleanly (loss converged) and produced REAL signal (recovered 6
h7 cases that R54c missed). But it lost 17 cases it had right. Net
negative across all subsets, including same-artist (−0.0296).

Same mechanism as R76's residual MLP failure, but less severe:
- R76 (small residual): same-artist Δ = −0.0455
- R80 (full listwise transformer over top-300): same-artist Δ = −0.0296

The richer architecture (transformer + cross-candidate attention) does
extract more signal — recovery=6 vs R76's recovery=0. But it still
over-weights semantic similarity over R54c's calibrated structural
features. Same fundamental issue: any neural model on this feature stack
chases semantic neighbors and breaks the LR's same-artist calibration.

## What R80 confirms

The neural ranker direction on top of the R54-stacked top-300 pool is
**empirically dead** on this dataset/feature set. Three independent
architectures tested:

| sprint | architecture | h7 Δ | same-art Δ |
|---|---|---:|---:|
| R71 (LightGBM stacker on top-30) | tree-based, 5 feats | −0.005 | −0.004 |
| R76 (neural residual MLP, top-30) | tiny MLP residual | −0.013 | −0.046 |
| **R80 (listwise transformer, top-300)** | full transformer | **−0.026** | **−0.030** |

All three close gates, all three lose net h7. The fundamental issue is
not the architecture — it's that the available features (LR + R68 BGE +
semantic scalars) don't carry enough new signal beyond what R54c LR
already extracts to overcome the cost of disrupting R54c's calibration.

## State of all paths

| direction | sprint(s) | result |
|---|---|---|
| Encoder upgrade (BGE-large retriever) | R68 | hit@300 ceiling, top-30 unreachable |
| Pool admission | R59 C3, R72 | only 2 h7 cases rescuable |
| LR substitution/addition | R68.1, R70 | OOF artifact |
| LR sibling reproducibility | R70b | "wall" was artifact + drift |
| Tree stacker | R58, R71 | no signal |
| LLM/cross-encoder rerank | R67, R69 | catastrophic |
| Neural residual (small MLP) | R76 | same-artist collapse |
| Hard-negative retriever | R79 | catastrophic collapse |
| **Listwise transformer (large)** | **R80** | **clean training, still loses** |

All credible nDCG-lift directions empirically exhausted.

## Production state unchanged

R78 holds at composite **0.6302**, position #4. No production impact.

## Honest recommendation

The retrieval/ranker direction is empirically closed on this dataset and
feature set. R80 was the cleanest architecture test possible — large
transformer, full feature stack, cross-candidate attention, listwise loss.
It still cannot beat R54c on standalone top-20 nDCG.

Possible remaining moves:
1. **Freeze R78 and defend.** All credible nDCG paths empirically closed.
2. **Try Phase 1 5-fold OOF anyway** — would confirm whether R80's small
   recovery signal (6 cases) persists across folds. ~$10-15. <20% chance
   of changing the verdict.
3. **Wait for Blind-B** if competition adds new tracks/data.
4. **End the sprint.** Production at 0.6302 (#4) is solid; further work
   has near-zero expected value given exhausted paths.

## Files

- `scripts/expR80_phase0a_build_listwise_dataset.py` — Mac data prep
- `scripts/expR80_phase0b_train_listwise.py` — Colab A100 train + eval
- `cache/r80/listwise_dataset_fold0.pkl.gz` — 36 MB compressed dataset
- `cache/r80/catalog_track_embs_fp16.npy` — 96 MB fp16 catalog
- `cache/r80/eval_baseline.json` — OOF R54c baseline
- `exp/eval/expR80_phase0a_stats.json` — Phase 0A audit
- `exp/eval/expR80_phase0b_result.json` — Phase 0B verdict + diagnostics (on Colab)
- `docs/r80_listwise_ranker_design.md` — design
- `docs/r80_phase0a_audit.md` — Phase 0A audit
- `docs/r80_phase0b_result.md` — Phase 0B result (on Colab)
- `docs/r80_sprint_summary.md` — this file
