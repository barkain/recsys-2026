# R79 Sprint — Top-Rank Discriminator Retriever — ARCHIVE_SPRINT

**Date:** 2026-05-23

## Verdict

ARCHIVE at Phase 0B. Per the predeclared rule: "If 0B fails, archive. No
A100-heavy Phase 0C / Phase 1 / Phase 2." Total A100 cost: **~$3** (well
under the $10 budget).

## Hypothesis tested

Train a BGE-large retriever with InfoNCE loss using **hard negatives from
R54c's top-20 false positives** (not random negatives). The intuition was:
R68's standard contrastive training found GT candidates at ranks 50-300
but couldn't surface them to top-20. R79 should learn to push GT *above*
the wrong R54c top-20 picks, directly attacking the conversion failure.

## Phase 0A (Mac, no cost)

- Built training pairs for all 8000 cases (5 sibling LRs, fully OOF top-20)
- Locked baseline: OOF R54c standalone fold-0 h7 = 0.2226, all = 0.2123
- 41.2% of cases have GT in OOF R54c top-20

## Phase 0B (Colab A100, ~$3)

- Fine-tuned BGE-large-en-v1.5 with InfoNCE + 16 hard negatives per case
  (R54c top-20 minus GT)
- 2 epochs, lr=1e-5, batch_cases=8, bf16 autocast
- Training time: 786 s (~13 min)
- Loss decreased 2.81 → 2.70 (poor convergence, only one decimal point)

## Result — catastrophic collapse

Standalone fold-0 eval against OOF R54c baseline:

| subset | n | baseline | R79 | Δ |
|---|---:|---:|---:|---:|
| all_fold0 | 1600 | 0.2123 | **0.0014** | −0.2109 |
| h7 | 200 | 0.2226 | **0.0000** | −0.2226 |
| same_artist | 531 | 0.4475 | **0.0000** | −0.4475 |
| diff_artist | 1069 | 0.0955 | 0.0021 | −0.0934 |
| h7_same | 83 | 0.4313 | 0.0000 | −0.4313 |
| h7_diff | 117 | 0.0745 | 0.0000 | −0.0745 |

- **h7 nDCG@20 = 0.0000** — not a single h7 case has GT in R79's top-20
- recovered = 0, lost = 83 (h7)
- top-1 churn 1600/1600 (100% disagreement with R54c)
- top-20 overlap: 0.01/20 (essentially zero)

ALL five gates fail. R79's standalone top-20 is essentially random with
respect to GT.

## Diagnosis

**Hard-negative collapse.** The model trained only on (GT vs 16 R54c-top-20
hard negatives) never saw the other ~47K random catalog tracks. At eval
time it must rank GT against 47K background candidates and has no signal
to discriminate. It learned local discrimination among "plausible
candidates R54c surfaces" without learning global structure.

The design doc explicitly mentioned mitigation: "+ M random negatives
(background)". Phase 0B simplified to hard-negs-only to limit cost. That
simplification was the fatal flaw.

Historical precedent matches: [[feedback_no_hard_negatives]] —
R23/R23a tried hard-negative mining and it hurt. R79 confirms the
mechanism more clearly: hard-only training is not just suboptimal, it
collapses the embedding space such that the model can't even maintain
baseline retrieval quality.

## What this rules out

- The R79 architecture as written (hard-negative-only fine-tune) is dead.
- A fix would require either (a) mixed positive/hard/random InfoNCE with
  careful weighting, OR (b) starting from a frozen good base and only
  fine-tuning a small head. Both add complexity.
- Even with that fix, the design risk remains: hard negatives are
  semantically very close to GT in this music domain, and contrastive
  training can over-push them away, hurting general retrieval.

Per predeclared rule, R79 archives. We do NOT attempt a Phase 0B-v2 with
random negatives without a fresh design review.

## Implications for the sprint

All credible nDCG-lift paths are now closed:

| direction | sprint | result |
|---|---|---|
| Encoder upgrade (BGE-large) | R68 | GT found at rank 50-300, not rescuable to top-20 |
| Pool admission | R59 C3, R72 | Only 2 h7 cases recoverable |
| Substitution LR | R68.1 | Δh7=−0.08 (artifact + drift) |
| Joint LR addition | R70 | Δh7=−0.08 (essentially identical to substitution) |
| LR sibling reproducibility | R70b | "Wall" is OOF artifact + recipe drift |
| Stacker over R54c top-K | R71, R76 | No signal beyond R54c |
| Naked rerank (rules/LightGBM/LLM/cross-enc) | R56/R58/R67/R69 | All regress |
| Neural residual ranker | R76 | Same-artist collapse |
| **Hard-negative retriever** | **R79** | **Catastrophic collapse** |

## Production state unchanged

R78 holds at composite **0.6302**, position #4. R79 made no submission;
no production impact.

## Honest recommendation

The retrieval/ranking direction is empirically exhausted for this sprint.
Possible remaining moves:

1. **Freeze R78 and defend.** Accept #4.
2. **Try R79 Phase 0B-v2 with random negatives** — would require fresh
   design + ~$10 + uncertain outcome. Honest estimate: <20% probability
   of clearing gates.
3. **Pivot to Blind-B** if the window has opened.
4. **Final response-side micro-push** — risk of regression given LLM 4.90
   ceiling and LexDiv near ceiling. Likely not worth a slot.

## Files

- `scripts/expR79_phase0a_build_data.py` — Mac data prep
- `scripts/expR79_phase0b_train_and_eval.py` — Colab A100 train + eval
- `cache/r79/training_pairs.pkl` — 8000 cases × hard negs
- `cache/r79/eval_baseline.json` — OOF R54c baseline
- `exp/eval/expR79_phase0a_stats.json` — Phase 0A audit
- `exp/eval/expR79_phase0b_result.json` — Phase 0B result + diagnosis
- `docs/r79_top_rank_retriever_design.md` — design + phased plan
- `docs/r79_phase0a_data_audit.md` — Phase 0A audit
- `docs/r79_sprint_summary.md` — this file

## Forward pointer

R79 closes the hard-negative retriever path. Future GPU-heavy nDCG work
moves to **R80 listwise neural ranker over top-300** — a learned
replacement for the R54c LR's top-rank decision surface, NOT a residual
or a fresh retriever. See `docs/r80_listwise_ranker_design.md`.
