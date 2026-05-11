# R51: Fine-Tuned Cross-Encoder Reranker on R39 Pool

## Background

R39 album-aware LambdaRank (h7=0.24298) is production best. Everything after R39 failed:

| Experiment | Approach | Result |
|---|---|---|
| R41a | Rare-tag features | Overfit, blind failed |
| R46 | R21-depth retrieval | Contaminated; OOF-clean only +0.002 |
| R49C | Structural metadata (duration, year) | All negative |
| R49A | Multi-query RRF | Diluted R21, negative |
| R50 | View-aware features | Near-zero, all configs fail gate |

LambdaRank cannot distinguish correct candidates from plausible wrong ones in the R39 pool. The missing capability is direct joint text reasoning over conversation + history + candidate metadata.

R29 tested zero-shot cross-encoder and failed -- but that does NOT invalidate a fine-tuned cross-encoder. R33c tested MLP over embeddings/features, not text-level interaction. R51 is genuinely different.

## 1. Objective

Train a fine-tuned cross-encoder that scores (conversation, candidate) pairs. Use R39 pool, no expansion. Blend cross-encoder scores with LambdaRank via z-score interpolation. Start with fold-0 smoke test only.

## 2. Architecture

```
Input:
  Conversation: [user messages, assistant responses]
  Recent tracks: [last 5-8 played tracks with title/artist]
  Candidate: title | artist | album | tags

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (22M params) or similar small reranker
Output: relevance score (single float)
```

## 3. Training Data Construction (smoke scope)

**Smoke test is intentionally small:**
- Train cases: **hist_5/6/7 only** from folds 1-4 (not all 6400 cases)
- Candidate depth: **top-100** from R39 pool
- Train on folds 1-4, evaluate fold 0
- Fold split: `grouped_session_folds(seed=0)` -- same as all previous experiments
- Skip cases where GT is not in top-100 pool (no positive available)

**Positive:** GT candidate (1 per case).

**Negatives:** 15 per positive.

| Type | Count | Purpose |
|---|---|---|
| LambdaRank top-5 wrong | 5 | Hard negatives -- what LR ranks high but is wrong |
| GT neighbors +/-3 positions | 5 | Medium negatives -- similar rank region |
| Random pool candidates | 5 | Easy negatives -- calibration anchors |

**Training set size:** ~3000 hist_5+ cases × ~60% pool hit × 16 pairs = ~29K pairs. Manageable on CPU.

## 4. Input Format

```text
Conversation:
User: I'm looking for something upbeat
Assistant: How about some pop music?
User: Yeah, something danceable from the 2010s

Recent tracks:
1. Shape of You | Ed Sheeran | / | pop, dance
2. Blinding Lights | The Weeknd | After Hours | synth-pop, 80s

Candidate:
Levitating | Dua Lipa | Future Nostalgia | pop, dance, disco
```

Truncation rules:
- Conversation: last 3-5 turns
- Track history: last 8 tracks
- Tags: 5 per track

## 5. Model and Loss

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params)
- **CPU only.** No MPS — Mac has crashed repeatedly on MPS transformer workloads.

**Loss: listwise cross-entropy (not BCE).**

BCE with 1:15 negatives often learns calibration, not ordering. Use listwise loss instead:

```python
# For each query case:
#   logits = model([query+GT, query+neg1, query+neg2, ...])  # shape: [16]
#   label = 0  (GT is first)
#   loss = cross_entropy(logits, label)
```

This is a listwise ranking loss within the sampled candidate set. Implement with `AutoModelForSequenceClassification` and a custom training loop, not `CrossEncoder.fit()`.

| Param | Value |
|---|---|
| Learning rate | 2e-5 |
| Batch size | 1 query (16 candidates per query) |
| Epochs | **1 (hard stop — see gates)** |
| Warmup | 10% of steps |
| Grad accumulation | 4 (effective batch = 4 queries) |
| Device | CPU only |

## 6. Scoring and Blending

At inference on fold-0:

1. Score all top-100 pool candidates with cross-encoder per case
2. Z-score normalize both LambdaRank and cross-encoder scores (per case)
3. Blend: `final_score = zscore(lr_score) + beta * zscore(ce_score)`
4. Sweep beta in {0.05, 0.1, 0.2, 0.5, 1.0}
5. Rank by final_score, take top-20

## 7. OOF Protocol

- Train on folds 1-4 (hist_5+ cases only), evaluate fold 0 ONLY
- Do NOT evaluate on folds used for training
- LambdaRank scores for fold 0 come from R39 CV5 model (fold-0 held out)
- If fold-0 smoke passes gate: proceed to full 5-fold CV5 in R51 Phase 2

## 8. Baselines and Evaluation (fold-0 only)

**Three baselines required** to isolate CE contribution from candidate truncation:

| Baseline | Description |
|---|---|
| `baseline_full_pool300` | R39 LambdaRank on full 300-candidate pool (the production baseline, h7=0.24298) |
| `baseline_top100_only` | R39 LambdaRank ranking only top-100 candidates (may lose some hits) |
| `CE_blend_top100` | CE + LR blend on top-100 candidates (the experiment) |

**Metrics:**

| Metric | Description |
|---|---|
| `h7_all` | nDCG@20 on all fold-0 hist_7 cases |
| `h7_gt_in_top100` | nDCG@20 only on cases where GT is in CE-scored top-100 |
| same/diff | Artist continuity metrics |
| recovered/lost | Cases that flip miss-to-hit and hit-to-miss vs baseline_top100_only |
| top-20 churn | Membership change count vs baseline_top100_only |

If `h7_gt_in_top100` improves but `h7_all` does not: the model works but candidate depth is too narrow.
If neither improves: stop.

## 9. Gates

| Gate | Threshold |
|---|---|
| Fold-0 `h7_gt_in_top100` after epoch 1 | Must improve over baseline_top100_only |
| **Hard stop:** if no improvement after epoch 1 | Do NOT continue to epochs 2-3 |
| Fold-0 delta-h7 (all cases) | >= +0.010 to proceed to Phase 2 |
| Recovered vs lost | Recovered meaningfully > lost |
| same/diff | Non-negative |

If fold-0 fails: stop. No full CV5. No larger models.

## 10. Risks

1. **Overfitting.** Small model on small dataset may memorize. Monitor train loss convergence.
2. **Score calibration.** Z-score blending assumes roughly normal distributions. May need alternative normalization.
3. **Negative sampling.** Too-easy negatives waste compute. Too-hard negatives may confuse the model.
4. **CPU training speed.** ~29K pairs with 22M param model at batch=1(×16) on CPU. Estimate ~1-2 hours for 1 epoch.
5. **Candidate truncation.** Top-100 loses some GT hits vs top-300. The `baseline_top100_only` comparison isolates this.

## 11. Expected Cost

| Phase | Time |
|---|---|
| Data preparation | ~10 min |
| Training (CPU, 1 epoch) | ~1-2 hours |
| Inference (fold-0, ~200 cases × 100) | ~15 min |
| Evaluation + beta sweep | ~5 min |
| **Total** | **~1.5-2.5 hours** |

## 12. What This Tests

Can a fine-tuned text-level scorer learn to distinguish correct from plausible-wrong candidates in the R39 pool, where LambdaRank's tabular features cannot?

- **If yes:** proceed to full CV5, then blind candidate.
- **If no:** ranking is not the bottleneck. The remaining gap is in retrieval or in cases where GT is not in pool at all.

## 13. R51 Phase 0 Results: ARCHIVED AS FAILED (2026-05-10)

**Gradient path:** Valid (grad_norm 200-230 across all LRs).

**LR sweep (200-example mini-overfit test):**

| LR | Final Loss | Below Random (2.77)? |
|----|-----------|---------------------|
| 1e-5 | 2.766 | Barely |
| 5e-5 | 2.863 | No |
| 1e-4 | 2.817 | No (diverged) |

**Full training (lr=2e-5, 1335 examples, 1 epoch):** Loss flat at 2.78-2.83 throughout. Eval best beta=0.05: h7_gt_in +0.002 (noise).

**Conclusion:** MS-MARCO MiniLM cross-encoder + structured text + listwise CE cannot even overfit 200 examples. The model cannot learn to distinguish correct from wrong music recommendations in this formulation. Scaling is unjustified.

**If CE is revisited, needs different formulation:**
- Start from BGE/R21-style embedding model or music/text model, not MS-MARCO reranker
- Use candidate metadata in R21 text format
- Add structural tokens: album match, artist match, source ranks
- Prove 200-example overfit before any full run

**Stop:** No full CV5, no blind, no larger models in this formulation.

## 14. If R51 Fails (directions)

Ranking improvement may be exhausted for tabular and text-level approaches. Possible directions:
- New retrieval source (not R21-based)
- LLM/LexDiv composite optimization
- Session-level data augmentation
