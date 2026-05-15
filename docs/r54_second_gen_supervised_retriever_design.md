# R54: Second-Generation Supervised Retriever

**Core hypothesis:** A supervised retriever trained with structured query/context text and richer target text can recover Bucket D/E cases better than R21/R22b. The main novelty is query/positive construction, not data volume.

**R54 is NOT "R22b again."** R22b already proved that scaling to 15,199 train sessions (121K pairs) with the same query/positive format gives only modest standalone improvement (+3.1% hit@300) and fails pool/h7 gates. Data volume alone is insufficient.

## 1. Prior Art and Lessons

### R21 (production retriever)
- Fine-tunes BGE-base on dev sessions with in-batch contrastive loss (temperature=0.05).
- Query: last 3 user utterances concatenated (`build_query_text`).
- Positive: `"{name} by {artist}. Album: {album}. Tags: {tags[:10]}"`.
- Loss: in-batch InfoNCE only. The script constructs `hard_examples` but never uses them in the training loop — production R21 is purely in-batch contrastive.
- OOF: full 5-fold for production LambdaRank integration (`grouped_session_folds(seed=0)`).
- Strongest single source in R39's 7-source RRF pool.

### R22b (all-train scaling — already done, negative)
- Trained BGE-base on 15,199 train-split sessions (121,592 pairs), 1 epoch, same query/positive format as R21.
- Results: standalone hit@300 = 0.564 vs R21's 0.533 (+3.1%). Unique recoveries: +602, lost: -415, net: +187.
- Pool hit@300: 0.602 vs R21 baseline 0.598 (Δ = +0.004). **Gate FAIL** (threshold 0.620).
- Proxy nDCG: 0.468 (essentially unchanged from R39 0.480).
- **Conclusion:** More data with same text format does not pass gates.

### R23a (hard negatives — already done, negative)
- Added explicit hard negatives: V3 pool, BM25, same-artist, popular/tag-matched (8 negatives per query).
- **Result: damaged retrieval quality.** Worse than R21 in-batch baseline.
- **Conclusion:** Hard/semi-hard negatives hurt this task. Do not make them a core phase.

### R41a–R52 (all negative)
- Ranking features, retrieval depth, multi-query, cross-encoder, multimodal retrieval — all failed to improve h7 beyond R39.
- Best post-R39 result: +0.005 (R52b audio quota), below +0.010 gate.
- The retrieval pool is the binding constraint.

## 2. What R54 Changes

The only untested axis is **how queries and positives are constructed**. R21/R22b both use the same flat text format. R54 tests whether structured, context-aware text changes what the model learns to retrieve.

### 2A. Structured Query Construction (Main Novelty)

R21 query: `" ".join(last_3_user_utterances)` — flat concatenation, no structure.

R54 query uses structured sections:

```
[QUERY] {current user utterance}
[HISTORY] {last 2 user utterances}
[CONTEXT] {last 2-3 played tracks as "track by artist"}
```

Rationale: when a user says "something like what I just heard", the played-track context is the real query. When they say "I want jazz", the explicit text is. Structured delimiters let the model attend to the right section.

**Ablation plan (fold-0 only):**
- Q0: R21 baseline (last 3 user utterances, flat)
- Q1: Q0 + played-track context appended
- Q2: Q1 with structured `[QUERY]`/`[HISTORY]`/`[CONTEXT]` delimiters

Only test Q3 (preference summary from conversation) if Q1 or Q2 shows improvement.

### 2B. Richer Positive (Track Text) Construction

R21 positive: `"{name} by {artist}. Album: {album}. Tags: {tags[:10]}"`.

R54 candidate enrichment (treat as ablation, not assumed helpful):

```
{name} by {artist}. Album: {album}. Year: {year}. Tags: {tags[:15]}
```

- `year`: release year (coverage TBD from Phase 0 inventory).
- Tags expanded from 10 to 15.
- Duration bucket: test separately only if year helps.

Year/duration features failed as LambdaRank features (R49C), but embedding-space semantics are different — the model may learn "user who listened to 2020s tracks wants 2020s tracks" without overfitting.

**Do NOT include:** ISRC, album_artist (redundant), popularity (leaks distribution).

### 2C. Training Data Strategy

R54 uses the same data as R21 by default (dev sessions, per fold). Adding train-split data is a **control comparison against R22b**, not the main novelty.

- **Phase 2 (main):** Structured query on dev data only, same data volume as R21.
- **Phase 3 (conditional):** Add all-train data only if Phase 2 shows improvement, to measure whether structured queries + more data compounds. Or as a clean R22b comparison if Phase 2 is flat.

### 2D. Negatives: In-Batch Only (Default)

R21 production uses in-batch contrastive. R23a proved hard negatives damage this task.

R54 default: **in-batch InfoNCE only**, same as R21 production.

**Optional Phase 5:** If earlier phases pass gates, test mild pool-based negatives (2 per positive from OOF pool) as a late experiment. Do not make the main experiment depend on this.

### 2E. Training Hyperparameters

| Parameter | R21 | R54 Default |
|-----------|-----|-------------|
| Model | BGE-base-en-v1.5 | BGE-base-en-v1.5 |
| Epochs | 2 | 2 |
| Batch size | 32 | 32 |
| Learning rate | 2e-5 | 2e-5 |
| Temperature | 0.05 | 0.05 |
| Max seq length | 256 | 256 |
| Negatives | in-batch only | in-batch only |
| Warmup | 0 | 10% of steps |

Only sweep hyperparameters after query/positive changes show signal. BGE-base only — if R54 fails with BGE-base, model scaling is a separate R55.

### 2F. Same-Session Positives (Conditional)

Multi-turn sessions may have earlier recommendations the user liked. Valid only when the user explicitly responded positively and the track is in catalog.

**Risk:** Coverage is likely very low. Investigate in Phase 0. Skip if < 50 valid cases total.

## 3. OOF Protocol (Non-Negotiable)

Inherits all 6 rules from R49 operating standard. Additionally:

1. **Fold-0 smoke first.** All phases run fold-0 only until signal is confirmed. Full 5-fold only if fold-0 shows meaningful improvement.
2. **Same folds as R21/R39:** `grouped_session_folds(sessions, seed=0)`.
3. **Train-split data is OOF-safe** (disjoint from dev sessions).
4. **No production model in dev evaluation.**
5. **Fold manifests:** fold indices, model path, training data, embedding cache path.
6. **OOF retrieval lists** for dev evaluation, never production lists.

## 4. Phased Execution

| Phase | What | Gate | Scope |
|-------|------|------|-------|
| 0 | Diagnostics: data inventory, metadata coverage, same-session positive count | — | No training |
| 1 | Fold-0 R21 reproduction under R39 h7 evaluation | h7 must reproduce R39 fold-0 baseline | Fold-0 only |
| 2 | Fold-0 structured-query retriever (Q1 or Q2), same data as R21 | Δh7 >= +0.010 vs Phase 1, OR Bucket E/D recovery clearly above R21 with h7 non-negative | Fold-0 only |
| 3 | Add all-train data (conditional: only if Phase 2 helps, or as R22b comparison if Phase 2 flat) | h7 non-negative; compare to R22b result | Fold-0 only |
| 4 | Richer positive text (+ year, tags[:15]) | Δh7 vs best prior phase | Fold-0 only |
| 5 | Optional: mild pool negatives (2 per positive from OOF pool). Only if Phases 2-4 pass. | Δh7 non-negative | Fold-0 only |
| 6 | Full 5-fold CV5 on best config from above | Δh7 >= +0.010 (full CV5 gate) | All 5 folds |

**Stop conditions:**
- Phase 2 shows zero or negative fold-0 h7 delta AND no meaningful Bucket E/D recovery → structured query hypothesis falsified. Archive R54.
- Phase 3 reproduces R22b's weak result → data volume confirmed insufficient, move on.
- No phase passes fold-0 gate → do not run full 5-fold.

## 5. Evaluation

### 5A. Fold-0 Evaluation (Phases 1-5)

Train on folds 1-4 (+ train-split if applicable), evaluate on fold-0.

Integrate R54 fold-0 retrieval lists into R39 RRF pipeline as 8th source. Run LambdaRank on fold-0 val cases. Compare fold-0 h7 against R39 fold-0 baseline.

**Fold-0 gate:**
- h7 Δ >= +0.010, OR
- Bucket E/D recovery clearly above R21 with h7 non-negative

### 5B. Full 5-Fold Evaluation (Phase 6)

Only after fold-0 passes. Run all 5 folds, aggregate CV5 h7.

**Full gate:**
- Δh7 >= +0.010
- Unique Bucket E recovery >= 20 cases
- Recovered/lost net positive
- No same/diff artist regression
- OOF-clean (all R49 rules)

### 5C. Bucket Recovery Diagnostics

For each fold evaluated, classify dev cases into buckets A-E from R39 baseline. Report:

| Metric | Definition |
|--------|------------|
| E→pool | Bucket E cases where R54 retrieves GT in top-300 |
| D→pool | Bucket D cases where R54 retrieves GT at better rank |
| E→hit@20 | Bucket E cases where R54 brings GT into LambdaRank top-20 |
| Unique vs R21 | Cases retrieved by R54 but NOT by R21 |

## 6. Integration with R39 Pipeline

R54 retrieval lists integrate as a new RRF source:

```
R39 sources:  A, B, C, D, F, ALS, R21  (7 sources)
R54 config:   A, B, C, D, F, ALS, R21, R54  (8 sources)
```

R54 RRF weight: start at 1.0, sweep 0.5–2.0.

New LambdaRank features: `r54_rank`, `r54_score`, `r54_in_pool` (34 + 3 = 37 total).

## 7. Compute Budget

| Phase | Estimated Time | Notes |
|-------|---------------|-------|
| Phase 0 (diagnostics) | ~30 min | Data inventory, no training |
| Phase 1 (fold-0 R21 repro) | ~1h | 1 fold, train + encode on CPU |
| Phase 2 (fold-0 structured query) | ~1h | Same data volume, new text format |
| Phase 3 (fold-0 + train data) | ~2h | 3x pairs, 1 fold |
| Phase 4 (fold-0 richer positive) | ~1h | Same data, new track text |
| Phase 5 (fold-0 optional negatives) | ~1h | Only if prior phases pass |
| Phase 6 (full 5-fold) | ~5-8h | Only if fold-0 passes |

Total if all phases run: ~12-15h. Fold-0-first design saves ~4x compute vs running full 5-fold at each phase.

**macOS constraint:** CPU only (MPS unreliable for sentence-transformers). Encoding is the bottleneck (~100k tracks per fold).

**Caching:** Cache catalog embeddings per fold. If only query construction changes (Phase 2 vs Phase 4), track embeddings from the same model can be reused.

## 8. What NOT to Do

1. **Do not treat data volume as the main hypothesis.** R22b already tested this. Structured text is the novelty.
2. **Do not make hard negatives a core phase.** R23a proved they damage this task. Optional late experiment only.
3. **Do not run full 5-fold before fold-0 passes.** Fold-0 smoke is mandatory.
4. **Do not change the base model.** BGE-base only. Model scaling is R55.
5. **Do not skip Phase 1 reproduction.** Fold-0 R21 baseline must match before modifications.
6. **Do not use production R21 model or lists in dev evaluation.**
7. **Do not import torch and lightgbm in the same script phase.** Use `--phase` argument.
8. **Do not train on dev-fold-val.** Evaluation only.
9. **Do not combine all changes at once.** Incremental ablation per §4.
10. **Do not use same-session positives without validating coverage (Phase 0).**

## 9. Success Criteria

| Criterion | Threshold | Required? |
|-----------|-----------|-----------|
| Fold-0 Δh7 vs R39 | >= +0.010 OR E/D recovery + h7 non-neg | YES (smoke gate) |
| Full CV5 Δh7 vs R39 | >= +0.010 | YES (final gate) |
| Unique Bucket E recovery vs R21 | >= 20 cases | YES (diagnostic) |
| Recovered/lost net | positive | Preferred |
| same/diff artist regression | No material regression | YES |
| OOF-clean | All 6 rules from R49 | YES |

If R54 passes all gates, build production model (train on all dev + train-split) and submit to blind.

## 10. First Step

**Phase 0: Data inventory and diagnostics.** No model training.

1. Count train-split sessions and extractable (query, GT) pairs. Verify the known 15,199 sessions / 121K pairs from R22b.
2. Verify GT track coverage in catalog metadata.
3. Count same-session positive candidates and assess coverage.
4. Inventory structural metadata fields (year, duration) coverage for enriched track text.
5. Output: `exp/eval/expR54_phase0_diagnostics.json`.
