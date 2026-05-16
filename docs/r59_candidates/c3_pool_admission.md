## 1. New information added beyond R54c

R54c's pool admission is `weighted_rrf`: for each source list, a track gets
`source_weight / (20 + source_rank)`, summed over sources, then the top 300
tracks become the LR candidate pool. The weights are fixed:
`A=1.0, B=1.0, C=1.0, D=0.5, F=1.0, ALS=1.0, R21=1.0, R54=1.0`
(`scripts/expR55_post_refresh_decomp.py`; implementation in
`scripts/expF1_cfbpr_retrieval.py`).

That is a strong hand-built prior, but it is still only a monotone rank-sum.
It cannot learn that different evidence shapes imply different GT-membership
probability. A learned pool-admission model could capture:

- Cross-source interaction: a track at ranks 80 and 95 in R54/R21 plus ranks
  140, 160, 180 in BM25/ALS may be better than a track ranked 18 in one noisy
  source and absent elsewhere.
- Source-combination reliability: `R54 + C`, `R21 + B`, or `ALS + R54` may
  have different precision than either source alone at the same RRF total.
- Session-conditional source trust: short-history, long-history,
  exploration-mode, and same-artist-heavy sessions may need different
  admission priors.
- Raw score shape where available: R54 Phase 2 carries cosine scores, ALS has
  scores, and ranks can be turned into depth bins, min/max rank, agreement
  counts, and rank dispersion. RRF throws most of that structure away.
- Nonlinear thresholds: "present in 4 sources, all deep" can be admitted even
  when the fixed RRF total is below the top-300 cutoff, while "one shallow
  source only" can be rejected.

This is different from R56 and R58 only if the boundary stays before final
ranking. R56 failed because it tried to protect source-strong candidates after
LR had already ranked them; even the true `diff_artist` ORACLE lost 26 HITs to
recover 15 DEMOTED at K=1. R58 failed because a learned top-50 specialist
reshuffled LR's already-calibrated ordering; every beta > 0 regressed h7 and
same-artist nDCG. A pool-admission model targets the 1163 POOL_MISS cases
where LR never got to see the GT in the pool, not the 1628 DEMOTED cases where
LR saw the GT and demoted it.

The strict limitation: with the current source union, this mechanism cannot
recover UNREACHABLE cases. If the GT is in no current source list, there is no
candidate row for the admission model to score. UNREACHABLE only becomes
addressable if a separate mechanism expands the candidate universe first
(deeper source lists, a new retriever, or catalog-wide scoring), which is a
different proposal.

## 2. Cached data/artifacts it can reuse

Primary dev evidence:

- `exp/eval/expR55_post_refresh_decomp.json` - bucket counts, pool_hit,
  source_union_hit, and POOL_MISS source coverage. Current values:
  `pool_hit=0.622`, `src_union_hit=0.767375`, `POOL_MISS=1163`.
- `scripts/expR55_post_refresh_decomp.py` - exact dev reconstruction of the
  R39+R54 baseline, bucket definitions, source assembly, and feature build.
- `scripts/expF1_cfbpr_retrieval.py` - `weighted_rrf` implementation being
  replaced or augmented.

Reusable candidate/source artifacts:

- `exp/eval/_R12_all_turns_payload.pkl` - dev cases, GT, session history,
  `src_a`, `src_b`, `src_c`, `src_d`, `src_f`, and token/artist/tag maps.
- `cache/r21_production/dev_r21_oof_lists.json` - R21 OOF source lists.
- `cache/r54/phase2_full/oof_r54_lists.json` - R54 Phase 2 OOF source lists
  and cosine scores; preserved proxy for deleted Phase 3 OOF.
- `cache/r54_phase3_als.npz` and
  `cache/r54_phase3_payload_maps.pkl` - ALS factors and payload maps used by
  the production LR feature path.
- `cache/r54_phase3_track_pop.json` - track popularity feature support.
- `cache/r21_production/{model,track_embeddings.npy,track_ids.json}` - R21
  production assets if a later blind path needs source refresh.
- `cache/r54_production/blind_r54_lists.json` - production R54 blind source
  lists, already used by R54c.
- `cache/blind_a/source_cache.pkl` - preserved blind source cache for future
  source/ranker work.
- `cache/r58/top50_dev.pkl` - top-50 dev table from R58; not enough for pool
  admission by itself, but useful for downstream conversion checks if an
  admission diagnostic passes.

POOL_MISS source coverage from
`exp/eval/expR55_post_refresh_decomp.json` shows the model would have real
source evidence to learn from. Counts overlap:

| Source with GT in POOL_MISS | Cases |
|---|---:|
| R54 | 394 |
| R21 | 368 |
| C | 362 |
| B | 286 |
| ALS | 193 |
| F | 65 |
| D | 27 |
| A | 24 |

These counts argue against a single-source rule. The recoverable signal is
likely in combinations and depth patterns, which fixed RRF cannot adapt to.

## 3. Leakage risks

The admission label is GT-derived, so the first risk is training on dev and
evaluating on the same cases. The diagnostic must be CV/OOF-clean:

- Use the same grouped CV5 session folds as the LR pipeline.
- For each held-out fold, train admission only on the other four folds and
  score the held-out fold's source-union candidates.
- Use OOF retrieval lists for learned sources on dev: R21 OOF and R54 Phase 2
  OOF. Do not use production full-data embeddings to construct dev source
  ranks.
- Keep all tuning decisions out of the held-out fold. If hyperparameters are
  tuned, use an inner split or fixed defaults.

The second risk is contaminating the final LR training signal. This proposal
is pool admission, not a new LR feature. The smallest diagnostic should not
feed `admission_score` into LR and should not retrain LR with it. If a later
phase uses `admission_score`, `admission_rank_inv`, or learned source
interactions as LR features, that becomes a new R57b-style LR feature
experiment and must carry its own OOF design.

The third risk is hidden distribution shift through LR's existing features.
The current LR feature set includes pool-rank-derived values, including the
base rank reciprocal. If a learned admission model replaces RRF order and LR
receives `admission_rank_inv` in a feature slot trained on RRF rank, the
experiment is no longer clean. For the first conversion check, either:

- keep RRF-derived rank features computed over the full source union, even for
  newly admitted candidates with RRF rank >300, or
- explicitly retrain LR OOF on admission-built pools, which is a separate,
  higher-cost phase.

The fourth risk is negative-label noise. For every session there is one GT
and many plausible non-GT tracks. Treating every non-GT source-union track as
a hard negative can teach the model to suppress good alternatives. The cheap
diagnostic should optimize admission recall and net pool recovery, not trust
raw classifier accuracy.

## 4. Expected ceiling

The hard pool-level ceiling with existing sources is exactly the POOL_MISS
bucket: 1163 cases, or 14.5% of dev. If all were admitted, pool_hit@300 would
rise from 0.622 to the source-union ceiling of 0.767375. That is not a
realistic expected outcome; it is only the maximum available from the current
candidate universe.

A plausible learned-admission recovery is lower:

- Conservative: 150-250 of 1163 POOL_MISS GTs admitted OOF without dropping
  too many previously covered GTs.
- Reasonable upside: 250-350 if cross-source depth patterns are genuinely
  predictive.
- Above 400 is unlikely without expanding source lists, because many
  POOL_MISS cases are probably single-source/deep hits that RRF drops for good
  reasons.

Final nDCG lift will be smaller than pool recovery. Newly admitted tracks are
likely low-RRF or weak-evidence candidates, and the frozen LR may still demote
them. If 150-350 extra GTs enter the pool and LR converts roughly 25-45% into
top-20 hits, the net top-20 gain is roughly 40-160 cases before accounting for
lost cases. That corresponds to an estimated nDCG lift around +0.003 to
+0.008, with +0.010 as an optimistic edge case if the admitted tracks are
LR-friendly and HIT losses are near zero.

Against Blind-A production, R54c is at nDCG 0.4925. The stated competitor
band is roughly 0.51-0.57, so the gap to 0.51 is +0.0175 and the gap to 0.57
is +0.0775. Pool admission alone is not a credible path to the upper end of
that range. Its best realistic role is a bounded +0.005 to +0.010 mechanism
that may close part of the gap if it passes strict dev gates.

UNREACHABLE contributes zero under the current source-union design. The 1861
UNREACHABLE cases require a new generator before an admission model can score
them. If another mechanism supplies additional candidates, this admission
model could become the gate for those candidates, but the recovery should be
credited to the new generator plus gate, not to source-rank admission alone.

## 5. Smallest falsifiable diagnostic

Run a dev-only, OOF-clean pool diagnostic before any LR conversion or blind
work.

Candidate universe:

- For each dev case, take the union of existing source lists:
  `src_a/src_b/src_c/src_d/src_f`, ALS, R21 OOF, and R54 Phase 2 OOF.
- Exclude already-played tracks exactly as the current retrieval path does.
- Do not add new retrievers or deeper uncached lists in the first diagnostic.

Features per candidate:

- Per-source rank features: rank, reciprocal rank, log-rank bin, top-10,
  top-20, top-50, top-100, top-300 indicators, and missing indicators for
  A/B/C/D/F/ALS/R21/R54.
- Source-agreement features: number of sources present, number of strong
  sources present, min rank, best dense rank, best lexical rank, rank
  dispersion, and shallow/deep combinations.
- Score features where available: R54 cosine, ALS score if rebuilt, and
  normalized source score fields that already exist in cached artifacts.
- Baseline features for comparison only: weighted RRF score and weighted RRF
  rank over the full source union.
- Session features available at inference: history length, unique artist
  count, last-artist repeat indicators, and source-list overlap summaries.

Label:

- Candidate-level training label: `candidate_track_id == gt_track_id`.
- Session-level diagnostic outcome: `is_in_top300_after_admission`, meaning
  whether the GT appears in the top 300 after sorting held-out candidates by
  the learned admission score.

Model:

- Small LightGBM binary classifier or LambdaRank admission model, fixed
  default hyperparameters for the first pass.
- Train on four folds, score the held-out fold. Concatenate held-out scores
  for a full OOF admission pool.
- No LR retraining. No blind data. No `admission_score` feature in LR.

Evaluation:

- Primary: does OOF `admission_score@300` beat `weighted_rrf@300` on
  pool_hit@300?
- Required breakdown: all-dev, h7, same-artist, diff-artist, and original
  bucket.
- Direct POOL_MISS metric: of the 1163 RRF POOL_MISS cases, how many have GT
  in learned admission@300?
- Direct loss metric: of cases where weighted RRF already had GT in pool
  (HIT + DEMOTED), how many lose GT under learned admission@300?
- Net pool recovery: gained POOL_MISS minus lost previously-covered cases.
- Stability: top-300 overlap with weighted RRF, distribution of lost cases,
  and source coverage of recovered cases.

The falsifier is simple: if OOF admission@300 does not materially improve
pool_hit@300 and POOL_MISS recovery over weighted RRF@300, there is no reason
to run final LR or build a blind path.

Only if the pool diagnostic passes should a second dev-only phase test frozen
LR conversion on the learned pool. That phase must report recovered/lost
top-20 cases and same/diff nDCG, using the same gate discipline that archived
R56 and R58.

## 6. Stop condition

Archive immediately after the pool diagnostic if any of these hold:

- OOF learned admission@300 has `pool_hit@300 <= weighted_rrf@300`.
- Net pool recovery is <= 0.
- Learned admission recovers fewer than 100 of the 1163 POOL_MISS GTs.
- Previously-covered GT loss is more than 25% of gained POOL_MISS cases.
- Gains are concentrated in one noisy source-only pattern, such as single
  deep `A`/`D` hits, with no robust cross-source pattern.
- The diagnostic requires non-OOF R21/R54 dev retrieval, GT-side routing, or
  full-dev training to show lift.

If the pool diagnostic passes and a later frozen-LR conversion is run, mirror
the R56/R57b/R58 gates:

- Production candidate only if h7 nDCG is at least +0.010 vs R54c/R39+R54
  dev baseline.
- Same-artist nDCG must not regress; same-artist regression >0.002 is an
  auto-archive canary.
- All-dev nDCG must not regress.
- Top-1 churn must stay within the R56 discipline: <=240/8000 for a production
  candidate, with h7-only churn reported separately.
- Recovered top-20 cases must exceed lost top-20 cases by a meaningful margin,
  not by the +1/-1 noise pattern seen in R58.
- A +0.005 to +0.010 h7 result is not enough for Blind-A submission after the
  eight post-R54c negatives; at most it justifies another design review.

Archive permanently if the mechanism starts behaving like a post-LR reranker
or LR feature addition. That would collapse back into the already failed R56,
R57b, and R58 territory rather than testing the distinct POOL_MISS admission
hypothesis.
