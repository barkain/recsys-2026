# R82 Sprint — LLM Intent Features — ARCHIVE_SPRINT (clean negative)

**Date:** 2026-05-23

## Verdict

ARCHIVE at Phase 0. The LLM-derived intent signal does NOT separate GT
from R54c's top-1 false positives — in fact, FP scores HIGHER than GT on
every match feature. **R54c has already absorbed the structured intent
signal that the LLM extracts.** $1.18 spent.

## What we tested

For 50 fold-0 h7 cases:
1. Used Opus 4.7 to extract structured intent JSON per query (mood, genre,
   era, language, energy, artist relation, novelty, constraints).
2. Built scalar match features per candidate using existing track metadata
   (tag_list, artist_name, release_date).
3. Pairwise compared GT vs R54c's top-20 false positives.

Gate (predeclared): GT > top-1 FP rate ≥ 55% AND (GT mean − FP mean) ≥ 0.1.

## Result

50/50 cases parsed cleanly (Opus reliably produced valid JSON).

| metric | value |
|---|---:|
| GT mean total match | 3.463 |
| R54c top-1 FP mean | **3.720** (HIGHER than GT) |
| R54c all FP mean | 3.689 |
| GT > top-1 FP rate | **24%** (worse than 50% chance) |
| GT > all FPs rate | 2% |
| GT > any FP rate | 62% |

Per-feature breakdown — EVERY feature has FP > GT:

| Feature | GT mean | FP mean | Δ |
|---|---:|---:|---:|
| mood_match | 0.113 | 0.164 | −0.051 |
| genre_match | 0.750 | 0.839 | −0.089 |
| era_match | 0.510 | 0.550 | −0.040 |
| language_match | 0.560 | 0.581 | −0.021 |
| energy_match | 0.190 | 0.215 | −0.025 |
| artist_rel_match | 0.840 | 0.841 | −0.001 |

## Why it fails

R54c's "false positives" at top-1 are not random tracks — they are the
tracks R54c judges as the BEST match for the query. They share mood,
genre, era, language with the query in ways that ALSO match the LLM's
structured intent. Our scoring scheme rewards exactly the pattern R54c
is already optimizing for.

The LLM intent extraction is doing the same job as R54c's LR features
(query-artist overlap, query-tag overlap, source-rank features), just
through a different mechanism. R54c's LightGBM has learned a calibrated
weighting that exceeds what a simple linear sum of match features can
achieve.

**The LLM doesn't see anything R54c doesn't.** Both work from the same
underlying signal (query text + candidate metadata). The LLM
intermediate representation (structured JSON) does not unlock new
information.

## The bigger lesson

This is the cleanest definitive negative on the nDCG question in this
sprint. After R71/R76/R80/R81/R82 we have:

- R71/R76/R80/R81: neural architectures over R54-stacked top-300 with
  existing feature stack → all net-negative or marginal
- R82: LLM-derived NEW feature stack → also negative, in fact GT scores
  LOWER than R54c's top picks

The feature ceiling isn't just real — R54c is also AT the ceiling. Any
intent-style feature we extract from query+metadata is already encoded
in R54c's calibrated score. R54c's "errors" (cases where GT is NOT at
rank 1) are not "easy" misses — they are cases where GT genuinely
matches the query less well than R54c's top pick by every interpretable
metric.

## What this means about R54c's "false positives"

When R54c ranks the wrong track at #1 instead of GT, it's usually
because:
1. R54c's top pick objectively matches the query intent better (per any
   measure we can extract)
2. GT happens to be what the user actually played but matches the
   structured intent LESS well than R54c's top pick

The signal for "which match would the user actually pick" is in user
preference data that doesn't exist in our features — preferences for
specific artists, sub-mood transitions, listening session arc, etc.
These cannot be extracted from query+catalog metadata alone, no matter
how sophisticated the extraction.

## Production state unchanged

R78 holds at composite 0.6302, position #4.

## Total spend summary

- R79 hard-negative retriever: ~$3 A100
- R80 listwise transformer: ~$3 A100
- R81 constrained-swap (4 configs): ~$12 A100
- R82 LLM intent signal test: ~$1 API
- **Total: ~$19**

All under any reasonable budget. Each negative caught cheaply via
predeclared gates.

## Honest recommendation

All credible nDCG paths empirically closed. The available query and
catalog metadata do not contain enough information to selectively
improve over R54c. Accept R78 (#4, composite 0.6302) and freeze.

If we want to push further, the ONLY remaining axis is data not yet in
the dataset: external metadata (Last.fm tags, Spotify genres, MusicBrainz
listening counts) or behavioral signals (which the dataset deliberately
withholds). Both speculative; neither in scope.

## Files

- `scripts/expR82_phase0_intent_signal_test.py` — main script (has LLM bug + cache-aware fix)
- `scripts/expR82_phase0_eval_only.py` — eval-only on cached intents
- `cache/r82/intents_fold0_h7.json` — 50 LLM-extracted intents
- `cache/r82/candidate_features_fold0_h7.json` — per-case match features
- `exp/eval/expR82_phase0_signal_test.json` — verdict + metrics
- `docs/r82_phase0_signal_test.md` — short result doc
- `docs/r82_sprint_summary.md` — this file
