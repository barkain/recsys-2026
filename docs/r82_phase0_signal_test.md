# R82 Phase 0 — LLM intent feature signal test

Sample: 50 cases, valid (intent parsed): 50
## Verdict: **ARCHIVE**

## Pairwise separation: GT vs R54c false positives

- GT mean total: **3.463**
- R54c top-1 FP mean: **3.720**
- All FPs mean: **3.689**
- GT > top-1 FP rate: **0.240** (chance = 0.5)
- GT > all FPs rate: **0.020**
- GT > any FP rate: **0.620**

| Feature | GT mean | FP mean | Δ |
|---|---:|---:|---:|
| mood_match | 0.113 | 0.164 | -0.051 |
| genre_match | 0.750 | 0.839 | -0.089 |
| era_match | 0.510 | 0.550 | -0.040 |
| language_match | 0.560 | 0.581 | -0.021 |
| energy_match | 0.190 | 0.215 | -0.025 |
| artist_rel_match | 0.840 | 0.841 | -0.001 |
