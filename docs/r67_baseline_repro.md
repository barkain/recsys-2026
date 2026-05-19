# R67 Baseline Reproduction (Wave 0)

Created: 2026-05-19T21:19:32.843862
Branch: `r67-llm-semantic-rerank`
HEAD: `349ad04ba341932ddbe6f5651a61d0d466a1207b`
Reference: `exp/eval/expR59_c3_phase2_frozen_lr.json`
Epsilon gate: 0.0005

## Verdict: **PASS**
max |delta| = 0.000000

## Metrics

| Metric | Reference | Reproduced | Delta |
|---|---:|---:|---:|
| all_dev_ndcg20 | 0.315875 | 0.315875 | +0.000000 |
| h7_ndcg20 | 0.348378 | 0.348378 | -0.000000 |
| same_artist_ndcg20 | 0.628214 | 0.628214 | +0.000000 |
| diff_artist_ndcg20 | 0.142367 | 0.142367 | +0.000000 |
| pool_hit_all | 0.622000 | 0.622000 | +0.000000 |
| pool_hit_h7 | 0.613000 | 0.613000 | +0.000000 |

## R63c response-only clarification

R63c-repair is response-side only. Retrieval/LR baseline is bitwise R54c. The R67 LLM reranker operates on the same R54c LR top-30.

## Top-30 extraction

- n_cases: 8000
- top_k per case: 30
- unique candidate tracks (top-30 union): 26344

### Metadata coverage (% of unique candidate tracks)

| Field | Coverage |
|---|---:|
| title | 1.0000 |
| artist | 1.0000 |
| album | 1.0000 |
| tags (>=1) | 0.9987 |
| release_year | 0.9791 |

## Artifacts

- Baseline JSON: `exp/eval/expR67_baseline_repro.json`
- Top-30 pickle: `exp/eval/expR67_top30_candidates.pkl`
- This document: `docs/r67_baseline_repro.md`
- Script: `scripts/expR67_baseline_repro.py`

## Notes

- Elapsed: 169.1s
- Wave 0 reuses cached intermediates and frozen production LR.
- Top-30 records include LR rank/score, per-source ranks (A/B/C/D/F/ALS/R21/R54), and metadata join (title/artist/album/tags[:5]/release_year).
