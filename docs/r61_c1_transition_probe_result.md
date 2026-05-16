# R61 C1 Transition-Memory Probe Result

Created: 2026-05-16T23:59:03.653424

## Gate Table

Verdict: **FAIL**

| Criterion | Threshold | Observed | Pass |
|---|---:|---:|---:|
| unique h7 GT outside current RRF pool@300 | >= 30 | 20 | False |
| best fused h7 nDCG@20 delta (w=0.25) | >= +0.003 | +0.00016 | False |
| novelty OR h7 lift | pass one above | False | False |
| best fused all-dev nDCG@20 delta | >= 0.00000 | -0.00097 | False |
| best fused same-artist nDCG@20 delta | >= -0.00200 | -0.00112 | True |
| best fused h7 top-1 churn | <= 1.5% (1.2/80 blind-eq) | 4.60% (3.68/80) | False |

## Standalone C1 Retrieval

| Split | n | hit@20 | hit@100 | hit@300 |
|---|---:|---:|---:|---:|
| all_dev | 8000 | 0.1373 (1098) | 0.2238 (1790) | 0.2636 (2109) |
| h7 | 1000 | 0.1320 (132) | 0.2330 (233) | 0.2750 (275) |
| same_artist | 2857 | 0.3185 (910) | 0.4792 (1369) | 0.5100 (1457) |
| diff_artist | 5143 | 0.0366 (188) | 0.0819 (421) | 0.1268 (652) |

## Novelty And Buckets

| Metric | Value |
|---|---:|
| unique h7 GT hits outside current RRF pool@300 | 20 |
| POOL_MISS recovered by C1 top-300 | 122 / 1163 |
| UNREACHABLE recovered by C1 top-300 | 63 / 1861 |
| same-artist outside-pool hits | 15 |
| diff-artist outside-pool hits | 170 |

## Top-300 Overlap With Current Weighted RRF

| Statistic | Overlap count | Overlap fraction |
|---|---:|---:|
| mean | 48.37 | 0.1612 |
| median | 41.00 | 0.1367 |
| p10 | 0.00 | 0.0000 |
| p90 | 106.00 | 0.3533 |

## Frozen-LR Fusion Sanity

Predeclared C1 source weights: `{0.25, 0.5, 1.0}`. LR was loaded from `cache/r54_phase3_lr_model.txt` and was not retrained.

| C1 weight | pool_hit@300 | delta | h7 nDCG delta | all-dev nDCG delta | same-artist delta | diff-artist delta | h7 top1 churn |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.62362 | +0.00162 | +0.00016 | -0.00097 | -0.00112 | -0.00089 | 4.60% |
| 0.5 | 0.62313 | +0.00112 | -0.00294 | -0.00148 | -0.00153 | -0.00145 | 6.60% |
| 1.0 | 0.62350 | +0.00150 | -0.00328 | -0.00363 | -0.00562 | -0.00253 | 10.10% |

## Train Split Audit

| Field | Value |
|---|---:|
| dataset | `talkpl-ai/TalkPlayData-Challenge-Dataset` |
| split | `train` |
| train arrow path | `/Users/nadavbarkai/dev/recsys-2026/.hf_cache/datasets/talkpl-ai___talk_play_data-challenge-dataset/default/0.0.0/8110a2cfda8f7cfd43805a09eca6c58e0f7b285c/talk_play_data-challenge-dataset-train.arrow` |
| train rows | 15199 |
| train unique session_ids | 15199 |
| dev unique session_ids | 1000 |
| excluded train sessions overlapping dev | 0 |
| included train sessions | 15199 |
| sessions with >=2 music turns | 15199 |
| transition rows emitted | 106393 |
| transition rows with current/previous user text context | 106393 |
| session_id schema | uuid-like string |
| sample UUID match | True |

## Implementation Notes

- Train source is official `train` split only; script requests `split="train"` and never loads Blind-A.
- Dev sessions found in train are excluded before counting transitions.
- Candidate generator is count-only: last-track counts, last-3 recency counts, last-artist counts, and artist/tag metadata backoff.
- Metadata backoff uses existing cached `track_artist` and `track_tags` maps from the R12 payload.
- No cached metadata-neighbor NN index was found, so `c1_metadata_neighbor` was skipped.
- Played tracks are excluded from C1 outputs before ranking.
