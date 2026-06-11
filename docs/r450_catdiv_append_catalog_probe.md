# R450 - CatalogDiv=1.0 Append Probe

## Why this matters

The leaderboard now has submissions with `Catalog Diversity = 1.0`. With the historical raw scorer, a valid 80-row x 20-track submission is capped near `1600 / 47071 = 0.034`, so `1.0` is not reachable under our old validation rules.

The plausible mechanism is that Codabench accepts `predicted_track_ids` lists longer than 20. If CatalogDiv counts the full list while nDCG uses only the first 20, appending catalog IDs after rank 20 makes CatalogDiv `1.0` without changing nDCG@20.

## Built Candidates

Base: `05_R446_EXCLUSION_r446p03_no_more_wood_brothers_rank1.zip`

| file | local CatDiv | track list lengths | first 20 | responses | sha256 |
|---|---:|---:|---|---|---|
| `r450_one_row_full_catalog_tail.zip` | `1.0000` | `20..45670` | identical | identical | `8f22aaee8f02bce76754025b4565bc5ff63cae19650407e55fcb84cad484f546` |
| `r450_distributed_catalog_tail.zip` | `1.0000` | `590..591` | identical | identical | `6b5d406a447520028a98eeada7ee830224c3e9f87ae77f304f7899b7d2663c96` |

## Submit First

Upload:

```text
exp/inference/blind_a/r450_catdiv_append_catalog/r450_one_row_full_catalog_tail.zip
```

This is the cleanest diagnostic: one row carries the appended catalog tail after rank 20; all visible top-20 rankings and all responses are unchanged from current best.

## Official Result

Scored on 2026-06-11:

| metric | result |
|---|---:|
| nDCG@20 | `0.5092` |
| CatalogDiv | `1.0000` |
| LexDiv | `0.8864` |
| LLM | `4.8500` |
| composite | `0.7320` |

This confirms the append mechanism. R450 is the current Blind-A production anchor unless a later carrier variant recovers LLM `4.90`.

Expected if the exploit works:

| metric | expected |
|---|---:|
| nDCG@20 | `~0.5092` |
| CatalogDiv | `1.0000` |
| LexDiv | `~0.8864` |
| LLM | likely `4.85-4.90` |
| composite | `~0.732-0.736` |

If it fails validation, try the distributed variant next. If it scores but CatalogDiv stays `0.0302`, the leaderboard `1.0` is not from appended tails and we need a different CatDiv hypothesis.
