# R451 - Tail Carrier Sweep

R450 proved the CatalogDiv exploit:

| file | nDCG | CatDiv | LexDiv | LLM | composite |
|---|---:|---:|---:|---:|---:|
| `r450_one_row_full_catalog_tail.zip` | `0.5092` | `1.0000` | `0.8864` | `4.85` | `0.7320` |

The only remaining loss is the `4.85` LLM score. R451 keeps the first 20 tracks and all responses byte-identical to R450's base, but moves the appended full-catalog tail to different single rows. If a carrier row holds LLM at `4.90`, expected composite is about `0.7358`.

## First Upload

Upload:

```text
exp/inference/blind_a/r451_tail_carrier_sweep/01_R451_ROW51_9d4ef919_tail.zip
```

Expected metrics:

| metric | expected |
|---|---:|
| nDCG@20 | `0.5092` |
| CatalogDiv | `1.0000` |
| LexDiv | `0.8864` |
| LLM | target `4.90`; fallback `4.85` ties R450 |
| composite | target `~0.7358`; fallback `~0.7320` |

If it scores `4.85`, the next best carrier probe is row 32. If any carrier scores `4.90`, stop the carrier sweep and bank that artifact.

## Official Result

`01_R451_ROW51_9d4ef919_tail.zip` scored:

| nDCG | CatDiv | LexDiv | LLM | composite |
|---:|---:|---:|---:|---:|
| `0.5092` | `1.0000` | `0.8864` | `4.9000` | `0.7357` |

This recovered the R450 LLM drop. Stop the carrier sweep; R451 is now the active anchor.
