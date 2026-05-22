# R72 — R68 pool-coverage scan (fold-0)

Elapsed: 24s

## Verdict: **POOL_HEADROOM_MARGINAL**

R68 admission surfaces GT into top-300 of augmented pool but rank is deep (>30 mostly). Needs ranker awareness to actually convert. Less promising for late-fusion only.

## Pool coverage categories (fold-0, n=1600)

| Category | Count | % |
|---|---:|---:|
| POOL_BOTH | 811 | 50.7% |
| POOL_R68_UNIQUE | 118 | 7.4% |
| POOL_BASE_ONLY | 156 | 9.8% |
| POOL_NEITHER | 515 | 32.2% |

## R68-unique-in-pool (R68 catches GT, baseline pool misses)

- fold-0 total: **118**
- h7: **16**

## Augmented (9-source) RRF — does admitting R68 surface GT?

- fold-0 GT in aug pool top-300: **35 / 118**
- fold-0 GT in aug top-30: **0**
- fold-0 GT in aug top-20: **0**
- h7 GT in aug pool top-300: **6 / 16**
- h7 GT in aug top-30: **0**
- h7 GT in aug top-20: **0**

### Aug-rank distribution for R68-unique cases

| Bucket | Count |
|---|---:|
| top-300 | 35 |

## Substitution loss (lose if we drop R54)

- fold-0 cases lost if R68 substitutes R54: **35**
