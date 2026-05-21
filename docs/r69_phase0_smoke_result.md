# R69 Phase 0 SMOKE — cross-encoder rerank (Mac MPS reduced scope)

Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`  Device: `mps`
Scope: fold-0 h7 only (n=200)  Pool: top-100
Pairs scored: 20000  Throughput: 227 pairs/s

## Verdict: SMOKE_NEGATIVE

| Subset | n | Baseline LR | R69 cross-enc | Delta |
|---|---:|---:|---:|---:|
| h7 | 200 | 0.3043 | 0.0507 | -0.2537 |
| h7_same | 83 | 0.5698 | 0.1094 | -0.4604 |
| h7_diff | 117 | 0.1160 | 0.0090 | -0.1070 |

- recovered=3  lost=73  net=-70
- top1_churn_per_80=79.60

## Caveats

- This is a SMOKE test with reduced scope (MiniLM-L-6 + POOL_K=100 + h7 only).
- A positive result here should be re-tested with bge-reranker-v2-m3 + POOL_K=300 + full fold-0 on A100.
- A negative result closes the cross-encoder-rerank hypothesis cheaply.
