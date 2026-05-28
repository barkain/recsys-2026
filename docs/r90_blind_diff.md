# R90 Blind-A Candidate

HEAD: `6dfe48e453`
Submission: `r90_submission.zip` (58333 bytes)
sha256: `510e62f4b3a7ede3a86d9ff6e252b96b48f34ce848c1652162871daaa0a1dd8d`

## Composition

- 80 cases total
- Tracks from R90 5-fold ensemble + production R90-LR
- Routing: R54c margin `<0.25` or `>=2.0` -> R90, else R54c
- Responses: **12 regenerated** (changed top-1 vs R84c), **68 reused from R84c**

## Audit

- Top-1 churn: 12/80
- Top-20 overlap mean: 16.94/20
- Route counts: R90=30, R54=50
- Sessions with >=6 changed tracks: 21/80
- Packaging gate: PASS

## Regenerated sessions

| session_id | turn | new top-1 | routed | margin | words | issues |
|---|---:|---|---|---:|---:|---|
| `0802ac4a-187` | 6 | `b93716ec-6c6` | R90 | 0.040 | 82 | - |
| `46faad58-58e` | 2 | `feeabd8f-32c` | R90 | 0.148 | 86 | - |
| `4b239a62-443` | 2 | `2ff2cb7b-bcc` | R90 | 0.179 | 75 | - |
| `5ad7094f-376` | 4 | `a7aa253a-301` | R90 | 0.239 | 77 | - |
| `6c54de37-9c5` | 2 | `d73e25b2-0cf` | R90 | 0.008 | 78 | - |
| `77faef85-566` | 1 | `48865b90-9a7` | R54 | 0.316 | 80 | - |
| `9cd93031-ae3` | 3 | `6b30834e-945` | R90 | 0.047 | 81 | vague_descriptor |
| `9d4ef919-504` | 3 | `e6b4b65e-335` | R90 | 0.045 | 79 | - |
| `ab87371b-9eb` | 1 | `989f8093-04e` | R90 | 0.040 | 80 | - |
| `c4f7d055-a3c` | 7 | `9d9ca4fe-fa4` | R90 | 0.195 | 77 | - |
| `db8ec85f-fa1` | 7 | `2b3bce2c-b23` | R90 | 0.015 | 77 | - |
| `dc3c1b72-052` | 7 | `71a00cb0-c93` | R90 | 0.008 | 84 | - |
