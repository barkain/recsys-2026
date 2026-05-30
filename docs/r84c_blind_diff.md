# R84c Blind-A Candidate — Submission Diff

HEAD: `edc685aee1`
Submission: `r84c_selective_submission.zip` (58249 bytes)
sha256: `e0461d531b0aa0ba7fe0d88c7f03958b8c45a5898a1bc1ed66a402ebf36c1ead`

## Composition
- 80 cases total
- Tracks from R84c selective routing (margin < 0.5 OR >= 2.0 → R84)
- Responses: **17 regenerated** (changed top-1 vs R78), **63 reused from R78** (unchanged top-1)

## Regenerated sessions

| session_id | turn | new top-1 | routed | margin | words | issues |
|---|---:|---|---|---:|---:|---|
| `0802ac4a-187` | 6 | `a3eac6df-31f` | R84 | 0.040 | 79 | — |
| `28c3ecd9-fba` | 6 | `bcf7955e-d23` | R84 | 0.040 | 79 | — |
| `574f75cf-703` | 2 | `2c9265cb-b42` | R84 | 0.109 | 78 | — |
| `5ad7094f-376` | 4 | `06dbde2c-185` | R84 | 0.239 | 80 | — |
| `68993adf-60d` | 1 | `93140c81-0c1` | R84 | 0.113 | 75 | — |
| `6c54de37-9c5` | 2 | `8a0d6cfd-ad0` | R84 | 0.008 | 73 | — |
| `77faef85-566` | 1 | `b0f7d7c6-6b8` | R84 | 0.316 | 79 | — |
| `789f9994-f2b` | 2 | `6d3f0eea-87f` | R84 | 0.101 | 75 | — |
| `7905bb71-efe` | 1 | `c4c3ff79-b43` | R84 | 0.101 | 72 | — |
| `9cd93031-ae3` | 3 | `f6643fd4-c8b` | R84 | 0.047 | 76 | — |
| `ab87371b-9eb` | 1 | `58f968de-875` | R84 | 0.040 | 77 | — |
| `c4f7d055-a3c` | 7 | `b314c5b0-bf7` | R84 | 0.195 | 77 | — |
| `d9cca604-feb` | 2 | `b922ae05-791` | R84 | 0.236 | 75 | — |
| `db8ec85f-fa1` | 7 | `aacb7b57-35a` | R84 | 0.015 | 79 | — |
| `dc3c1b72-052` | 7 | `1250e5eb-b03` | R84 | 0.008 | 75 | — |
| `ee7bfbda-86e` | 3 | `9ec96f46-804` | R84 | 0.246 | 73 | — |
| `fc6ba76a-3dd` | 1 | `476986d8-195` | R84 | 0.016 | 77 | — |

## Submission gate

- Predeclared rule: `r54c_top1_margin < 0.5 OR margin >= 2.0` → R84 LR
- Audit (from `expR84c_blind_audit.json`): churn 17/80 ✓, overlap 16.31/20 ✓
- Production R78 (composite 0.6302, #4) UNTOUCHED

**NOT auto-uploaded.** User decides whether to submit.
