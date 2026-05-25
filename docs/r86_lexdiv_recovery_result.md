# R86 LexDiv Recovery on R84c — Result

HEAD: `a597ee23f5`
Base: `r84c_selective_submission.zip` (production)
R86: `r86_lexdiv_recovery_submission.zip` (58354 bytes, sha256 `a51ca4c8f376ca06`)

## Composition
- 80 cases (tracks identical to R84c, hash matches)
- **12** responses regenerated (top-12 by R84c bigram-repeat-density)
- **68** responses unchanged from R84c

## Lexical diversity

| metric | R84c | R86 | Δ |
|---|---:|---:|---:|
| local corpus Distinct-2 | 0.9347 | 0.9422 | +0.0076 |

Gate (lift >= +0.010): **FAIL**

## Regenerated rows (top-density)

| session_id | turn | density_before | words | issues |
|---|---:|---:|---:|---|
| `0802ac4a-187` | 6 | 8 | 84 | — |
| `28c3ecd9-fba` | 6 | 6 | 77 | — |
| `ee7bfbda-86e` | 3 | 6 | 83 | — |
| `6c54de37-9c5` | 2 | 4 | 75 | — |
| `7905bb71-efe` | 1 | 4 | 77 | — |
| `9cd93031-ae3` | 3 | 4 | 78 | — |
| `d9cca604-feb` | 2 | 4 | 74 | — |
| `db8ec85f-fa1` | 7 | 4 | 78 | — |
| `68993adf-60d` | 1 | 3 | 81 | — |
| `574f75cf-703` | 2 | 2 | 75 | — |
| `5ad7094f-376` | 4 | 2 | 79 | — |
| `77faef85-566` | 1 | 2 | 79 | — |

## Submission gate

- Tracks bitwise identical to R84c (zero nDCG risk).
- Local Distinct-2 proxy: gate FAILED.
- No banned-phrase failures expected (validate caught only inline issues).

**NOT auto-uploaded.** User decides whether to submit.
