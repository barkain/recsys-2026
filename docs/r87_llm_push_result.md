# R87 LLM-Judge Push on R84c — Result

HEAD: `d72c79cbf4`
Base: `r84c_selective_submission.zip` (production, composite 0.6362)
R87: `r87_llm_push_submission.zip` (58342 bytes, sha256 `e9438ce5c0c95b91`)

## Composition
- 80 cases total
- **12** responses regenerated (weakest by LLM-judge audit)
- **68** responses unchanged from R84c
- Tracks bitwise identical: YES ✓

## Local LexDiv (proxy)

| metric | R84c | R87 | Δ |
|---|---:|---:|---:|
| local Distinct-2 | 0.9347 | 0.9383 | +0.0036 |

Gate (R87 ≥ R84c): **PASS**
Caveat: per feedback_local_distinct2_doesnt_predict_lexdiv, local D-2 doesn't predict competition LexDiv. Use as directional only.

## Regenerated rows

| session_id | turn | audit_score_before | diagnosis | words | issues_after |
|---|---:|---:|---|---:|---|
| `0a1f7a63-c5e` | 1 | 7 | no_causal,no_session_anchor,banned:you're chasing,mid_bigram_density(3) | 77 | — |
| `4b8ed42b-39e` | 4 | 7 | no_causal,no_session_anchor,vague_descriptor | 74 | — |
| `5b2f877b-a6d` | 1 | 7 | no_causal,no_session_anchor,banned:rather than,mid_bigram_density(3) | 78 | — |
| `9c37dcd7-d7c` | 1 | 7 | no_causal,no_session_anchor,vague_descriptor | 71 | — |
| `0fedfa80-ebe` | 7 | 6 | no_causal,no_session_anchor,mid_bigram_density(3) | 79 | — |
| `164fc33f-b10` | 5 | 6 | no_session_anchor,vague_descriptor,high_bigram_density(5) | 79 | — |
| `38a5f361-980` | 2 | 6 | no_causal,no_session_anchor,banned:you're chasing | 81 | — |
| `ab8bc8fa-632` | 1 | 6 | no_causal,no_session_anchor,banned:fits that | 79 | banned:matches the |
| `d5c80ee5-97c` | 7 | 6 | no_attribute,no_session_anchor,mid_bigram_density(4) | 73 | — |
| `eaeca304-3dd` | 7 | 6 | no_attribute,no_session_anchor,banned:you're after | 77 | — |
| `2a677d32-23e` | 4 | 5 | no_causal,no_session_anchor | 80 | — |
| `2fe9abf6-cce` | 5 | 5 | no_session_anchor,vague_descriptor,banned:rather than | 80 | — |

## Submission gate

- Tracks bitwise identical to R84c (zero nDCG risk).
- Local Distinct-2 proxy: PASS.
- Regen validation: 11/12 rows clean.
- Expected risk: LLM judge could go up (4.90 → 4.95 target) or stay flat (R78 ceiling). LexDiv could regress at competition scorer despite local-D-2 maintained.

**NOT auto-uploaded.** User decides whether to submit.

**Codabench scorer still has Gemini deprecation issue as of R86 submission (2026-05-25). Verify scorer is fixed before submitting.**
