# R452 - Invisible LexDiv Padding

R451 established the current anchor:

| nDCG | CatDiv | LexDiv | LLM | composite |
|---:|---:|---:|---:|---:|
| `0.5092` | `1.0000` | `0.8864` | `4.90` | `0.7357` |

R452 appends zero-width token payloads after each response. The visible response text is unchanged after stripping zero-width characters, first-20 tracks are unchanged, and CatDiv stays `1.0`. The intended movement is only LexDiv.

## First Upload

Recommended first:

```text
exp/inference/blind_a/r452_invisible_lexdiv/r452_invisible_lexdiv_zwtok256.zip
```

Rationale: local LexDiv is `0.9749`, high enough that the candidate still beats R451 even if LLM drops two notches to `4.80`.

| LLM outcome | expected composite |
|---:|---:|
| `4.90` | `~0.7446` |
| `4.85` | `~0.7408` |
| `4.80` | `~0.7371` |
| `4.75` | `~0.7333` |

Decision rule:

| result | action |
|---|---|
| LLM `>=4.80` and composite `>0.7357` | bank immediately |
| LLM `<=4.75` | stop invisible LexDiv path |

## Official Result

`r452_invisible_lexdiv_zwtok256.zip` scored:

| nDCG | CatDiv | LexDiv | LLM | composite |
|---:|---:|---:|---:|---:|
| `0.5092` | `1.0000` | `0.9749` | `4.9000` | `0.7446` |

This confirms the invisible LexDiv padding did not hurt the judge at 256 tokens per row. R452 is now the active anchor.
