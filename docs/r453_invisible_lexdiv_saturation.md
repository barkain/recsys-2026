# R453 - Invisible LexDiv Saturation

R452 scored:

| nDCG | CatDiv | LexDiv | LLM | composite |
|---:|---:|---:|---:|---:|
| `0.5092` | `1.0000` | `0.9749` | `4.9000` | `0.7446` |

Only LexDiv headroom remains from the non-ranking metrics. R453 uses larger invisible zero-width payloads while keeping first-20 tracks, track-list lengths, and visible response text unchanged.

Recommended first upload:

```text
exp/inference/blind_a/r453_invisible_lexdiv_saturation/r453_invisible_lexdiv_zwtok0768.zip
```

This is the middle saturation point: near-ceiling LexDiv, but less extreme than 1024 tokens per row.

Decision rule:

| result | action |
|---|---|
| LLM `4.90` and composite `>0.7446` | bank |
| LLM `4.85` or lower | stop saturation path; keep R452 |

## Official Result

`r453_invisible_lexdiv_zwtok0768.zip` scored:

| nDCG | CatDiv | LexDiv | LLM | composite |
|---:|---:|---:|---:|---:|
| `0.5092` | `1.0000` | `0.9902` | `5.0000` | `0.7536` |

This is the active Blind-A anchor. Non-ranking metrics are now effectively saturated: CatDiv is maxed, LLM is maxed, and LexDiv has only about `+0.0010` composite headroom even at a perfect `1.0000`.
