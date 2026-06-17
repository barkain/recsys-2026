# R513 - R510 Candidate Set, R500 In-Set Order

**Status:** staged candidate after R512 failure.

## Decision

Recommended upload:

`exp/inference/blind_a/r513_r510_r500_common_order/r513_r510_r500_common_top5_ov16_conf65_submission.zip`

`sha256: 525e917bb425aedd1ae938eba9f9d115707824f580aa082cd3506d6d3db8029f`

## Why This Candidate

R512 failed because it admitted new tail candidates chosen by GPT-4.1 and lost official nDCG (`0.5107` vs R510 `0.5149`). R513 removes that failure mode. It starts from R510 and only reorders tracks already present in R510.

The selected variant applies the R500 top-20 GPT-4.1 order only when R500 and R510 already strongly agree on the candidate set:

- mode: top-5 reorder only, preserving rank 1
- overlap gate: at least 16 of 20 tracks shared between R500 and R510
- confidence gate: R500 confidence at least 0.65
- changed rows: 4
- top-1 changes: 0
- response changes: 0
- candidate membership changes: 0

## Changed Rows

| row | session | summary |
| ---: | --- | --- |
| 39 | `70a0e5ad` | Laura Pausini same-artist reorder; promotes `Lo sabias antes tu` and `200 notas` inside the top 5. |
| 50 | `9cd93031` | Ryan Adams & The Cardinals same-artist reorder; promotes `A Kiss Before I Go`, `Peaceful Valley`, `Let It Ride`. |
| 51 | `9d4ef919` | Myrkur same-artist reorder; swaps `Gladiatrix`, `De Tre Piker`, `Norn` inside the head. |
| 61 | `c75f8e41` | holiday-jazz row; promotes `O Christmas Tree - Tony Bennett` and Oscar Peterson holiday standards. |

## Rejected Alternatives

- `top5_ov14_conf70`: changes 7 rows, but row 0 has a severe head rewrite despite only 14/20 overlap.
- full-common variants: change fewer rows but reorder many tail positions; previous Blind-A evidence says broad list churn can hurt the judge without reliable nDCG conversion.
- append-tail variants: R512 official result rejected this mechanism.

## Validation

All staged R513 zips pass structural validation: 80 rows, exactly 20 unique tracks per row, no invisible response characters, no empty responses. The recommended zip has no top-1, response, or top-20 membership changes relative to R510.
