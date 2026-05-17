| Diagnostic | Value | Gate |
|---|---:|---|
| Best category-conditioned source-choice ceiling on all-dev pool_hit | +0.0028 (0.6220 -> 0.6248) | ARCHIVE (< +0.010) |
| C/K source-specific advantage present | Yes (C=N, K=Y) | PROCEED (Yes) |

**Verdict**

ARCHIVE_G1: oracle ceiling +0.0028 is below +0.010.

**Coverage Matrix**

Cached-depth source coverage counts GT membership in each cached list as-is; B/C have depth 500, R21/R54 depth 300, ALS/A/D/F max depth 200. The oracle sections below enforce pool@300 for source-choice comparisons.

| category | n | A | B | C | D | F | ALS | R21 | R54 | source-union | weighted_rrf@300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 488 | 161/488 (0.330) | 251/488 (0.514) | 283/488 (0.580) | 151/488 (0.309) | 125/488 (0.256) | 120/488 (0.246) | 271/488 (0.555) | 290/488 (0.594) | 384/488 (0.787) | 314/488 (0.643) |
| B | 1136 | 326/1136 (0.287) | 569/1136 (0.501) | 626/1136 (0.551) | 291/1136 (0.256) | 269/1136 (0.237) | 302/1136 (0.266) | 571/1136 (0.503) | 677/1136 (0.596) | 872/1136 (0.768) | 707/1136 (0.622) |
| C | 464 | 116/464 (0.250) | 190/464 (0.409) | 214/464 (0.461) | 101/464 (0.218) | 110/464 (0.237) | 124/464 (0.267) | 202/464 (0.435) | 216/464 (0.466) | 303/464 (0.653) | 244/464 (0.526) |
| D | 688 | 192/688 (0.279) | 324/688 (0.471) | 355/688 (0.516) | 184/688 (0.267) | 196/688 (0.285) | 208/688 (0.302) | 341/688 (0.496) | 398/688 (0.578) | 517/688 (0.751) | 411/688 (0.597) |
| E | 760 | 210/760 (0.276) | 399/760 (0.525) | 413/760 (0.543) | 210/760 (0.276) | 175/760 (0.230) | 177/760 (0.233) | 389/760 (0.512) | 477/760 (0.628) | 590/760 (0.776) | 484/760 (0.637) |
| F | 760 | 231/760 (0.304) | 420/760 (0.553) | 492/760 (0.647) | 204/760 (0.268) | 198/760 (0.261) | 207/760 (0.272) | 491/760 (0.646) | 522/760 (0.687) | 629/760 (0.828) | 528/760 (0.695) |
| G | 616 | 154/616 (0.250) | 286/616 (0.464) | 315/616 (0.511) | 143/616 (0.232) | 121/616 (0.196) | 167/616 (0.271) | 318/616 (0.516) | 352/616 (0.571) | 477/616 (0.774) | 368/616 (0.597) |
| H | 1080 | 310/1080 (0.287) | 587/1080 (0.544) | 691/1080 (0.640) | 302/1080 (0.280) | 268/1080 (0.248) | 351/1080 (0.325) | 653/1080 (0.605) | 681/1080 (0.631) | 883/1080 (0.818) | 740/1080 (0.685) |
| I | 144 | 19/144 (0.132) | 50/144 (0.347) | 64/144 (0.444) | 21/144 (0.146) | 12/144 (0.083) | 33/144 (0.229) | 71/144 (0.493) | 77/144 (0.535) | 108/144 (0.750) | 87/144 (0.604) |
| J | 616 | 142/616 (0.231) | 264/616 (0.429) | 346/616 (0.562) | 121/616 (0.196) | 130/616 (0.211) | 166/616 (0.269) | 306/616 (0.497) | 339/616 (0.550) | 458/616 (0.744) | 372/616 (0.604) |
| K | 1248 | 258/1248 (0.207) | 512/1248 (0.410) | 655/1248 (0.525) | 238/1248 (0.191) | 264/1248 (0.212) | 344/1248 (0.276) | 650/1248 (0.521) | 698/1248 (0.559) | 918/1248 (0.736) | 721/1248 (0.578) |

**Oracle Ceiling**

Best single source per category: overall 4727/8000 (0.5909), delta -249 cases / -0.0311.

| category | source | hits/n | rate | delta rate |
|---|---|---:|---:|---:|
| A | single:R54 | 290/488 | 0.5943 | -0.0492 |
| B | single:R54 | 677/1136 | 0.5960 | -0.0264 |
| C | single:R54 | 216/464 | 0.4655 | -0.0603 |
| D | single:R54 | 398/688 | 0.5785 | -0.0189 |
| E | single:R54 | 477/760 | 0.6276 | -0.0092 |
| F | single:R54 | 522/760 | 0.6868 | -0.0079 |
| G | single:R54 | 352/616 | 0.5714 | -0.0260 |
| H | single:R54 | 681/1080 | 0.6306 | -0.0546 |
| I | single:R54 | 77/144 | 0.5347 | -0.0694 |
| J | single:R54 | 339/616 | 0.5503 | -0.0536 |
| K | single:R54 | 698/1248 | 0.5593 | -0.0184 |

Best 2-source RRF per category: overall 4929/8000 (0.6161), delta -47 cases / -0.0059.

| category | pair | hits/n | rate | delta rate |
|---|---|---:|---:|---:|
| A | pair:C+R54 | 310/488 | 0.6352 | -0.0082 |
| B | pair:ALS+R54 | 691/1136 | 0.6083 | -0.0141 |
| C | pair:C+R54 | 224/464 | 0.4828 | -0.0431 |
| D | pair:ALS+R54 | 418/688 | 0.6076 | +0.0102 |
| E | pair:ALS+R54 | 481/760 | 0.6329 | -0.0039 |
| F | pair:R21+R54 | 525/760 | 0.6908 | -0.0039 |
| G | pair:ALS+R54 | 367/616 | 0.5958 | -0.0016 |
| H | pair:ALS+R54 | 731/1080 | 0.6769 | -0.0083 |
| I | pair:ALS+R54 | 87/144 | 0.6042 | +0.0000 |
| J | pair:ALS+R54 | 359/616 | 0.5828 | -0.0211 |
| K | pair:ALS+R54 | 736/1248 | 0.5897 | +0.0120 |

Best conditioned choice per category (baseline, single, or pair): overall 4998/8000 (0.6248), delta +22 cases / +0.0028.

| category | option | hits/n | rate | delta rate |
|---|---|---:|---:|---:|
| A | baseline:weighted_rrf@300 | 314/488 | 0.6434 | +0.0000 |
| B | baseline:weighted_rrf@300 | 707/1136 | 0.6224 | +0.0000 |
| C | baseline:weighted_rrf@300 | 244/464 | 0.5259 | +0.0000 |
| D | pair:ALS+R54 | 418/688 | 0.6076 | +0.0102 |
| E | baseline:weighted_rrf@300 | 484/760 | 0.6368 | +0.0000 |
| F | baseline:weighted_rrf@300 | 528/760 | 0.6947 | +0.0000 |
| G | baseline:weighted_rrf@300 | 368/616 | 0.5974 | +0.0000 |
| H | baseline:weighted_rrf@300 | 740/1080 | 0.6852 | +0.0000 |
| I | baseline:weighted_rrf@300 | 87/144 | 0.6042 | +0.0000 |
| J | baseline:weighted_rrf@300 | 372/616 | 0.6039 | +0.0000 |
| K | pair:ALS+R54 | 736/1248 | 0.5897 | +0.0120 |

**C/K vs F/H Focus**

| category | baseline | best single | delta | best pair | delta | best conditioned | delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| C | 0.5259 | single:R54 0.4655 | -0.0603 | pair:C+R54 0.4828 | -0.0431 | baseline:weighted_rrf@300 0.5259 | +0.0000 |
| K | 0.5777 | single:R54 0.5593 | -0.0184 | pair:ALS+R54 0.5897 | +0.0120 | pair:ALS+R54 0.5897 | +0.0120 |
| F | 0.6947 | single:R54 0.6868 | -0.0079 | pair:R21+R54 0.6908 | -0.0039 | baseline:weighted_rrf@300 0.6947 | +0.0000 |
| H | 0.6852 | single:R54 0.6306 | -0.0546 | pair:ALS+R54 0.6769 | -0.0083 | baseline:weighted_rrf@300 0.6852 | +0.0000 |

**LR-Bury Analysis**

For C/K current RRF@300 misses, `source_union_top300` means the GT is already in at least one existing source within the pool-comparable top-300 depth but was diluted out of the current all-source RRF@300. `cached_union_only` is mostly B/C depth beyond 300. `unscored` means absent from the current expR55 RRF@300 pool and therefore absent from its LR top-50 capture.

| group | baseline misses | in source_union_top300 | cached_union_only | outside cached union | appears in expR55 top50 | unscored | best-conditioned recovered | recovered in expR55 top50 | recovered unscored |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 220 | 49 | 10 | 161 | 0 | 220 | 0 | 0 | 0 |
| K | 527 | 177 | 20 | 330 | 0 | 527 | 80 | 0 | 80 |
| C+K | 747 | 226 | 30 | 491 | 0 | 747 | 80 | 0 | 80 |

**Sanity**

- Recomputed weighted_rrf@300 pool_hit: 4976/8000 (0.6220).
- Cached-depth source-union hit: 6139/8000 (0.7674).
- Top50 `gt_in_pool` mismatches vs recomputed baseline: 0.

**Verdict Line**

ARCHIVE_G1 with reason: oracle ceiling +0.0028 is below +0.010.
