# R95 — Oracle Probe Forensics

Date: 2026-05-29

## Question

After ~14 Codabench single-row probes (R92 full-swaps, R93 reorders, R94
injections) we have **one** positive (R92 `p11`, +0.0004) and it is production
(0.5073). Is there any offline feature that separates the win from the 10 flat
and 3 negative probes — i.e. a sharper policy worth more slots?

Source: `scripts/expR95_oracle_forensics.py` → `exp/eval/expR95_oracle_forensics.json`.

## The full labeled ledger

```
probe_id                sign    delta t1c ov prior R90 R84 R54 cons a@1 a@h  mgn
r92p11_c4f7d055_t7      POS   +0.0004   1 19    4    0   8   2    3   1   1  0.20   <- the only win
r92p01_77faef85_t1      flat  +0.0000   1 10    8   64 147 129    2   0   0  0.32
r92p03_0802ac4a_t6      flat  +0.0000   1 14    8   13   7   4    5   0   1  0.04
r92p04_46faad58_t2      flat  +0.0000   1 14    3   10   9  18    4   1   1  0.15
r92p05_4b239a62_t2      flat  +0.0000   1 15    1    3   4   1    6   1   1  0.18
r92p07_dc3c1b72_t7      flat  +0.0000   1 15    2    2   1   2    3   1   1  0.01
r92p12_d9cca604_t2      flat  +0.0000   0 20  abs    -   -   -    0   0   0  0.24
r93p01_574f75cf_t2      flat  +0.0000   0 19  abs   44  57   -    0   0   0  0.11
r93p02_68993adf_t1      flat  +0.0000   0 18  abs    -   -   -    0   0   0  0.11
r93p03_ee7bfbda_t3      flat  +0.0000   0 18  abs   11  61 206    1   0   0  0.25
r94p05_ca8cbe02_t6      flat  +0.0000   0 19  abs    7   8   5    5   1   1  1.80
r94p01_9d4ef919_t3      NEG   -0.0001   1 19  abs   13  14   7    5   1   1  0.04
r92p02_ab87371b_t1      NEG   -0.0023   1 13    3    -   -   -    2   0   0  0.04
r92p06_db8ec85f_t7      NEG   -0.0078   1 15    1    6   4   4    3   1   1  0.01
```
`prior` = key track's rank in the production list before the change (abs =
absent); `R90/R84/R54` = its rank in those cosine pools; `cons` = orthogonal-
source consensus; `a@1/a@h` = same artist as old top-1 / as history; `mgn` =
R54 top-1 margin.

## Findings

1. **No feature separates the win.** `p11` is mid-distribution on every signal.
   Flat/negative probes beat it on each one we trusted:
   - **Consensus is falsified.** The three highest-consensus probes (cons=5:
     `r94p01`, `r94p05`, `p03`) are NEG/flat/flat; `p11` had cons=3.
   - **BGE rank doesn't predict.** `p07`'s promoted track was R84-cos **rank 1**
     (better than `p11`'s rank 8) yet flat; `p05` R54-rank 1, flat.
   - **Same-artist doesn't predict.** Six flat/negative probes share `p11`'s
     `artist=top1 ∧ artist∈history`.
   - **Margin doesn't predict the win** (p11 mgn 0.20, several lower-margin flat).

2. **The only robust pattern is the negative one.** All three real losses are
   **margin < 0.05 top-1 swaps** (`p06` −0.0078, `p02` −0.0023, `r94p01`
   −0.0001). Very-low-margin top-1 swaps are pure downside.

3. **Why `p11` actually won.** Its GT (`9d9ca4fe`) was already in production's
   pool at **rank 5**; R90 promoted it to rank 1. The win was recovering an
   in-pool, mis-ranked GT — not introducing a new track. The losing probes
   promoted other tracks to #1 that simply were not GT.

4. **Flat top-1-preserving probes (R93 reorders, `r94p05`, `p12`) imply** that
   for those rows the GT is either already at #1 (locked) or absent from our
   entire candidate universe — reordering/injecting from current sources can't
   move it.

5. **EV is poor.** 14 slots → 1 win → **+0.0004** cumulative. Max single-row
   impact on the 80-case mean is 0.0125; metric resolution is 0.0001. We are
   extracting ~3% of one row's headroom per ~14 slots, and the search is
   feature-blind (≈7% hit rate, unpredictable).

## Verdict & proposed policy

**Single-row blind probing guided by our current features is exhausted.** The
positive signal is unlearnable from this data; the orthogonal-consensus,
same-artist, and BGE-rank heuristics are all falsified as *positive* predictors.
The R90-vs-R84c top-1-disagreement vein (where `p11` came from) is fully mined.

Recommended:
- **Keep R92 p11 (0.5073) as production.** It is the banked ceiling of this method.
- **Stop feature-guided single-row uploads.** The only data-supported rule is
  *exclusionary*: never probe margin < 0.05 top-1 swaps (downside only).
- **To move nDCG materially, change the candidate universe, not the ordering.**
  Most missed GTs appear absent from every pool we have (BGE, BM25, R21, ALS,
  CFBPR, qwen3). A genuinely new retrieval signal (e.g. the A100 BGE-large
  direction, or an external/metadata source) is the only lever with headroom —
  micro-probing is not.

This is an n=1-positive analysis; the conclusion is about EV and the absence of
a separating feature, not a claim that no win exists.
