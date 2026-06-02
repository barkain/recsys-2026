# R107 — Personalization-as-content (Lever 2) — ARCHIVE (blind-negative)

**Verdict:** ARCHIVE. Production stays **R106 A-clean (composite 0.6377)**.

## What it was
The leaderboard reframe's 3-lever plan: (1) build a calibrated Gemini-2.5-Pro local
judge gate, (2) use it to add back listener-specific personalization the response
evolution had eroded, (3) an nDCG selection-policy probe. R107 executed levers 1–2.

- **Lever 1 (judge gate):** an absolute per-aspect Gemini-2.5-Pro judge FAILED the strict
  ordering gate (couldn't resolve the 0.05 band between our 4.85/4.90 submissions). A
  PAIRWISE judge (`scripts/judge_gemini_pairwise.py`, order-counterbalanced, consistent-only
  wins) PASSED validation: it correctly ranked R106 (4.90) > R74 (4.85) and revealed R106
  had *traded personalization for explanation* vs R74 (pers −11, expl +18).
- **Lever 2 (R107):** 50 headroom cases rewritten by Opus subagents (operationalize
  `listener_goal` + `preferred_musical_culture`, strip bolted-on demographic recitals,
  facts intact, ±15% length), gated vs R106 → **19 accepted** (win ≥1 aspect, lose 0),
  zero local regressions, batch-2 personalization net +11 (recovering the exact deficit).

## Blind result (official scorer)
| metric | R106 | R107 | Δ |
|---|---|---|---|
| nDCG@20 | 0.5073 | 0.5073 | 0 (tracks identical) |
| CatalogDiv | 0.0301 | 0.0301 | 0 |
| LexDiv | 0.8859 | 0.8821 | −0.0038 |
| **LLM judge** | **4.90** | **4.85** | **−0.05** |
| **composite** | **0.6377** | **0.6336** | **−0.0041** |

Composite formula held exactly (−0.05·0.0753 − 0.0038·0.095 = −0.0041). We got the **LLM
direction wrong**: the local gate said the 19 edits were ≥ R106; the official judge scored
them lower on average.

## Why it failed (the durable lesson)
A pairwise LLM judge can DISCRIMINATE two genuinely-different existing submissions but
CANNOT safely GATE generated edits: (1) **Goodhart** — optimizing edits to win "which is
MORE personalized?" decouples from the real ABSOLUTE 1-5 score; (2) the official judge is
absolute + naturalness/volume-sensitive, which isolated per-response pairwise comparison
can't see. This reproduces the R106b pattern (4.90→4.85 once >~15 rows edited) and extends
the proxy-judge warning to the RIGHT model (Gemini-2.5-Pro). See memory
`feedback_pairwise_gate_no_transfer`.

## Consequences
- **LLM-judge response lever CLOSED at 4.90** (R78/R84c/R86/R87/R106b/R107 all confirm).
- Do NOT submit response variants unless near-byte-identical to A-clean.
- Reusable assets kept: `scripts/judge_gemini_pro.py`, `scripts/judge_gemini_pairwise.py`
  (good for RANKING existing submissions, not optimizing edits), key at `~/.gemini_key_recsys`.
- Next Blind-A effort → nDCG via the new-evidence direction (official user embeddings /
  Gemini selection-policy modeling), NOT response edits.
