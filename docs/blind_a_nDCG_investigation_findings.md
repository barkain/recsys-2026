# Blind-A nDCG Investigation — Findings & Strategic Conclusion

**Date:** 2026-06-04
**Status:** nDCG comprehensively closed on provided data. Production = **R106 A-clean, composite 0.6377**.
**TL;DR:** Our nDCG (0.5073) is near-optimal on the data *as intended*. The leaders' +0.12 nDCG
comes from reconstructing the hidden candidate pool out of the public LFM-2b source dataset —
which is (a) against the challenge's spirit and (b) practically blocked for us because LFM-2b
has been taken down. We compete clean on our composite-optimal profile (best LexDiv + tied-best
LLM) and shift effort to Blind-B (the final ranking, Jun 23).

---

## 1. The leaderboard situation (2026-06-03 snapshot)

| # | team | composite | nDCG@20 | LexDiv | LLM |
|---|------|-----------|---------|--------|-----|
| 1 | vkost | 0.68 | **0.63** | 0.75 | 4.85 |
| 2 | grishaplod | 0.68 | **0.63** | 0.84 | 4.70 |
| 3 | semintelligence | 0.65 | 0.53 | 0.86 | 4.95 |
| 4 | volart | 0.65 | 0.54 | 0.82 | 4.85 |
| 5 | theoviel | 0.64 | 0.60 | 0.79 | 4.45 |
| 8 | **dirac (us)** | 0.64 | **0.51** | **0.89** | **4.90** |

The entire gap to the top is **nDCG**. We own the best LexDiv and tied-best LLM in the field.
Composite formula (recovered, RMSE 5.6e-5): `0.506·nDCG + 0.095·LexDiv + 0.305·((LLM−1)/4)`.

## 2. The composite levers, and which are closed

### Response side (LLM judge 30%, LexDiv ~10%)
- **LexDiv** saturated at 0.8859; edits beyond the A-clean baseline cost LLM (R106b).
- **LLM judge** confirmed hard-capped at **4.90** across R78/R84c/R86/R87/R106b/**R107**.
- **R107 (personalization-as-content)** — BLIND-NEGATIVE. We built a validated Gemini-2.5-Pro
  *pairwise* judge (it correctly ranked R106>R74 and revealed R106 had traded personalization
  for explanation). Generated 19 gate-accepted personalization rows (zero local regressions,
  pers net +11). Blind result: **LLM 4.90→4.85, composite 0.6377→0.6336 (−0.0041)**. The local
  pairwise gate does NOT transfer to the official judge on *generated edits* (Goodhart +
  naturalness-sensitivity). **LLM response lever is closed.** See `feedback_pairwise_gate_no_transfer`.

### nDCG side (50%) — every lever tested and closed
| lever | experiment(s) | result |
|-------|---------------|--------|
| text retrieval (query→track) | R96, and R422 would repeat it | E5/BGE/Qwen recover 0 missing GTs |
| text ranking / cross-encoder | R69, **R421 v1+v2** | discrimination fixable but signal redundant |
| collaborative filtering / behavioral | R104, R400/R401 | re-rank-dominated, zero conversion |
| popularity / selection-policy | **R108/R109/R420** | real GT-correlate, non-convertible |
| pool broadening / admission | R59–R62, R103 | recovered < lost through frozen LR |

## 3. R421 — the cross-encoder deep-dive (the most thorough ranking attempt)

Hypothesis: our production ranker is **conversation-text-blind** (it ranks on source-ranks and
cosines, never reads the dialog against each candidate). A cross-encoder that jointly reads
(conversation + candidate metadata) should crack the 0.238 ranking headroom (GT in pool@300 but
ranked outside top-20).

- **v1** (BGE-reranker-v2-m3, 1 epoch, easy negatives): FAIL. Pure-CE ΔnDCG −0.224; it demoted
  even *explicitly-named* GTs (user asked for "Heart-Shaped Box", CE ranked "Lithium" #1). It
  learned coarse genre/artist matching, not track-identity discrimination.
- **v2** (hard negatives = metadata-NN + same-album + same-artist): discrimination **fixed** —
  named-track exact-GT-#1 0%→54%, GT-rank median 44→21, easy-case preserve 0.518→0.765. BUT all
  three deployments fail the +0.02 gate: blend +0.0002, selective +0.0002, **CE-as-feature
  −0.0010** (the correct deployment per `feedback_rerank_only_closed`). Epoch 1 overfit (dev
  regressed while train loss fell).
- **Root cause:** the CE signal is **redundant with production retrieval**. Where the CE is
  confident and right, production already has the GT (named/easy cases); where production fails
  (the hard headroom), the CE is *also* wrong because those GTs are **text-undetermined**.

## 4. Why nDCG is genuinely capped — the recall-ceiling diagnostic

Dev OOF: **28% of GTs (2,261/8,000) are not in our union@300** (recall ceiling). Of those:
- **0%** are cold (all have embeddings — retrievable in principle)
- **only 1% (18/2,241)** have their title named/described in the conversation

→ **99% of the unreachable GTs are vibe-only.** The conversation says "something dark and
atmospheric"; Gemini picked one specific track among many that fit, and nothing in the text
points to it. No retriever or ranker can extract what the text doesn't determine.

## 5. How the leaders actually do it (the mechanism)

From the **TalkPlayData-2 paper** (arXiv 2509.09685, Algorithm 1 + Appendix B.4):
- Each real **LFM-2b** listening session is reduced to a **recommendation pool of 16–32 tracks**
  (sampled from that user's session, minus 5 profile tracks).
- The Recsys LLM (Gemini-2.5-Flash) is instructed *"recommend ONLY from the provided available
  tracks"* → **the GT is always one of those 16–32 tracks.**

So the GT is always inside a 16–32-track pool drawn from the user's **public Last.fm session**.
**Reconstruct that pool → recall ~100% and ranking collapses from 47k candidates to ~24** →
exactly the +0.12 nDCG the leaders show. This is not a better model; it is recovering the hidden
candidate set from the source dataset. The "vibe-only" GTs are only undetermined to *us* — the
reconstruction teams simply have the 24-item candidate list.

**Our fingerprint is strong:** `user_profile` (age/country/gender — LFM-2b has exactly these) +
`session_date` (matches LFM-2b timestamps) + the conversation's played tracks (specific
name+artist listening events) + ISRC codes in our Track-Metadata. Listening to ~4+ specific
tracks on a specific date is information-theoretically a near-unique fingerprint.

## 6. Rules & feasibility verdict

**Rules** (music-crs-challenge `terms.html`, checked 2026-06-04): **no explicit prohibition** on
external data (LFM-2b/Last.fm/Spotify) or on pool/GT reconstruction. Dataset is CC BY-NC 4.0,
no redistribution. Organizers provide embeddings *"to focus on model architecture"* → reconstruction
is clearly **against the spirit**, even if not the letter.

**Feasibility:** **BLOCKED for us.** The official LFM-2b host (cp.jku.at/datasets/LFM-2b) states:
*"The dataset is not available for download anymore due to license issues."* Our own data exposes
only the ~8 recommended tracks per session (not the full 16–32 pool), so reconstruction genuinely
needs LFM-2b — which is gone. The leaders who reached 0.63 likely had **pre-takedown or
institutional** access. Pursuing it now would require an unofficial mirror of a dataset removed
for license/privacy reasons — worse on every axis.

## 7. Strategic conclusion

- Our **nDCG 0.5073 is near-optimal** on the data as intended; the gap to the leaders is
  **source-data access, not modeling skill.**
- We hold the **composite-optimal profile** (best LexDiv + tied-best LLM). R106 A-clean (0.6377)
  is a legitimately strong, defensible position.
- **Do NOT chase the leakage** via unofficial LFM-2b mirrors or de-anonymization: reconstructing
  test candidate pools is effectively answer-key leakage and likely disqualifying if audited,
  even though the rules don't explicitly forbid external data. Top nDCG here may reflect
  source-session access, not better recommender modeling.
- **Stop the nDCG chase** (conclusively closed) and **focus on Blind-B** (the final ranking, Jun
  23). The prior hard blocker — R54 phase-3 fold 1–4 model weights — is **RESOLVED (2026-06-04)**:
  the weights were restored from a Drive backup (`r54_phase3_full_folds1_4.zip`, not lost), and the
  ensemble now reproduces the Blind-A R54 lists **80/80 sessions bit-for-bit**. All 5 ensemble folds
  are present locally and backed up off-machine (see `docs/blind_b_artifact_backup_manifest.md`).
  Blind-B replay is a one-line run: `expR54_phase3_ensemble_blind.py --blind-name blind_b`.

## 8. Reusable assets from this investigation
- `scripts/judge_gemini_pro.py`, `scripts/judge_gemini_pairwise.py` — Gemini-2.5-Pro judges
  (good for *ranking existing* submissions, not for gating generated edits).
- `scripts/expR421_phase0_data.py` / `_phase1_train_crossencoder.py` / `_phase2_blend_gate.py`
  / `_phase2b_ce_feature.py` — cross-encoder reranker pipeline + trained `model_v2` (on Drive).
- `scripts/expR108/109/420_*` — selection-policy / popularity probes.
- `scripts/exp_goal65_eval.py` — the dev OOF nDCG harness + gate suite.

### Key sources
- TalkPlayData-2 paper: https://arxiv.org/html/2509.09685v1
- Challenge: https://nlp4musa.github.io/music-crs-challenge/ ; https://www.codabench.org/competitions/15786/
- LFM-2b (taken down): http://www.cp.jku.at/datasets/LFM-2b/
- Evaluator: https://github.com/nlp4musa/music-crs-evaluator
