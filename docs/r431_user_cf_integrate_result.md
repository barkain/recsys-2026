# R431 — Public User-CF Full Integration Result (ARCHIVE)

**Date:** 2026-06-05
**Verdict:** **FAIL primary gate → ARCHIVE.** Production stays **R106 A-clean (0.6377)**.
**One-line:** Official public user embeddings (cf-bpr) are **recall-positive but non-converting** —
the full 3-arm OOF LR integration gains only **+0.0017 all-cases nDCG@20**, 5.8× below the +0.010
transfer-safe bar, and is fold-unstable in the R90/R103 non-transfer signature.

## What was run

The R103 added-source harness (`scripts/expR103_integrate_eval.py`) with the official user-cf
source lists as the added source (`cache/r431_user_cf/user_cf_oof_lists.json`, weight 1.0).
5-fold OOF, three sibling-LR arms (OOF-vs-OOF, never vs in-sample R54c):

- **A** base — pool = R54-stacked RRF, feats = R39+r84 (37). OOF analog of current best system.
- **B** aug-pool — pool += user-cf source, feats = R39+r84 (37). Isolates the **pool** effect.
- **C** full — pool += user-cf, feats = R39+r84 + `usercf_{rank_inv,presence,cosine}` (40). Full treatment.

Output: `exp/eval/expR431_user_cf_integrate_w1.json` (elapsed 7611s on Mac).

## Result — C vs A (primary) and the decomposition

| metric | A | C | **ΔC vs A** | ΔB vs A (pool) | ΔC vs B (features) | gate | pass |
|---|---|---|---|---|---|---|---|
| **nDCG@20 all (8000)** | 0.22559 | 0.22733 | **+0.00173** | +0.00132 | +0.00041 | ≥+0.010 | ❌ |
| h7 nDCG (1000) | 0.24930 | 0.24974 | +0.00044 | +0.00422 | −0.00378 | ≥0 | ✅ |
| same-artist (2857) | 0.44579 | 0.44700 | +0.00121 | −0.00063 | +0.00184 | ≥−0.005 | ✅ |
| diff-artist (5143) | 0.10327 | 0.10530 | +0.00203 | +0.00241 | −0.00038 | ≥0 | ✅ |
| top-1 churn /80 | | | 29.21 | | | ≤30 | ✅ |
| overlap@20 | | | 16.55 | | | ≥16 | ✅ |
| h7 recovered/lost | | | 31 / 15 (+16) | | | rec≥lost | ✅ |

**The primary gate fails decisively** (+0.0017 ≪ +0.010). Every secondary gate passes, but only
barely (churn 29/30, overlap 16.6/16) — and they are moot once the headline gain is 5.8× short.

## Why it fails

1. **Almost all the (tiny) gain is pool admission, not user-cf intelligence.** B-vs-A = +0.00132
   of the +0.00173 total; the user-cf *features* add only +0.00041 all-cases and **−0.0038 on h7**
   (they hurt the hardest slice). `usercf_presence` LR importance is trivial (30 vs ~15k for
   rank/cosine); total user-cf feature fraction 1.8%.
2. **Unique conversion is poor.** Of 45 GTs admitted to the pool *only* because of user-cf, just
   **9 reach top-20** (20%); h7: 3/5. Pool_hit rose only +25/8000.
3. **Fold-unstable — R90/R103 non-transfer signature.** Per-fold h7 ΔC-vs-A:
   +0.0177 / +0.0042 / **−0.0109** / +0.0001 / **−0.0089**. Folds 2 & 4 strongly negative.
   A +0.0017 all-cases dev gain with this spread is exactly the regime that did **not** survive
   blind before (R90 blind −0.0185; R91/R103 had no churn-safe delivery rule). See
   [[feedback_r90_blind_nontransfer]], [[project_r103_plan]].

## Note on the script verdict

`expR103_integrate_eval.py` writes `"verdict": "PROCEED_TO_BLIND"`, but that field uses the
**R103 GTE gate** (h7 recovered>lost OR h7 Δ≥+0.005), *not* the R431 task's +0.010 all-cases
nDCG bar. Against the R431 gate the run **fails**. Do not be misled by the embedded field.

## Consistency with prior R431 evidence

- R180 (`exp/eval/expR180_user_embedding_audit.json`): user-cf recovers 257 GTs absent from
  union@300 — **real recall** (recall@300 0.157).
- R431 fast policy (`docs/r431_user_cf_fast_policy.md`): first user-cf candidate outside prod
  top-20 is the GT only ~0.06% of the time; direct blend dNDCG +0.000016.
- R431 full integration (this doc): the sibling LR **cannot convert** that recall — +0.0017.

All three agree: **recall-positive, precision-poor, non-converting.**

## Conclusion

- Public user embeddings are **closed** for Blind-A nDCG. They join the exhausted nDCG levers
  (text retrieval, cross-encoder, CF/behavioral, popularity/selection-policy, pool admission).
- This is the same wall as R103 GTE: orthogonal recall that frozen-/sibling-LR ranking cannot
  monetize because the unique GTs are text-/feature-undetermined.
- The remaining leaders' nDCG gap is the hidden source-session/pool reconstruction documented in
  `docs/blind_a_nDCG_investigation_findings.md` — **not** an ordinary public-asset lever.
- **Production remains R106 A-clean (0.6377).** No more Mac integration sweeps; no weight sweep
  (w=1.0 is clearly negative, so per spec we do not sweep {0.15,0.3,0.5}).

## R431b — pool-admission-only (Arm B) + GT-independent selective rule (2026-06-05)

Arm B (user-cf expands the pool, LR ranks with the existing 37 features — no user-cf features)
was the cleaner half of R431 (ΔB-vs-A h7 +0.0042 vs C's +0.0004). R431b recomputed only arms
A/B (`scripts/expR431b_pool_only_selective.py`, 5-fold OOF, reproduced ΔB-A all +0.0013 / h7
+0.0042 exactly), dumped per-case A/B nDCG + **B-side GT-independent signals**, and searched for
a selective deployment that keeps B's wins (`scripts/expR431b_rule_search.py`).

**The oracle reframes it: this is NOT a ceiling problem — it is a discoverability problem.**

| selection | dAll | dH7 | churn/80 | note |
|---|---|---|---|---|
| always-B | +0.0013 | +0.0042 | 28.4 | baseline B |
| **ORACLE** (keep ndB>ndA) | **+0.0236** | **+0.0252** | 5.2 | the wins exist, well above targets |
| best GT-indep rule (dAll) | +0.0014 | +0.0042 | 28.4 | = always-B; selection adds nothing |
| `has_ucf_in_b20` (cleanest) | +0.0010 | +0.0039 | 2.6 | safe but 4× short of +0.004 |
| target | ≥+0.004 | ≥+0.006 | safe | — |

The oracle (1143 cases where B truly beats A) reaches **+0.0236 all / +0.0252 h7** — comfortably
above both submit targets. But **no GT-independent signal locates those cases.** Every rule built
on user-cf cosine / user-cf rank / presence / production confidence margin (`a_margin`) either
collapses to always-B (+0.0013, churn 28) or cuts churn while shrinking the gain proportionally
(+0.0002–0.0010) — it never *concentrates* the wins. always-B's +0.0042 h7 comes with 28.4/80
churn and slightly negative same-artist: the unsafe, non-transferable R90/R107 profile.

**Verdict: FAIL → ARCHIVE R431b.** The cases where user-cf admission helps are, at submit time,
indistinguishable from where it hurts — the fast-policy 0.06%-precision wall, confirmed one level
deeper (after LR re-ranking, not just raw injection). **Public user embeddings are fully closed
for nDCG.** Production remains R106 A-clean (0.6377). No selective user-cf candidate is built.

### Artifacts
- `exp/eval/expR431_user_cf_integrate_w1.json` — full 3-arm (A/B/C) metrics
- `exp/eval/expR431b_pool_only.json` + `_percase.json` — R431b A/B aggregate + per-case dump
- `exp/eval/expR431b_rule_search.json` — selective-rule search (oracle + all rules)
- `scripts/expR431b_pool_only_selective.py`, `scripts/expR431b_rule_search.py`
- `exp/eval/expR180_user_embedding_audit.json` — recall audit
- `exp/eval/expR431_fast_policy_eval.json`, `docs/r431_user_cf_fast_policy.md` — fast policy
- `cache/r431_user_cf/user_cf_oof_lists.json` — user-cf source lists (119 MB, local only)
