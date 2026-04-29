# CV5 Leakage Audit — Phase A1

## Summary

The CV5 protocol has **no session-level leakage**. Folds are session-level, indices are track-metadata-only, features are session-independent.

## A1.2 Fold Construction

**Location:** `scripts/r3_confirm_400_deterministic.py` lines 250-259, `scripts/tune_postrank_v23.py` lines 344-363.

**Verification:**

- [x] Folds split at the **session level**. `cv_folds()` hashes on `session_id`, round-robin assigns each session to exactly one fold.
- [x] Held-out sessions are excluded from training via complement construction: `train = [i for i in range(n) if i not in held]`.
- [x] Only one turn per session is evaluated (last user turn), so no intra-session leakage is possible.

## A1.3 BM25 Corpus

**Confirmed:** BM25 index is built from `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` using only `track_name, artist_name, album_name, tag_list`. No conversation text.

**Index sharing:** BM25 and FAISS indices are built over the global track catalog, not session data. Sharing across folds is safe — the track catalog exists independently of which sessions reference the tracks.

## Known Issues (non-blocking)

1. `r3_validation_bundle.py` E3 "fast" mode fits Powell on all data including held-out fold. Acknowledged in code; used only for relative config ranking, not final metrics.
2. `post_rerank_sweep.py` does full-data grid search without CV. Diagnostic script only.

## A1.1 Ablation-Based Bounds

See table below (computed by `scripts/expA1_ablation_cv5.py`).

| Config | CV5 nDCG@20 | Δ vs full | Flag |
|---|---|---|---|
| Full (A'+B+C+D+F) | 0.1551 | — | — |
| No F (CF-BPR) | 0.1537 | -0.0014 | OK |
| No A' (qwen3 max-recent) | 0.1542 | -0.0009 | OK |
| No D (qwen3 neighbors) | 0.1543 | -0.0008 | OK |
| No A' no D | 0.1526 | -0.0025 | OK |
| B+C only (BM25) | 0.1513 | -0.0039 | OK |

All sources below the 0.02 flag threshold. No leakage detected.
Evaluated on 8000 all-turn cases, 5-fold CV × 5 seeds.
Artifact: `exp/eval/expA1_ablation_cv5.json`
