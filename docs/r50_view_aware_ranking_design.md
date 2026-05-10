# R50: View-Aware Multi-Query Ranking

**Status:** Design (approved for implementation)
**Depends on:** R39 baseline, R49A-Opt2 post-mortem

## 1. Motivation

R49A-Opt2 showed that RRF-fusing 3 query views into one source dilutes R21 signal. But multi-query is not dead — the failure was architectural. The fused approach hides per-view evidence from LambdaRank, which cannot learn when to trust each view.

**Core insight:** Alternate query views should be **conditional evidence** (per-candidate features the ranker learns from), not a blind fused retrieval source.

## 2. Architecture

```
R39 pool (300 candidates, unchanged)
    ↓
For each candidate, compute per-view R21 signals:
    ├── current_view:    cosine only (rank/presence already in R39 as r21_rank_inv, r21_presence)
    ├── last_query_view: rank_inv, presence, cosine
    └── history_view:    rank_inv, presence, cosine
    ↓
Add cross-view features:
    ├── max_cosine across views
    ├── n_views_present
    ├── rank_std / disagreement
    └── current_minus_lastquery cosine gap
    ↓
LambdaRank over R39 album features + view features
```

**Key difference from R49A:** Do NOT fuse views before learning. Preserve each view as separate evidence.

## 3. Feature Set

### Per-view features (7 new features, no current-view duplication)

R39 already has `r21_rank_inv` and `r21_presence` for the current view. Do NOT duplicate these.

- `current_cosine` — cosine(current_query_emb, candidate_track_emb). Dense signal for all pool candidates.
- `lastquery_rank_inv` — 1/rank if candidate in last_query top-300, else 0
- `lastquery_presence` — 1 if candidate in last_query top-300, else 0
- `lastquery_cosine` — cosine(last_query_emb, candidate_track_emb). Dense for all pool candidates.
- `history_rank_inv` — 1/rank if candidate in history top-300, else 0
- `history_presence` — 1 if candidate in history top-300, else 0
- `history_cosine` — cosine(history_query_emb, candidate_track_emb). Dense for all pool candidates.

**Cosine is computed for EVERY R39 pool candidate**, not only top-300 hits. This preserves "almost relevant" information for candidates absent from a view's top-300.

### Cross-view features (4 new features)

- `max_cosine_across_views` — max(current_cosine, lastquery_cosine, history_cosine)
- `n_views_present` — count of views where candidate is in top-300 (0-3, using existing r21_presence for current)
- `rank_std` — std of ranks across views where present (0 if only 1 view)
- `current_minus_lastquery_cosine` — cosine gap between current and last_query view

### Total: R39 34 features + 11 view features = 45 features

## 4. Feature-Family Ablations (REQUIRED)

Full feature set may fail due to noise even if one family works. Test these configs:

| Config | Features | Total |
|--------|----------|-------|
| baseline | R39 (34) | 34 |
| cosine_only | R39 + current/lastquery/history cosine + max_cosine | 38 |
| alt_rank_only | R39 + lastquery/history rank_inv + presence | 38 |
| cross_view_only | R39 + n_views_present + max_cosine + rank_std + cosine_gap | 38 |
| full_safe | R39 + all 11 view features (no current rank/presence duplication) | 45 |

## 5. Data Requirements

Per-view OOF top-300 lists from R49A-Opt2:
- `cache/r49a_view_current_oof_lists.json`
- `cache/r49a_view_lastquery_oof_lists.json`
- `cache/r49a_view_history_oof_lists.json`

Per-fold track embeddings from R49A:
- `cache/r49a_fold_{0-4}_track_embs.npy`

**Missing:** Per-view per-fold query embeddings (R49A saved lists but not raw embeddings). Must re-encode queries and cache them (~15 min total).

### Cache Manifest (REQUIRED)

Every embedding cache must have a manifest recording:
- `fold_id`, `model_dir`, `track_text_format` (must be `r21_album_tags`), `view_name`, `created_at`, `n_tracks`, parity stats vs existing OOF
- Refuse to run if manifest is missing or stale

## 6. OOF Protocol

- Same 5-fold split: `grouped_session_folds(seed=0)`
- Per-view lists verified OOF-clean (99.9% parity across all folds)
- R39 pool construction unchanged (7-source RRF, R21 current view at weight 1.0)
- Album features unchanged
- LambdaRank params unchanged (300 rounds, 31 leaves, lr=0.05)

## 7. Implementation Plan

### Phase 1: View-aware features only (no pool change)

1. Re-encode per-view query embeddings per fold, save with manifest
2. Load R49A per-view OOF lists + track/query embeddings
3. Build R39 pool (unchanged)
4. For each candidate in pool, compute view features (cosine for all candidates, rank/presence for top-300 hits)
5. Run CV5 LambdaRank for all 5 configs (baseline + 4 ablations)
6. Report h7, same/diff, recovered/lost, feature importance per config

### Phase 2: Pool expansion (only if Phase 1 passes gate)

If view features improve h7, test expanding pool with candidates found by non-current views but missed by current-view RRF.

## 8. Gates

| Gate | Metric | Threshold |
|------|--------|-----------|
| Primary | OOF-clean Δh7 | >= +0.010 |
| Required | Baseline reproduction | h7 = 0.24298 |
| Required | Recovered > lost | net >= +20 |
| Check | same/diff | non-negative |
| Check | Feature importance | View features should appear in top-15 |

If Phase 1 fails: do not proceed to Phase 2. Conclude "view-aware R21 features do not help LambdaRank." A learned view router or neural scorer would still be conceptually different, but not the next step unless signal is seen.

## 9. Risks

1. **Feature dilution:** 11 new features may give LightGBM overfit surface (R49C pattern). Mitigated by ablation configs testing smaller families.
2. **Cosine computation cost:** Need per-fold embeddings for 8000 cases × 300 pool × 3 views. Manageable with embedding cache.
3. **Partial redundancy:** `current_cosine` may correlate with existing `r21_rank_inv`. But cosine is dense (all candidates) while rank_inv is sparse (top-300 only), so there is new information.

## 10. Expected Cost

- Query re-encoding: ~15 min
- Feature building: ~10 min
- CV5 LambdaRank × 5 configs: ~25 min
- **Total: ~50 min**

## 11. What This Tests

**The real question:** Can LambdaRank learn to use per-view R21 signals conditionally — trusting the history view for same-vibe cases but ignoring it for explicit-request cases?

If yes: view-aware ranking is a viable path forward.
If no: view-aware R21 features do not help LambdaRank. Focus shifts to entirely different approaches.

## 12. R50 Phase 1 Results: ARCHIVED AS NEGATIVE (2026-05-10)

| Config | h7 | Δh7 | cv5 | n_feat |
|--------|------|------|------|--------|
| baseline | 0.24298 | — | 0.21872 | 34 |
| cosine_only | 0.24158 | -0.001 | 0.21883 | 38 |
| alt_rank_only | 0.24073 | -0.002 | 0.21949 | 38 |
| cross_view_only | 0.24318 | +0.0002 | 0.22151 | 38 |
| full_safe | 0.24250 | -0.0005 | 0.22026 | 45 |

- Per-view and cross-view R21 evidence is learnable by LightGBM (max_cosine = #2 feature by gain) but not useful for h7 ranking.
- The issue is not RRF fusion alone; alternate R21 query views do not provide robust ranking signal inside the R39 pool.
- 994/1000 top-20 churn for near-zero gain confirms instability.
- **Stop: no pool expansion, no blind, no further multi-query R21 variants.**
