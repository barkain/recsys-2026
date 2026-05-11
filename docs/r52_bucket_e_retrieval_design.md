# R52: Bucket E/D Retrieval Recovery

## 1. Objective

Identify new retrieval sources that recover GT tracks currently unreachable by the 7-source pool. Primary metric: unique Bucket E/D recovery, not global CV5. Only integrate sources that improve h7 pool coverage without same-artist collapse.

## 2. Background

R39 album-aware LambdaRank (h7=0.24298) is production best. R42 error decomposition on 1000 h7 cases:

| Bucket | Description | Count | % |
|--------|-------------|------:|---:|
| A (hit) | LR rank <= 20 | 422 | 42.2 |
| B | rank 21-100 | 149 | 14.9 |
| C | rank > 100 | 27 | 2.7 |
| D | pool miss, in source union | 145 | 14.5 |
| E | unretrievable by any source | 257 | 25.7 |

Ranking improvements are exhausted (R41a, R46, R49C, R49A, R50, R51 all failed). The next nDCG gain requires bringing new candidates into the pool.

Current 7 sources: A (Qwen3 track sim), B (BM25 title), C (BM25 combined), D (track neighbors), F (CF-BPR), ALS, R21 (supervised BGE dense).

**IMPORTANT:** Bucket labels must be reproduced fresh from the current R39 pipeline, not from stale R42 artifacts. First implementation step must reproduce R39 h7=0.24298 and rebuild bucket assignments.

## 3. Evaluation Framework

- Reproduce R39 h7 and rebuild bucket labels from current R39 pipeline.
- **Primary:** how many Bucket E cases does the new source bring into hit@300?
- **Secondary:** how many Bucket D cases move to pool-hit?
- **Tertiary:** after LambdaRank re-ranking with expanded pool, does h7 improve?
- Always OOF-clean on dev.
- **Gate:** pool_hit improvement >= +0.02 AND h7 non-negative after LambdaRank, OR direct h7 +0.010.

## 4. Candidate Retrieval Sources

### A. Provided embedding spaces (diagnostic first)

Check what embedding types are available in the competition dataset:

- **Audio CLAP embeddings** — R36 tested CLAP as both features and retrieval/pool expansion globally and it failed integration. Re-evaluate specifically against Bucket D/E; only consider if it recovers unreachable cases without same-artist damage.
- **Lyrics embeddings** (if available).
- **Attribute/genre embeddings** (if available).
- **Metadata embeddings** (if different from current BM25).

For each: compute nearest-neighbor retrieval from user's recent tracks, report GT hit rate specifically on Bucket E/D cases.

### B. User-profile retrieval

If user embeddings or listening profiles are available:

- User embedding -> nearest tracks in embedding space.
- Report blind user coverage separately.
- **Leakage warning:** If user embeddings/profiles are derived from all interactions (including dev labels), treat them as potentially transductive. Diagnostic is allowed, but no blind unless the same feature is available for blind users without using hidden labels.

### C. Catalog-neighbor expansion

Multi-hop retrieval: user -> recent tracks -> neighbor tracks.

For each of the user's recent played tracks, find neighbors in multiple embedding spaces (CLAP audio, lyrics/semantic, tag/genre). Aggregate across recent tracks, weighted by recency.

### D. Cross-modal retrieval

Query text -> audio/lyrics embedding space (if bridge exists). Catches cases where the user describes a mood/vibe that text retrieval misses but audio/lyrics captures.

## 5. Diagnostic Protocol (Phase 1)

Before building any new source:

1. Reproduce R39 h7=0.24298 and rebuild bucket E/D labels from current pipeline.
2. Inventory all available embedding types in the competition dataset.
3. For each candidate source, compute top-300 retrieval lists on dev (OOF where applicable).
4. Report per source:
   - h7 Bucket E hit@300 (primary)
   - h7 Bucket D hit@300
   - unique recoveries vs R39 source union
   - same-artist / diff-artist split of recovered cases
   - overlap with R21 / R39 pool
   - source standalone h7@20, h7@100, h7@300
5. Rank sources by unique Bucket E recovery.
6. **Pass threshold:** >= 40 unique Bucket E recoveries, OR >= 25 Bucket E recoveries with strong diff-artist skew and low same-artist damage.

## 6. Integration Protocol (Phase 2, only if Phase 1 passes)

For sources that pass Phase 1 diagnostic, test separately:

**6a. Feature-only (no pool change):**
- Add source rank/presence/similarity as features on unchanged R39 pool
- Tests whether the source's ranking signal helps within existing candidates

**6b. Pool-add (expanded candidate set):**
- Add source candidates to RRF fusion at low weights (0.3, 0.5, 1.0)
- Rebuild LambdaRank with new source features
- Gate: h7 +0.010 or pool_hit +0.02 with h7 non-negative

**6c. Gated pool-add (conditional):**
- Only add source candidates for cases matching diff-artist / discovery profile
- Avoids same-artist damage on cases where current pool is already good

## 7. OOF Protocol

- Any learned retriever must use 5-fold OOF (same folds as R21).
- Provided embeddings (CLAP, lyrics, etc.) are allowed as-is (not learned on dev).
- User profiles: if derived from all interactions, treat as transductive. No blind use unless available for blind users without hidden labels.
- Save fold manifests for all new sources.

## 8. What NOT to Do

- Do not optimize global CV5 without checking bucket-specific recovery.
- Do not add sources that mostly retrieve same-artist tracks.
- Do not skip OOF for learned sources.
- Do not combine sources before proving individual Bucket E recovery.
- Do not submit blind unless h7 improvement is clear.
- Do not include hard-negative supervised retriever training in R52 — that belongs to a separate R53 if needed. R52 focuses on source discovery from existing/provided signals.

## 9. Expected Cost

| Phase | Effort |
|-------|--------|
| Phase 1 (diagnostics) | ~2-4 hours (depends on available embeddings) |
| Phase 2 (integration) | ~1-2 hours per source that passes diagnostic |

## 10. First Step

Reproduce R39 h7 and rebuild bucket labels. Then inventory all available embedding types and retrieval signals in the competition dataset. Many may already be cached or loadable from HuggingFace. The diagnostic is cheap if embeddings exist.
