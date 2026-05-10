# R49 OOF-Clean Next Architecture

Reset document for post-R39 development. All future experiments must follow the OOF discipline defined here.

## 1. Current Production

| Metric | Value |
|--------|-------|
| Model | R39 album-aware LambdaRank |
| Blind nDCG | 0.4798 |
| Composite | 0.6024 |
| Features | 34 (29 base + 5 album) |
| Retrieval sources | 7 |
| Validation | CV5 (5-fold cross-validation) |

## 2. Lessons Learned

### R39 (SUCCESS -- album features transferred)

- Album is a structural metadata attribute: categorical, sparse, interpretable.
- Album features are distribution-agnostic -- they work identically on dev and blind.
- Dev h7 delta = +0.010, transferred cleanly to blind with first h7 improvement since R21.

### R41a (FAILURE -- rare tags did NOT transfer)

- Dev h7 delta = +0.005, passed local gate.
- Blind nDCG dropped from 0.4798 to 0.4569 (delta = -0.023).
- Root cause: 99.8% of tags have df < 500, so rare-tag features are effectively full-tag overlap -- high-dimensional noise that overfits dev distribution.
- Lesson: noisy lexical/tag features can overfit even with modest dev gains.

### R46 (FAILURE -- contaminated by data leakage)

- Original R46 showed h7 delta = +0.01486 expanding R21 depth 300 to 1000.
- Investigation revealed: expanded configs used production R21 embeddings (trained on ALL dev), baseline used OOF lists -- apples-to-oranges comparison.
- OOF-clean rebuild showed h7 delta = +0.002 -- gate FAIL (threshold +0.010).
- The contaminated gain was 7x overstated.
- Pool_hit barely moved with OOF lists (0.6000 to 0.6016 vs contaminated 0.6000 to 0.6616).
- Lesson: any retriever trained on dev and evaluated on dev creates false pool gains. OOF discipline is mandatory.

## 3. Non-Negotiable OOF Rules

1. **Fold-specific models required.** Any learned retriever evaluated on dev MUST use fold-specific models, never the production model.
2. **OOF source lists required.** Any generated source list used in dev training MUST be out-of-fold.
3. **Production model = blind only.** The production model is allowed ONLY for blind inference on unseen test data.
4. **No mixed comparisons.** Comparing OOF baseline vs production candidate is INVALID. All configs in a comparison must use the same OOF source.
5. **Fold manifests mandatory.** Store a fold manifest with every experiment: model path, fold indices, and training data used per fold.
6. **Reproduce before extending.** Verify that the OOF-clean baseline reproduces known results before trusting any expanded configuration.

## 4. Candidate R49 Directions

### A. Multi-query R21 retraining / augmentation (OOF 5-fold)

| Aspect | Detail |
|--------|--------|
| What | Retrain R21 BGE with query augmentation (multiple query views per session) |
| Why | R47 showed oracle best-view brings 12/61 hard cases into top-300, 37/61 into top-1000 |
| OOF feasibility | Must be 5-fold OOF from the start; each fold trains on 4/5 of dev |
| Effort | ~10 hours (5 folds x 2h training each) |
| Risk | Query augmentation may not help if the embedding space is already saturated |

### B. Official user embeddings / profile-aware rank features (OOF-safe)

| Aspect | Detail |
|--------|--------|
| What | Use user profile signals (listening history summary, genre preferences) as rank features |
| Why | Official static user/profile embeddings are allowed as provided resources, but any user aggregation we compute from dev/train interactions must be fold-clean. If a user-level feature is learned or aggregated from dev labels/sessions, it requires OOF construction. |
| OOF feasibility | High -- no learned retriever involved, features are static per user |
| Effort | Medium (feature engineering + CV5 evaluation) |
| Risk | Profile signals may not discriminate at the track level |

### C. Stable structural metadata features only

| Aspect | Detail |
|--------|--------|
| What | Release year/decade, album artist vs track artist, record label, ISRC adjacency if available |
| Why | Like album features (R39), these are categorical/structural and distribution-agnostic |
| OOF feasibility | High -- metadata features require no learned model |
| Effort | Low (feature engineering + CV5) |
| Risk | Limited metadata availability in the catalog |

### D. Conservative expert stacking with OOF expert outputs

| Aspect | Detail |
|--------|--------|
| What | Train a stacking model over OOF outputs from existing sources (A, B, C, D, F, ALS, R21) |
| Why | Current RRF fusion uses fixed weights; a learned stacker could improve combination |
| OOF feasibility | Requires all expert dev predictions to be OOF -- no production model outputs in training |
| Effort | Medium (requires OOF outputs from all 7 sources) |
| Risk | Current sources may already be well-fused by RRF; stacking overhead may not pay off |

## 5. Gates

| Feature Type | h7 Gate | Additional Requirements |
|---|---|---|
| Non-structural (learned, lexical, tag-based) | h7 delta >= +0.010 | Must be OOF-clean |
| Highly structural (categorical metadata) | h7 delta >= +0.005 | Must be distribution-agnostic |
| Retrieval depth/source changes | h7 delta >= +0.010 | MUST use OOF-clean models for ALL configs |
| Any blind candidate | -- | recovered/lost net >= +20 preferred |
| Any blind candidate | -- | same/diff no material regression |
| Any blind candidate | -- | Artifact review by Codex before submission |

### Historical Blind Transfer Calibration

Before any submission, check against transfer history:
- R39 structural album features transferred successfully.
- R41a rare-tag features failed despite passing local h7 gate (+0.005).
- R46 contaminated depth failed after OOF cleanup revealed only +0.002.
- Any candidate with local h7 delta between +0.005 and +0.010 must have a clear, articulable reason it should transfer (e.g., structural/categorical nature, distribution-agnostic computation).

## 6. Process

1. No implementation until this document is reviewed and approved.
2. Each R49 sub-experiment gets its own script with OOF protocol baked in.
3. Results include fold manifest and OOF verification.
4. Blind submissions require passing all applicable gates from Section 5.
