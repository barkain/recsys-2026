## 1. New information added beyond R54c

The only credible new signal here is ordered behavior from train-split session transitions:

```text
prior music_turns + user utterances so far -> next music track
```

R54c already uses train-split data, but `scripts/expR54_phase3_full5fold_train.py` reduces it to independent BGE contrastive pairs. It samples 20,000 train-split `(structured query, positive track text)` pairs with `MAX_PAIRS_PER_SESSION=2`, where the query contains the current user message, up to three prior user utterances, and up to five prior played track references. The loss is in-batch query/text contrastive retrieval against track metadata text. That teaches semantic compatibility between a conversation string and a GT track description.

It does not explicitly learn:

- ordered item-to-item transitions such as "after A -> B -> C, D is likely next";
- whether a user utterance is a continuation, pivot, exclusion, or refinement relative to the immediately prior recommendations;
- session-local trajectory shape across all music turns, since most train transitions are discarded by the 20k cap and max 2/session sample;
- behavior-native item identity, because the output side is track metadata text, not a learned ID+metadata item vector trained to predict next behavior;
- repeated transition statistics across train sessions, such as artist-to-artist, tag-to-tag, popularity-conditioned, or context-conditioned next-track movement.

The current R39+R54+LambdaRank stack already sees strong content and source-rank evidence: R54/R21 dense ranks, BM25 ranks, ALS/popularity, album/tag/artist-style features, and R54 cosine/presence features. R56 and R58 show that rearranging those existing signals is not enough. R56 source-rank protection had every variant fail; even the ORACLE true `diff_artist` gate recovered 15 DEMOTED cases but lost 26 HIT cases. R58's learned top-50 LightGBM specialist also failed across 4 configs x 7 betas; every beta > 0 regressed h7 and same-artist nDCG. So a behavioral sequence model must not be another wrapper around the existing top-50 ordering.

The viable mechanism is a new first-stage retrieval source or a candidate score that changes pool membership by adding behavior-derived candidates. The strongest version would be hybrid rather than sparse R31:

```text
input tokens: prior track ID embedding + fixed metadata embedding + turn/user-utterance embedding + position
encoder: small causal Transformer/GRU over the session
output: dot product against content-initialized trainable item vectors, or dual ID+metadata heads
```

This differs from the previous R31 sparse path. The archived R31/R31v1 diagnostics were too ID-sparse: `exp/eval/expR31_stage1_standalone.json` shows h7 hit@200 of 0.025 for R31 versus 0.435 for R21, and `exp/eval/expR31v1_stage1.json` improved only to h7 hit@200 0.105 with 8 unique h7 GTs versus R21 and 74 lost. A new attempt only makes sense if content initialization and utterance-conditioned transitions prevent that sparse-ID collapse.

It also differs from the older Qwen-space sequence source S. `exp/eval/expS2_neural_candidate.json` showed S had real but weak signal: 398 unique hits over all dev, h7 hit@200 144/1000, best fusion CV5 lift +0.0021, and the S2 design notes record Blind-A regression. That suggests the sequence idea is not empty, but predicting into a fixed metadata embedding space was too close to existing semantic retrieval. The R59 version would need behavior-native item vectors with metadata only as initialization/backoff.

## 2. Cached data/artifacts it can reuse

Reusable dev and production baseline artifacts:

- `exp/eval/_R12_all_turns_payload.pkl` (114M): 8,000 dev cases, `session_id`, `history`, `music_turns`, `user_query`, GT.
- `cache/r54/phase2_full/oof_r54_lists.json` (142M): preserved R54 Phase 2 OOF proxy for dev-side R54 source ranks.
- `cache/r21_production/dev_r21_oof_lists.json` (92M), `cache/r21_production/oof_manifest.json`, `cache/r21_production/fold_indices.json`: R21 OOF retrieval and fold metadata.
- `cache/r58/top50_dev.pkl` (188M): top-50 dev table with baseline candidate rows and features; useful to add sequence scores and test whether the score is informative inside the existing pool.
- `cache/r54_phase3_lr_model.txt`, `cache/r54_phase3_als.npz`, `cache/r54_phase3_track_pop.json`, `cache/r54_phase3_payload_maps.pkl`: production LR and supporting metadata/ALS/popularity maps.
- `cache/r54_production/blind_r54_lists.json`: current blind R54 retrieval list. It should not be touched for a dev diagnostic.
- `exp/eval/expR55_post_refresh_decomp.json`, `exp/eval/expR57b_structural_implementation.json`, `exp/eval/expR58_stage2_results.json`: baseline decomposition and failure comparisons.

Reusable content/item representations:

- `cache/track_sim/metadata-qwen3_embedding_0.6b/{track_ids.json,vectors.npy}` (vectors 182M): fixed metadata embeddings for content initialization or a metadata output head.
- `cache/dense/bge-base-en-v1.5/{track_ids.json,embeddings.npy}`: BGE catalog embeddings if a BGE-initialized item head is preferred.
- `cache/track_sim/cf-bpr/{track_ids.json,vectors.npy}` (vectors 23M): collaborative item vectors for initialization or diagnostic backoff.
- `cache/r21_production/{model,track_embeddings.npy,track_ids.json}`: R21 text model and catalog embeddings.

Reusable sequence-era artifacts, only after coverage validation:

- `cache/seq_model/utt_embeddings.npy` and `cache/seq_model/utt_embedding_index.json` (190M + 6.1M): frozen utterance embeddings from the prior sequence source. Reuse only if keys cover the current train/dev examples and encoder choice is still acceptable.
- `cache/seq_model/hard_negatives.pkl`: should not be used by default. The `feedback_no_hard_negatives.md` memory says plausible music hard negatives hurt; only reuse for analysis, not first-pass training.
- `cache/r31/{vocab.json,track_counts.json,fold0_r31_lists.json}` and `exp/eval/expR31v1_stage1.json`: useful as a negative baseline, not as an implementation template.

Train-split data is available locally via Hugging Face cache:

- `.hf_cache/datasets/talkpl-ai___talk_play_data-challenge-dataset/.../talk_play_data-challenge-dataset-train.arrow`
- `~/.cache/huggingface/datasets/talkpl-ai___talk_play_data-challenge-dataset/.../talk_play_data-challenge-dataset-train.arrow`
- track metadata in `.hf_cache/datasets/talkpl-ai___talk_play_data-challenge-track-metadata/.../talk_play_data-challenge-track-metadata-all_tracks.arrow`

Regeneration costs:

- Extract full train-split transition table: minutes on CPU. R22/R54 records indicate about 15,199 train sessions and about 121k trainable transitions before sampling.
- Audit train/dev/blind `session_id` overlap: minutes.
- Build a no-neural transition-memory diagnostic: minutes to tens of minutes, depending on whether nearest-neighbor metadata backoff is used.
- Train one small hybrid sequence model on train-split only: likely 1-3 hours on local MPS/CPU-class hardware, much less on a CUDA GPU. Full 5-model OOF with dev-fold training would be several hours local or roughly an hour-class GPU job.
- Generate dev top-300 sequence lists by catalog dot product: minutes if item vectors are in memory and scoring is chunked.

## 3. Leakage risks

The biggest leakage risk is splitting a conversation by turn. A sequence model trained on turns 1-6 of a session and evaluated on turn 7 of the same session has already seen the session trajectory. All folds must be grouped by `session_id`, matching `grouped_session_folds(seed=0)`.

Train-split versus dev:

- Train-split transitions are safe only after excluding any `session_id` that appears in `exp/eval/_R12_all_turns_payload.pkl`. R54 already does this in `build_train_split_sample()` by skipping `sid in dev_session_ids`; the sequence version must do the same.
- If the diagnostic trains only on official train-split sessions and evaluates on all dev sessions, it is OOF-clean with respect to dev labels.
- If the sequence model also trains on dev folds, it requires five fold-specific models. For fold k, no transition, item-ID update, user/session aggregate, or early-stopping signal from held-out sessions may be used.

Train-split versus Blind-A:

- Blind-A test queries and Blind-A session histories may be used only as inference inputs. They cannot be used for model selection, early stopping, prompt/architecture tuning, transition-table smoothing choices, or row-level inspection-driven edits.
- Before any production path, audit whether any Blind-A `session_id` appears in official train data. If overlap exists, those train sessions must be excluded for a Blind-A model or the model is contaminated.

Artifact leakage:

- Production R21/R54 models trained on all dev cannot justify dev improvements for a new retrieval source. The `feedback_oof_contamination.md` memory documents R46: using production R21 for extended lists overstated h7 lift by about 7x (+0.01486 contaminated versus +0.002 OOF-clean).
- Content embeddings derived only from catalog metadata are safe across folds. Item embeddings learned from interactions are not fold-safe if they include dev interactions from held-out sessions.
- `cache/r58/top50_dev.pkl` is safe for analysis of existing candidates, but a sequence retriever diagnostic must also evaluate candidates outside that top-50; otherwise it collapses into the R58 failure mode.

Tuning leakage:

- A cheap diagnostic should use fixed, predeclared weights and gates. If several source weights, smoothing constants, or architectures are swept on dev, selection must be fold-clean or treated as exploratory only.
- Same-artist and diff-artist splits are reporting metrics, not routing labels. R56's ORACLE result is diagnostic evidence, not a deployable gating pattern.

## 4. Expected ceiling

R54c's Blind-A nDCG@20 is 0.4925. The visible top competitor range is roughly 0.51-0.57, so matching even the low end requires about +0.0175 absolute nDCG, and matching the high end requires about +0.0775. This candidate is unlikely to close that gap alone.

Honest expected lift: +0.002 to +0.006 h7 nDCG if it works as a weak additive source. Optimistic ceiling: +0.010 to +0.015, but only if the hybrid sequence source adds genuinely new h7 candidates that are absent from the current RRF pool and does not regress same-artist cases. In Blind-A score terms, that would put R54c roughly in the 0.495-0.508 range for the expected case and perhaps around 0.503-0.508 in the optimistic but still plausible case. It probably does not reach 0.51 without unusually clean transfer, and 0.57 is out of scope.

The reasons for a low prior:

- R54 Phase 3 already added train-split supervision and produced real but modest local gain: h7 +0.0055 versus R39, below the +0.010 gate, with fold 3 regressing -0.0169. The project memory explicitly calls out diminishing returns: Phase 2 structured query gave +0.0034; adding 20k train-split pairs raised that to +0.0055.
- R55's all-data retriever trained on the same dev + train-split family was flat composite and -0.0067 nDCG on Blind-A versus R54c, so "more supervised retrieval" is not automatically beneficial.
- R31 sparse ID sequence failed badly on h7 coverage, and S/Qwen sequence showed weak dev signal but poor Blind-A transfer.
- R56 and R58 show the existing LR top-50 is calibrated. If the sequence model only reshuffles current top-50 candidates, expected lift is approximately zero or negative.

The reason not to archive immediately:

- R54's train-split use discards most transition events and optimizes text matching, not behavior-native next-track prediction.
- The current stack still has large non-hit buckets in the frozen record: DEMOTED 1,628, POOL_MISS 1,163, UNREACHABLE 1,861. A first-stage sequence source can only matter if it attacks POOL_MISS/UNREACHABLE by admitting new candidates. It should not target DEMOTED.

## 5. Smallest falsifiable diagnostic

Cheapest diagnostic: a train-split-only transition probe, no neural training and no blind data.

Build a fixed sequence candidate source from official train-split sessions after excluding dev `session_id`s:

- Extract every train music event with context `(previous 1-5 track IDs, previous/current user utterances, target next track)`.
- Compute predeclared candidate scores for each dev case:
  - exact last-track transition count: targets observed after the same last track;
  - last-3 transition aggregation: recency-weighted targets following any of the last three tracks;
  - artist/tag transition backoff: targets following train contexts whose prior track artist/tag pattern matches the dev context;
  - metadata-neighbor backoff: find nearest train prior-track contexts using fixed Qwen/BGE track embeddings, then aggregate their next targets;
  - utterance bucket multiplier for simple observable buckets such as continuation ("more like this"), pivot ("different", "something else"), and constraint/refinement terms.
- Exclude already played tracks and emit top-300 sequence candidates plus a scalar `seq_transition_score` for candidates already present in `cache/r58/top50_dev.pkl`.

Labels and evaluation:

- Label is `candidate_track_id == gt` from `exp/eval/_R12_all_turns_payload.pkl`.
- Primary eval is h7 nDCG@20 after adding this source to the existing RRF pool with one predeclared low weight, then applying the existing baseline LR without retraining where possible.
- Secondary eval: standalone h7 hit@20/100/300, all-dev nDCG@20, same-artist nDCG, diff-artist nDCG, POOL_MISS/UNREACHABLE recoveries, unique h7 GT hits versus current R39+R54 pool, top-1 churn, median top-20 overlap.
- Inside-pool diagnostic: on `cache/r58/top50_dev.pkl`, report whether GT rows in DEMOTED cases have higher `seq_transition_score` than LR rank-18..25 competitors. This is not sufficient for a pass, because R58 already failed top-50 specialization, but it can explain whether the source is pure new-admission signal or also has calibration value.

Pass condition for continuing to a real hybrid sequence model:

- h7 nDCG improves by at least +0.003 in this cheap fixed-source test, or the source contributes at least 30 unique h7 GT hits outside the current RRF pool@300 with no all-dev regression.
- same-artist h7 nDCG does not regress by more than 0.002.
- top-1 churn is below 1.5% for exploratory status and below 3.0% for any future production candidate, mirroring the R56/R58 stability discipline.

Fail condition:

- If the transition probe has h7 hit@300 near the R31/S range without unique pool recoveries, do not train a neural sequence model. The data is not showing enough behavioral transition signal to justify a heavier version.

## 6. Stop condition

Archive this direction immediately if the cheap transition probe fails both candidate-coverage and nDCG gates. Specifically, stop if unique h7 GT additions outside the current RRF pool are fewer than 30, h7 nDCG lift is below +0.003, or all-dev nDCG regresses.

Archive even after a neural implementation if any of the R56/R57b/R58 canaries fire:

- same-artist nDCG regresses by more than 0.002;
- net recovery is <= 0, or recovered POOL_MISS/UNREACHABLE cases are offset by lost HIT cases;
- top-1 churn exceeds 1.5% for exploratory or 3.0% for production;
- the model mainly changes LR top-50 ordering rather than adding new pool candidates;
- feature/attention/gain diagnostics look strong but nDCG is flat or negative, matching the R57b "high LightGBM gain, bad transfer" failure pattern;
- any beta/weight > 0 behaves like R58, where beta=0 reproduces baseline and every nonzero intervention regresses.

For a full OOF hybrid sequence model, the production gate should remain strict: h7 nDCG +0.010 over the R54c dev baseline, no same-artist regression, all-dev nonnegative, stable churn, and clear evidence that recovered cases come from POOL_MISS/UNREACHABLE rather than DEMOTED. A +0.005 exploratory lift is no longer enough to justify a Blind-A burn after eight post-R54c negatives. If it lands in the +0.003 to +0.009 range, document it for Blind-B and keep R54c production.
