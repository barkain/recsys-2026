# R400/R401 — Large-Scale Session/Choice Recommender

**Goal:** beat R106 A-clean (composite 0.6377, nDCG@20 0.5073) on nDCG directly, via a
full supervised model trained on ALL official train data — not another probe. GPU/cost
unconstrained (Colab A100 + Drive). Avoid closed variants (standalone BGE/E5/GTE/Qwen
retriever swap, naked cross-encoder rerank, LexDiv-only, CatalogDiv).

## The central thesis (Phase 0 breakthrough)

The GT next-track is a Gemini pick from the user's hidden 16–32 track LFM-2b session pool.
Phase 0 user-overlap audit (NEW): **74.2% of dev users (371/500) and 43.1% of blind users
(25/58) also appear in TRAIN sessions** — same users, *different* sessions (0 session
leakage), with **~104 train-history tracks each** on average. Prior work missed this: it
used only the within-conversation played tracks (R104), never the user's full cross-session
listening history. The user's train history reveals their taste, and the GT is drawn from
their pool → **a user→item history lever that is real, large, and untested.** This is the
nDCG signal class beyond similarity that justifies a large run. Corpus: **106,393 train
prefix→next transitions**, 8,591 train users, 47,071-track catalog.

## Architecture

**Phase 0 — data + recall ceiling (GATE).** Canonical corpus (train prefix→next + dev OOF),
leakage audit (session-disjoint ✓, user-overlap ✓), per-user train-history index. High-recall
candidate universe = production R54/R84 union + official embedding NN (cf-bpr/audio/lyrics/
metadata/image) + **user-history retrieval (the new lever)** + BM25, depth 1000–3000. Measure
dev OOF recall@20/100/300/1000 and user-history UNIQUE recall over the production union.
GO if recall ceiling materially above 0.777 OR user-history adds large unique recall.

**Model A — session/user→track catalog retriever.** Large sequence encoder over
{conversation text + music history + user demographics + **user train-history / user
embedding**}; track tower = official embeddings + metadata + learned id embedding; sampled/
full softmax over 47k + hard negatives (production top candidates, same-artist confounders).
Train on all transitions; OOF fine-tune per fold for dev eval. Eval: retrieval @20/100/300 +
unique recoveries.

**Model B — listwise Gemini-choice imitator.** Input: session context + candidate list with
metadata/source-rank/**user-history** features. Large listwise reranker (LambdaRank/ListMLE/
top-20 nDCG surrogate, NOT pairwise GT-vs-one-FP). Hard negs: prod top-20, same-artist,
similar-tag, high-source-rank FPs. Learns feature interactions the 37-feat LR couldn't.

**Model C — selective gate/blender.** Inputs: production scores, Model A/B scores, source
ranks, margin, same/diff-artist, n_prior, user-overlap flag, novelty. Output: production /
model / bounded-patch ranking. Optimize OOF nDCG@20. Three operating points: conservative
(proj. blind churn ≤20/80), medium (≤35/80), aggressive (max OOF if gates strong).

## Validation gates (strict)

Primary: 5-fold OOF nDCG@20 vs production. PASS requires ≥1 of:
- all-case nDCG Δ ≥ **+0.020** OOF, no catastrophic h7/same-artist regression; OR
- **h7 Δ ≥ +0.008 AND diff-artist Δ > 0**; OR
- a strong last-turn/dev-blind proxy improvement.
Report same/diff-artist, h7, n_prior buckets, churn, top-20 overlap, recovered/lost top-20
GTs. **Compare to the R90/R103 failure signature** (high churn + weak h7); if the model has
that shape, do NOT submit full rerank — build the selective gate. Package r400_submission.zip
ONLY after gates pass (A-clean response style unless top-1 changes force regen). Else archive
with the exact failure mode.

## Agent team (workflow-orchestrated)

data · retrieval/model (A) · listwise (B) · gate (C) · eval/adversary · packaging.
Execution: run Phase 0 first; if recall ceiling real, train A+B in parallel; if OOF gates
pass, build selective candidate; if fail, archive. Do not pause for approval between small
steps.
