# R84 → R88 Sprint Summary (Post-R63c nDCG Arc)

**Window:** 2026-05-23 → 2026-05-26
**Total cost:** ~$19
**Outcome:** R84c becomes production at composite **0.6362** (+0.0060 vs R78). All multimodal-as-features paths archived.

## Headline

Before this sprint: 17 consecutive post-R63c experiments (R64 through R83 + R86) failed to move nDCG. R78 production at composite 0.6302 was the plateau.

After: R84c production at 0.6362. **First nDCG-side composite gain in the cycle.**

| | composite | nDCG@20 | LexDiv | LLM |
|---|---:|---:|---:|---:|
| R78 (prev prod) | 0.6302 | 0.4925 | 0.8845 | 4.90 |
| **R84c (new prod)** | **0.6362** | **0.5069** | 0.8720 | 4.90 |
| Δ | +0.0060 | **+0.0144** | −0.0125 | 0 |

## The 4-experiment closure that finally opened nDCG

| sprint | mechanism | result | cost |
|---|---|---:|---:|
| R84 Phase 0A | Full-corpus pair census | 5.58× more pairs vs R54's sampled 26K | $0 |
| R84 Phase 0B | BGE-large fold-0 fine-tune | Strong retrieval signal, partial-OOF probe failed same-artist | $1.50 |
| R84 Phase 1 | 5-fold proper OOF | Same-artist canary closes; h7 +0.0042 (just below gate) | $6 |
| R84b CPU sweep | Feature interface + LR hyperparams + segment diagnostics | Revealed selective deployment opportunity | $0 |
| **R84c** | **Selective routing (R54c LR top-1 margin < 0.5 OR ≥ 2.0 → R84)** | **Dev h7 +0.0056, blind +0.0144** | **$1.50 + $7.50 ensemble + $1.70 Opus** |

R84c's key insight: the conversion problem solved itself when we stopped trying to make R84 win on every case and instead **only routed to R84 where R54c's confidence was either very low or very high**. R84 wins in the mid-confidence band were never going to materialize; the segment diagnostics caught this and selective routing extracted only the safe wins.

## What R84c blind taught us (vs dev predictions)

| | dev h7 Δ | blind nDCG@20 Δ | ratio |
|---|---:|---:|---:|
| R84c | +0.0042 (sub-gate by 0.0008) | +0.0144 | **3.4× over-delivery** |

The OOF sibling LR baseline ([[feedback_lr_wall_was_artifact]]) systematically under-estimates blind transfer. Useful calibration for future sub-gate candidates — but we should not trust over-delivery to repeat for non-text-retrieval mechanisms.

## R85/R86/R88 — what's been definitively ruled out

After R84c shipped, four follow-up sprints all failed to move composite further:

### R86 — LexDiv recovery on R84c
- 12 of 17 R84c regen'd responses re-regenerated with banned-bigram constraint.
- Local Distinct-2 lift +0.0076. Tracks bitwise identical.
- **Competition LexDiv = 0.8720 (= R84c's). Zero recovery at scorer.**
- Lesson: local stopword-filtered Distinct-2 is NOT a reliable proxy for Codabench LexDiv. [[feedback_local_distinct2_doesnt_predict_lexdiv]]
- Codabench submission 755267 also failed externally (Gemini scorer 404).

### R85 Phase 0 — multimodal inventory
Vs R84 5-fold OOF, all 4 modalities passed the headroom gate:

| modality | unique h7 top-30 recoveries | same/diff |
|---|---:|---:|
| image_siglip | **86** | 77/9 |
| attributes_qwen | 59 | 56/3 |
| audio_clap | 29 | 15/14 |
| lyrics_qwen | 22 | 17/5 |

### R85 Phase 1 — multimodal conversion
3 integration variants tested:
- R85a (10-source RRF pool): h7 Δ = +0.0002 (sub-gate, all canaries positive)
- R85b (43-col LR with raw IMG/META features): **h7 Δ = −0.0075** (LR can't use raw features)
- R85c (selective: margin × IMG top-1): IMG saturated at 1.0, sweep collapses

### R88 — constrained multimodal ranker
22 configs across 3 variants:
- V1 monotone LightGBM: best h7 Δ = +0.0012 with same +0.0088 (strongest same-artist in cycle), but h7 sub-gate
- V2 guarded additive boost: h7 +0.0012 with churn 2/80 (safe but small)
- V3 quota injection: REGRESSES (R84c's own top-50 is better-ordered than what multimodal candidates would replace)

**Conclusion: multimodal source signal is real (86 unique recoveries) but does NOT convert at the LR layer.** The conversion wall is structural for cross-feature-class signals. R84-class signal converts because text retrieval upgrades the SAME feature class R54 LR was already calibrated for.

### R89 — learned multimodal dual-encoder retriever (Colab A100)
Frozen everything + ~3.5M-param fusion MLP. **All gates fail catastrophically.**
- h7 nDCG Δ = −0.041 (worse than R84 source-alone)
- same-artist Δ = **−0.119** (catastrophic regression)
- Unique h7 top-30 = 13 vs LOST 39 (net −26)

**Diagnosis:** fusion MLP bottleneck destroys signal each modality had alone. R85 image_siglip standalone produced 86 unique recoveries; R89 with ALL 5 modalities fused produced only 13. Compression through small projections is lossy.

**Multimodal CLOSED across all 3 architectures** (R85 features, R88 constrained LR, R89 learned fusion).

### R87 — evidence-injection LLM push (blind sub 757667, 2026-05-26)
On R84c production base, regenerated 12 weakest responses with stronger evidence prompt.

| metric | R84c | R87 | Δ |
|---|---:|---:|---:|
| composite | 0.6362 | **0.6360** | −0.0002 |
| LLM judge | 4.90 | 4.90 | 0 |
| LexDiv | 0.8720 | 0.8706 | −0.0014 |
| nDCG@20 | 0.5069 | 0.5069 | 0 |

**LLM 4.90 ceiling CONFIRMED** across R78 / R84c / R86 / R87. semintelligence's 4.95 uses a categorically different prompt shape our audit+regen pattern cannot reach. Evidence-injection (R87) slightly REGRESSED LexDiv vs R78-style.

**Empirical lesson for Blind-B:** use **R78-style prompting** for fresh response generation. NOT R87-style.

**Response side fully saturated under R84c tracks. No more Blind-A response polish.**

## Cost ledger (the entire post-R63c → R84c → R88 arc)

| sprint | $ | what |
|---|---:|---|
| R79 hard-neg retriever | $3 | catastrophic collapse |
| R80 listwise Transformer | $3 | same-artist canary |
| R81 constrained-swap | $12 | feature ceiling |
| R82 LLM intent features | $1 | GT < FP |
| R83 SASRec | $0.50 | sessions too short |
| R84 Phase 0A census | $0 | 5.58× pair gain |
| R84 Phase 0B fold-0 | $1.50 | partial-OOF false negative |
| R84 Phase 1 | $7.50 | canary closes |
| R84b CPU sweep | $0 | segment diagnostics |
| R84c blind 5-fold ensemble | $7.50 | production candidate |
| R84c Opus regen | $1.70 | response packaging |
| R85 Phase 0 inventory | $0 | multimodal headroom |
| R85 Phase 1 | $0 | conversion fails |
| R86 LexDiv recovery | $1.20 | competition LexDiv no-op |
| R88 constrained MM | $0 | 22/22 configs fail |
| **TOTAL** | **~$39** | from R78 (0.6302) to R84c (0.6362) |

## Mechanism map at the end of R88

Every nDCG-side mechanism tested in this cycle, with outcome:

| signal class | mechanism | result |
|---|---|---|
| Retrieval (text) | R84 BGE-large full-corpus | **WIN (R84c)** |
| Retrieval (multimodal) | image/lyrics/attr/audio anchored top-300 | source signal real, doesn't convert |
| Pool admission | R59 C3 matched retrain | LR buries 91.7% |
| LR feature addition | R85b (43-col with IMG/META) | regresses |
| LR architecture | R88 V1 monotone shallow trees | best same-artist (+0.0088), h7 sub-gate |
| Rerank (rules) | R56 | flat |
| Rerank (LightGBM) | R58 | flat |
| Rerank (LLM) | R67 | catastrophic |
| Rerank (cross-encoder) | R69 | catastrophic |
| Score blend | R84b score blend | sub-gate |
| Quota injection | R88 V3 | regresses |
| Hard-neg retriever | R79 | catastrophic |
| Behavior-native seq | R83 | catastrophic (sessions too short) |
| Listwise Transformer | R80 | same-artist canary |
| Constrained-swap | R81 | feature ceiling |
| LLM intent features | R82 | GT < FP |
| Selective routing on observable margin | **R84c** | **WIN** |

## What still hasn't been tested

R88 closes *multimodal-as-ranker-features*. It does NOT close **learned multimodal representation training end-to-end on full-corpus pairs**.

Per user direction, next sprint is **R89 — learned multimodal dual-encoder retriever**:
- Query tower from R84 BGE-large
- Track tower fuses BGE-text + image SigLIP + attributes Qwen + lyrics Qwen + (maybe) audio CLAP via learned gated MLP
- Phase 0: freeze base embeddings, train only projections + gates (isolates whether fusion has signal without wrecking R84)
- Loss: in-batch InfoNCE + random catalog negatives
- Phase 0 fold-0 only on A100

R89 is structurally different from R85/R88 because it doesn't bolt multimodal onto the existing ranker — it learns a *new track representation* that fuses modalities end-to-end against query-text. This is the missing experiment: does learned fusion at the encoder layer produce signal that converts where bolted-on features did not?

## Files (per sprint)

- R84: `scripts/expR84_phase0a_census.py`, `expR84_phase0b_train.py`, `expR84_phase0b_eval.py`, `expR84b_sweep.py`, `expR84c_selective_deployment.py`, `expR84c_blind_candidate.py`, `expR84c_response_regen.py`
- R85: `scripts/expR85_phase0_inventory.py`, `expR85_phase1_sweep.py`, `expR85c_selective_routing.py`
- R86: `scripts/expR86_lexdiv_recovery.py`
- R88: `scripts/expR88_constrained_mm.py`
- Docs: `docs/r84_phase0a_census.md`, `r84_phase0b_result.md`, `r84_phase1_result.md`, `r84c_blind_diff.md`, `r84c_blind_result.md`, `r85_phase0_inventory_result.md`, `r85_phase1_result.md`, `r86_lexdiv_recovery_result.md`, `r84_to_r88_sprint_summary.md` (this file)
- Artifacts: `cache/r84/`, `cache/r84_production/`, `cache/r84c_production_lr.txt`, `cache/r85/multimodal_lists/`, blind submission ZIPs in `exp/inference/blind_a/`
