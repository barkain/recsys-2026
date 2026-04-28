# Echo — Version Registry

**Metric**: composite = 0.50×nDCG@20 + 0.10×CatDiv + 0.10×LexDiv + 0.30×norm(LLM)  
**Primary metric**: nDCG@20 (last turn, 100 devset sessions)  
**Blind correlation factor**: ~2.87× offline nDCG → blind nDCG (calibrated from v16)

---

## Versioning Convention

| Artifact | Pattern | Example |
|----------|---------|---------|
| Devset config | `config/echo_v{N}_devset.yaml` | `echo_v22_devset.yaml` |
| Blind config | `config/echo_v{N}_blind.yaml` | `echo_v22_blind.yaml` |
| Submission zip | `echo_v{N}_blind_submit.zip` | `echo_v22_blind_submit.zip` |
| Git tag | `v{N}` | `v22` — points to commit that produced the results |
| Inference output | `exp/inference/devset/echo_v{N}_devset.json` | auto-generated |
| Eval output | `exp/eval/echo_v{N}_devset_last_turn.json` | auto-generated |

**Rule**: tag the commit *after* eval confirms the result. Never tag before running.

---

## Version History

| Ver | Git commit | Off-nDCG | LexDiv | Blind nDCG | LB Comp | Notes |
|-----|-----------|----------|--------|------------|---------|-------|
| v14 | — | 0.1025 | — | ~0.287 | ~0.47 | dual-QR + Sonnet w=10 |
| v15 | 4144113 | 0.1037 | — | ~0.290 | ~0.47 | Sonnet w=20, first to beat #1 offline |
| v16 | 76fea10 | 0.1104 | 0.748 | 0.3176 | **0.503** | w=20 v3 prompt, **LB #1** |
| v17 | 28f673b | — | 0.748 | 0.3109 | 0.492 | LexDiv experiment, regression |
| v18 | — | 0.0974 | — | — | — | track_sim_query, rank noise, dropped |
| v19 | eabba44 | **0.1046** | **0.904** | — | ~0.510 est | NLQ hint + w=50; best offline |
| v20 | 42bb57f | 0.0936 | 0.893 | — | — | w=20 ablation; confirmed w=50 better |
| v21 | f6bb803 | 0.0747 | 0.894 | — | — | o4-mini QR; worse, not pursuing |
| v22 | 68fc872 | pending | — | — | — | generative retrieval (LLM→25 tracks) |
| v23 | 48d238d | pending | — | — | — | v22 + Qwen3 track embedding sim |

---

## How to Create a New Version

```bash
# 1. Write config
cp config/echo_v23_devset.yaml config/echo_v24_devset.yaml
# edit flags

# 2. Run devset eval
uv run python3 run_inference_devset.py --tid echo_v24_devset --batch_size 8 --last_turn_only
uv run python3 eval_devset.py --tid echo_v24_devset --last_turn_only

# 3. Record result in VERSIONS.md, commit, tag
git add config/echo_v24_devset.yaml VERSIONS.md
git commit -m "feat: v24 — <description>"
git tag v24

# 4. If submitting, create blind config and generate submission
cp config/echo_v24_devset.yaml config/echo_v24_blind.yaml
# edit: test_dataset_name → blind dataset
uv run python3 run_inference_blind.py --tid echo_v24_blind
# zip the prediction.json as echo_v24_blind_submit.zip

# 5. Get Nadav's approval before merging/tagging release
```

---

## Submission History

| Submission | LB Comp | nDCG | LexDiv | LLM | Rank | Status |
|-----------|---------|------|--------|-----|------|--------|
| v14 (old) | 0.44 | 0.21 | — | — | #5 | superseded |
| v16 | **0.503** | 0.3176 | 0.7478 | 4.55 | **#1** | best |
| v17 | 0.492 | 0.3109 | 0.7482 | 4.45 | #2 | regression |
