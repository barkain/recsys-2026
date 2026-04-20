#!/bin/bash
export USER=echo
export LOGNAME=echo
export TORCHINDUCTOR_CACHE_DIR=/workspace/group/torch_cache
export HF_DATASETS_CACHE=/workspace/group/hf_cache
export HF_HOME=/workspace/group/hf_home
export TRANSFORMERS_CACHE=/workspace/group/hf_home

cd /workspace/group/recsys-work
echo "[$(date)] Starting BM25+CF-BPR Blind-A inference..."
.venv/bin/python run_inference_blind_bm25cf.py --tid echo_bm25_cf_blind_a 2>&1
echo "[$(date)] Done. Exit: $?"
