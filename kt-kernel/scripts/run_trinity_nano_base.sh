#!/bin/bash
# Launch SGLang server for Trinity-Nano-Base with kt-kernel MoE acceleration
# Usage: bash scripts/run_trinity_nano_base.sh

set -e

# === Configuration (adjust to your environment) ===
MODEL_PATH="/mnt/data/models/Trinity-Nano-Base"
KT_WEIGHT_PATH="${MODEL_PATH}"  # For BF16 method, same as model path
KT_METHOD="BF16"                # Options: BF16, FP8, AMXINT4, AMXINT8, LLAMAFILE
KT_CPUINFER=64                  # Physical CPU cores (not hyperthreads)
KT_THREADPOOL_COUNT=2           # Number of NUMA nodes
KT_NUM_GPU_EXPERTS=16           # Experts to keep on GPU (out of 64 total)
KT_MAX_DEFERRED=2               # Deferred experts for pipelining (0 to disable)
PORT=8000

# === Download model (uncomment if needed) ===
# pip install huggingface-hub
# huggingface-cli download arcee-ai/Trinity-Nano-Base --local-dir ${MODEL_PATH}

# === For AMX quantized weights (uncomment if using AMXINT4/AMXINT8) ===
# python scripts/convert_cpu_weights.py \
#   --input-path ${MODEL_PATH} \
#   --input-type bf16 \
#   --output /mnt/data/models/Trinity-Nano-Base-INT8 \
#   --quant-method int8
# KT_WEIGHT_PATH="/mnt/data/models/Trinity-Nano-Base-INT8"
# KT_METHOD="AMXINT8"

# === Launch server ===
python -m sglang.launch_server \
    --host 0.0.0.0 \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --trust-remote-code \
    --mem-fraction-static 0.85 \
    --chunked-prefill-size 4096 \
    --served-model-name Trinity-Nano-Base \
    --enable-mixed-chunk \
    --kt-method ${KT_METHOD} \
    --kt-weight-path ${KT_WEIGHT_PATH} \
    --kt-cpuinfer ${KT_CPUINFER} \
    --kt-threadpool-count ${KT_THREADPOOL_COUNT} \
    --kt-num-gpu-experts ${KT_NUM_GPU_EXPERTS} \
    --kt-max-deferred-experts-per-token ${KT_MAX_DEFERRED} \
    --attention-backend flashinfer \
    --max-running-requests 4 \
    --tensor-parallel-size 1 \
    --enable-p2p-check
