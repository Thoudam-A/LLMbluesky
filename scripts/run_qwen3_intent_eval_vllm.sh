#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/home/gbh/gubinhao/atc_intent_eval_qwen}"
PYTHON="${PYTHON:-/home/gbh/miniconda3/envs/LlamaFactory/bin/python}"
MODEL="${MODEL:-/home/gbh/gubinhao/LLM/Qwen3-4B-Instruct-2507}"
GPU="${GPU:-0}"
START="${1:-0}"
LIMIT="${2:-25}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CUDA_HOME="${CUDA_HOME:-/home/gbh/miniconda3/envs/LlamaFactory/lib/python3.11/site-packages/nvidia/cu13}"
CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
CC="${CC:-/usr/bin/gcc}"
CXX="${CXX:-/usr/bin/g++}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

SAMPLE="${SAMPLE:-$WORKDIR/data/gold/annotation_sample_500_gpt55_preannotated_validated.jsonl}"
SCHEMA="${SCHEMA:-$WORKDIR/schemas/atc_intent_schema.json}"
OUT_DIR="$WORKDIR/outputs/qwen3_4b_vllm"
LOG_DIR="$WORKDIR/logs/qwen3_4b_vllm"
REPORT_DIR="$WORKDIR/reports"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR"

PRED="$OUT_DIR/predictions_start${START}_limit${LIMIT}_${STAMP}.jsonl"
PRED_REPORT="$REPORT_DIR/qwen3_4b_vllm_prediction_start${START}_limit${LIMIT}_${STAMP}.md"
LOG="$LOG_DIR/run_start${START}_limit${LIMIT}_${STAMP}.log"

echo "[$(date -Is)] model=$MODEL gpu=$GPU start=$START limit=$LIMIT batch_size=$BATCH_SIZE max_new_tokens=$MAX_NEW_TOKENS backend=vllm" | tee "$LOG"

export CUDA_HOME CUDA_PATH CUDACXX CC CXX VLLM_USE_FLASHINFER_SAMPLER
export PATH="$CUDA_HOME/bin:$(dirname "$PYTHON"):/usr/bin:$PATH"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$WORKDIR/scripts/run_qwen3_intent_predictions_vllm.py" \
  --model "$MODEL" \
  --sample "$SAMPLE" \
  --schema "$SCHEMA" \
  --out "$PRED" \
  --report "$PRED_REPORT" \
  --start "$START" \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] done" | tee -a "$LOG"
echo "prediction=$PRED"
echo "prediction_report=$PRED_REPORT"
echo "log=$LOG"
