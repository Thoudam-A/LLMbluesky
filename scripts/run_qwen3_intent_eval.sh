#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/home/cjj/atc_intent_eval}"
PYTHON="${PYTHON:-/home/cjj/.virtualenvs/Autonomous-ATC-N_Closest/bin/python}"
MODEL="${MODEL:-/home/gbh/gubinhao/LLM/Qwen3-4B-Instruct-2507}"
GPU="${GPU:-0}"
START="${1:-0}"
LIMIT="${2:-25}"
STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"

SAMPLE="${SAMPLE:-$WORKDIR/data/gold/annotation_sample_500.jsonl}"
SCHEMA="${SCHEMA:-$WORKDIR/schemas/atc_intent_schema.json}"
OUT_DIR="$WORKDIR/outputs/qwen3_4b"
LOG_DIR="$WORKDIR/logs/qwen3_4b"
REPORT_DIR="$WORKDIR/reports"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR"

PRED="$OUT_DIR/predictions_start${START}_limit${LIMIT}_${STAMP}.jsonl"
PRED_LINK="$OUT_DIR/predictions_latest.jsonl"
PRED_REPORT="$REPORT_DIR/qwen3_4b_prediction_start${START}_limit${LIMIT}_${STAMP}.md"
SCORE_JSON="$OUT_DIR/score_start${START}_limit${LIMIT}_${STAMP}.json"
SCORE_REPORT="$REPORT_DIR/qwen3_4b_score_start${START}_limit${LIMIT}_${STAMP}.md"
LOG="$LOG_DIR/run_start${START}_limit${LIMIT}_${STAMP}.log"

echo "[$(date -Is)] model=$MODEL gpu=$GPU start=$START limit=$LIMIT batch_size=$BATCH_SIZE max_new_tokens=$MAX_NEW_TOKENS" | tee "$LOG"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$WORKDIR/scripts/run_qwen3_intent_predictions.py" \
  --model "$MODEL" \
  --sample "$SAMPLE" \
  --schema "$SCHEMA" \
  --out "$PRED" \
  --report "$PRED_REPORT" \
  --start "$START" \
  --limit "$LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" 2>&1 | tee -a "$LOG"

ln -sfn "$PRED" "$PRED_LINK"

"$PYTHON" "$WORKDIR/scripts/score_atc_intent_predictions.py" \
  --gold "$SAMPLE" \
  --pred "$PRED" \
  --schema "$SCHEMA" \
  --out-json "$SCORE_JSON" \
  --report "$SCORE_REPORT" 2>&1 | tee -a "$LOG"

echo "[$(date -Is)] done" | tee -a "$LOG"
echo "prediction=$PRED"
echo "prediction_report=$PRED_REPORT"
echo "score_json=$SCORE_JSON"
echo "score_report=$SCORE_REPORT"
echo "log=$LOG"
