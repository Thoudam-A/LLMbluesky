# ATC Intent Evaluation Workspace

Server workspace:

```text
/home/cjj/atc_intent_eval
├── data/raw
├── data/processed
├── data/gold
├── logs
├── outputs
├── reports
├── schemas
└── scripts
```

Raw transcription data remains read-only at:

```text
/home/cjj/instruction_data
```

First cleaning command:

```bash
python3 /home/cjj/atc_intent_eval/scripts/clean_atc_instruction_data.py \
  --raw-dir /home/cjj/instruction_data \
  --out-dir /home/cjj/atc_intent_eval/data/processed \
  --report-dir /home/cjj/atc_intent_eval/reports
```

Primary output for the intent-understanding metric:

```text
/home/cjj/atc_intent_eval/data/processed/atc_utterances.jsonl
```

Qwen3-4B local-model evaluation:

```bash
# Usage: bash run_qwen3_intent_eval.sh <start> <limit>
cd /home/cjj/atc_intent_eval
GPU=0 bash scripts/run_qwen3_intent_eval.sh 0 25
```

Default model path:

```text
/home/gbh/gubinhao/LLM/Qwen3-4B-Instruct-2507
```

Outputs:

```text
/home/cjj/atc_intent_eval/outputs/qwen3_4b/predictions_*.jsonl
/home/cjj/atc_intent_eval/outputs/qwen3_4b/score_*.json
/home/cjj/atc_intent_eval/reports/qwen3_4b_prediction_*.md
/home/cjj/atc_intent_eval/reports/qwen3_4b_score_*.md
/home/cjj/atc_intent_eval/logs/qwen3_4b/run_*.log
```

Important scoring note: `annotation_sample_500.jsonl` starts as `pending` with
empty `intents`, so official accuracy is `NA` until reviewed gold annotations
are written with `annotation_status="done"`. The prediction script can still be
used immediately to inspect Qwen3-4B outputs; the scorer will compute official
frame accuracy once a reviewed gold JSONL is supplied with `--gold`.
