#!/usr/bin/env python3
"""Run Qwen3-4B on ATC intent-recognition samples with vLLM."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from run_qwen3_intent_predictions import (
    build_prompt,
    compact_intent_types,
    find_json_object,
    normalize_prediction,
    read_jsonl,
    write_jsonl,
    write_report,
)


def build_chat_text(tokenizer, system_prompt: str, prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{system_prompt}\n\n{prompt}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/gbh/gubinhao/LLM/Qwen3-4B-Instruct-2507")
    parser.add_argument("--sample", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.sample))
    batch = rows[args.start : args.start + args.limit if args.limit >= 0 else None]
    schema = json.load(open(args.schema, encoding="utf-8"))
    intent_types = compact_intent_types(schema)

    import run_qwen3_intent_predictions as base

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    predictions = []
    failures = []
    start_time = time.time()
    out_path = Path(args.out)
    prompts = [
        build_chat_text(tokenizer, base.SYSTEM_PROMPT, build_prompt(row, intent_types))
        for row in batch
    ]
    batch_size = max(1, args.batch_size)

    for offset in range(0, len(batch), batch_size):
        chunk_rows = batch[offset : offset + batch_size]
        chunk_prompts = prompts[offset : offset + batch_size]
        try:
            outputs = llm.generate(chunk_prompts, sampling_params)
        except Exception as exc:
            failures.extend(
                {
                    "annotation_id": row.get("annotation_id"),
                    "utterance_id": row.get("utterance_id"),
                    "text": row.get("text"),
                    "error": f"vllm_generate_failed: {exc}",
                    "raw_response": "",
                }
                for row in chunk_rows
            )
            continue

        for row, output in zip(chunk_rows, outputs):
            raw_response = output.outputs[0].text.strip() if output.outputs else ""
            try:
                obj = find_json_object(raw_response)
                pred = normalize_prediction(obj, row, raw_response)
                pred["model"] = "qwen3-4b"
                pred["model_path"] = args.model
                pred["backend"] = "vllm"
                predictions.append(pred)
            except Exception as exc:
                failures.append(
                    {
                        "annotation_id": row.get("annotation_id"),
                        "utterance_id": row.get("utterance_id"),
                        "text": row.get("text"),
                        "error": str(exc),
                        "raw_response": raw_response,
                    }
                )

        processed = min(offset + len(chunk_rows), len(batch))
        if args.save_every > 0 and processed % args.save_every == 0:
            write_jsonl(out_path, predictions)

    write_jsonl(out_path, predictions)
    if failures:
        write_jsonl(out_path.with_suffix(".failures.jsonl"), failures)
    elapsed = time.time() - start_time
    write_report(Path(args.report), predictions, failures, elapsed, args.model)
    print(
        json.dumps(
            {
                "sample": args.sample,
                "start": args.start,
                "limit": args.limit,
                "predictions": len(predictions),
                "failures": len(failures),
                "out": str(out_path),
                "report": args.report,
                "elapsed_seconds": round(elapsed, 1),
                "backend": "vllm",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
