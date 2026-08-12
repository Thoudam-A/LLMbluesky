#!/usr/bin/env python3
"""Run Qwen3-4B on ATC intent-recognition samples."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """你是民航空管指令意图识别评测模型。必须严格按给定 schema 抽取所有管制意图。只输出 JSON，不输出解释。"""


TASK_TEMPLATE = """请根据给定 schema 对一条 ATC 管制员话语进行意图识别。

硬性要求：
1. 输出必须是一个 JSON object；字段必须包含 annotation_id, utterance_id, annotation_status, intents, note。
2. annotation_status 只能是 done 或 skip。只要话语中有任何明确管制指令，就必须 annotation_status=done。
3. 必须抽取话语里的所有独立管制意图；同一句里的进近许可、雷达服务终止、联系频率、速度/高度/航向限制等都要分别列为 intent，不能只标第一条。
4. intent_type 必须从下方 intent_types 白名单中选择；禁止输出 schema 外类别。无法归类但仍是管制行为时用 other_control/other。
5. 每个 intent 必须包含 callsign, intent_type, action，并按该 intent_type 的 required_slots 补齐关键槽位。required slot 能从原文推断时绝不能填 null。
6. 不要只输出大类，必须输出具体 action。不要把 raw_span/raw_value 当作 required slot 的替代。
7. 无明确管制意图、纯寒暄、残缺无法判断时才 annotation_status=skip 且 intents=[]。

槽位强制规则：
- frequency_transfer/contact_frequency: 必须填写 frequency。中文频率逐位归一化，例如 幺幺八点幺=118.1，幺二零点三=120.3，幺二幺点三七五=121.375。
- approach_clearance/cleared_approach: 若出现 三六右/三六左/一八右 等，必须填写 runway；盲降/ILS 填 approach_type=ILS，目视填 visual，不确定才填 unknown。
- landing_clearance/cleared_to_land: 必须填写 runway。
- direct_to_fix/direct_to: 出现“直飞/飞向 + 点名”优先标 direct_to_fix，不要误标为 heading_change。fix 要规范化，例如 SIERRA SIERRA 两洞五 = SIERRA SIERRA 205。
- altitude_change/altitude_maintain: 必须填写 target_value 和 unit=m。常见高度：五百五=550，九百=900，幺二=1200，幺五=1500，幺八=1800，两幺=2100，三六=3600。
- speed_adjust: 必须填写 target_value 和 unit=kt。逐位数字归一化，例如 one seven zero=170，两洞洞=200，幺八洞=180。
- heading_change: 明确航向或转弯角度时必须填写 target_value 和 unit=degree。航向用三位字符串，例如 洞九洞=090，三六洞=360。
- qnh_setting/set_qnh: 必须填写 target_value 和 unit=hPa，例如 幺洞三幺=1031。
- restriction_cancel: 必须填写 restriction_type，可取 altitude, speed, heading, procedure, unspecified。
- report_requirement/report: 必须填写 report_type，不确定填 unspecified。
- 英文呼号中的数字要规范成阿拉伯数字，例如 Six three four zero -> 6340；中文呼号保持中文可读形式。

类别优先级：
- “联系/转频 + 单位 + 频率”一定是 frequency_transfer，不要漏掉。
- “雷达服务终止/雷达看到/雷达引导”是 radar_service 下的具体 action。
- “右转直飞/左转直飞 + 点名”优先 direct_to_fix。
- “盲降/ILS/进近许可”是 approach_clearance，并补 runway 和 approach_type。
- “再见”不单独标 intent。

intent_types:
{intent_types}

待识别话语：
annotation_id: {annotation_id}
utterance_id: {utterance_id}
language: {language}
text: {text}

只输出 JSON object，格式如下：
{{
  "annotation_id": "...",
  "utterance_id": "...",
  "annotation_status": "done",
  "intents": [
    {{
      "callsign": "...",
      "intent_type": "...",
      "action": "...",
      "target_value": null,
      "unit": null,
      "frequency": null,
      "runway": null,
      "approach_type": null,
      "fix": null,
      "restriction_type": null,
      "report_type": null,
      "raw_value": "...",
      "raw_span": "..."
    }}
  ],
  "note": ""
}}
"""


# The prompt block above may contain mojibake from earlier Windows/SSH syncs.
# These clean ASCII prompts override it and are the active prompts used below.
SYSTEM_PROMPT = """You are an ATC intent recognition evaluator. Extract every control intent from one controller utterance according to the provided schema. Return only one valid JSON object."""


TASK_TEMPLATE = """Task: identify all ATC control intents in one utterance.

Return a single JSON object with:
- annotation_id
- utterance_id
- annotation_status: "done" or "skip"
- intents: a list of intent frames
- note

Hard rules:
1. Use annotation_status="done" when the utterance contains any control, coordination, confirmation, traffic, report, holding, or operational instruction. Use "skip" only when there is no actionable or classifiable ATC intent.
2. Extract every independent intent in the utterance. Do not stop after the first one.
3. Every intent must use exactly one intent_type from the schema and one valid action for that intent_type.
4. If an utterance is operational/control-related but does not fit a specific class, use other_control/other. Do not drop it.
5. Do not infer an approach clearance from a forecast, possibility, question, or another aircraft's approach. Use approach_clearance only for actual clearance or instruction to continue/perform approach.
6. Callsign should be inherited from the utterance prefix when omitted in later spans.
7. Fill required slots when stated or directly inferable. Use null only when the slot is genuinely absent.

Intent boundary guide:
- altitude_change: climb/descend instructions, including "up to", "down to", "climb", "descend", and Chinese equivalents.
- altitude_maintain: maintain/keep a specific altitude without climb/descend wording.
- heading_change: turn/fly/maintain heading or turn by degrees.
- direct_to_fix: "direct", "direct to", "fly direct", "straight to" a named fix/point. This has priority over heading_change.
- speed_adjust: set/reduce/increase/maintain a speed.
- speed_procedure: speed as procedure, no numeric speed target.
- frequency_transfer: contact/switch/transfer to a unit or frequency. Must include frequency if present.
- qnh_setting: QNH/pressure setting, including "use 1008/1013" when it is a pressure setting rather than a radio frequency.
- approach_clearance: cleared/continue/perform approach, with runway and approach_type when available.
- landing_clearance: cleared to land.
- takeoff_or_departure_instruction: takeoff clearance, hold for takeoff, or departure climb.
- restriction_cancel: cancel altitude/speed/heading/procedure restriction.
- radar_service: radar identified, radar service terminated, radar vectoring.
- holding_instruction: hold, orbit, wait, expect holding/waiting time.
- traffic_information: traffic advisory or situational traffic warning, e.g. traffic, relative traffic, similar callsign, listen out for similar flight number.
- confirmation_request: controller asks for confirmation or information, e.g. "confirm", "is that right", "can you", "whether".
- report_requirement: explicit requirement to report/inform/notify a condition or position. Do not use this for a forecast that merely says something is expected.
- procedure_assignment: follow/join a named procedure, route, SID, STAR, arrival/departure procedure.
- weather_deviation: approve/request weather deviation.
- other_control: operational or control-related content not covered above, such as "okay", "pay attention to fuel", expected runway without clearance, spacing/flow coordination, or general controller coordination.

Important negative examples:
- "expected runway 36R" is other_control, not approach_clearance.
- "can you approach?" is confirmation_request, not approach_clearance.
- "previous aircraft is approaching" is not approach_clearance.
- "expect to wait one hour" is holding_instruction, not report_requirement.
- "similar callsign" or "relative traffic" is traffic_information.
- "okay/can do" as a controller response is other_control unless it clearly grants a specific clearance.

intent_types:
{intent_types}

Utterance:
annotation_id: {annotation_id}
utterance_id: {utterance_id}
language: {language}
text: {text}

Return only JSON in this shape:
{{
  "annotation_id": "...",
  "utterance_id": "...",
  "annotation_status": "done",
  "intents": [
    {{
      "callsign": "...",
      "intent_type": "...",
      "action": "...",
      "target_value": null,
      "unit": null,
      "frequency": null,
      "runway": null,
      "approach_type": null,
      "fix": null,
      "restriction_type": null,
      "report_type": null,
      "raw_value": "...",
      "raw_span": "..."
    }}
  ],
  "note": ""
}}
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def compact_intent_types(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "actions": spec.get("actions", []),
            "required_slots": spec.get("required_slots", []),
            "conditional_required_slots": spec.get("conditional_required_slots", {}),
        }
        for name, spec in schema.get("intent_types", {}).items()
    }


def build_prompt(row: dict[str, Any], intent_types: dict[str, Any]) -> str:
    return TASK_TEMPLATE.format(
        intent_types=json.dumps(intent_types, ensure_ascii=False, indent=2),
        annotation_id=row.get("annotation_id") or "",
        utterance_id=row.get("utterance_id") or row.get("atc_instruction_id") or row.get("uid") or "",
        language=row.get("language") or "",
        text=row.get("text") or "",
    )


def find_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(text[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in model response")


def normalize_prediction(obj: dict[str, Any], source: dict[str, Any], raw_response: str) -> dict[str, Any]:
    annotation_id = source.get("annotation_id") or obj.get("annotation_id") or ""
    utterance_id = source.get("utterance_id") or source.get("atc_instruction_id") or source.get("uid") or obj.get("utterance_id") or ""
    status = obj.get("annotation_status")
    intents = obj.get("intents")
    if status not in {"done", "skip"}:
        status = "done" if isinstance(intents, list) and intents else "skip"
    if not isinstance(intents, list):
        intents = []
    clean_intents: list[dict[str, Any]] = []
    for item in intents:
        if isinstance(item, dict):
            clean_intents.append(item)
    if status == "skip":
        clean_intents = []

    return {
        "annotation_id": annotation_id,
        "utterance_id": utterance_id,
        "source_file": source.get("source_file"),
        "audio": source.get("audio"),
        "start": source.get("start"),
        "end": source.get("end"),
        "language": source.get("language"),
        "text": source.get("text"),
        "annotation_status": status,
        "intents": clean_intents,
        "note": obj.get("note") if isinstance(obj.get("note"), str) else "",
        "raw_response": raw_response,
    }


def load_model(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, prompt: str, max_new_tokens: int, temperature: float) -> str:
    return generate_batch(tokenizer, model, [prompt], max_new_tokens, temperature)[0]


def build_chat_text(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
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
        return f"{SYSTEM_PROMPT}\n\n{prompt}"


def generate_batch(tokenizer, model, prompts: list[str], max_new_tokens: int, temperature: float) -> list[str]:
    import torch

    texts = [build_chat_text(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "temperature": temperature if temperature > 0 else None,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **kwargs)

    prompt_width = inputs["input_ids"].shape[1]
    results: list[str] = []
    for idx in range(len(prompts)):
        new_tokens = output_ids[idx][prompt_width:]
        results.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return results


def write_report(path: Path, rows: list[dict[str, Any]], failures: list[dict[str, Any]], elapsed: float, model_path: str) -> None:
    total_intents = sum(len(row.get("intents", [])) for row in rows)
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("annotation_status"))
        status_counts[status] = status_counts.get(status, 0) + 1
    lines = [
        "# Qwen3-4B Intent Prediction Report",
        "",
        f"- Model path: `{model_path}`",
        f"- Rows predicted: {len(rows)}",
        f"- Failed rows: {len(failures)}",
        f"- Predicted intents: {total_intents}",
        f"- Elapsed seconds: {elapsed:.1f}",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(status_counts.items()):
        lines.append(f"- {key}: {value}")
    if failures:
        lines.extend(["", "## Failures", ""])
        for failure in failures[:50]:
            lines.append(f"- {failure.get('annotation_id')}: {failure.get('error')}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/gbh/gubinhao/LLM/Qwen3-4B-Instruct-2507")
    parser.add_argument("--sample", default="/home/cjj/atc_intent_eval/data/gold/annotation_sample_500.jsonl")
    parser.add_argument("--schema", default="/home/cjj/atc_intent_eval/schemas/atc_intent_schema.json")
    parser.add_argument("--out", default="/home/cjj/atc_intent_eval/outputs/qwen3_4b/predictions.jsonl")
    parser.add_argument("--report", default="/home/cjj/atc_intent_eval/reports/qwen3_4b_prediction_report.md")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=5)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.sample))
    batch = rows[args.start : args.start + args.limit if args.limit >= 0 else None]
    schema = json.load(open(args.schema, encoding="utf-8"))
    intent_types = compact_intent_types(schema)

    tokenizer, model = load_model(args.model)

    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    start_time = time.time()
    out_path = Path(args.out)
    batch_size = max(1, args.batch_size)
    for offset in range(0, len(batch), batch_size):
        chunk = batch[offset : offset + batch_size]
        prompts = [build_prompt(row, intent_types) for row in chunk]
        raw_responses: list[str] = []
        try:
            raw_responses = generate_batch(tokenizer, model, prompts, args.max_new_tokens, args.temperature)
        except Exception as exc:  # pragma: no cover - operational path
            raw_responses = ["" for _ in chunk]
            failures.extend(
                {
                    "annotation_id": row.get("annotation_id"),
                    "utterance_id": row.get("utterance_id"),
                    "text": row.get("text"),
                    "error": f"batch_generate_failed: {exc}",
                    "raw_response": "",
                }
                for row in chunk
            )
            continue

        for row, raw_response in zip(chunk, raw_responses):
            try:
                obj = find_json_object(raw_response)
                pred = normalize_prediction(obj, row, raw_response)
                pred["model"] = "qwen3-4b"
                pred["model_path"] = args.model
                predictions.append(pred)
            except Exception as exc:  # pragma: no cover - operational path
                failures.append(
                    {
                        "annotation_id": row.get("annotation_id"),
                        "utterance_id": row.get("utterance_id"),
                        "text": row.get("text"),
                        "error": str(exc),
                        "raw_response": raw_response,
                    }
                )

        processed = min(offset + len(chunk), len(batch))
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
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
