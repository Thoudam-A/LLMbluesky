#!/usr/bin/env python3
"""Post-process ATC intent predictions with conservative task rules."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ZH_DIGITS = {
    "\u6d1e": "0",
    "\u96f6": "0",
    "\u5e7a": "1",
    "\u4e00": "1",
    "\u4e8c": "2",
    "\u4e24": "2",
    "\u4e09": "3",
    "\u56db": "4",
    "\u4e94": "5",
    "\u516d": "6",
    "\u62d0": "7",
    "\u4e03": "7",
    "\u516b": "8",
    "\u4e5d": "9",
}
EN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "tree": "3",
    "three": "3",
    "fower": "4",
    "four": "4",
    "fife": "5",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "niner": "9",
    "nine": "9",
}

CH_DIGIT_RE = "[\u6d1e\u96f6\u5e7a\u4e00\u4e8c\u4e24\u4e09\u56db\u4e94\u516d\u62d0\u4e03\u516b\u4e5d]"


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


def has(text: str, needle: str) -> bool:
    return needle in text


def first_callsign(text: str) -> str:
    for sep in ("\uff0c", ","):
        if sep in text:
            return text.split(sep, 1)[0].strip()
    return ""


def ch_digits_to_number(text: str) -> str:
    return "".join(ZH_DIGITS.get(ch, ch) for ch in text)


def spoken_english_callsign(text: str) -> str:
    words = re.findall(r"[A-Za-z]+", text.lower())
    if words and all(word in EN_DIGITS for word in words):
        return "".join(EN_DIGITS[word] for word in words)
    return text


def en_spoken_digits(words: list[str]) -> str:
    return "".join(EN_DIGITS[word] for word in words if word in EN_DIGITS)


def callsign_from_text(text: str) -> str:
    prefix = first_callsign(text)
    return spoken_english_callsign(prefix) if prefix else ""


def qnh_from_text(text: str) -> str | None:
    # "now use 1008" is QNH in this annotation task unless explicitly framed as radio contact.
    pattern = rf"(?:\u73b0\u5728\u7528|\u4f7f\u7528|\u7528)({CH_DIGIT_RE}{{4}})"
    match = re.search(pattern, text)
    if not match:
        return None
    return ch_digits_to_number(match.group(1))


def frequency_from_text(text: str) -> float | None:
    match = re.search(rf"({CH_DIGIT_RE}{{3}})\u70b9({CH_DIGIT_RE}{{1,3}})", text)
    if match:
        left = ch_digits_to_number(match.group(1))
        right = ch_digits_to_number(match.group(2))
    else:
        digit = r"(?:zero|oh|one|two|tree|three|fower|four|fife|five|six|seven|eight|niner|nine)"
        pattern = rf"\b(?:contact|tower|approach|control|shanghai|on)\b(?:\W+\w+){{0,5}}?\W+((?:{digit}\W+){{3}})(?:decimal|point)\W+((?:{digit}\W*){{1,3}})"
        match = re.search(pattern, text.lower())
        if not match:
            return None
        left = en_spoken_digits(re.findall(digit, match.group(1)))
        right = en_spoken_digits(re.findall(digit, match.group(2)))
    try:
        return float(f"{int(left)}.{right}")
    except ValueError:
        return None


def heading_from_text(text: str) -> str | None:
    match = re.search(rf"\u822a\u5411(?:\u98de)?({CH_DIGIT_RE}{{2,3}})", text)
    if match:
        return ch_digits_to_number(match.group(1)).zfill(3)
    digit = r"(?:zero|oh|one|two|tree|three|fower|four|fife|five|six|seven|eight|niner|nine)"
    match = re.search(rf"\bheading\W+((?:{digit}\W*){{2,3}})", text.lower())
    if match:
        return en_spoken_digits(re.findall(digit, match.group(1))).zfill(3)
    return None


def has_heading_command(text: str) -> bool:
    lower = text.lower()
    return bool(
        heading_from_text(text)
        or re.search(r"[\u5de6\u53f3]\u8f6c[^，,\u3002]{0,8}\u5ea6", text)
        or has(text, "\u822a\u5411\u98de")
        or has(text, "\u822a\u98de")
        or re.search(r"\bturn\s+(?:left|right)\s+heading\b", lower)
    )


def has_speed_adjust(text: str) -> bool:
    lower = text.lower()
    if has(text, "\u6309\u7a0b\u5e8f") or has(text, "\u901f\u5ea6\u6b63\u5e38"):
        return False
    zh_speed_word = has(text, "\u901f\u5ea6") or has(text, "\u51cf\u901f") or has(text, "\u589e\u901f") or has(text, "\u8868\u901f")
    if zh_speed_word and re.search(CH_DIGIT_RE, text):
        return True
    digit = r"(?:zero|oh|one|two|tree|three|fower|four|fife|five|six|seven|eight|niner|nine)"
    return bool(
        re.search(rf"\b(?:speed|reduce speed|increase speed)\W+(?:{digit}\W*){{2,3}}", lower)
        or re.search(r"\baccelerate\b", lower)
    )


def has_altitude_maintain(text: str) -> bool:
    lower = text.lower()
    if (
        has(text, "\u4e0a\u5230")
        or has(text, "\u4e0a\u5347")
        or has(text, "\u4e0a\u9ad8")
        or has(text, "\u4e0b\u5230")
        or has(text, "\u4e0b\u964d")
        or has(text, "\u4e0b\u9ad8")
        or "climb" in lower
        or "descend" in lower
    ):
        return False
    if has(text, "\u901f\u5ea6") or has(text, "\u8868\u901f") or has(text, "\u822a\u5411"):
        return False
    if has(text, "\u4fdd\u6301\u6807\u51c6\u6c14\u538b") or (
        has(text, "\u4fdd\u6301") and re.search(CH_DIGIT_RE, text)
    ):
        return True
    if re.search(r"\bmaintain\b.*\bmeters?\b", lower):
        return True
    altitude_word = r"(?:zero|oh|one|two|tree|three|fower|four|fife|five|six|seven|eight|niner|nine|hundred|thousand)"
    return bool(re.search(rf"\b(?:{altitude_word}\W+){{2,6}}meters?\b", lower))


def runway_from_text(text: str) -> str | None:
    match = re.search(rf"({CH_DIGIT_RE}{{2}})([\u5de6\u53f3\u4e2d])?\u8dd1\u9053", text)
    if not match:
        return None
    side_map = {"\u5de6": "L", "\u53f3": "R", "\u4e2d": "C"}
    return ch_digits_to_number(match.group(1)) + side_map.get(match.group(2) or "", "")


def has_intent(intents: list[dict[str, Any]], intent_type: str, action: str | None = None) -> bool:
    for item in intents:
        if item.get("intent_type") != intent_type:
            continue
        if action is None or item.get("action") == action:
            return True
    return False


def add_intent(intents: list[dict[str, Any]], text: str, intent_type: str, action: str, **slots: Any) -> None:
    if has_intent(intents, intent_type, action):
        return
    item = base_intent(text, intent_type, action)
    item.update({key: value for key, value in slots.items() if value is not None})
    intents.append(item)


def base_intent(text: str, intent_type: str, action: str) -> dict[str, Any]:
    return {
        "callsign": callsign_from_text(text),
        "intent_type": intent_type,
        "action": action,
    }


def is_interrogative_approach(text: str) -> bool:
    return has(text, "\u80fd\u4e0d\u80fd\u8fdb\u8fd1") or has(text, "\u80fd\u4e0d\u80fd\u52a0\u5165")


def is_forecast_runway(text: str) -> bool:
    return has(text, "\u9884\u8ba1") and has(text, "\u8dd1\u9053") and not (
        has(text, "\u53ef\u4ee5\u8fdb\u8fd1") or has(text, "\u8fdb\u8fd1\u8bb8\u53ef") or "cleared" in text.lower()
    )


def rewrite_intent(intent: dict[str, Any], text: str) -> dict[str, Any] | None:
    item = dict(intent)
    item_type = str(item.get("intent_type") or "")

    if not item.get("callsign"):
        cs = callsign_from_text(text)
        if cs:
            item["callsign"] = cs

    if item_type == "altitude_maintain":
        if has(text, "\u4e0a\u5230"):
            item["intent_type"] = "altitude_change"
            item["action"] = "climb"
        elif has(text, "\u4e0b\u5230"):
            item["intent_type"] = "altitude_change"
            item["action"] = "descend"

    if item.get("intent_type") == "frequency_transfer":
        qnh = qnh_from_text(text)
        if qnh and not any(word in text.lower() for word in ("contact", "frequency")):
            item["intent_type"] = "qnh_setting"
            item["action"] = "set_qnh"
            item["target_value"] = qnh
            item["unit"] = "hPa"
            item.pop("frequency", None)

    if item.get("intent_type") == "report_requirement" and has(text, "\u9884\u8ba1\u9700\u8981\u7b49\u5f85"):
        item["intent_type"] = "holding_instruction"
        item["action"] = "expect_hold"

    if item.get("intent_type") == "approach_clearance":
        if is_interrogative_approach(text):
            item["intent_type"] = "confirmation_request"
            item["action"] = "request_information"
        elif is_forecast_runway(text):
            item["intent_type"] = "other_control"
            item["action"] = "other"
        elif has(text, "\u524d\u673a\u8fdb\u8fd1") and not has(text, "\u8fdb\u8fd1\u8bb8\u53ef"):
            return None
        elif text.rstrip().endswith("\u53ef\u4ee5\u3002") or text.rstrip().endswith("\u53ef\u4ee5"):
            item["intent_type"] = "other_control"
            item["action"] = "other"

    if item.get("intent_type") == "confirmation_request" and has(text, "\u542c\u4f60") and has(text, "\u4e2a"):
        item["intent_type"] = "other_control"
        item["action"] = "other"

    if item.get("intent_type") == "heading_change" and has(text, "\u76f4\u98de"):
        item["intent_type"] = "direct_to_fix"
        item["action"] = "direct_to"
        raw = str(item.get("raw_span") or text)
        if "SIERRA SIERRA" in raw:
            item["fix"] = "SIERRA SIERRA"

    runway = runway_from_text(text)
    if runway and item.get("intent_type") in {"approach_clearance", "landing_clearance"}:
        item["runway"] = runway

    return item


def append_missing_intents(intents: list[dict[str, Any]], text: str) -> None:
    lower = text.lower()

    if (
        re.search(r"\u96f7\u8fbe\s*\u770b\u5230", text)
        or has(text, "\u96f7\u8fbe\u670d\u52a1\u7ec8\u6b62")
        or has(text, "\u96f7\u8fbe\u5f15\u5bfc")
        or re.search(r"radar,?\s*contact", lower)
        or "radar service terminated" in lower
    ):
        add_intent(intents, text, "radar_service", "radar_identified")

    heading = heading_from_text(text)
    if heading:
        add_intent(intents, text, "heading_change", "fly_heading", target_value=heading, unit="degree")
    elif has_heading_command(text):
        add_intent(intents, text, "heading_change", "fly_heading", unit="degree")

    frequency = frequency_from_text(text)
    if frequency is not None and (
        has(text, "\u5854\u53f0") or has(text, "\u8054\u7cfb") or has(text, "\u79fb\u4ea4") or "contact" in lower
    ):
        add_intent(intents, text, "frequency_transfer", "contact_frequency", frequency=frequency)

    if (
        has(text, "\u76f2\u964d\u8fdb\u8fd1")
        or has(text, "\u7ee7\u7eed\u8fdb\u8fd1")
        or has(text, "\u53ef\u4ee5\u8fdb\u8fd1")
        or has(text, "\u8fdb\u8fd1\u8bb8\u53ef")
        or re.search(r"\bclear(?:ed)?(?:\s+for)?\s+(?:ils\s+)?approach\b", lower)
        or re.search(r"\bcontinue\s+approach\b", lower)
    ) and not is_interrogative_approach(text) and not is_forecast_runway(text):
        add_intent(intents, text, "approach_clearance", "cleared_approach", runway=runway_from_text(text))

    if has(text, "\u53ef\u4ee5\u843d\u5730") or re.search(r"\bclear(?:ed)?\s+to\s+land\b", lower):
        add_intent(intents, text, "landing_clearance", "cleared_to_land", runway=runway_from_text(text))

    if has_speed_adjust(text):
        add_intent(intents, text, "speed_adjust", "set_speed", unit="kt")

    if has(text, "\u8c03\u901f\u6309\u7a0b\u5e8f") or has(text, "\u901f\u5ea6\u6309\u7a0b\u5e8f") or has(text, "\u901f\u5ea6\u6b63\u5e38"):
        add_intent(intents, text, "speed_procedure", "resume_normal_speed")

    if (
        has(text, "\u4e0a\u5230")
        or has(text, "\u4e0a\u6807\u51c6")
        or has(text, "\u4e0a\u5230\u6807\u51c6")
        or has(text, "\u4e0a\u5347\u5230")
        or has(text, "\u4e0a\u9ad8\u5ea6\u5230")
        or has(text, "\u4e0a\u9ad8")
        or "climb and maintain" in lower
    ):
        add_intent(intents, text, "altitude_change", "climb", unit="m")

    if (
        has(text, "\u4e0b\u5230")
        or has(text, "\u4e0b\u964d\u5230")
        or has(text, "\u4e0b\u9ad8\u5ea6")
        or has(text, "\u4e0b\u9ad8")
        or "descend and maintain" in lower
    ):
        add_intent(intents, text, "altitude_change", "descend", unit="m")

    if has_altitude_maintain(text) and not (
        "descend" in lower or has(text, "\u4e0b\u5230")
    ):
        add_intent(intents, text, "altitude_maintain", "maintain_altitude", unit="m")

    if has(text, "\u4fee\u6b63\u6d77\u538b") or " qnh " in f" {lower} ":
        qnh = qnh_from_text(text)
        add_intent(intents, text, "qnh_setting", "set_qnh", target_value=qnh, unit="hPa")

    if (
        re.search(r"\b(cancel|cancelled)\b.*\brestriction\b", lower)
        or has(text, "\u53d6\u6d88") and has(text, "\u9650\u5236")
        or has(text, "\u6062\u590d\u81ea\u4e3b\u9886\u822a")
        or has(text, "\u53d6\u6d88\u504f\u7f6e")
    ):
        add_intent(intents, text, "restriction_cancel", "cancel_unspecified_restriction", restriction_type="unspecified")

    if (
        "direct to" in lower
        or re.search(r"\bdirect\b", lower)
        or has(text, "\u98de\u5411")
        or has(text, "\u76f4\u98de")
        or re.search(r"\u98de\s+[A-Z][A-Z ]{2,}\b", text)
    ):
        add_intent(intents, text, "direct_to_fix", "direct_to")

    if has(text, "\u7a0b\u5e8f\u79bb\u6e2f") or has(text, "\u8fdb\u6e2f") or has(text, "\u8fdb\u573a") or has(text, "\u52a0\u5165\u8ba1\u5212\u822a\u8def") or " arrival" in lower or "departure procedure" in lower or "follow " in lower:
        add_intent(intents, text, "procedure_assignment", "follow_procedure")

    if has(text, "\u76d8\u65cb") or "hold left" in lower or "hold right" in lower or "outbound time" in lower:
        add_intent(intents, text, "holding_instruction", "hold")

    if has(text, "\u7b49\u5f85") and (has(text, "\u9884\u8ba1") or has(text, "\u9700\u8981")) and not has(text, "\u524d\u9762\u8fd8\u6709"):
        add_intent(intents, text, "holding_instruction", "expect_hold")

    if (has(text, "\u8dd1\u9053") and (has(text, "\u7b49\u5f85") or has(text, "\u5916\u7b49"))) or "hold short at runway" in lower:
        add_intent(intents, text, "takeoff_or_departure_instruction", "hold_for_takeoff")

    if has(text, "\u80fd") and (has(text, "\u5417") or has(text, "\uff1f")):
        add_intent(intents, text, "confirmation_request", "request_information")

    if (
        has(text, "\u8bc1\u5b9e")
        or has(text, "\u786e\u8ba4")
        or has(text, "\u5bf9\u5427")
        or has(text, "\u662f\u5427")
        or has(text, "\u662f\u4e0d\u662f")
        or has(text, "\u4ec0\u4e48\u610f\u56fe")
        or has(text, "\u600e\u4e48\u98de")
        or has(text, "\u6b63\u786e")
        or has(text, "\u5417")
        or has(text, "\uff1f")
        or re.search(r"\bconfirm\b", lower)
        or "do you copy" in lower
        or "how about" in lower
    ) and not has_intent(intents, "confirmation_request"):
        add_intent(intents, text, "confirmation_request", "confirm")

    if (
        re.search(r"\breport\b", lower)
        or (first_callsign(text) and not has(text, "\u6709\u62a5\u544a") and (has(text, "\u62a5\u544a") or re.search(r"\u62a5(?:[\u3002\uff0c,]|$)", text)))
    ):
        add_intent(intents, text, "report_requirement", "report")

    if has(text, "\u6709\u7a7f\u8d8a") or has(text, "\u6709\u4ea4\u53c9") or has(text, "\u6709\u76f8\u5bf9") or has(text, "\u76f8\u4f3c\u822a\u73ed\u53f7"):
        add_intent(intents, text, "traffic_information", "traffic_advisory")

    if (
        has(text, "\u7ed5\u98de")
        or has(text, "\u53f3\u504f")
        or has(text, "\u5f80\u897f\u5357\u4fa7")
        or has(text, "\u5f80\u897f\u4fa7")
        or (has(text, "\u8131\u79bb") and has(text, "\u524d\u65b9") and not has(text, "\u8054\u7cfb"))
    ):
        add_intent(intents, text, "weather_deviation", "deviate_weather")

    if (
        has(text, "\u8bf7\u8bb2")
        or re.search(r"\uff0c\s*\u53ef\u4ee5\u3002?$", text)
        or has(text, "\u542c\u4f60")
        or has(text, "\u9884\u8ba1") and has(text, "\u8dd1\u9053")
        or has(text, "\u4e0d\u884c")
        or has(text, "\u6709\u5f71\u54cd")
        or has(text, "\u7a0d\u7b49") and first_callsign(text)
        or has(text, "\u53ef\u4ee5\u7a7f\u9ad8\u5ea6")
        or has(text, "\u4e0a\u5347\u7387")
        or has(text, "\u4e0b\u964d\u7387")
        or has(text, "\u63a8\u51fa\u5f00\u8f66")
        or has(text, "\u673a\u576a")
        or has(text, "\u6302\u62d6\u8f66")
        or has(text, "\u62d6\u8f66")
        or has(text, "\u6ed1\u884c")
        or has(text, "\u6ed1\u51fa")
        or has(text, "\u7ee7\u7eed\u6ed1")
        or has(text, "\u8fdb\u4f4d")
        or has(text, "\u673a\u4f4d")
        or has(text, "\u524d\u7b49")
        or has(text, "\u5916\u7b49")
        or has(text, "\u8ddf\u5f15\u5bfc")
        or has(text, "\u52a8\u4f5c\u5feb")
        or has(text, "\u8fdb\u8dd1\u9053")
        or has(text, "\u7a7f\u8d8a\u8dd1\u9053")
        or has(text, "\u4e0d\u80fd\u5207\u8fc7")
        or "go ahead" in lower
        or "taxi via" in lower
        or "apron" in lower
        or "parking bay" in lower
        or "push-back" in lower
        or re.search(r"\bhold short at (?!runway\b)", lower)
        or "vacate" in lower
    ):
        add_intent(intents, text, "other_control", "other")

    if (has(text, "\u6709\u76f8\u5bf9") or has(text, "\u76f8\u4f3c\u822a\u73ed\u53f7") or has(text, "\u6ce8\u610f\u5b88\u542c")) and not has_intent(
        intents, "traffic_information"
    ):
        intents.append(base_intent(text, "traffic_information", "traffic_advisory"))

    if has(text, "\u4e5f\u662f\u7b49\u4e00\u4e0b\u662f\u5427") and not has_intent(intents, "confirmation_request"):
        intents.append(base_intent(text, "confirmation_request", "confirm"))

    if has(text, "\u4e5f\u6ce8\u610f\u4e0b\u6cb9\u91cf") and not has_intent(intents, "other_control"):
        intents.append(base_intent(text, "other_control", "other"))

    if has(text, "\u53cd\u9988") and has(text, "\u5206\u949f") and not has_intent(intents, "other_control"):
        intents.append(base_intent(text, "other_control", "other"))


def postprocess_row(row: dict[str, Any]) -> dict[str, Any]:
    text = str(row.get("text") or "")
    processed: list[dict[str, Any]] = []
    for original in row.get("intents", []):
        if not isinstance(original, dict):
            continue
        rewritten = rewrite_intent(original, text)
        if rewritten is not None:
            processed.append(rewritten)

    append_missing_intents(processed, text)
    row = dict(row)
    row["intents"] = processed
    row["annotation_status"] = "done" if processed else "skip"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = [postprocess_row(row) for row in read_jsonl(Path(args.pred))]
    write_jsonl(Path(args.out), rows)
    print(json.dumps({"input_rows": len(rows), "out": args.out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
