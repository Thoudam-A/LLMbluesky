#!/usr/bin/env python3
"""Build deterministic rule parses for historical ATC utterances.

This is the first stage of the controller-habit dataset pipeline.  It does not
call an LLM and its output is not gold data.  Every extracted intent keeps the
matched text span and rule id so later LLM review can audit or overturn it.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RULE_VERSION = "habit_rule_v0.6"
SPACE_RE = re.compile(r"\s+")
ZH_DIGITS = {
    "洞": "0",
    "零": "0",
    "幺": "1",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "拐": "7",
    "七": "7",
    "八": "8",
    "九": "9",
}
ZH_DIGIT_CHARS = "洞零幺一二两三四五六拐七八九"
ZH_NUMBER_CHARS = ZH_DIGIT_CHARS + "十百千"
NUM_TOKEN = rf"(?:[0-9]{{1,5}}|[{ZH_NUMBER_CHARS}]{{1,8}})"
SPOKEN_DIGIT_RE = re.compile(rf"[{ZH_DIGIT_CHARS}]")
EN_DIGITS = {
    "zero": "0", "oh": "0", "one": "1", "two": "2", "tree": "3", "three": "3",
    "fower": "4", "four": "4", "fife": "5", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "niner": "9", "nine": "9",
}
EN_NUMBER_WORD = r"(?:zero|oh|one|two|tree|three|fower|four|fife|five|six|seven|eight|niner|nine|hundred|thousand|and)"
EN_NUMBER_TOKEN = rf"(?:{EN_NUMBER_WORD})(?:[ -]+{EN_NUMBER_WORD}){{0,8}}"
EN_CALLSIGN_FORBIDDEN_PREFIX_WORDS = {
    "turn", "left", "right", "heading", "descend", "climb", "maintain", "speed", "reduce",
    "increase", "contact", "approach", "radar", "continue", "clear", "cleared", "runway", "qnh",
    "report", "direct", "hold", "fly", "level", "altitude",
}

AIRLINE_ALIASES = (
    "中国国际", "中国货运", "东方", "东航", "南方", "南航", "国航", "上航",
    "四川", "川航", "山东", "山航", "厦航", "厦门", "深圳", "深航", "海南",
    "海航", "吉祥", "春秋", "白鹭", "成都", "联合", "首都", "天津", "华夏",
    "长龙", "顺丰", "邮政", "金鹏", "西藏", "河北", "奥凯", "瑞丽", "昆明",
    "多彩", "幸福", "福州", "重庆", "青岛", "九元", "东海", "警航", "联航", "国泰",
    "港龙", "汉莎", "货航",
)
AIRLINE_PATTERN = "|".join(sorted(map(re.escape, AIRLINE_ALIASES), key=len, reverse=True))
CALLSIGN_WITH_ALIAS_RE = re.compile(
    rf"(?P<raw>(?:{AIRLINE_PATTERN})(?:{AIRLINE_PATTERN})?的?[{ZH_DIGIT_CHARS}0-9]{{2,6}})"
)
CALLSIGN_BARE_RE = re.compile(rf"^(?P<raw>[{ZH_DIGIT_CHARS}0-9]{{2,6}})(?=[，,])")
CALLSIGN_LATIN_RE = re.compile(r"\b(?P<raw>[A-Z]{2,3}\s?[0-9]{2,5})\b", re.IGNORECASE)
CALLSIGN_LATIN_ZH_RE = re.compile(rf"\b(?P<prefix>[A-Za-z]{{2,15}})\s*(?P<digits>[{ZH_DIGIT_CHARS}]{{2,6}})")

MAIN_TYPES = {
    "altitude_change",
    "altitude_maintain",
    "speed_adjust",
    "speed_procedure",
    "heading_change",
}
TARGET_REQUIRED = {
    ("altitude_change", "climb"),
    ("altitude_change", "descend"),
    ("altitude_maintain", "maintain_altitude"),
    ("speed_adjust", "set_speed"),
    ("speed_adjust", "reduce_speed"),
    ("speed_adjust", "increase_speed"),
    ("heading_change", "fly_heading"),
    ("heading_change", "turn_left_heading"),
    ("heading_change", "turn_right_heading"),
    ("heading_change", "turn_left_degrees"),
    ("heading_change", "turn_right_degrees"),
}
CORRECTION_MARKERS = ("更正", "改为", "纠正", "correction", "correcting", "rather", "i say again")


def validation_flags_for_intent(
    intent: dict[str, Any],
    text: str,
    *,
    intent_count: int,
    distinct_callsign_count: int,
    conflicting_target_keys: set[tuple[str, str]],
) -> list[str]:
    flags: list[str] = []
    intent_type = str(intent.get("intent_type") or "")
    action = str(intent.get("action") or "")
    target = intent.get("target_value")
    unit = intent.get("unit")
    if not intent.get("callsign"):
        flags.append("missing_callsign")
    if (intent_type, action) in TARGET_REQUIRED and target is None:
        flags.append("missing_target")
    if isinstance(target, (int, float)):
        if intent_type.startswith("altitude_"):
            if unit == "m" and not 300 <= target <= 15000:
                flags.append("implausible_altitude_m")
            elif unit == "ft" and not 1000 <= target <= 50000:
                flags.append("implausible_altitude_ft")
            elif unit == "flight_level" and not 20 <= target <= 500:
                flags.append("implausible_flight_level")
        elif intent_type == "speed_adjust" and not 80 <= target <= 500:
            flags.append("implausible_speed_kt")
        elif intent_type == "heading_change" and not 0 <= target <= 360:
            flags.append("implausible_heading_degree")
    lower = text.lower()
    correction_scan = re.sub(r"修正(?:海压|气压)", "", lower)
    if any(marker in correction_scan for marker in CORRECTION_MARKERS):
        flags.append("utterance_contains_correction")
    if intent_count >= 3:
        flags.append("three_or_more_intents")
    if distinct_callsign_count >= 2:
        flags.append("multiple_distinct_callsigns")
    if (intent_type, action) in conflicting_target_keys:
        flags.append("multiple_targets_same_action")
    return flags


@dataclass(frozen=True)
class NumberValue:
    value: int
    raw: str
    explicit_magnitude: bool
    source: str


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return SPACE_RE.sub(" ", str(value).strip())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def chinese_integer(raw: str) -> NumberValue | None:
    """Parse ATC digit strings plus common colloquial 百/千 forms.

    Examples: 幺八洞 -> 180, 五百五 -> 550, 三千六 -> 3600.
    Context-specific scaling such as terminal altitude 幺八 -> 1800 is applied
    separately by ``number_for_slot``.
    """

    token = raw.strip()
    if not token:
        return None
    if token.isdigit():
        return NumberValue(int(token), token, False, "arabic")
    if any(ch not in ZH_NUMBER_CHARS for ch in token):
        return None

    def digit(ch: str) -> int:
        return int(ZH_DIGITS[ch])

    if not any(unit in token for unit in "十百千"):
        digits = "".join(ZH_DIGITS[ch] for ch in token)
        return NumberValue(int(digits), token, False, "spoken_digits")

    zero_chars = {"零", "洞"}

    def plain_digits(value: str) -> int:
        digits = "".join(ZH_DIGITS[ch] for ch in value if ch in ZH_DIGITS)
        return int(digits) if digits else 0

    def parse_magnitude(value: str) -> int:
        if "千" in value:
            left, right = value.split("千", 1)
            base = (plain_digits(left) or 1) * 1000
            if not right:
                return base
            if right[0] in zero_chars:
                return base + parse_magnitude(right[1:])
            if "百" in right or "十" in right:
                return base + parse_magnitude(right)
            digits = plain_digits(right)
            return base + digits * (100 if len(right) == 1 else 10)
        if "百" in value:
            left, right = value.split("百", 1)
            base = (plain_digits(left) or 1) * 100
            if not right:
                return base
            if right[0] in zero_chars:
                return base + parse_magnitude(right[1:])
            if "十" in right:
                return base + parse_magnitude(right)
            digits = plain_digits(right)
            return base + digits * (10 if len(right) == 1 else 1)
        if "十" in value:
            left, right = value.split("十", 1)
            return (plain_digits(left) or 1) * 10 + plain_digits(right)
        return plain_digits(value)

    return NumberValue(parse_magnitude(token), token, True, "chinese_number")


def english_integer(raw: str) -> NumberValue | None:
    words = re.findall(r"[A-Za-z]+", raw.lower())
    words = [word for word in words if word != "and"]
    if not words or any(word not in EN_DIGITS and word not in {"hundred", "thousand"} for word in words):
        return None
    if "hundred" not in words and "thousand" not in words:
        digits = "".join(EN_DIGITS[word] for word in words)
        return NumberValue(int(digits), raw, False, "english_spoken_digits")
    total = 0
    current = 0
    pending: int | None = None
    for word in words:
        if word in EN_DIGITS:
            pending = int(EN_DIGITS[word])
        elif word == "hundred":
            current += (1 if pending is None else pending) * 100
            pending = None
        elif word == "thousand":
            current += 1 if pending is None else pending
            total += current * 1000
            current = 0
            pending = None
    if pending is not None:
        current += pending
    return NumberValue(total + current, raw, True, "english_number")


def number_for_slot(
    raw: str, slot: str, standard_level: bool = False, explicit_unit: str | None = None
) -> tuple[int | None, str | None, bool]:
    parsed = chinese_integer(raw)
    if parsed is None:
        return None, None, False
    value = parsed.value
    inferred = False
    if slot == "altitude":
        if explicit_unit:
            unit = "ft" if explicit_unit in {"英尺", "feet", "foot"} else "m"
            return value, unit, False
        if standard_level:
            # In Shanghai transcripts, “标准气压五千四” is a metric altitude
            # referenced to standard pressure, not FL5400. Only compact spoken
            # digit forms such as “标准幺两洞” denote a flight level.
            if parsed.explicit_magnitude:
                return value, "m", False
            if parsed.source == "spoken_digits" and len(raw) == 2:
                value *= 100
                return value, "m", True
            return value, "flight_level", inferred
        if parsed.source == "spoken_digits" and len(raw) in {2, 3}:
            value *= 100
            inferred = True
        return value, "m", inferred
    if slot == "speed":
        return value, "kt", False
    if slot == "heading":
        return value, "degree", False
    if slot == "qnh":
        return value, "hPa", False
    return value, None, False


def normalize_callsign(raw: str) -> str:
    compact = SPACE_RE.sub("", raw).upper().replace("的", "")
    for alias in AIRLINE_ALIASES:
        doubled = alias + alias
        if compact.startswith(doubled):
            compact = alias + compact[len(doubled) :]
            break
    return "".join(ZH_DIGITS.get(ch, ch) for ch in compact)


def english_callsign_from_prefix(text: str) -> tuple[str | None, str | None]:
    if "," not in text and "，" not in text:
        return None, None
    prefix = re.split(r"[，,]", text, maxsplit=1)[0].strip()
    words = re.findall(r"[A-Za-z]+", prefix)
    if not words:
        return None, None
    first_digit = next((index for index, word in enumerate(words) if word.lower() in EN_DIGITS), None)
    if first_digit is None:
        return None, None
    digit_words = words[first_digit:]
    if not digit_words or any(word.lower() not in EN_DIGITS for word in digit_words):
        return None, None
    airline_words = words[:first_digit]
    if len(airline_words) > 3 or any(word.lower() in EN_CALLSIGN_FORBIDDEN_PREFIX_WORDS for word in airline_words):
        return None, None
    digits = "".join(EN_DIGITS[word.lower()] for word in digit_words)
    airline = " ".join(airline_words).upper()
    normalized = f"{airline} {digits}".strip()
    return normalized, prefix


def callsign_from_text(text: str) -> tuple[str | None, str | None, str | None]:
    search_zone = text[:50]
    match = CALLSIGN_WITH_ALIAS_RE.search(search_zone)
    rule_id = "CALLSIGN_ALIAS"
    if match is None:
        match = CALLSIGN_BARE_RE.search(search_zone)
        rule_id = "CALLSIGN_BARE_PREFIX"
    if match is None:
        mixed = CALLSIGN_LATIN_ZH_RE.search(search_zone)
        if mixed is not None:
            raw = mixed.group(0)
            normalized = f"{mixed.group('prefix').upper()} {normalize_callsign(mixed.group('digits'))}"
            return normalized, raw, "CALLSIGN_LATIN_ZH"
    if match is None:
        match = CALLSIGN_LATIN_RE.search(search_zone)
        rule_id = "CALLSIGN_LATIN"
    if match is None:
        normalized, raw = english_callsign_from_prefix(search_zone)
        if normalized:
            return normalized, raw, "CALLSIGN_ENGLISH_SPOKEN"
        return None, None, None
    raw = match.group("raw")
    return normalize_callsign(raw), raw, rule_id


def make_intent(
    *,
    callsign: str | None,
    callsign_raw: str | None,
    intent_type: str,
    action: str,
    rule_id: str,
    raw_span: str,
    target_value: int | float | str | None = None,
    unit: str | None = None,
    raw_value: str | None = None,
    unit_inferred: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "callsign": callsign,
        "callsign_raw": callsign_raw,
        "intent_type": intent_type,
        "action": action,
        "target_value": target_value,
        "unit": unit,
        "raw_value": raw_value,
        "raw_span": raw_span,
        "rule_id": rule_id,
        "unit_inferred": unit_inferred,
    }
    result.update(extra)
    return result


def add_unique(intents: list[dict[str, Any]], intent: dict[str, Any]) -> None:
    key = (
        intent.get("intent_type"),
        intent.get("action"),
        intent.get("target_value"),
        intent.get("unit"),
        intent.get("raw_span"),
    )
    for existing in intents:
        existing_key = (
            existing.get("intent_type"),
            existing.get("action"),
            existing.get("target_value"),
            existing.get("unit"),
            existing.get("raw_span"),
        )
        if key == existing_key:
            return
    intents.append(intent)


def extract_altitude(text: str, callsign: str | None, callsign_raw: str | None) -> list[dict[str, Any]]:
    intents: list[dict[str, Any]] = []
    standard_pressure = r"(?:标准(?:气压|压)?|标压)"
    altitude_boundary = rf"(?=\s*(?:保持|听|有交叉|以上|以下|速度|航向|联系|再见|上|下|高度|下降率|上升率|尽快|增速|减速|取消|有穿越|修正海压|修正气压|修压|海压|{standard_pressure}|{AIRLINE_PATTERN}|[A-Z]|，|,|。|$))"
    specs = [
        ("climb", "ALT_CLIMB", rf"(?<!速度)(?<!速度控制)(?:上到(?:{standard_pressure}的?)?|上{standard_pressure}的?|上升到|爬升到|高度上(?:到)?|上高度(?:到)?(?:{standard_pressure}的?)?)\s*(?:呃|啊)?\s*({NUM_TOKEN})\s*(米|英尺)?{altitude_boundary}"),
        ("descend", "ALT_DESCEND", rf"(?<!速度)(?<!速度控制)(?:下到(?:{standard_pressure}的?)?|下{standard_pressure}的?|下降到|下降至|高度(?:下|下降)(?:到)?|下高度(?:到)?(?:{standard_pressure}的?)?|下降高度到|下降高度|下高度)\s*(?:呃|啊)?\s*({NUM_TOKEN})\s*(米|英尺)?{altitude_boundary}"),
        ("maintain_altitude", "ALT_MAINTAIN_EXPLICIT", rf"(?:保持高度|高度保持)\s*({NUM_TOKEN})\s*(米|英尺)?"),
        ("maintain_altitude", "ALT_MAINTAIN_IMPLICIT", rf"保持\s*({NUM_TOKEN})\s*(米|英尺)?(?=\s*(?:，|,|。|$))"),
    ]
    for action, rule_id, pattern in specs:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw_value = match.group(1)
            explicit_unit = match.group(2)
            standard_level = "标准" in match.group(0) or "标压" in match.group(0)
            target, unit, inferred = number_for_slot(
                raw_value, "altitude", standard_level=standard_level, explicit_unit=explicit_unit
            )
            intent_type = "altitude_maintain" if action == "maintain_altitude" else "altitude_change"
            add_unique(
                intents,
                make_intent(
                    callsign=callsign,
                    callsign_raw=callsign_raw,
                    intent_type=intent_type,
                    action=action,
                    rule_id=rule_id,
                    raw_span=match.group(0),
                    target_value=target,
                    unit=unit,
                    raw_value=raw_value,
                    unit_inferred=inferred or (unit == "m" and explicit_unit is None),
                ),
            )
    english_specs = [
        ("climb", "ALT_CLIMB_EN", rf"\bclimb(?: and maintain)?\s+(?P<value>{EN_NUMBER_TOKEN})\s+(?P<unit>meters?|feet)\b"),
        ("descend", "ALT_DESCEND_EN", rf"\bdescend(?: and maintain)?\s+(?P<value>{EN_NUMBER_TOKEN})\s+(?P<unit>meters?|feet)\b"),
        ("maintain_altitude", "ALT_MAINTAIN_EN", rf"(?<!and )\bmaintain\s+(?P<value>{EN_NUMBER_TOKEN})\s+(?P<unit>meters?|feet)\b"),
    ]
    for action, rule_id, pattern in english_specs:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            parsed = english_integer(match.group("value"))
            if parsed is None:
                continue
            intent_type = "altitude_maintain" if action == "maintain_altitude" else "altitude_change"
            unit = "ft" if match.group("unit").lower().startswith("f") else "m"
            add_unique(
                intents,
                make_intent(
                    callsign=callsign,
                    callsign_raw=callsign_raw,
                    intent_type=intent_type,
                    action=action,
                    rule_id=rule_id,
                    raw_span=match.group(0),
                    target_value=parsed.value,
                    unit=unit,
                    raw_value=match.group("value"),
                ),
            )
    return intents


def extract_speed(text: str, callsign: str | None, callsign_raw: str | None) -> list[dict[str, Any]]:
    intents: list[dict[str, Any]] = []
    for match in re.finditer(r"(?:调速|速度(?:控制)?)按程序|速度正常", text):
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="speed_procedure",
                action="speed_as_procedure",
                rule_id="SPD_PROCEDURE",
                raw_span=match.group(0),
            ),
        )

    pattern = rf"(?P<prefix>速度(?:控制)?(?:在)?(?:减(?:到)?|下(?:到)?|加(?:到)?|上(?:到)?|保持)?|减速(?:到)?|增速(?:到)?|加速(?:到)?|表速(?:保持)?)\s*(?P<value>{NUM_TOKEN})\s*(?P<explicit_unit>节)?\s*(?P<bound>以上|以下)?"
    for match in re.finditer(pattern, text):
        span = match.group(0)
        if "按程序" in span or "速度正常" in span:
            continue
        prefix = match.group("prefix")
        if "减" in prefix or "下" in prefix:
            action, rule_id = "reduce_speed", "SPD_REDUCE"
        elif "增" in prefix or "加" in prefix or "上" in prefix:
            action, rule_id = "increase_speed", "SPD_INCREASE"
        elif "保持" in prefix:
            action, rule_id = "maintain_speed", "SPD_MAINTAIN"
        else:
            action, rule_id = "set_speed", "SPD_SET"
        raw_value = match.group("value")
        target, unit, inferred = number_for_slot(raw_value, "speed")
        bound = match.group("bound")
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="speed_adjust",
                action=action,
                rule_id=rule_id,
                raw_span=span,
                target_value=target,
                unit=unit,
                raw_value=raw_value,
                unit_inferred=inferred or match.group("explicit_unit") is None,
                constraint_operator=">=" if bound == "以上" else "<=" if bound == "以下" else "=",
                target_semantics="lower_bound" if bound == "以上" else "upper_bound" if bound == "以下" else "exact_value",
            ),
        )
    english_pattern = rf"\b(?P<prefix>reduce speed|increase speed|maintain speed|speed)\s+(?P<value>{EN_NUMBER_TOKEN})\s*(?:knots?|kts?)\b"
    for match in re.finditer(english_pattern, text, flags=re.IGNORECASE):
        parsed = english_integer(match.group("value"))
        if parsed is None:
            continue
        prefix = match.group("prefix").lower()
        if prefix.startswith("reduce"):
            action, rule_id = "reduce_speed", "SPD_REDUCE_EN"
        elif prefix.startswith("increase"):
            action, rule_id = "increase_speed", "SPD_INCREASE_EN"
        elif prefix.startswith("maintain"):
            action, rule_id = "maintain_speed", "SPD_MAINTAIN_EN"
        else:
            action, rule_id = "set_speed", "SPD_SET_EN"
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="speed_adjust",
                action=action,
                rule_id=rule_id,
                raw_span=match.group(0),
                target_value=parsed.value,
                unit="kt",
                raw_value=match.group("value"),
            ),
        )
    return intents


def extract_heading(text: str, callsign: str | None, callsign_raw: str | None) -> list[dict[str, Any]]:
    intents: list[dict[str, Any]] = []
    heading_value = rf"(?:[0-9]{{1,3}}|[{ZH_DIGIT_CHARS}]{{2,3}})"
    heading_pattern = rf"(?P<turn>左转|右转)?(?P<maintain>保持)?航向(?:飞)?\s*(?P<value>{heading_value})"
    for match in re.finditer(heading_pattern, text):
        raw_value = match.group("value")
        target, unit, inferred = number_for_slot(raw_value, "heading")
        if match.group("turn") == "左转":
            action, rule_id = "turn_left_heading", "HDG_LEFT_TO"
        elif match.group("turn") == "右转":
            action, rule_id = "turn_right_heading", "HDG_RIGHT_TO"
        elif match.group("maintain"):
            action, rule_id = "maintain_heading", "HDG_MAINTAIN_VALUE"
        else:
            action, rule_id = "fly_heading", "HDG_FLY"
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="heading_change",
                action=action,
                rule_id=rule_id,
                raw_span=match.group(0),
                target_value=target,
                unit=unit,
                raw_value=raw_value,
                unit_inferred=inferred,
            ),
        )

    turn_angle_pattern = rf"(?P<turn>左转|右转)[^，,。]{{0,5}}?(?P<value>{NUM_TOKEN})度"
    for match in re.finditer(turn_angle_pattern, text):
        raw_value = match.group("value")
        target, unit, inferred = number_for_slot(raw_value, "heading")
        side = "left" if match.group("turn") == "左转" else "right"
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="heading_change",
                action=f"turn_{side}_degrees",
                rule_id=f"HDG_{side.upper()}_DEGREES",
                raw_span=match.group(0),
                target_value=target,
                unit=unit,
                raw_value=raw_value,
                unit_inferred=inferred,
            ),
        )

    if not any(item["intent_type"] == "heading_change" for item in intents):
        for match in re.finditer(r"保持(?:三边|四边|当前)?(?:航向|航迹)", text):
            add_unique(
                intents,
                make_intent(
                    callsign=callsign,
                    callsign_raw=callsign_raw,
                    intent_type="heading_change",
                    action="maintain_heading",
                    rule_id="HDG_MAINTAIN",
                    raw_span=match.group(0),
                ),
            )
    english_heading = rf"\b(?:(?P<turn>turn left|turn right)\s+)?heading\s+(?P<value>{EN_NUMBER_TOKEN})\b"
    for match in re.finditer(english_heading, text, flags=re.IGNORECASE):
        parsed = english_integer(match.group("value"))
        if parsed is None:
            continue
        turn = (match.group("turn") or "").lower()
        if turn == "turn left":
            action, rule_id = "turn_left_heading", "HDG_LEFT_TO_EN"
        elif turn == "turn right":
            action, rule_id = "turn_right_heading", "HDG_RIGHT_TO_EN"
        else:
            action, rule_id = "fly_heading", "HDG_FLY_EN"
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="heading_change",
                action=action,
                rule_id=rule_id,
                raw_span=match.group(0),
                target_value=parsed.value,
                unit="degree",
                raw_value=match.group("value"),
            ),
        )
    if re.search(r"\b(?:maintain|continue) (?:present|current) heading\b", text, flags=re.IGNORECASE):
        match = re.search(r"\b(?:maintain|continue) (?:present|current) heading\b", text, flags=re.IGNORECASE)
        assert match is not None
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="heading_change",
                action="maintain_heading",
                rule_id="HDG_MAINTAIN_EN",
                raw_span=match.group(0),
            ),
        )
    return intents


def extract_auxiliary(text: str, callsign: str | None, callsign_raw: str | None) -> list[dict[str, Any]]:
    intents: list[dict[str, Any]] = []

    for match in re.finditer(r"(?:右转|左转)?直飞\s*(?P<fix>[A-Z](?:[A-Z ]{1,30}[A-Z])|[\u4e00-\u9fff]{2,8})(?=\s*[，,。]|\s*[洞零幺一二两三四五六拐七八九0-9]{2,5}|入航|$)", text):
        fix = SPACE_RE.sub(" ", match.group("fix").strip())
        fix = re.sub(r"(入航|加入)$", "", fix)
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="direct_to_fix",
                action="direct_to",
                rule_id="DIRECT_TO_FIX",
                raw_span=match.group(0),
                fix=fix or None,
            ),
        )

    for match in re.finditer(r"(?:调速|速度(?:控制)?)按程序|按程序[^，,。]{0,15}(?:进近|离港)|(?:加入|接)[^，,。]{1,16}(?:入航|程序|航路)", text):
        if "速" in match.group(0):
            continue
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="procedure_assignment",
                action="follow_procedure",
                rule_id="PROCEDURE_ASSIGNMENT",
                raw_span=match.group(0),
            ),
        )

    for match in re.finditer(r"(?:当前位置)?(?P<turn>左转|右转)?盘旋(?:等待)?", text):
        action = "orbit_hold" if match.group("turn") else "hold"
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="holding_instruction",
                action=action,
                rule_id="HOLD_ORBIT",
                raw_span=match.group(0),
            ),
        )

    approach_pattern = r"(?:可以|许可|继续)[^，,。]{0,12}(?:盲降)?进近|进近许可"
    for match in re.finditer(approach_pattern, text):
        action = "continue_approach" if "继续" in match.group(0) else "cleared_approach"
        runway_match = re.search(rf"([{ZH_DIGIT_CHARS}0-9]{{2}})(左|右|中)?", match.group(0))
        runway = None
        if runway_match:
            raw_runway = runway_match.group(1)
            parsed = chinese_integer(raw_runway)
            if parsed:
                runway = f"{parsed.value:02d}" + {"左": "L", "右": "R", "中": "C"}.get(runway_match.group(2) or "", "")
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="approach_clearance",
                action=action,
                rule_id="APPROACH_CLEARANCE",
                raw_span=match.group(0),
                runway=runway,
            ),
        )

    qnh_pattern = rf"(?:修正海压(?:是)?|QNH\s*)\s*(?P<value>{NUM_TOKEN})"
    for match in re.finditer(qnh_pattern, text, flags=re.IGNORECASE):
        raw_value = match.group("value")
        target, unit, inferred = number_for_slot(raw_value, "qnh")
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="qnh_setting",
                action="set_qnh",
                rule_id="QNH_SET",
                raw_span=match.group(0),
                target_value=target,
                unit=unit,
                raw_value=raw_value,
                unit_inferred=inferred,
            ),
        )

    if "雷达服务终止" in text:
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="radar_service",
                action="radar_service_terminated",
                rule_id="RADAR_SERVICE_TERMINATED",
                raw_span="雷达服务终止",
            ),
        )
    elif re.search(r"雷达\s*(?:看到了|识别)", text):
        match = re.search(r"雷达\s*(?:看到了|识别)", text)
        assert match is not None
        add_unique(
            intents,
            make_intent(
                callsign=callsign,
                callsign_raw=callsign_raw,
                intent_type="radar_service",
                action="radar_identified",
                rule_id="RADAR_IDENTIFIED",
                raw_span=match.group(0),
            ),
        )
    return intents


def parse_rule_intents(text: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    callsign, callsign_raw, callsign_rule = callsign_from_text(text)
    intents: list[dict[str, Any]] = []
    for extractor in (extract_altitude, extract_speed, extract_heading, extract_auxiliary):
        for intent in extractor(text, callsign, callsign_raw):
            add_unique(intents, intent)

    all_callsigns = {
        normalize_callsign(match.group("raw")) for match in CALLSIGN_WITH_ALIAS_RE.finditer(text)
    }
    if callsign:
        all_callsigns.add(callsign)
    distinct_callsign_count = len(all_callsigns)
    targets_by_key: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for intent in intents:
        if intent.get("target_value") is not None:
            key = (str(intent.get("intent_type")), str(intent.get("action")))
            targets_by_key[key].add(f"{intent.get('target_value')}|{intent.get('unit')}")
    conflicting_target_keys = {key for key, values in targets_by_key.items() if len(values) >= 2}
    review_reasons: list[str] = []
    if intents and not callsign:
        review_reasons.append("missing_callsign")
    if len(intents) >= 3:
        review_reasons.append("three_or_more_intents")
    for intent in intents:
        validation_flags = validation_flags_for_intent(
            intent,
            text,
            intent_count=len(intents),
            distinct_callsign_count=distinct_callsign_count,
            conflicting_target_keys=conflicting_target_keys,
        )
        intent["validation_flags"] = validation_flags
        intent["rule_eligible_main"] = intent.get("intent_type") in MAIN_TYPES and not validation_flags
        review_reasons.extend(f"validation:{flag}" for flag in validation_flags)
        pair = (str(intent.get("intent_type")), str(intent.get("action")))
        if pair in TARGET_REQUIRED and intent.get("target_value") is None:
            review_reasons.append(f"missing_target:{pair[0]}:{pair[1]}")
        if intent.get("unit_inferred"):
            review_reasons.append(f"inferred_unit:{intent.get('intent_type')}")
    if not intents:
        review_reasons.append("rule_no_match")

    critical_complete = bool(intents) and bool(callsign) and not any(
        "missing_target" in reason or "implausible_" in reason or "utterance_contains_correction" in reason
        for reason in review_reasons
    )
    if critical_complete and not review_reasons:
        confidence = "high"
    elif critical_complete:
        confidence = "medium"
    else:
        confidence = "low"
    metadata = {
        "callsign": callsign,
        "callsign_raw": callsign_raw,
        "callsign_rule_id": callsign_rule,
        "distinct_callsign_count": distinct_callsign_count,
        "review_required": bool(review_reasons),
        "review_reasons": sorted(set(review_reasons)),
        "rule_confidence": confidence,
    }
    return intents, metadata


def load_json_list(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected top-level JSON list")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path}: all rows must be JSON objects")
    return value


def read_weak_labels(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def comparison_time_key(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def comparison_key(row: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(row.get("source_file") or ""),
        str(row.get("audio") or ""),
        comparison_time_key(row.get("start")),
        comparison_time_key(row.get("end")),
        norm_text(row.get("text")),
    )


def weak_label_comparison(rule_rows: list[dict[str, Any]], weak_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not weak_rows:
        return None

    rule_by_key: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rule_rows:
        rule_by_key[comparison_key(row)].append(row)

    matched = 0
    exact_type_set = 0
    weak_intents = 0
    rule_intents = 0
    type_overlap = 0
    unmatched = 0
    per_type: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for weak in weak_rows:
        candidates = rule_by_key.get(comparison_key(weak), [])
        if not candidates:
            unmatched += 1
            continue
        matched += 1
        rule = candidates[0]
        weak_types = collections.Counter(
            str(intent.get("intent_type")) for intent in (weak.get("intents") or []) if isinstance(intent, dict)
        )
        rule_types = collections.Counter(
            str(intent.get("intent_type")) for intent in (rule.get("rule_intents") or []) if isinstance(intent, dict)
        )
        weak_intents += sum(weak_types.values())
        rule_intents += sum(rule_types.values())
        type_overlap += sum((weak_types & rule_types).values())
        for intent_type in set(weak_types) | set(rule_types):
            per_type[intent_type]["weak"] += weak_types[intent_type]
            per_type[intent_type]["rule"] += rule_types[intent_type]
            per_type[intent_type]["overlap"] += min(weak_types[intent_type], rule_types[intent_type])
        exact_type_set += int(weak_types == rule_types)
    per_type_result: dict[str, Any] = {}
    for intent_type, values in sorted(per_type.items()):
        weak_count = values["weak"]
        rule_count = values["rule"]
        overlap_count = values["overlap"]
        per_type_result[intent_type] = {
            "weak": weak_count,
            "rule": rule_count,
            "overlap": overlap_count,
            "overlap_over_weak": overlap_count / weak_count if weak_count else None,
            "overlap_over_rule": overlap_count / rule_count if rule_count else None,
        }
    return {
        "warning": "Comparison target is GPT-5.5 weak preannotation without explicit human review; values are diagnostic only.",
        "weak_rows": len(weak_rows),
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "exact_intent_multiset_rows": exact_type_set,
        "exact_intent_multiset_rate": exact_type_set / matched if matched else None,
        "weak_intents": weak_intents,
        "rule_intents": rule_intents,
        "intent_type_overlap": type_overlap,
        "overlap_over_weak": type_overlap / weak_intents if weak_intents else None,
        "overlap_over_rule": type_overlap / rule_intents if rule_intents else None,
        "per_intent_type": per_type_result,
    }


def normalized_field_value(field: str, value: Any) -> Any:
    if value in {None, ""}:
        return None
    if field == "callsign":
        return normalize_callsign(str(value))
    if field == "unit":
        aliases = {"knots": "kt", "kts": "kt", "meters": "m", "meter": "m", "degrees": "degree"}
        compact = str(value).strip().lower()
        return aliases.get(compact, compact)
    if field == "target_value":
        try:
            numeric = float(value)
            return int(numeric) if numeric.is_integer() else numeric
        except (TypeError, ValueError):
            return str(value).strip()
    return str(value).strip()


def weak_label_field_conflicts(
    rule_rows: list[dict[str, Any]], weak_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not weak_rows:
        return [], {}
    rule_by_key = {comparison_key(row): row for row in rule_rows}
    fields = ("callsign", "intent_type", "action", "target_value", "unit", "fix", "runway")
    conflicts: list[dict[str, Any]] = []
    difference_counts: collections.Counter[str] = collections.Counter()
    for weak in weak_rows:
        rule = rule_by_key.get(comparison_key(weak))
        if rule is None:
            conflicts.append(
                {
                    "utterance_id": weak.get("utterance_id"),
                    "source_file": weak.get("source_file"),
                    "audio": weak.get("audio"),
                    "start": weak.get("start"),
                    "text": weak.get("text"),
                    "conflict_type": "missing_rule_row",
                }
            )
            difference_counts["missing_rule_row"] += 1
            continue
        rule_intents = [item for item in rule.get("rule_intents", []) if isinstance(item, dict)]
        unused = set(range(len(rule_intents)))
        for weak_intent in [item for item in weak.get("intents", []) if isinstance(item, dict)]:
            same_type = [
                index for index in unused if rule_intents[index].get("intent_type") == weak_intent.get("intent_type")
            ]
            same_action = [
                index for index in same_type if rule_intents[index].get("action") == weak_intent.get("action")
            ]
            candidates = same_action or same_type
            if not candidates:
                conflicts.append(
                    {
                        "utterance_id": weak.get("utterance_id"),
                        "source_file": weak.get("source_file"),
                        "audio": weak.get("audio"),
                        "start": weak.get("start"),
                        "text": weak.get("text"),
                        "conflict_type": "missing_rule_intent",
                        "weak_intent": weak_intent,
                    }
                )
                difference_counts["missing_rule_intent"] += 1
                continue
            index = candidates[0]
            unused.remove(index)
            rule_intent = rule_intents[index]
            differences: dict[str, Any] = {}
            for field in fields:
                weak_value = normalized_field_value(field, weak_intent.get(field))
                rule_value = normalized_field_value(field, rule_intent.get(field))
                if weak_value != rule_value:
                    differences[field] = {"weak": weak_intent.get(field), "rule": rule_intent.get(field)}
                    difference_counts[field] += 1
            if differences:
                conflicts.append(
                    {
                        "utterance_id": weak.get("utterance_id"),
                        "source_file": weak.get("source_file"),
                        "audio": weak.get("audio"),
                        "start": weak.get("start"),
                        "text": weak.get("text"),
                        "conflict_type": "field_difference",
                        "differences": differences,
                        "weak_intent": weak_intent,
                        "rule_intent": rule_intent,
                    }
                )
        for index in sorted(unused):
            conflicts.append(
                {
                    "utterance_id": weak.get("utterance_id"),
                    "source_file": weak.get("source_file"),
                    "audio": weak.get("audio"),
                    "start": weak.get("start"),
                    "text": weak.get("text"),
                    "conflict_type": "extra_rule_intent",
                    "rule_intent": rule_intents[index],
                }
            )
            difference_counts["extra_rule_intent"] += 1
    return conflicts, dict(difference_counts.most_common())


def build_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# 历史管制指令规则抽取报告",
        "",
        f"- 规则版本：`{summary['rule_version']}`",
        f"- 原始JSON文件：{counts['json_files']}",
        f"- 全部话语：{counts['all_utterances']}",
        f"- ATC话语：{counts['atc_utterances']}",
        f"- 有效ATC话语：{counts['valid_atc_utterances']}",
        f"- 排除ATC话语：{counts['excluded_atc_utterances']}",
        f"- 规则命中话语：{counts['matched_utterances']}",
        f"- 规则未命中话语：{counts['unmatched_utterances']}",
        f"- 原子意图：{counts['atomic_intents']}",
        f"- 检测到的主动作意图：{counts['detected_main_intents']}",
        f"- 主指标候选意图：{counts['main_metric_intents']}",
        "",
        "> 本报告只描述确定性规则输出。规则结果不是人工真值，也不用于直接报告最终模仿精度。",
        "",
        "## 意图类型分布",
        "",
        "| 类型 | 数量 |",
        "|---|---:|",
    ]
    for name, count in summary["intent_type_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## 规则置信等级", "", "| 等级 | 话语数 |", "|---|---:|"])
    for name, count in summary["confidence_counts"].items():
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## 待复核原因", "", "| 原因 | 话语数 |", "|---|---:|"])
    for name, count in summary["review_reason_counts"].items():
        lines.append(f"| {name} | {count} |")
    comparison = summary.get("weak_label_comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## 与现有500条弱标注的诊断性比较",
                "",
                f"- 匹配话语：{comparison['matched_rows']}/{comparison['weak_rows']}",
                f"- 意图类型多重集完全一致：{comparison['exact_intent_multiset_rows']}（{comparison['exact_intent_multiset_rate']:.2%}）",
                f"- 弱标注意图覆盖：{comparison['intent_type_overlap']}/{comparison['weak_intents']}（{comparison['overlap_over_weak']:.2%}）",
                f"- 规则意图被弱标注覆盖：{comparison['intent_type_overlap']}/{comparison['rule_intents']}（{comparison['overlap_over_rule']:.2%}）",
                "- 注意：比较对象没有明确人工复核字段，这些比例不是正式准确率。",
                "",
                "| 意图类型 | 弱标注 | 规则 | 重合 | 弱标注覆盖 | 规则支持率 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for intent_type, values in comparison["per_intent_type"].items():
            weak_coverage = values["overlap_over_weak"]
            rule_support = values["overlap_over_rule"]
            weak_text = "NA" if weak_coverage is None else f"{weak_coverage:.2%}"
            rule_text = "NA" if rule_support is None else f"{rule_support:.2%}"
            lines.append(
                f"| {intent_type} | {values['weak']} | {values['rule']} | {values['overlap']} | {weak_text} | {rule_text} |"
            )
        field_audit = summary.get("weak_label_field_audit") or {}
        lines.extend(
            [
                "",
                f"- 字段级冲突记录：{field_audit.get('conflict_rows', 0)}",
                "- 字段级差异分布：",
            ]
        )
        for field, count in (field_audit.get("difference_counts") or {}).items():
            lines.append(f"  - {field}: {count}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="instruction_data")
    parser.add_argument("--out-dir", default="datasets/controller_habit_v1")
    parser.add_argument("--report", default="reports/controller_habit_rule_extraction_report.md")
    parser.add_argument("--weak-labels", default="_tmp_gold_validated_500.jsonl")
    parser.add_argument("--limit", type=int, help="Process only the first N valid ATC utterances for sanity testing.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)
    weak_path = Path(args.weak_labels) if args.weak_labels else None
    files = sorted(raw_dir.glob("*.json"), key=lambda path: path.name)
    if not files:
        raise SystemExit(f"No JSON files found under {raw_dir}")

    inventory: list[dict[str, Any]] = []
    parses: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    all_count = 0
    atc_count = 0
    valid_count = 0
    seen_valid = 0

    for path in files:
        raw_bytes = path.read_bytes()
        rows = load_json_list(path)
        file_atc = sum(1 for row in rows if norm_text(row.get("speaker")) == "ATC")
        file_valid_atc = sum(
            1 for row in rows if norm_text(row.get("speaker")) == "ATC" and norm_text(row.get("text"))
        )
        inventory.append(
            {
                "source_file": path.name,
                "sha256": sha256_bytes(raw_bytes),
                "bytes": len(raw_bytes),
                "rows": len(rows),
                "atc_rows": file_atc,
                "valid_atc_rows": file_valid_atc,
                "distinct_audio": len({str(row.get("audio") or "") for row in rows if row.get("audio")}),
            }
        )
        all_count += len(rows)
        atc_count += file_atc
        valid_count += file_valid_atc

        for row_index, row in enumerate(rows):
            if norm_text(row.get("speaker")) != "ATC":
                continue
            text = norm_text(row.get("text"))
            uid = f"{path.stem}#{row_index:05d}"
            base = {
                "utterance_id": row.get("id") or uid,
                "source_file": path.name,
                "row_index": row_index,
                "audio": row.get("audio"),
                "historical_session_id": "session_" + sha256_text(str(row.get("audio") or path.name))[:16],
                "start": row.get("start"),
                "end": row.get("end"),
                "language": row.get("language"),
                "raw_text": row.get("text"),
                "text": text,
                "normalized_text_sha256": sha256_text(SPACE_RE.sub("", text)),
                "rule_version": RULE_VERSION,
            }
            if not text:
                excluded.append({**base, "exclusion_reason": "empty_atc_text"})
                continue
            if args.limit is not None and seen_valid >= args.limit:
                continue
            seen_valid += 1
            intents, metadata = parse_rule_intents(text)
            detected_main_intents = [intent for intent in intents if intent.get("intent_type") in MAIN_TYPES]
            main_intents = [intent for intent in detected_main_intents if intent.get("rule_eligible_main")]
            parses.append(
                {
                    **base,
                    **metadata,
                    "rule_status": "matched" if intents else "no_match",
                    "rule_intents": intents,
                    "detected_main_rule_intents": detected_main_intents,
                    "main_metric_rule_intents": main_intents,
                    "main_metric_rule_candidate": bool(main_intents),
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = out_dir / "raw_inventory.csv"
    with inventory_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    write_jsonl(out_dir / "utterance_rule_parse.jsonl", parses)
    write_jsonl(out_dir / "excluded_events.jsonl", excluded)

    intent_types = collections.Counter(
        str(intent.get("intent_type")) for row in parses for intent in row.get("rule_intents", [])
    )
    confidence = collections.Counter(str(row.get("rule_confidence")) for row in parses)
    review_reasons = collections.Counter(
        str(reason) for row in parses for reason in row.get("review_reasons", [])
    )
    detected_main_intent_count = sum(len(row.get("detected_main_rule_intents", [])) for row in parses)
    main_intent_count = sum(len(row.get("main_metric_rule_intents", [])) for row in parses)
    weak_rows = read_weak_labels(weak_path)
    comparison = weak_label_comparison(parses, weak_rows)
    weak_conflicts, weak_difference_counts = weak_label_field_conflicts(parses, weak_rows)
    weak_conflict_path = out_dir / "weak_label_field_conflicts.jsonl"
    write_jsonl(weak_conflict_path, weak_conflicts)
    summary: dict[str, Any] = {
        "rule_version": RULE_VERSION,
        "limited_run": args.limit is not None,
        "limit": args.limit,
        "counts": {
            "json_files": len(files),
            "all_utterances": all_count,
            "atc_utterances": atc_count,
            "valid_atc_utterances": valid_count,
            "processed_valid_atc_utterances": len(parses),
            "excluded_atc_utterances": len(excluded),
            "matched_utterances": sum(1 for row in parses if row["rule_status"] == "matched"),
            "unmatched_utterances": sum(1 for row in parses if row["rule_status"] == "no_match"),
            "atomic_intents": sum(len(row.get("rule_intents", [])) for row in parses),
            "detected_main_intents": detected_main_intent_count,
            "main_metric_intents": main_intent_count,
            "main_metric_candidate_utterances": sum(1 for row in parses if row["main_metric_rule_candidate"]),
            "distinct_audio": len({str(row.get("audio")) for row in parses if row.get("audio")}),
            "unique_normalized_text_hashes": len({str(row["normalized_text_sha256"]) for row in parses}),
        },
        "intent_type_counts": dict(intent_types.most_common()),
        "confidence_counts": dict(confidence.most_common()),
        "review_reason_counts": dict(review_reasons.most_common()),
        "weak_label_comparison": comparison,
        "weak_label_field_audit": {
            "conflict_rows": len(weak_conflicts),
            "difference_counts": weak_difference_counts,
            "warning": "The comparison target is unreviewed GPT-5.5 weak preannotation, not final gold.",
        },
        "outputs": {
            "raw_inventory": str(inventory_path),
            "utterance_rule_parse": str(out_dir / "utterance_rule_parse.jsonl"),
            "excluded_events": str(out_dir / "excluded_events.jsonl"),
            "weak_label_field_conflicts": str(weak_conflict_path),
            "summary": str(out_dir / "rule_extraction_summary.json"),
            "manifest": str(out_dir / "manifest.json"),
            "report": str(report_path),
        },
        "scope_warning": "Rule output is deterministic weak extraction, not reviewed gold and not a final imitation metric.",
    }
    json_dump(out_dir / "rule_extraction_summary.json", summary)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary), encoding="utf-8")
    output_paths = [
        inventory_path,
        out_dir / "utterance_rule_parse.jsonl",
        out_dir / "excluded_events.jsonl",
        weak_conflict_path,
        out_dir / "rule_extraction_summary.json",
        report_path,
    ]
    aggregate_input = "\n".join(f"{item['source_file']}|{item['sha256']}" for item in inventory)
    manifest = {
        "rule_version": RULE_VERSION,
        "script": {
            "path": str(Path(__file__)),
            "sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "run_arguments": {
            "raw_dir": str(raw_dir),
            "out_dir": str(out_dir),
            "report": str(report_path),
            "weak_labels": str(weak_path) if weak_path else None,
            "limit": args.limit,
        },
        "inputs": {
            "raw_json_files": len(inventory),
            "aggregate_raw_inventory_sha256": sha256_text(aggregate_input),
            "weak_labels_sha256": sha256_bytes(weak_path.read_bytes()) if weak_path and weak_path.exists() else None,
        },
        "outputs": {
            str(path): {"sha256": sha256_bytes(path.read_bytes()), "bytes": path.stat().st_size}
            for path in output_paths
        },
        "scope_warning": summary["scope_warning"],
    }
    json_dump(out_dir / "manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
