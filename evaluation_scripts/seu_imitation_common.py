#!/usr/bin/env python3
"""Shared helpers for the SEU controller-imitation data and metric pipeline."""

from __future__ import annotations

import bisect
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


LOCAL_TZ = dt.timezone(dt.timedelta(hours=8), name="Asia/Shanghai")
UTC = dt.timezone.utc

# Chinese radiotelephony aliases observed in the SEU transcripts.  The value is
# the ICAO operator designator used by the trajectory parquet.
AIRLINE_ALIAS_TO_ICAO = {
    "中国国际": "CCA",
    "国际": "CCA",
    "国航": "CCA",
    "东方": "CES",
    "东航": "CES",
    "南方": "CSN",
    "南航": "CSN",
    "上海": "CSH",
    "上航": "CSH",
    "吉祥": "DKH",
    "春秋": "CQH",
    "海南": "CHH",
    "海航": "CHH",
    "四川": "CSC",
    "川航": "CSC",
    "山东": "CDG",
    "山航": "CDG",
    "深圳": "CSZ",
    "深航": "CSZ",
    "厦门": "CXA",
    "厦航": "CXA",
    "白鹭": "CXA",
    "成都": "UEA",
    "联合": "CUA",
    "中联航": "CUA",
    "河北": "HBH",
    "首都": "CBJ",
    "天津": "GCR",
    "奥凯": "OKA",
    "昆明": "KNA",
    "长龙": "CDC",
    "华夏": "HXA",
    "祥鹏": "LKE",
    "东海": "EPA",
    "西部": "CHB",
    "瑞丽": "RLH",
    "青岛": "QDA",
    "邮政": "CYZ",
}

ZH_DIGITS = {
    "零": "0",
    "洞": "0",
    "幺": "1",
    "一": "1",
    "两": "2",
    "二": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "拐": "7",
    "七": "7",
    "八": "8",
    "九": "9",
}

MAIN_INTENT_TYPES = {
    "altitude_change",
    "altitude_maintain",
    "speed_adjust",
    "speed_procedure",
    "heading_change",
}

TYPE_FAMILY = {
    "altitude_change": "altitude",
    "altitude_maintain": "altitude",
    "altitude": "altitude",
    "level": "altitude",
    "speed_adjust": "speed",
    "speed_procedure": "speed",
    "speed": "speed",
    "heading_change": "heading",
    "heading": "heading",
    "course": "heading",
}

PARAMETER_TOLERANCE = {
    "altitude": {"m": 100.0, "ft": 300.0, "fl": 3.0},
    "speed": {"kt": 10.0, "mps": 5.0},
    "heading": {"deg": 10.0},
}

_ALIASES = sorted(AIRLINE_ALIAS_TO_ICAO, key=len, reverse=True)
_ALIAS_PATTERN = "|".join(re.escape(x) for x in _ALIASES)
_DIGIT_TOKEN = r"[0-9零洞幺一两二三四五六拐七八九]{2,6}"
_VAD_PATH_RE = re.compile(
    r"(?P<stamp>\d{14})(?:\.(?P<base_ms>\d{1,6}))?_"
    r"(?P<start>\d+)_(?P<end>\d+)_spk(?P<speaker>\d+)\.wav$",
    re.IGNORECASE,
)


def compact_text(text: Any) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def spoken_digits_to_ascii(value: str) -> str | None:
    value = compact_text(value)
    if not value:
        return None
    out: list[str] = []
    for char in value:
        if char.isdigit():
            out.append(char)
        elif char in ZH_DIGITS:
            out.append(ZH_DIGITS[char])
        else:
            return None
    result = "".join(out)
    return result if 2 <= len(result) <= 6 else None


def normalize_callsign(value: Any) -> str | None:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()
    if (
        re.fullmatch(r"[A-Z0-9]{2,8}", text) is None
        or not any(char.isalpha() for char in text)
        or not any(char.isdigit() for char in text)
    ):
        return None
    return text


def extract_seu_callsign(text: Any) -> dict[str, Any] | None:
    """Extract one callsign without treating frequencies/QNH as flight numbers."""

    normalized = compact_text(text).upper()
    candidates: list[dict[str, Any]] = []

    for match in re.finditer(
        rf"(?P<alias>{_ALIAS_PATTERN})(?P<number>{_DIGIT_TOKEN})",
        normalized,
    ):
        digits = spoken_digits_to_ascii(match.group("number"))
        if digits:
            candidates.append(
                {
                    "callsign": AIRLINE_ALIAS_TO_ICAO[match.group("alias")] + digits,
                    "raw": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "rule": "zh_alias_then_number",
                }
            )

    for match in re.finditer(
        rf"(?P<number>{_DIGIT_TOKEN})(?P<alias>{_ALIAS_PATTERN})",
        normalized,
    ):
        digits = spoken_digits_to_ascii(match.group("number"))
        if digits:
            candidates.append(
                {
                    "callsign": AIRLINE_ALIAS_TO_ICAO[match.group("alias")] + digits,
                    "raw": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "rule": "zh_number_then_alias",
                }
            )

    # Direct ICAO callsigns are retained for English or ASR-normalized speech.
    for match in re.finditer(r"(?<![A-Z0-9])([A-Z]{2,3})[\s-]?(\d{2,6})(?![A-Z0-9])", str(text or "").upper()):
        callsign = normalize_callsign(match.group(0))
        if callsign:
            candidates.append(
                {
                    "callsign": callsign,
                    "raw": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                    "rule": "icao_letters_then_number",
                }
            )

    if not candidates:
        return None
    # Longer aliases are more specific.  If several remain, choose the one
    # nearest an utterance edge; controller calls and readbacks commonly occur
    # at the beginning/end rather than around a frequency.
    candidates.sort(
        key=lambda row: (
            -len(str(row["raw"])),
            min(int(row["start"]), max(0, len(normalized) - int(row["end"]))),
            int(row["start"]),
        )
    )
    chosen = dict(candidates[0])
    chosen["candidate_count"] = len({x["callsign"] for x in candidates})
    return chosen


def parse_vad_segment_time(path_value: Any, utt_value: Any = None) -> dict[str, Any]:
    """Parse a VAD path into local and UTC midpoint timestamps."""

    name = Path(str(path_value or "")).name
    match = _VAD_PATH_RE.search(name)
    if match:
        stamp = match.group("stamp")
        base_fraction = str(match.group("base_ms") or "")
        start_ms = int(match.group("start"))
        end_ms = int(match.group("end"))
        speaker = int(match.group("speaker"))
    else:
        utt_match = re.search(r"OpenASR(?P<stamp>\d{14})", str(utt_value or ""))
        if not utt_match:
            raise ValueError(f"cannot parse VAD timestamp from path={path_value!r}, utt={utt_value!r}")
        stamp = utt_match.group("stamp")
        base_fraction = ""
        start_ms = 0
        end_ms = 0
        speaker = None

    base = dt.datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=LOCAL_TZ)
    if base_fraction:
        base += dt.timedelta(seconds=float(f"0.{base_fraction}"))
    start_local = base + dt.timedelta(milliseconds=start_ms)
    end_local = base + dt.timedelta(milliseconds=end_ms)
    midpoint_local = start_local + (end_local - start_local) / 2
    midpoint_utc = midpoint_local.astimezone(UTC)
    return {
        "segment_start_local": start_local.isoformat(timespec="milliseconds"),
        "segment_end_local": end_local.isoformat(timespec="milliseconds"),
        "event_time_local": midpoint_local.isoformat(timespec="milliseconds"),
        "event_time_utc": midpoint_utc.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "event_time_epoch": midpoint_utc.timestamp(),
        "speaker_cluster": speaker,
        "segment_duration_sec": max(0.0, (end_ms - start_ms) / 1000.0),
    }


def classify_speaker_role(text: Any, callsign: dict[str, Any] | None) -> dict[str, Any]:
    """Conservative lexical/position heuristic; it is not a human role label."""

    normalized = compact_text(text)
    score = 0
    reasons: list[str] = []
    if callsign:
        start = int(callsign["start"])
        end = int(callsign["end"])
        if start <= 3:
            score += 2
            reasons.append("callsign_near_start")
        if len(normalized) - end <= 3:
            score -= 2
            reasons.append("callsign_near_end")

    controller_cues = (
        "雷达看到",
        "雷达识别",
        "上到",
        "下降到",
        "下到",
        "保持高度",
        "速度",
        "左转",
        "右转",
        "航向",
        "联系",
        "修正海压",
        "可以进近",
    )
    readback_cues = ("收到", "明白", "抄收", "再见")
    if any(cue in normalized for cue in controller_cues):
        score += 1
        reasons.append("controller_lexical_cue")
    if any(cue in normalized for cue in readback_cues):
        score -= 1
        reasons.append("readback_lexical_cue")

    if score >= 1:
        role = "controller_candidate"
    elif score <= -1:
        role = "readback_candidate"
    else:
        role = "uncertain"
    return {
        "speaker_role_rule": role,
        "speaker_role_score": score,
        "speaker_role_reasons": reasons,
        "speaker_role_human_verified": False,
    }


def family_for_intent(value: Any) -> str | None:
    return TYPE_FAMILY.get(str(value or "").strip().lower())


def jsonl_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            yield value


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def iso_to_epoch(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def event_epoch(row: dict[str, Any]) -> float:
    for key in ("event_time_epoch", "timestamp_epoch", "time_epoch"):
        value = row.get(key)
        if value is not None:
            return float(value)
    for key in ("event_time_utc", "timestamp", "time"):
        if row.get(key):
            return iso_to_epoch(row[key])
    raise ValueError(f"row has no supported timestamp field: {row}")


def nearest_index(times: list[float], target: float) -> int | None:
    if not times:
        return None
    insertion = bisect.bisect_left(times, target)
    candidates = [idx for idx in (insertion - 1, insertion) if 0 <= idx < len(times)]
    return min(candidates, key=lambda idx: abs(times[idx] - target)) if candidates else None


def finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def target_matches(reference: dict[str, Any], prediction: dict[str, Any]) -> bool | None:
    """Return None when either side lacks a comparable numeric target."""

    ref_value = finite_or_none(reference.get("target_value"))
    pred_value = finite_or_none(prediction.get("target_value"))
    if ref_value is None or pred_value is None:
        return None
    family = family_for_intent(reference.get("intent_type") or reference.get("command_type"))
    unit_aliases = {
        "degree": "deg",
        "degrees": "deg",
        "knot": "kt",
        "knots": "kt",
        "kts": "kt",
        "meter": "m",
        "meters": "m",
        "feet": "ft",
    }
    ref_unit_raw = str(reference.get("unit") or "").lower()
    pred_unit_raw = str(prediction.get("unit") or "").lower()
    ref_unit = unit_aliases.get(ref_unit_raw, ref_unit_raw)
    pred_unit = unit_aliases.get(pred_unit_raw, pred_unit_raw)
    if not family or ref_unit != pred_unit:
        return False
    tolerance = PARAMETER_TOLERANCE.get(family, {}).get(ref_unit)
    if tolerance is None:
        return False
    if family == "heading":
        difference = abs((ref_value - pred_value + 180.0) % 360.0 - 180.0)
    else:
        difference = abs(ref_value - pred_value)
    return difference <= tolerance
