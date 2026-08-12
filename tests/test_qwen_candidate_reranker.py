from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "evaluation_scripts"
sys.path.insert(0, str(SCRIPTS))

from qwen_candidate_reranker import build_prompt, compact_state, parse_choice  # noqa: E402


def test_compact_prompt_excludes_unknown_fields() -> None:
    state = {
        "callsign": "CES1234",
        "altitude_m": 3000.12345,
        "raw_instruction_text": "must not leak",
    }
    compact = compact_state(state)
    assert compact == {"callsign": "CES1234", "altitude_m": 3000.123}
    prompt = build_prompt(
        state,
        [
            {
                "candidate_id": "ALT_2700",
                "candidate_kind": "altitude",
                "candidate_target": 2700,
                "candidate_score": 0.7,
                "margin": 0.3,
            }
        ],
    )
    assert "raw_instruction_text" not in prompt
    assert "ALT_2700" in prompt


def test_parse_choice_accepts_fenced_json() -> None:
    text = '```json\n{"candidate_id":"SPD_250","reason_codes":["STAR_SEQUENCE"]}\n```'
    choice = parse_choice(text, {"ALT_3000", "SPD_250"}, 0.25)
    assert choice.candidate_id == "SPD_250"
    assert choice.reason_codes == ("STAR_SEQUENCE",)


def test_parse_choice_accepts_direct_candidate_id() -> None:
    choice = parse_choice("SPD_250", {"ALT_3000", "SPD_250"}, 0.25)
    assert choice.candidate_id == "SPD_250"


def test_parse_choice_rejects_hallucinated_candidate() -> None:
    with pytest.raises(ValueError, match="unknown candidate"):
        parse_choice(json.dumps({"candidate_id": "SPD_999"}), {"SPD_250"}, 0.1)
