from __future__ import annotations

import json
import sys
from pathlib import Path

USE_CASE_ROOT = (
    Path(__file__).resolve().parents[4] / "use-cases" / "cursor-agent-memory"
)
if str(USE_CASE_ROOT) not in sys.path:
    sys.path.insert(0, str(USE_CASE_ROOT))

from hooklib.transcript import TurnContent, extract_last_turn  # noqa: E402


def _line(entry: dict) -> str:
    return json.dumps(entry)


def test_extract_last_turn_single_exchange() -> None:
    lines = [
        _line(
            {
                "type": "user",
                "message": {"content": "Remember I like dark mode."},
            }
        ),
        _line(
            {
                "type": "assistant",
                "message": {
                    "content": [{"type": "text", "text": "Got it — dark mode."}]
                },
            }
        ),
    ]
    turn = extract_last_turn(lines)
    assert turn.user == "Remember I like dark mode."
    assert turn.assistant == "Got it — dark mode."
    assert turn.has_content


def test_extract_last_turn_ignores_prior_turn_after_marker() -> None:
    lines = [
        _line({"type": "user", "message": {"content": "old question"}}),
        _line({"type": "system", "subtype": "turn_duration"}),
        _line({"type": "user", "message": {"content": "new question"}}),
        _line(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "new answer"}]},
            }
        ),
    ]
    turn = extract_last_turn(lines)
    assert turn.user == "new question"
    assert turn.assistant == "new answer"


def test_turn_fingerprint_stable() -> None:
    turn = TurnContent(user="a", assistant="b")
    assert turn.fingerprint == "a\n---\nb"
