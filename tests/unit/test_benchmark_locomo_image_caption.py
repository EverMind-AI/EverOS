"""A LoCoMo turn that shares an image reaches the extractor with the image described.

910 of LoCoMo's 5,882 turns share an image and only their ``blip_caption`` says what is
in it. The harness used to send ``text`` alone, so those facts never reached the store,
while the everalgo benchmark loader writes ``[<speaker> shared an image: <caption>]``
in front of the text. These tests pin that wording, so both harnesses feed the
extractor the same message.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import run as run_mod  # noqa: E402
from pydantic import TypeAdapter  # noqa: E402

_IMAGE_TURN = {
    "speaker": "Caroline",
    "dia_id": "D1:5",
    "text": "The transgender stories were so inspiring!",
    "img_url": ["https://example.invalid/a.jpg"],
    "blip_caption": "a photo of a dog walking past a wall with a painting of a woman",
}


def test_a_shared_image_is_described_in_front_of_the_text() -> None:
    assert run_mod._message_text(_IMAGE_TURN) == (
        "[Caroline shared an image: a photo of a dog walking past a wall with a "
        "painting of a woman] The transgender stories were so inspiring!"
    )


def test_a_turn_without_an_image_is_unchanged() -> None:
    turn = {"speaker": "Melanie", "dia_id": "D1:6", "text": "Thanks, Caroline!"}
    assert run_mod._message_text(turn) == "Thanks, Caroline!"


def test_an_image_without_a_caption_still_says_an_image_was_shared() -> None:
    turn = {k: v for k, v in _IMAGE_TURN.items() if k != "blip_caption"}
    text = run_mod._message_text(turn)
    assert text.startswith("[Caroline shared an image: an image] ")


def test_an_image_with_no_text_is_kept_rather_than_skipped() -> None:
    turn = {**_IMAGE_TURN, "text": ""}
    assert run_mod._message_text(turn).startswith("[Caroline shared an image: a photo")


def test_the_loader_sends_the_described_turn(tmp_path: Path) -> None:
    data = [
        {
            "conversation": {
                "speaker_a": "Caroline",
                "speaker_b": "Melanie",
                "session_1_date_time": "1:56 pm on 8 May, 2023",
                "session_1": [
                    {"speaker": "Melanie", "dia_id": "D1:4", "text": "How are you?"},
                    _IMAGE_TURN,
                ],
            },
            "qa": [],
        }
    ]
    path = tmp_path / "locomo10.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    sessions, _qa, _a, _b = run_mod.load_conversation(str(path), 0)
    texts = [m["text"] for m in sessions[0]["messages"]]
    assert texts[0] == "How are you?"
    assert texts[1].startswith("[Caroline shared an image: a photo of a dog")
    assert texts[1].endswith("The transgender stories were so inspiring!")


def test_the_described_text_is_part_of_the_ingest_identity() -> None:
    """A store extracted without captions must not be reused as if it had them."""
    from config import MESSAGE_FORMAT, BenchmarkConfig, RunSpec, ingest_identity

    identity = ingest_identity(BenchmarkConfig(), "locomo")
    assert identity["message_format"] == MESSAGE_FORMAT == "2"
    # The run record takes it as written: an int here stopped the run at startup.
    record_type = TypeAdapter(RunSpec.model_fields["ingest_identity"].annotation)
    assert record_type.validate_python(identity) == identity
