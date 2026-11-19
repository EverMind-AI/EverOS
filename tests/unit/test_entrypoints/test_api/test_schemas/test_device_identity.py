"""``POST /memory/device/add`` request shapes: what the route refuses (422).

Who owns each utterance is the domain's (``tests/unit/test_memory/test_device.py``);
these pin the input rules that keep an owner id and a label safe.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.entrypoints.api.schemas import (
    DeviceAddRequest,
    DeviceSpeaker,
    DeviceUtterance,
)
from everos.entrypoints.api.schemas.device import DeviceAddResponseData
from everos.service import DEVICE_OWNER_SEPARATOR, DeviceLine

_TS = 1_700_000_000_000


def _request(
    *utterances: DeviceUtterance, device: str = "dev_kitchen"
) -> DeviceAddRequest:
    return DeviceAddRequest(
        device_id=device, session_id="s_kitchen", utterances=list(utterances)
    )


def _says(
    slot: str | None = None,
    *,
    name: str | None = None,
    consented: bool = True,
    text: str = "hi",
) -> DeviceUtterance:
    speaker = (
        None
        if slot is None
        else DeviceSpeaker(slot=slot, display_name=name, consented=consented)
    )
    return DeviceUtterance(speaker=speaker, text=text, timestamp=_TS)


def test_neither_half_may_contain_the_separator() -> None:
    """Otherwise ``a@b`` + ``c`` and ``a`` + ``b@c`` would be one owner."""
    with pytest.raises(ValidationError):
        _says(f"spk{DEVICE_OWNER_SEPARATOR}1")
    with pytest.raises(ValidationError):
        _request(_says(None), device=f"dev{DEVICE_OWNER_SEPARATOR}x")


def test_a_batch_whose_owner_would_not_fit_is_refused() -> None:
    with pytest.raises(ValidationError):
        _request(_says("s" * 61 + "_1"), device="d" * 120)


@pytest.mark.parametrize(
    "slot",
    [
        "",
        "a/b",
        "../x",
        ".hidden",
        "-x",
        "a" * 64,
        "1",
        "007",
        "_x",
        "alice",
        "corp.com1",
    ],
)
def test_a_slot_must_be_a_safe_distinct_label(slot: str) -> None:
    """Starts with a letter, holds a digit or ``_``, no ``.``: never a date, a
    plain word, or (after ``device@``) an e-mail address."""
    with pytest.raises(ValidationError):
        DeviceSpeaker(slot=slot)


@pytest.mark.parametrize("slot", ["spk_01", "guest2", "a1", "voice-print_7"])
def test_usual_slots_pass(slot: str) -> None:
    assert DeviceSpeaker(slot=slot).slot == slot


@pytest.mark.parametrize("device", ["", "a/b", "..", "dev::x", "dev@x"])
def test_a_device_id_goes_through_the_existing_path_safe_id(device: str) -> None:
    with pytest.raises(ValidationError):
        _request(_says("spk_1"), device=device)


def test_a_blank_display_name_is_no_name() -> None:
    """Whitespace must not stand in for a name downstream."""
    assert DeviceSpeaker(slot="spk_1", display_name="   ").display_name is None


def test_lines_carry_every_field_to_the_domain() -> None:
    request = _request(
        _says("spk_1", name="Alice", consented=False, text="a"), _says(None, text="b")
    )
    assert request.lines() == [
        DeviceLine(
            text="a", timestamp=_TS, slot="spk_1", display_name="Alice", consented=False
        ),
        DeviceLine(text="b", timestamp=_TS),
    ]


def test_response_counts_the_batch_and_names_its_owners() -> None:
    data = DeviceAddResponseData(
        message_count=2, status="accumulated", owners=["dev_kitchen@spk_1"]
    )
    assert data.model_dump() == {
        "message_count": 2,
        "status": "accumulated",
        "owners": ["dev_kitchen@spk_1"],
    }


def test_a_batch_needs_at_least_one_utterance() -> None:
    with pytest.raises(ValidationError):
        DeviceAddRequest(device_id="dev_kitchen", session_id="s", utterances=[])


def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        DeviceUtterance(text="hi", timestamp=_TS, owner_id="u1")  # type: ignore[call-arg]
