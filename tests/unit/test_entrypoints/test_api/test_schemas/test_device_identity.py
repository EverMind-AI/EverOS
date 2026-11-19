"""The device-assistant identity mapping: device id + speaker slot -> owner.

A device serves several people with one agent instance. These tests pin the
rules the hardware path depends on, so a later route or a refactor of the
schemas cannot quietly change whose memory an utterance lands in:

* an owner id is ``<device_id>@<slot>``, deterministic and path-safe;
* neither half may contain the separator, so one owner id names one
  (device, speaker) pair;
* an unidentified speaker has no per-person owner (the utterance is the
  device's);
* a non-consenting speaker gets no owner and none of their words is written;
* ``display_name`` never participates: renaming a person moves no memory.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.entrypoints.api.schemas import (
    DeviceAddRequest,
    DeviceSpeaker,
    DeviceUtterance,
    speaker_owner_id,
)
from everos.entrypoints.api.schemas.device import (
    DEVICE_OWNER_SEPARATOR,
    DeviceAddResponseData,
)

_TS = 1_700_000_000_000


def _request(*utterances: DeviceUtterance, device: str = "dev_kitchen") -> dict:
    return DeviceAddRequest(
        device_id=device,
        session_id="s_kitchen",
        utterances=list(utterances),
    )


def _says(
    slot: str | None = None,
    *,
    name: str | None = None,
    consented: bool = True,
    text: str = "hi",
    timestamp: int = _TS,
) -> DeviceUtterance:
    speaker = (
        None
        if slot is None
        else DeviceSpeaker(slot=slot, display_name=name, consented=consented)
    )
    return DeviceUtterance(speaker=speaker, text=text, timestamp=timestamp)


# ── The owner id ─────────────────────────────────────────────────────────


def test_owner_is_the_device_and_the_slot() -> None:
    assert speaker_owner_id("dev_kitchen", "spk_a7") == "dev_kitchen@spk_a7"


def test_the_same_slot_on_two_devices_is_two_owners() -> None:
    """A roster slot is device-local: one person, two rooms, two memories."""
    assert speaker_owner_id("dev_a", "spk_1") != speaker_owner_id("dev_b", "spk_1")


def test_owner_is_deterministic_and_independent_of_the_display_name() -> None:
    anonymous = _request(_says("spk_1"))
    renamed = _request(_says("spk_1", name="Alice"))
    assert (
        anonymous.speaker_owners() == renamed.speaker_owners() == ["dev_kitchen@spk_1"]
    )


def test_neither_half_may_contain_the_separator() -> None:
    """Otherwise ``a@b`` + ``c`` and ``a`` + ``b@c`` would be one owner."""
    with pytest.raises(ValidationError):
        _says(f"spk{DEVICE_OWNER_SEPARATOR}other")
    with pytest.raises(ValidationError):
        _request(_says(None), device=f"dev{DEVICE_OWNER_SEPARATOR}x")


def test_the_composed_owner_fits_the_owner_limit() -> None:
    with pytest.raises(ValueError, match="limit is 128"):
        speaker_owner_id("d" * 120, "s" * 20)


def test_a_batch_whose_owner_would_not_fit_is_refused() -> None:
    with pytest.raises(ValidationError):
        _request(_says("s" * 63), device="d" * 120)


@pytest.mark.parametrize("slot", ["", "a/b", "../x", ".hidden", "-x", "a" * 64])
def test_a_slot_must_be_a_safe_roster_id(slot: str) -> None:
    with pytest.raises(ValidationError):
        DeviceSpeaker(slot=slot)


@pytest.mark.parametrize("device", ["", "a/b", "..", "dev::x", "dev@x"])
def test_a_device_id_goes_through_the_existing_path_safe_id(device: str) -> None:
    with pytest.raises(ValidationError):
        _request(_says("spk_1"), device=device)


# ── Who the batch writes to ──────────────────────────────────────────────


def test_an_unidentified_speaker_has_no_owner_of_their_own() -> None:
    request = _request(_says(None, text="who left this here"))
    assert request.speaker_owners() == []


def test_a_non_consenting_speaker_gets_no_owner() -> None:
    request = _request(_says("spk_2", consented=False))
    assert request.speaker_owners() == []


def test_a_non_consenting_speakers_words_are_not_written() -> None:
    """Not even under the device: dropped, while the others' words stay."""
    request = _request(
        _says("spk_1", text="we need milk"),
        _says("spk_2", consented=False, text="do not keep this"),
        _says(None, text="who left this here"),
    )
    assert [m["content"] for m in request.projected_messages()] == [
        "we need milk",
        "who left this here",
    ]


def test_owners_come_back_once_each_in_first_spoken_order() -> None:
    request = _request(
        _says("spk_b"),
        _says("spk_a"),
        _says("spk_b"),
        _says("spk_c"),
    )
    assert request.speaker_owners() == [
        "dev_kitchen@spk_b",
        "dev_kitchen@spk_a",
        "dev_kitchen@spk_c",
    ]


def test_a_two_person_exchange_becomes_two_owners() -> None:
    """What one memcell fans out to: one episode copy per distinct sender."""
    request = _request(
        _says("spk_a", name="Alice", text="we need milk"),
        _says("spk_b", name="Bob", text="and bread"),
    )
    assert request.speaker_owners() == [
        "dev_kitchen@spk_a",
        "dev_kitchen@spk_b",
    ]


# ── Request / response shape ─────────────────────────────────────────────


def test_response_datacounts_the_batch_and_names_its_owners() -> None:
    data = DeviceAddResponseData(
        message_count=2,
        status="accumulated",
        owners=["dev_kitchen@spk_1"],
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


def test_an_unnamed_speaker_is_labelled_by_the_slot_not_the_owner_id() -> None:
    """The label is what the narrative calls the person; the owner id
    ("dev@slot") gets shortened to the device id by the model."""
    request = _request(_says("spk_1"), _says("spk_2", name="Bob"), _says(None))
    assert [m["sender_name"] for m in request.projected_messages()] == [
        "spk_1",
        "Bob",
        None,
    ]
