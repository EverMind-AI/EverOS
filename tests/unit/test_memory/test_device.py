"""Who owns each utterance of a device batch (``everos.memory.device``)."""

from __future__ import annotations

import pytest

from everos.memory import DeviceLine, project_device_batch, speaker_owner_id
from everos.memory.device import SENDER_NAME_FALLBACK_KEY

_TS = 1_700_000_000_000


def _says(
    slot: str | None = None,
    *,
    name: str | None = None,
    consented: bool = True,
    text: str = "hi",
) -> DeviceLine:
    return DeviceLine(
        text=text, timestamp=_TS, slot=slot, display_name=name, consented=consented
    )


def _batch(*lines: DeviceLine):  # type: ignore[no-untyped-def]
    return project_device_batch("dev_kitchen", lines)


def test_owner_is_the_device_and_the_slot() -> None:
    assert speaker_owner_id("dev_kitchen", "spk_a7") == "dev_kitchen@spk_a7"


def test_the_same_slot_on_two_devices_is_two_owners() -> None:
    """A roster slot is device-local: one person, two rooms, two memories."""
    assert speaker_owner_id("dev_a", "spk_1") != speaker_owner_id("dev_b", "spk_1")


def test_the_composed_owner_fits_the_owner_limit() -> None:
    with pytest.raises(ValueError, match="limit is 128"):
        speaker_owner_id("d" * 120, "s" * 20)


def test_owner_is_independent_of_the_display_name() -> None:
    assert _batch(_says("spk_1")).owners == _batch(_says("spk_1", name="Alice")).owners


def test_owners_come_back_once_each_in_first_spoken_order() -> None:
    batch = _batch(
        _says("spk_b"), _says("spk_a"), _says("spk_b"), _says(None), _says("spk_c")
    )
    assert batch.owners == [
        "dev_kitchen@spk_b",
        "dev_kitchen@spk_a",
        "dev_kitchen@spk_c",
    ]


def test_an_unidentified_line_is_the_devices_and_owns_no_person() -> None:
    batch = _batch(_says(None, text="who left this here"))
    assert batch.owners == []
    ((row),) = batch.messages
    assert (row["sender_id"], row["sender_name"], row["role"]) == (
        "dev_kitchen",
        None,
        "user",
    )
    assert SENDER_NAME_FALLBACK_KEY not in row


def test_one_decline_drops_all_of_that_speakers_batch() -> None:
    """Consent is the person's: a decline anywhere drops their earlier lines
    too, and nothing of theirs is filed under the device."""
    batch = _batch(
        _says("spk_1", text="keep this?"),
        _says("spk_1", consented=False, text="no, do not keep me"),
        _says("spk_2", text="fine for me"),
        _says(None, text="who left this here"),
    )
    assert [m["content"] for m in batch.messages] == [
        "fine for me",
        "who left this here",
    ]
    assert batch.owners == ["dev_kitchen@spk_2"]


def test_a_name_is_sent_as_is_and_the_slot_only_as_a_fallback() -> None:
    """The slot never overwrites a stored name: it rides as the fallback the
    ingest uses only when the sender has no name at all."""
    named, unnamed = _batch(_says("spk_1", name="Alice"), _says("spk_2")).messages
    assert (named["sender_name"], named[SENDER_NAME_FALLBACK_KEY]) == ("Alice", "spk_1")
    assert (unnamed["sender_name"], unnamed[SENDER_NAME_FALLBACK_KEY]) == (
        None,
        "spk_2",
    )
