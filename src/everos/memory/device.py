"""One hardware device, several people: who owns what each utterance says.

A device runs ONE agent instance and is spoken to by several people. The
device resolves each speaker to a stable *slot* (voiceprint slot, roster
index); a person's memory owner is derived from the device and the slot, so
the same pipeline that fans a memcell out per ``role="user"`` sender keeps
each person's episodes and profile apart, and utterances nobody was
identified for stay with the device's own owner.

Rules (``docs/device-assistant.md``):

- The owner is ``<device_id>@<slot>``; neither half may contain ``@``.
- A slot that declined (``consented=False``) anywhere in the batch has all of
  its utterances dropped; none is filed under the device.
- The label a narrative calls a person by is their ``display_name``. Without
  one, the slot is offered as a *fallback* label: used only while the sender
  has no stored name, never stored, so a name learned earlier is not
  overwritten. The owner id is never the label: a model shortens
  ``dev@spk_1`` to ``dev`` -- the device's own owner -- and the profile,
  which picks episodes by label, then hands the person's facts to the device.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Final

__all__ = [
    "DEVICE_OWNER_SEPARATOR",
    "MAX_OWNER_ID_LENGTH",
    "SENDER_NAME_FALLBACK_KEY",
    "DeviceBatch",
    "DeviceLine",
    "project_device_batch",
    "speaker_owner_id",
]

DEVICE_OWNER_SEPARATOR: Final = "@"
MAX_OWNER_ID_LENGTH: Final = 128
"""Longest owner id (``sender_id`` / ``user_id``) any route accepts."""
SENDER_NAME_FALLBACK_KEY: Final = "sender_name_fallback"
"""Message key the ingest reads as a label of last resort (``sender_names``)."""


@dataclasses.dataclass(frozen=True)
class DeviceLine:
    """One utterance, attributed (``slot``) or not (``slot is None``)."""

    text: str
    timestamp: int
    slot: str | None = None
    display_name: str | None = None
    consented: bool = True


@dataclasses.dataclass(frozen=True)
class DeviceBatch:
    """A device batch as ``/memory/add`` message rows, and who it writes to."""

    messages: list[dict[str, Any]]
    owners: list[str]


def speaker_owner_id(device_id: str, slot: str) -> str:
    """The memory owner of one person on one device: ``<device_id>@<slot>``.

    Raises:
        ValueError: Either half is empty or contains the separator, or the
            composed id is longer than :data:`MAX_OWNER_ID_LENGTH`.
    """
    for name, part in (("device_id", device_id), ("slot", slot)):
        if not part:
            raise ValueError(f"{name} must be non-empty")
        if DEVICE_OWNER_SEPARATOR in part:
            raise ValueError(
                f"{name} may not contain {DEVICE_OWNER_SEPARATOR!r}; it is "
                "reserved for device-qualified owner ids"
            )
    owner = f"{device_id}{DEVICE_OWNER_SEPARATOR}{slot}"
    if len(owner) > MAX_OWNER_ID_LENGTH:
        raise ValueError(
            f"device-qualified owner id is {len(owner)} characters; the owner "
            f"limit is {MAX_OWNER_ID_LENGTH} (shorten the device id or the slot)"
        )
    return owner


def project_device_batch(device_id: str, lines: Sequence[DeviceLine]) -> DeviceBatch:
    """The batch as message rows (``role`` always ``"user"``) and its owners.

    ``owners`` lists the per-person owners written to, in first-spoken order;
    the device's own owner is not among them.
    """
    declined = {
        line.slot for line in lines if line.slot is not None and not line.consented
    }
    messages: list[dict[str, Any]] = []
    owners: dict[str, None] = {}
    for line in lines:
        row: dict[str, Any] = {
            "role": "user",
            "timestamp": line.timestamp,
            "content": line.text,
        }
        if line.slot is None:
            row |= {"sender_id": device_id, "sender_name": None}
        elif line.slot in declined:
            continue
        else:
            owner = speaker_owner_id(device_id, line.slot)
            owners[owner] = None
            row |= {
                "sender_id": owner,
                "sender_name": line.display_name,
                SENDER_NAME_FALLBACK_KEY: line.slot,
            }
        messages.append(row)
    return DeviceBatch(messages=messages, owners=list(owners))
