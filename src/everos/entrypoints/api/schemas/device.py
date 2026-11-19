"""Identity and request models for a hardware device that serves several people.

A device runs ONE EverOS agent instance and is spoken to by several people in
the same room. Two identities therefore travel with every utterance:

* the **device** — the ``device_id``, plus the ``session_id`` the utterances
  accumulate under;
* the **speaker** — the person talking, resolved upstream (voiceprint slot,
  roster index, wake-word profile) to a stable slot id.

The per-person memory owner is derived, not supplied. A user-track episode is
written once per distinct ``role="user"`` ``sender_id`` of a memcell
(``everos.memory.extract.pipeline.user_memory._unique_user_senders``), each
copy under that sender's owner directory, and the profile is keyed the same
way. Giving every speaker of a device a *device-qualified* owner id therefore
keeps each person's episodes and profile in their own directory, while
utterances nobody was identified for stay with the device.

``POST /memory/device/add`` (``routes.memorize``) projects a
:class:`DeviceAddRequest` onto the ``/memory/add`` request; nothing below the
route knows about devices.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .ids import PATH_SAFE_CHARSET, AppScopeId, PathSafeId, ProjectScopeId, SessionId
from .memorize import AcceptedResponseData

__all__ = [
    "DEVICE_ID_CHARSET",
    "DEVICE_OWNER_SEPARATOR",
    "MAX_OWNER_ID_LENGTH",
    "SPEAKER_SLOT_CHARSET",
    "DeviceAcceptedResponseData",
    "DeviceAddRequest",
    "DeviceAddResponseData",
    "DeviceSpeaker",
    "DeviceUtterance",
    "speaker_owner_id",
]

DEVICE_OWNER_SEPARATOR = "@"
"""Separates the device id from the speaker slot in a per-person owner id.

Inside :data:`~.ids.PATH_SAFE_CHARSET`, so a composed owner is an ordinary
``user_id`` / ``sender_id`` on every existing route and a valid directory
name on every platform (``:`` is not one on Windows). Neither a device id nor
a slot may contain it, so one owner id names exactly one (device, speaker)
pair. A client can still send such an id to ``/memory/add`` itself: like every
other owner id, it is not a credential.
"""

MAX_OWNER_ID_LENGTH = 128
"""The owner-id length the existing routes accept (``sender_id`` max_length)."""

DEVICE_ID_CHARSET = r"^[a-zA-Z0-9_.+-]+$"
""":data:`~.ids.PATH_SAFE_CHARSET` without the separator."""

SPEAKER_SLOT_CHARSET = r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,62}$"
"""A roster slot id: filesystem-safe, no separator, no leading ``.``/``-``."""


class DeviceSpeaker(BaseModel):
    """One person the device can recognize, resolved upstream.

    Attributes:
        slot: The device's stable per-person id (voiceprint slot, roster
            index). Never a name: names change, slots do not.
        display_name: How the person is called in the room; ``None`` when the
            device has not been told, and the slot is the label then. Only
            ever a label — it never selects which memory directory an
            utterance lands in.
        consented: Whether this person agreed to the device keeping their
            memory. ``False`` means their utterances are not written at all.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot: str = Field(..., min_length=1, max_length=63, pattern=SPEAKER_SLOT_CHARSET)
    display_name: str | None = Field(default=None, max_length=128)
    consented: bool = True


class DeviceUtterance(BaseModel):
    """One thing said in front of the device, attributed to one speaker.

    Attributes:
        speaker: The resolved speaker; ``None`` when the device could not
            identify who is talking. Such an utterance is filed under the
            device, never guessed onto a person.
        text: What was said, as transcribed.
        timestamp: Event time as Unix epoch milliseconds, like
            :class:`~.memorize.MessageItemDTO.timestamp`.
    """

    model_config = ConfigDict(extra="forbid")

    speaker: DeviceSpeaker | None = None
    text: str = Field(..., min_length=1)
    timestamp: int = Field(..., gt=0)


class DeviceAddRequest(BaseModel):
    """A batch of utterances from one device.

    Attributes:
        device_id: The device's id: the owner of utterances nobody was
            identified for, and the prefix of every speaker owner id.
        session_id: The conversation the utterances belong to, as
            ``/memory/add`` takes it.
        app_id / project_id: Scope segments, as ``/memory/add`` takes them.
        utterances: The batch, in spoken order.
        async_mode: As ``/memory/add`` takes it: queue the batch and answer
            ``202`` at once.
    """

    model_config = ConfigDict(extra="forbid")

    device_id: PathSafeId = Field(
        ..., min_length=1, max_length=MAX_OWNER_ID_LENGTH, pattern=DEVICE_ID_CHARSET
    )
    session_id: SessionId = Field(..., min_length=1, max_length=128)
    app_id: AppScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    project_id: ProjectScopeId = Field(
        default="default", min_length=1, max_length=128, pattern=PATH_SAFE_CHARSET
    )
    utterances: list[DeviceUtterance] = Field(..., min_length=1, max_length=500)
    async_mode: bool = Field(
        default=False,
        description=(
            "As /memory/add: persist the batch to the memorize queue and return "
            "202 at once; poll GET /tasks/{request_id}."
        ),
    )

    def speaker_owners(self) -> list[str]:
        """The per-person owners this batch writes to, in first-spoken order.

        Unknown and non-consenting speakers are left out: they produce no
        owner of their own.
        """
        owners: list[str] = []
        for utterance in self.utterances:
            speaker = utterance.speaker
            if speaker is None or not speaker.consented:
                continue
            owner = speaker_owner_id(self.device_id, speaker.slot)
            if owner not in owners:
                owners.append(owner)
        return owners

    def projected_messages(self) -> list[dict[str, object]]:
        """The batch as ``/memory/add`` message rows.

        ``sender_id`` is the device-qualified owner (the device id alone for
        an unidentified speaker), ``sender_name`` the speaker's label (the
        slot when the device sent no ``display_name``),
        ``role`` always ``"user"`` — the device is a human-facing surface, so
        nothing it hears is an assistant or tool turn. A non-consenting
        speaker's utterances are dropped, not filed under the device.
        """
        rows: list[dict[str, object]] = []
        for utterance in self.utterances:
            speaker = utterance.speaker
            if speaker is None:
                sender, name = self.device_id, None
            elif not speaker.consented:
                continue
            else:
                sender = speaker_owner_id(self.device_id, speaker.slot)
                # Never let the owner id be the label: a model writing the
                # narrative shortens "dev@spk_b" to "dev" -- the device's own
                # id -- and the person's profile then misses the episode.
                name = speaker.display_name or speaker.slot
            rows.append(
                {
                    "sender_id": sender,
                    "sender_name": name,
                    "role": "user",
                    "timestamp": utterance.timestamp,
                    "content": utterance.text,
                }
            )
        return rows

    @model_validator(mode="after")
    def _device_qualifies_every_speaker(self) -> DeviceAddRequest:
        # A device id may not be long enough to prefix a slot: refuse the
        # batch here rather than halfway through its write.
        for utterance in self.utterances:
            if utterance.speaker is not None:
                speaker_owner_id(self.device_id, utterance.speaker.slot)
        return self


class DeviceAddResponseData(BaseModel):
    """What one device batch wrote.

    Attributes:
        message_count: Utterances written, unattributed ones included and
            non-consenting speakers' left out.
        status: As ``/memory/add`` returns it.
        owners: The per-person owners the batch wrote to; empty when every
            utterance was unattributed or unconsented.
    """

    message_count: int
    status: Literal["accumulated", "extracted"]
    owners: list[str] = Field(default_factory=list)


class DeviceAcceptedResponseData(AcceptedResponseData):
    """``202`` body: ``/memory/add``'s, plus the owners the batch will write to."""

    owners: list[str] = Field(default_factory=list)


def speaker_owner_id(device_id: str, slot: str) -> str:
    """The memory owner of one person on one device.

    The same person talking to two devices keeps two memories: a device's
    roster slot is its own identity and is resolved per device.

    Args:
        device_id: The device's id.
        slot: The device's stable per-person slot.

    Returns:
        ``<device_id>@<slot>``, a path-safe owner id of at most 128
        characters.

    Raises:
        ValueError: Either half is empty or contains the reserved separator,
            or the composed id is longer than
            :data:`MAX_OWNER_ID_LENGTH`.
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
