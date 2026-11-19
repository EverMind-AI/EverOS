"""Request / response shapes of ``POST /memory/device/add``.

Shape and charset validation only. Who owns each utterance -- the owner id,
the consent rule, the label a narrative uses -- is the domain's
(:mod:`everos.memory.device`), run by ``MemoryService.add_device``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from everos.service import DeviceLine, speaker_owner_id

from .ids import (
    DEVICE_ID_CHARSET,
    OWNER_ID_MAX_LENGTH,
    PATH_SAFE_CHARSET,
    AppScopeId,
    PathSafeId,
    ProjectScopeId,
    SessionId,
)
from .memorize import AcceptedResponseData, AddResponseData

__all__ = [
    "SPEAKER_SLOT_CHARSET",
    "DeviceAcceptedResponseData",
    "DeviceAddRequest",
    "DeviceAddResponseData",
    "DeviceSpeaker",
    "DeviceUtterance",
]

SPEAKER_SLOT_CHARSET = r"^[a-zA-Z][a-zA-Z0-9_-]{0,62}$"
"""A roster slot id: starts with a letter; no ``@`` (the owner separator) and
no ``.``, so ``device@slot`` never reads as an e-mail address another user id
could already be.

The slot is the person's label when no ``display_name`` was ever given, and a
profile picks its episodes by whether the narrative contains the label: a
numeric slot would match every date, a plain word every sentence using it. So
a slot must also hold a digit or ``_`` (``spk_01``, ``guest2``).
"""


class DeviceSpeaker(BaseModel):
    """One person the device can recognize, resolved upstream.

    Attributes:
        slot: The device's stable per-person id (voiceprint slot, roster
            index). Never a name: names change, slots do not.
        display_name: How the person is called in the room; ``None`` (or
            blank) when the device has not been told. Only ever a label -- it
            never selects which memory directory an utterance lands in.
        consented: Whether this person agreed to the device keeping their
            memory. ``False`` on any utterance of a batch means none of that
            slot's utterances in the batch is written.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    slot: str = Field(..., min_length=1, max_length=63, pattern=SPEAKER_SLOT_CHARSET)
    display_name: str | None = Field(default=None, max_length=128)
    consented: bool = True

    @field_validator("slot")
    @classmethod
    def _slot_is_a_distinct_label(cls, value: str) -> str:
        if not any(ch.isdigit() or ch == "_" for ch in value):
            raise ValueError("a slot must contain a digit or '_' (e.g. spk_01)")
        return value

    @field_validator("display_name")
    @classmethod
    def _blank_name_is_none(cls, value: str | None) -> str | None:
        # A blank name must not stand in for a real one downstream.
        return (value.strip() or None) if value is not None else None


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
        ..., min_length=1, max_length=OWNER_ID_MAX_LENGTH, pattern=DEVICE_ID_CHARSET
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

    def lines(self) -> list[DeviceLine]:
        """The batch in the domain's terms, for ``MemoryService.add_device``."""
        return [
            DeviceLine(
                text=u.text,
                timestamp=u.timestamp,
                slot=u.speaker.slot if u.speaker else None,
                display_name=u.speaker.display_name if u.speaker else None,
                consented=u.speaker.consented if u.speaker else True,
            )
            for u in self.utterances
        ]

    @model_validator(mode="after")
    def _device_qualifies_every_speaker(self) -> DeviceAddRequest:
        # A device id may not be long enough to prefix a slot: refuse the
        # batch here (422) rather than halfway through its write.
        for utterance in self.utterances:
            if utterance.speaker is not None:
                speaker_owner_id(self.device_id, utterance.speaker.slot)
        return self


class DeviceAddResponseData(AddResponseData):
    """What one device batch wrote: ``/memory/add``'s body plus ``owners``.

    ``message_count`` counts utterances written, unattributed ones included
    and non-consenting speakers' left out; ``owners`` is empty when every
    utterance was unattributed or unconsented.
    """

    owners: list[str] = Field(default_factory=list)


class DeviceAcceptedResponseData(AcceptedResponseData):
    """``202`` body: ``/memory/add``'s, plus the owners the batch will write to."""

    owners: list[str] = Field(default_factory=list)
