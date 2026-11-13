"""The forgetting curve: how a memory's age and version state weigh its score.

The 2.0 plan's retrieval section ("forgetting curve", "recalling old
memories"); docs/configuration.md ``[search]``. Nothing here runs unless
``[search] forgetting_curve`` is on or a request asks about the past
(``include_past``): without either, the multi-round loop gets no
:class:`CurveRanking` and ranks exactly as before.

* **Retention.** Every memory has a strength ``S`` in days and a clock. After
  ``t`` days its retention is ``R = 1 / (1 + t / (9 S))``: 0.9 when ``t = S``,
  falling ever more slowly and never to zero -- a power law, the family FSRS
  uses. With ``S = 30`` a two-year-old memory keeps 0.27 where an exponential
  would keep 0.08, so old memories stay reachable.
* **Score.** A candidate's fused (RRF) score is multiplied by
  ``(1 - w + w R) * state``: ``w`` is ``[search] retention_weight`` (0.3: age
  takes at most 30%), ``state`` is 1 for a current version and ``[search]
  past_state_factor`` (0.3) for a superseded version or a forgotten memory a
  search for the past brought back. The candidates are sorted again; none is
  ever dropped, so the curve only reorders.
* **Reinforcement.** When the decider keeps a memory as core, or ``get``
  returns it, ``S <- min(cap, S (1 + g (1 - R)))`` and its clock restarts
  (:func:`strengthen`): one recalled after a long gap gains much, one used a moment
  ago almost nothing. Reads only queue it (:class:`ReinforcementSink`).
* **One strength per memory**, shared by its versions (:func:`strength_key`).
  A memory never reinforced has ``[search] initial_strength_days`` and its
  clock starts at its own time -- an episode's or a picture's (when it
  happened), a scene version's (its newest member's: every scene update
  restarts the scene's clock). Scenes, episodes and pictures follow the same
  curve with the same start, so they compete on equal terms.

Pure scoring: no model call. :class:`CurveRanking` reads the store once per
round, for the memories the search has not met yet, and never writes; the
loop (:mod:`.llm_multiround`) does the rescoring of its candidates.
"""

from __future__ import annotations

__all__ = [
    "Curve",
    "CurveRanking",
    "reinforced",
    "reinforcements",
    "retention",
    "state_of",
    "strength_key",
    "strengthen",
]

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from everos.memory.models import FORGOTTEN_MARK, Episode, ImageMemory
from everos.memory.ports import (
    Reinforcement,
    Stored,
    Strength,
    StrengthKey,
    StrengthStore,
)

if TYPE_CHECKING:
    from everos.config import GroupSearchSettings

DAY_MS: Final = 86_400_000
"""Milliseconds in a day: ``S`` and the clock's reading are in days."""

type Record = Stored[Episode] | Stored[ImageMemory]
type VersionState = Literal["current", "superseded", "forgotten"]


@dataclass(frozen=True)
class Curve:
    """The curve's parameters (``[search]``).

    Attributes:
        initial_days: ``S`` of a memory never reinforced.
        max_days: The most ``S`` can grow to.
        gain: ``g`` in ``S <- S (1 + g (1 - R))``.
        weight: ``w``, the share of the score retention can take away.
        past_factor: The state factor of a superseded or forgotten version.
    """

    initial_days: float = 30.0
    max_days: float = 3650.0
    gain: float = 2.0
    weight: float = 0.3
    past_factor: float = 0.3

    @classmethod
    def of(cls, settings: GroupSearchSettings) -> Curve:
        """The curve ``[search]`` configures."""
        return cls(
            initial_days=settings.initial_strength_days,
            max_days=settings.max_strength_days,
            gain=settings.reinforce_gain,
            weight=settings.retention_weight,
            past_factor=settings.past_state_factor,
        )


def retention(days: float, strength_days: float) -> float:
    """``R = 1 / (1 + t / (9 S))``; a clock that reads less than zero reads zero."""
    return 1.0 / (1.0 + max(days, 0.0) / (9.0 * strength_days))


def reinforced(strength_days: float, days: float, *, gain: float, cap: float) -> float:
    """``S`` after one use ``days`` after the clock last restarted, at most ``cap``."""
    kept = retention(days, strength_days)
    return min(cap, strength_days * (1.0 + gain * (1.0 - kept)))


def strengthen(curve: Curve, row: Strength | None, event: Reinforcement) -> Strength:
    """A memory's strength after ``event``, from its stored ``row`` (or none).

    The clock reads from the later of the last reinforcement and the memory's
    own time (a scene version newer than the last reinforcement restarted
    it), and restarts at the use; it never runs backwards.
    """
    if row is None:
        days, start, count, last = curve.initial_days, event.since, 0, event.at
    else:
        days = row.days
        start = max(row.reinforced_at, event.since)
        count, last = row.reinforcements, row.reinforced_at
    grown = reinforced(
        days, (event.at - start) / DAY_MS, gain=curve.gain, cap=curve.max_days
    )
    return Strength(
        days=grown, reinforced_at=max(event.at, last), reinforcements=count + 1
    )


def strength_key(record: Record) -> StrengthKey:
    """The memory a record is a version of, in its owner's space.

    A version keeps what its memory rests on, so the key is read from that:
    an episode by its segment (``parent_type`` and ``parent_id``; one episode
    per segment and owner, every version of it on the same segment), a scene
    version by its scene (``parent_id``), a picture memory by its picture and
    segment. A record without a parent is its own memory.
    """
    item = record.item
    if isinstance(item, ImageMemory):
        memory = f"im:{item.image_id}:{item.parent_id}" if item.image_id else ""
    elif item.parent_type == "scene":
        memory = f"sc:{item.parent_id}" if item.parent_id else ""
    else:
        memory = f"ep:{item.parent_type}:{item.parent_id}" if item.parent_id else ""
    scope = record.scope
    return StrengthKey(
        app_id=scope.app_id,
        project_id=scope.project_id,
        owner_type=scope.owner_type,
        owner_id=scope.owner_id,
        memory=memory or f"id:{record.ref.id}",
    )


def state_of(record: Record) -> VersionState:
    """``current``, ``superseded`` (a later version replaced it) or ``forgotten``."""
    if record.mark is None:
        return "current"
    return "forgotten" if record.mark == FORGOTTEN_MARK else "superseded"


def reinforcements(records: Iterable[Record], *, at: int) -> list[Reinforcement]:
    """One reinforcement per memory among ``records``, used at ``at`` (epoch ms).

    Two versions of one memory read together reinforce it once, from the
    newer version's time.
    """
    found: dict[StrengthKey, Reinforcement] = {}
    for record in records:
        key = strength_key(record)
        since = record.item.timestamp
        prior = found.get(key)
        if prior is None or since > prior.since:
            found[key] = Reinforcement(key=key, since=since, at=at)
    return list(found.values())


class CurveRanking:
    """One search's weighing by the curve: its reference time and the strengths met.

    Built per request by the search manager and handed to the multi-round
    loop, which reweighs every fused block by :meth:`factor` (after
    :meth:`load`) before it cuts it.

    Args:
        curve: The parameters.
        now: The reference time, epoch ms: when the question is asked.
        retain: Apply retention (``[search] forgetting_curve``). Without it only
            the state factor applies -- a search for the past on a deployment
            with the curve off.
        store: Where strengths are read; ``None`` reads every memory as never
            reinforced.
    """

    def __init__(
        self,
        curve: Curve,
        *,
        now: int,
        retain: bool,
        store: StrengthStore | None,
    ) -> None:
        self._curve = curve
        self._now = now
        self._retain = retain
        self._store = store
        self._known: dict[StrengthKey, Strength | None] = {}

    async def load(self, records: Iterable[Record]) -> None:
        """Read the strengths of the memories not met yet, in one store read.

        The loop calls it with everything a round recalled, before it asks
        :meth:`factor` of any block; without retention there is nothing to read.
        """
        if not self._retain:
            return
        wanted = {strength_key(r) for r in records} - self._known.keys()
        if not wanted:
            return
        found = await self._store.read(wanted) if self._store is not None else {}
        for key in wanted:
            self._known[key] = found.get(key)

    def factor(self, record: Record | None) -> float:
        """What a record's fused score is multiplied by."""
        if record is None:
            return 1.0
        factor = 1.0 if state_of(record) == "current" else self._curve.past_factor
        if self._retain:
            weight = self._curve.weight
            factor *= 1.0 - weight + weight * self.retention_of(record)
        return factor

    def retention_of(self, record: Record) -> float:
        """``R`` of the record's memory at the reference time."""
        own = record.item.timestamp
        row = self._known.get(strength_key(record))
        if row is None:
            days, start = self._curve.initial_days, own
        else:
            days, start = row.days, max(row.reinforced_at, own)
        return retention((self._now - start) / DAY_MS, days)

    def describe(self, record: Record | None) -> dict[str, object]:
        """What the round trace records of one candidate's weighting."""
        if record is None:
            return {"factor": 1.0}
        out: dict[str, object] = {
            "state": state_of(record),
            "factor": self.factor(record),
        }
        if self._retain:
            out["retention"] = self.retention_of(record)
        return out
