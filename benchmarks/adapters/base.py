"""Dataset adapters: the one place a benchmark's own shape is allowed to live.

`run.py` drives ADD -> SEARCH -> ANSWER -> JUDGE and knows nothing about any particular
benchmark. Everything that differs between benchmarks answers four questions, and an
adapter is exactly those four answers:

  1. load_units()  -- how to read the conversations and questions off disk
  2. owner_of()    -- what memory owner a question's answer lives under. Owner naming is
                      decided when the store is BUILT, so this must reproduce the
                      builder's convention rather than invent one. Getting it wrong
                      returns zero episodes and scores 0% with no error anywhere.
  3. gold_of()     -- gold session ids, in the form THE STORE uses. Every benchmark
  cites
                      evidence differently (haystack positions, D<session>:<turn> dia
                      ids, original session names) and none of them matches the store
                      directly.
  4. judge_spec()  -- which judge, and the answer prompt it grades against. The hybrid
                      judge's leniency clauses are keyed to LongMemEval's categories and
                      would mis-fire on any other benchmark, so the judge belongs to the
                      adapter, not to a shared scoring layer.

Anything that is NOT one of those four belongs in run.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DatasetAdapter(Protocol):
    """Four questions, one benchmark."""

    name: str

    def load_units(self, data_path: str) -> list[dict[str, Any]]:
        """Return one entry per conversation/topic, each carrying its own questions.

        The shape run.py consumes is ``{"index": int, "sessions": [...], "qa": [...]}``;
        an adapter is free to derive that however its source data is organised.
        """
        ...

    def owner_of(self, unit: dict[str, Any], eval_owner: str) -> str:
        """Memory owner id to query for this unit.

        ``eval_owner`` carries the config's preference where a benchmark has more than
        one candidate (LoCoMo has two speakers); benchmarks with a single owner per unit
        ignore it.
        """
        ...

    def gold_of(self, unit: dict[str, Any], qa: dict[str, Any]) -> set[str]:
        """Gold session ids for one question, already translated into store ids."""
        ...

    def sessions_of(self, unit: dict[str, Any]) -> list[dict[str, Any]]:
        """Ingestible sessions for the ADD stage.

        Each session is ``{"session_idx": int, "messages": [...], "timestamp_ms": int}``
        and each message carries ``speaker`` / ``text`` / ``dia_id``. Only the ADD stage
        needs this; a benchmark scored against a pre-built store never calls it.

        Splitting this out is what lets ADD run for anything other than LoCoMo: the
        loader used to read ``unit["conversation"]`` directly, so every other dataset
        died with ``KeyError: 'conversation'`` the moment ingestion started.
        """
        raise NotImplementedError

    def judge_spec(self) -> dict[str, Any]:
        """``{"judge": <name>, "answer_prompt": <template>}``.

        ``judge`` selects the grading function; ``answer_prompt`` is the template the
        answer model is given, because a judge's leniency assumes a particular answer
        protocol (e.g. the enumeration / temporal protocol LongMemEval's clauses
        expect).
        """
        ...

    def categories(self) -> dict[str, str]:
        """Category id -> human label, for per-category reporting.

        LoCoMo's mapping is counter-intuitive (cat1 = multi-hop, cat4 = single-hop) and
        this is the single place that fact is recorded.
        """
        ...


def normalize_speaker(name: str) -> str:
    """Make a speaker usable as a `sender_id`.

    The API validates sender_id against ``^[a-zA-Z0-9_.@+-]+$``, so any real name with a
    space ("Bo Chen") is rejected with a 422 -- after ADD has already paid for
    extraction on that batch. Collapsing the disallowed characters keeps names
    distinguishable without inventing an id mapping the gold would not recognise.
    """
    import re

    cleaned = re.sub(r"[^a-zA-Z0-9_.@+-]+", "_", str(name or "").strip())
    return cleaned.strip("_") or "speaker"


def leniency_clause(qa: dict[str, Any]) -> str:
    """Extra grading rule for this question, prepended to the judge prompt.

    There is one judge and one answer prompt in this harness -- no modes, no flags. What
    varies is only this clause, and each benchmark decides it from its OWN question
    metadata. That last part is the whole point: the original hybrid judge indexed
    LongMemEval's question types by conversation number, so pointing it at another
    dataset applied LongMemEval's clauses to unrelated questions. Deriving the clause
    from the question at hand cannot mis-fire, and a benchmark that defines none simply
    gets the plain judge.
    """
    return ""


# Every benchmark stamps its sessions with a real clock, in its own format. Ingesting a
# synthetic clock instead puts wrong timestamps on every episode, and those timestamps
# are rendered into the answer prompt -- which is what temporal questions are graded on.
_SESSION_TS_FORMATS = (
    "%I:%M %p on %d %B, %Y",  # LoCoMo / EverMemBench: "10:02 am on 4 March, 2025"
    "%Y/%m/%d (%a) %H:%M",  # LongMemEval haystack_dates: "2023/05/20 (Sat) 02:21"
)


def session_epoch_ms(raw: str) -> int | None:
    """Parse a session timestamp to epoch milliseconds, or None if unparseable.

    The wall clock is pinned to UTC whether or not the value carries an offset. Naive
    values need it for the reason everalgo's LoCoMo loader gives -- without an explicit
    zone the same dataset yields different epochs on machines in different zones.

    Values that DO carry an offset need it because of how the reference reaches them.
    Every reference harness reads a locomo-style conversion of its dataset, and that
    conversion renders the wall clock and drops the zone: SubtleMemory's raw
    ``2025-04-01T10:01:36+08:00`` arrives at the reference as
    ``"10:01 am on 1 April, 2025"``, which it then pins to UTC. Honouring the offset
    here instead moves every one of that dataset's 2364 sessions 8 hours earlier, and
    carries 397 of them into the previous UTC day. Session timestamps are rendered into
    the answer prompt and are what temporal questions are graded on, so the two must
    agree on the wall clock, not on the instant it denotes.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    try:  # ISO-8601, possibly with an offset (SubtleMemory)
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = None
        for fmt in _SESSION_TS_FORMATS:
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            return None
    # replace(), not astimezone(): the wall clock is the value, the offset is discarded.
    return int(dt.replace(tzinfo=UTC).timestamp() * 1000)
