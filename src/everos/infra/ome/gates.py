"""OME gate types — declarative configuration only.

Counter is the only built-in gate. The actual N-counting lives in
_stores/counter.py keyed by (strategy_name, bucket_key).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class Counter(BaseModel):
    """Counter gate: batch trigger by accumulated event count per bucket.
    Each event increments the bucket counter; the `threshold`-th event
    passes and resets.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: Annotated[
        int,
        Field(
            gt=0,
            description=(
                "Pass once every `threshold` events; threshold=1 lets every event pass."
            ),
        ),
    ]
    cooldown_seconds: Annotated[
        int,
        Field(
            ge=0,
            description=(
                "Minimum seconds between consecutive passes per bucket; 0 disables."
            ),
        ),
    ] = 0
    event_field: Annotated[
        str | None,
        Field(
            description=(
                'Bucket dimension on the event (e.g. "user_id"); '
                "None means a single global bucket."
            ),
        ),
    ] = None


# Single-member alias today; becomes a union as more gate types land.
Gate = Counter
