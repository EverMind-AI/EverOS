from __future__ import annotations

import pytest
from pydantic import ValidationError

from everos.infra.ome.gates import Counter


def test_counter_accepts_threshold() -> None:
    c = Counter(threshold=5)
    assert c.threshold == 5
    assert c.cooldown_seconds == 0
    assert c.event_field is None


def test_counter_with_bucket_field() -> None:
    c = Counter(threshold=5, event_field="user_id", cooldown_seconds=120)
    assert c.event_field == "user_id"
    assert c.cooldown_seconds == 120


def test_counter_rejects_zero_threshold() -> None:
    with pytest.raises(ValidationError):
        Counter(threshold=0)


def test_counter_rejects_negative_cooldown() -> None:
    with pytest.raises(ValidationError):
        Counter(threshold=5, cooldown_seconds=-1)
