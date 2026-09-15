"""Pin exact datetime, JSON, vector, and limit behavior in SeekDB row codecs."""

from __future__ import annotations

import datetime as dt
import struct

import pytest

from everos.infra.persistence.index import Episode
from everos.infra.persistence.index.schema import schema_for
from everos.infra.persistence.seekdb.codec import from_row, to_row, write_field_value
from everos.infra.persistence.seekdb.errors import SeekdbValueLimitError


def _episode(**overrides: object) -> Episode:
    values: dict[str, object] = {
        "id": "u1_ep1",
        "entry_id": "ep1",
        "owner_id": "u1",
        "owner_type": "user",
        "session_id": "session",
        "timestamp": dt.datetime(1999, 1, 1, tzinfo=dt.UTC),
        "parent_id": "mc1",
        "sender_ids": ["u1"],
        "episode": "red apple memory",
        "episode_tokens": "red apple memory",
        "md_path": "users/u1/episodes/day.md",
        "content_sha256": "a" * 64,
        "vector": [1.0] + [0.0] * 1023,
        "subject_vector": None,
    }
    values.update(overrides)
    return Episode(**values)  # type: ignore[arg-type]


def test_record_round_trip_preserves_epoch_ms_json_and_null_vector() -> None:
    schema = schema_for(Episode)
    stored = to_row(_episode(), schema)
    assert stored["timestamp_ms"] == 915148800000
    assert stored["sender_ids"] == '["u1"]'
    assert stored["subject_vector"] is None
    restored = from_row(stored, schema)
    assert restored["timestamp"] == dt.datetime(1999, 1, 1, tzinfo=dt.UTC)
    assert restored["sender_ids"] == ["u1"]
    assert restored["vector"][:2] == [1.0, 0.0]


def test_binary_float32_vectors_are_accepted() -> None:
    schema = schema_for(Episode)
    raw = {"vector": struct.pack("<1024f", *([0.5] * 1024))}
    restored = from_row(raw, schema)
    assert restored["vector"][0] == pytest.approx(0.5)


def test_binary_vector_starting_with_json_marker_is_not_misclassified() -> None:
    schema = schema_for(Episode)
    raw = {"vector": bytes([0x5B, 0, 0, 0]) * 1024}
    restored = from_row(raw, schema)
    assert len(restored["vector"]) == 1024


@pytest.mark.parametrize(
    ("moment", "epoch_ms"),
    [
        (dt.datetime(1970, 1, 1, tzinfo=dt.UTC), 0),
        (dt.datetime(2001, 9, 9, 1, 46, 40, 123000, tzinfo=dt.UTC), 1000000000123),
    ],
)
def test_datetime_boundaries_round_trip_exactly(
    moment: dt.datetime, epoch_ms: int
) -> None:
    schema = schema_for(Episode)
    stored = to_row(_episode(timestamp=moment), schema)
    assert stored["timestamp_ms"] == epoch_ms
    assert from_row(stored, schema)["timestamp"] == moment


def test_short_string_array_and_vector_limits_are_loud() -> None:
    schema = schema_for(Episode)
    with pytest.raises(SeekdbValueLimitError, match="id is 513 characters"):
        to_row(_episode(id="x" * 513), schema)
    with pytest.raises(SeekdbValueLimitError, match="sender_ids has 257 items"):
        to_row(_episode(sender_ids=["u"] * 257), schema)
    with pytest.raises(SeekdbValueLimitError, match="dimension 2"):
        write_field_value(schema.field("vector"), [1.0, 0.0], schema)
    with pytest.raises(SeekdbValueLimitError, match="non-finite"):
        write_field_value(
            schema.field("vector"),
            [float("nan")] + [0.0] * 1023,
            schema,
        )
