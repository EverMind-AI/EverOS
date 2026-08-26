"""Decision LanceDB table schema validation."""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from everos.infra.persistence.lancedb import Decision


def _kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "u_jason_dc_20260826_0001",
        "entry_id": "dc_20260826_0001",
        "owner_id": "u_jason",
        "owner_type": "user",
        "timestamp": dt.datetime(2026, 8, 26, 12, 0, tzinfo=dt.UTC),
        "parent_id": "mc_1",
        "title": "Use Rust on device",
        "decision": "Device Runtime uses Rust.",
        "reason": "Need deterministic latency on the edge.",
        "tags": ["runtime", "rust"],
        "decision_tokens": "Device Runtime uses Rust",
        "reason_tokens": "Need deterministic latency on the edge",
        "md_path": "users/u_jason/decisions/decision-2026-08-26.md",
        "content_sha256": "a" * 64,
    }
    base.update(overrides)
    return base


class TestDecisionSchema:
    def test_table_name(self) -> None:
        assert Decision.TABLE_NAME == "decision"

    def test_bm25_fields_dual_column(self) -> None:
        assert Decision.BM25_FIELDS == ["decision_tokens", "reason_tokens"]

    def test_has_required_fields(self) -> None:
        fields = set(Decision.model_fields.keys())
        required = {
            "id",
            "entry_id",
            "owner_id",
            "owner_type",
            "app_id",
            "project_id",
            "session_id",
            "timestamp",
            "parent_type",
            "parent_id",
            "title",
            "decision",
            "reason",
            "impact",
            "tags",
            "decision_tokens",
            "reason_tokens",
            "md_path",
            "content_sha256",
            "vector",
            "deprecated_by",
        }
        assert required.issubset(fields), f"Missing: {required - fields}"
        assert "sender_ids" not in fields

    def test_vector_nullable_at_pyarrow_layer(self) -> None:
        arrow = Decision.to_arrow_schema()
        assert arrow.field("vector").nullable is True

    def test_constructs_minimal_row(self) -> None:
        row = Decision(**_kwargs())  # type: ignore[arg-type]
        assert row.impact is None
        assert row.deprecated_by is None
        assert row.vector is None
        assert row.parent_type == "memcell"
        assert row.tags == ["runtime", "rust"]

    def test_missing_decision_raises(self) -> None:
        bad = _kwargs()
        del bad["decision"]
        with pytest.raises(ValidationError):
            Decision(**bad)  # type: ignore[arg-type]

    def test_missing_content_sha256_raises(self) -> None:
        bad = _kwargs()
        del bad["content_sha256"]
        with pytest.raises(ValidationError):
            Decision(**bad)  # type: ignore[arg-type]
