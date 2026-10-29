"""``IndexRecord`` declares columns and keys, and marks vectors without a width."""

from __future__ import annotations

from typing import ClassVar

from everos.infra.persistence.records import DenseVector, IndexRecord, IndexRecordMeta


def test_index_record_mixes_in_index_record_meta() -> None:
    assert IndexRecordMeta in IndexRecord.__mro__
    assert IndexRecord.TABLE_NAME == ""
    assert IndexRecord.BM25_FIELDS == []


def test_primary_key_has_no_default() -> None:
    # A declared-but-unset ClassVar: present in annotations, absent as a value.
    assert "PRIMARY_KEY" in IndexRecordMeta.__annotations__
    assert not hasattr(IndexRecordMeta, "PRIMARY_KEY")
    assert not hasattr(IndexRecord, "PRIMARY_KEY")


def test_subclass_declaration_is_read_back() -> None:
    class _T(IndexRecord):
        TABLE_NAME: ClassVar[str] = "_t"
        PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("id",)
        id: str

    assert _T.PRIMARY_KEY == ("id",)
    assert _T.TABLE_NAME == "_t"


def test_vector_columns_are_the_dense_vector_fields() -> None:
    """Both spellings pydantic stores differently are recognised."""

    class _T(IndexRecord):
        PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("id",)
        id: str
        embeddings: list[float] = []
        required: DenseVector
        optional: DenseVector | None = None

    assert _T.vector_columns() == ["required", "optional"]


def test_a_record_carries_no_width() -> None:
    """Any length validates: the width belongs to the backend, not the record."""

    class _T(IndexRecord):
        PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("id",)
        id: str
        vector: DenseVector | None = None

    assert len(_T(id="a", vector=[0.0] * 3).vector or []) == 3
    assert len(_T(id="b", vector=[0.0] * 3072).vector or []) == 3072
