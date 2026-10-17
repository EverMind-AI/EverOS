"""IndexRecordMeta is the backend-neutral home of the three declarations."""

from __future__ import annotations

from typing import ClassVar

from everos.core.persistence import BaseLanceTable, IndexRecordMeta


def test_base_lance_table_mixes_in_index_record_meta() -> None:
    assert IndexRecordMeta in BaseLanceTable.__mro__
    assert BaseLanceTable.TABLE_NAME == ""
    assert BaseLanceTable.BM25_FIELDS == []


def test_primary_key_has_no_default() -> None:
    # A declared-but-unset ClassVar: present in annotations, absent as a value.
    assert "PRIMARY_KEY" in IndexRecordMeta.__annotations__
    assert not hasattr(IndexRecordMeta, "PRIMARY_KEY")
    assert not hasattr(BaseLanceTable, "PRIMARY_KEY")


def test_subclass_declaration_is_read_back() -> None:
    class _T(BaseLanceTable):
        TABLE_NAME: ClassVar[str] = "_t"
        PRIMARY_KEY: ClassVar[tuple[str, ...]] = ("id",)
        id: str

    assert _T.PRIMARY_KEY == ("id",)
    assert _T.TABLE_NAME == "_t"
