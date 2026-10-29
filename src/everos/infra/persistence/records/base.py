"""The shared base of every derived-index record declaration.

:class:`IndexRecord` adds the ``created_at`` / ``updated_at`` columns to the
:class:`IndexRecordMeta` declarations (``TABLE_NAME`` / ``BM25_FIELDS`` /
``PRIMARY_KEY``). It is a plain pydantic model: it declares columns and marks
the embedding ones (:data:`~.vector.DenseVector`), and knows no index engine.
Each backend derives its physical schema from
:func:`everos.infra.persistence.index.schema_for`, which is also where the
vector width comes in.

Note:
    Neither backend has a SQL ``onupdate`` equivalent — the application must
    explicitly set ``updated_at = get_utc_now()`` before a write.

    Datetime columns are declared as plain ``datetime``; storage in UTC is
    each backend's mapping of the ``datetime`` kind (LanceDB:
    ``timestamp[us, tz=UTC]``), not something a record opts into.
"""

from __future__ import annotations

__all__ = ["IndexRecord"]

import datetime as dt

from pydantic import BaseModel, Field

from everos.core.utils.datetime import get_utc_now
from everos.infra.persistence.records.index_record import IndexRecordMeta
from everos.infra.persistence.records.vector import is_vector_field


class IndexRecord(IndexRecordMeta, BaseModel):
    """Timestamp columns shared by every derived-index record.

    The declarations every backend reads (``TABLE_NAME`` / ``BM25_FIELDS`` /
    ``PRIMARY_KEY``) live on :class:`IndexRecordMeta`; this class adds the
    ``created_at`` / ``updated_at`` columns. Repos read the ClassVars; they do
    not duplicate them.
    """

    created_at: dt.datetime = Field(default_factory=get_utc_now)
    updated_at: dt.datetime = Field(default_factory=get_utc_now)

    @classmethod
    def vector_columns(cls) -> list[str]:
        """Names of the fields declared as :data:`~.vector.DenseVector`."""
        return [
            name for name, field in cls.model_fields.items() if is_vector_field(field)
        ]
