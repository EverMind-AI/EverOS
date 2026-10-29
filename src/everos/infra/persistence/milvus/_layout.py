"""The physical Milvus collection EverOS derives from one index schema.

:class:`CollectionLayout` declares every column (:class:`PhysicalField`), the
BM25 functions and the derived indexes of a table, and checks a server's
``describe_collection`` / ``describe_index`` replies against that same
declaration — so a reported mismatch always means the server disagrees with
us, never that the builder and the checker drifted apart. Pure: the repository
performs the describe calls and hands the replies over.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pymilvus import DataType, Function, FunctionType, MilvusClient

from everos.core.observability.logging import get_logger
from everos.infra.persistence.index.schema import (
    IndexField,
    IndexFieldKind,
    IndexSchema,
)

from ._columns import (
    DUMMY_VECTOR_DIMENSION,
    DUMMY_VECTOR_FIELD,
    PK_FIELD,
    PK_MAX_LENGTH,
    SPARSE_SUFFIX,
    datetime_storage_field,
    present_field,
    sparse_field,
)
from .milvus_manager import MilvusSchemaMismatchError

logger = get_logger(__name__)


@dataclass(frozen=True)
class PhysicalField:
    """One Milvus column exactly as EverOS declares it.

    The same descriptor drives collection creation and startup verification,
    so a reported mismatch always means the server disagrees with us — never
    that the builder and the checker have drifted apart from each other.
    """

    name: str
    datatype: DataType
    is_primary: bool = False
    nullable: bool = False
    dim: int | None = None
    element_type: DataType | None = None
    max_length: int | None = None
    max_capacity: int | None = None
    enable_analyzer: bool = False

    def create_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"field_name": self.name, "datatype": self.datatype}
        if self.is_primary:
            kwargs["is_primary"] = True
        if self.element_type is not None:
            kwargs["element_type"] = self.element_type
        if self.max_length is not None:
            kwargs["max_length"] = self.max_length
        if self.max_capacity is not None:
            kwargs["max_capacity"] = self.max_capacity
        if self.dim is not None:
            kwargs["dim"] = self.dim
        if self.enable_analyzer:
            kwargs["enable_analyzer"] = True
        if self.nullable:
            kwargs["nullable"] = True
        return kwargs

    def mismatches(self, actual: dict[str, Any]) -> list[str]:
        """Report how a server-reported field differs from this declaration.

        ``describe_collection`` omits ``nullable`` and ``is_primary`` when they
        are false and omits ``element_type`` for non-array columns, so absence
        is unambiguous and can be compared strictly. ``max_length`` and
        ``max_capacity`` are reported only for the types that carry them and
        the server may normalize them, so they are advisory: a wrong length
        surfaces as a loud write rejection anyway, whereas a wrong datatype or
        dimension is the kind that fails opaquely much later.
        """
        params = actual.get("params") or {}
        out: list[str] = []
        if actual.get("type") != self.datatype:
            out.append(
                f"{self.name}: datatype {_name_of(actual.get('type'))} "
                f"!= expected {_name_of(self.datatype)}"
            )
        if bool(actual.get("is_primary", False)) != self.is_primary:
            out.append(
                f"{self.name}: is_primary {actual.get('is_primary', False)} "
                f"!= expected {self.is_primary}"
            )
        if bool(actual.get("nullable", False)) != self.nullable:
            out.append(
                f"{self.name}: nullable {actual.get('nullable', False)} "
                f"!= expected {self.nullable}"
            )
        if self.dim is not None and params.get("dim") != self.dim:
            out.append(f"{self.name}: dim {params.get('dim')} != expected {self.dim}")
        if self.element_type is not None and actual.get("element_type") != (
            self.element_type
        ):
            out.append(
                f"{self.name}: element_type {_name_of(actual.get('element_type'))} "
                f"!= expected {_name_of(self.element_type)}"
            )
        if self.enable_analyzer and not _is_true(params.get("enable_analyzer")):
            out.append(
                f"{self.name}: enable_analyzer "
                f"{params.get('enable_analyzer', False)} != expected True"
            )
        return out

    def soft_mismatches(self, actual: dict[str, Any]) -> list[str]:
        """Advisory-only limit drift (see :meth:`mismatches`)."""
        params = actual.get("params") or {}
        out: list[str] = []
        for key, want in (
            ("max_length", self.max_length),
            ("max_capacity", self.max_capacity),
        ):
            got = params.get(key)
            if want is not None and got is not None and int(got) != want:
                out.append(f"{self.name}: {key} {got} != declared {want}")
        return out


def _name_of(datatype: Any) -> str:
    return getattr(datatype, "name", repr(datatype))


def _is_true(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.casefold() == "true")


def _function_key(
    function: dict[str, Any],
) -> tuple[Any, tuple[str, ...], tuple[str, ...]]:
    return (
        function.get("type"),
        tuple(function.get("input_field_names") or ()),
        tuple(function.get("output_field_names") or ()),
    )


def _format_function_key(
    key: tuple[Any, tuple[str, ...], tuple[str, ...]],
) -> str:
    function_type, inputs, outputs = key
    return (
        f"type={_name_of(function_type)}, inputs={list(inputs)}, "
        f"outputs={list(outputs)}"
    )


def _metric_name(metric: Any) -> str | None:
    if metric is None:
        return None
    return str(getattr(metric, "name", metric)).upper()


class CollectionLayout:
    """Declares and verifies one table's Milvus collection.

    Args:
        index_schema: The table's neutral index schema.
    """

    def __init__(self, index_schema: IndexSchema) -> None:
        self._index_schema = index_schema

    def build_schema(self) -> Any:
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        for physical in self.physical_fields():
            schema.add_field(**physical.create_kwargs())
        for field in self._index_schema.bm25_fields:
            sparse_name = sparse_field(field)
            schema.add_function(
                Function(
                    name=f"{field}_bm25",
                    function_type=FunctionType.BM25,
                    input_field_names=[field],
                    output_field_names=[sparse_name],
                )
            )
        return schema

    def physical_fields(self) -> tuple[PhysicalField, ...]:
        """Every Milvus column this table declares, in creation order."""
        out: list[PhysicalField] = [
            PhysicalField(
                name=PK_FIELD,
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=PK_MAX_LENGTH,
            )
        ]
        for field in self._index_schema.fields:
            out.extend(self._physical_for(field))
        if not self._index_schema.vector_fields:
            out.append(
                PhysicalField(
                    name=DUMMY_VECTOR_FIELD,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=DUMMY_VECTOR_DIMENSION,
                )
            )
        out.extend(
            PhysicalField(
                name=sparse_field(field),
                datatype=DataType.SPARSE_FLOAT_VECTOR,
            )
            for field in self._index_schema.bm25_fields
        )
        return tuple(out)

    def _physical_for(self, field: IndexField) -> list[PhysicalField]:
        is_bm25 = field.name in self._index_schema.bm25_fields
        if field.kind is IndexFieldKind.STRING:
            return [
                PhysicalField(
                    name=field.name,
                    datatype=DataType.VARCHAR,
                    # A BM25 input is written as "" rather than null so the
                    # analyzer always has something to tokenize.
                    nullable=field.nullable and not is_bm25,
                    max_length=field.max_length,
                    enable_analyzer=is_bm25,
                )
            ]
        if field.kind is IndexFieldKind.STRING_ARRAY:
            return [
                PhysicalField(
                    name=field.name,
                    datatype=DataType.ARRAY,
                    nullable=field.nullable,
                    element_type=DataType.VARCHAR,
                    max_length=field.max_length,
                    max_capacity=field.max_capacity,
                )
            ]
        if field.kind is IndexFieldKind.FLOAT:
            return [
                PhysicalField(
                    name=field.name,
                    datatype=DataType.DOUBLE,
                    nullable=field.nullable,
                )
            ]
        if field.kind is IndexFieldKind.INTEGER:
            return [
                PhysicalField(
                    name=field.name,
                    datatype=DataType.INT64,
                    nullable=field.nullable,
                )
            ]
        if field.kind is IndexFieldKind.DATETIME:
            return [
                PhysicalField(
                    name=datetime_storage_field(field.name),
                    datatype=DataType.INT64,
                    nullable=field.nullable,
                )
            ]
        if field.kind is IndexFieldKind.DENSE_VECTOR:
            return [
                PhysicalField(
                    name=field.name,
                    datatype=DataType.FLOAT_VECTOR,
                    dim=field.dimension,
                ),
                # Milvus cannot store a null dense vector, so a logical null is
                # a zero vector plus this presence marker.
                PhysicalField(
                    name=present_field(field.name),
                    datatype=DataType.BOOL,
                ),
            ]
        # pragma: no cover - enum exhaustiveness guard
        raise TypeError(f"unsupported index field kind: {field.kind}")

    def _stored_field_names(self) -> list[str]:
        return [physical.name for physical in self.physical_fields()]

    def output_fields(self, *, include_vectors: bool) -> list[str]:
        fields: list[str] = []
        vector_names = {field.name for field in self._index_schema.vector_fields}
        present_names = {present_field(name) for name in vector_names}
        for name in self._stored_field_names():
            if name.endswith(SPARSE_SUFFIX):
                continue
            if name == DUMMY_VECTOR_FIELD:
                if include_vectors:
                    fields.append(name)
                continue
            if not include_vectors and (name in vector_names or name in present_names):
                continue
            fields.append(name)
        return fields

    def add_indexes(self, index_params: Any) -> None:
        """Declare every derived index on ``index_params``.

        ``index_params`` comes from ``prepare_index_params``. One COSINE
        AUTOINDEX per dense vector (or on the dummy vector when the
        table has none) and one BM25 AUTOINDEX per sparse column.
        """
        vector_fields = self._index_schema.vector_fields
        if vector_fields:
            for field in vector_fields:
                index_params.add_index(
                    field_name=field.name,
                    index_type="AUTOINDEX",
                    metric_type="COSINE",
                )
        else:
            index_params.add_index(
                field_name=DUMMY_VECTOR_FIELD,
                index_type="AUTOINDEX",
                metric_type="COSINE",
            )
        for field in self._index_schema.bm25_fields:
            index_params.add_index(
                field_name=sparse_field(field),
                index_type="AUTOINDEX",
                metric_type="BM25",
            )

    def verify_fields(
        self, description: Mapping[str, Any], *, collection_name: str
    ) -> None:
        """Reject a collection whose physical metadata disagrees with ours.

        ``description`` is the server's ``describe_collection`` reply.

        A name-only check waves through a collection whose columns happen to
        share our names but carry the wrong datatype, primary key, nullability
        or vector dimension — a stale model, or a ``collection_prefix``
        collision with someone else's data. That collection starts fine and
        then fails opaquely on the first write or search. Comparing the
        physical shape, BM25 function mapping, and field-matched vector index
        metrics turns it back into a startup error with a recovery path.
        Index metrics are checked separately by :meth:`verify_indexes`.

        Raises:
            MilvusSchemaMismatchError: A column, its shape, or a BM25 function
                differs from the declaration.
        """
        reported = {
            field["name"]: field
            for field in description.get("fields", [])
            if "name" in field
        }
        expected = {physical.name: physical for physical in self.physical_fields()}

        missing = sorted(set(expected) - set(reported))
        stale = sorted(set(reported) - set(expected))
        drift: list[str] = []
        advisory: list[str] = []
        for name, physical in expected.items():
            actual = reported.get(name)
            if actual is None:
                continue
            drift.extend(physical.mismatches(actual))
            advisory.extend(physical.soft_mismatches(actual))

        expected_functions = {
            (
                FunctionType.BM25,
                (field,),
                (sparse_field(field),),
            )
            for field in self._index_schema.bm25_fields
        }
        reported_functions = {
            _function_key(function)
            for function in description.get("functions", [])
            if isinstance(function, dict)
        }
        missing_functions = sorted(
            _format_function_key(key) for key in expected_functions - reported_functions
        )
        stale_functions = sorted(
            _format_function_key(key) for key in reported_functions - expected_functions
        )

        if advisory:
            logger.warning(
                "milvus_collection_limit_drift",
                collection=collection_name,
                details=advisory,
            )
        if missing or stale or drift or missing_functions or stale_functions:
            raise MilvusSchemaMismatchError(
                f"Milvus collection {collection_name!r} schema drift: "
                f"missing={missing}, stale={stale}, incompatible={drift}, "
                f"missing_functions={missing_functions}, "
                f"stale_functions={stale_functions}. "
                "The index is rebuildable from markdown; run "
                "`everos cascade rebuild`."
            )

    def verify_indexes(
        self,
        index_descriptions: Sequence[tuple[str, Any]],
        *,
        collection_name: str,
    ) -> None:
        """Verify only the field and metric contracts of derived indexes.

        ``index_descriptions`` pairs each name from ``list_indexes`` with its
        ``describe_index`` reply, in the server's order.

        Raises:
            MilvusSchemaMismatchError: An index is missing, duplicated, on an
                unexpected vector field, or uses the wrong metric.
        """
        expected = {field.name: "COSINE" for field in self._index_schema.vector_fields}
        if not expected:
            expected[DUMMY_VECTOR_FIELD] = "COSINE"
        expected.update(
            {sparse_field(field): "BM25" for field in self._index_schema.bm25_fields}
        )

        reported: dict[str, list[tuple[str, str | None]]] = {}
        incompatible: list[str] = []
        physical = {field.name: field for field in self.physical_fields()}
        vector_types = {DataType.FLOAT_VECTOR, DataType.SPARSE_FLOAT_VECTOR}
        for index_name, description in index_descriptions:
            if not isinstance(description, dict):
                incompatible.append(f"index {index_name!r}: description unavailable")
                continue
            field_name = description.get("field_name")
            if not isinstance(field_name, str):
                incompatible.append(f"index {index_name!r}: field_name unavailable")
                continue
            declared = physical.get(field_name)
            if field_name not in expected:
                if declared is None or declared.datatype in vector_types:
                    incompatible.append(
                        f"index {index_name!r}: unexpected vector field {field_name!r}"
                    )
                continue
            metric = description.get("metric_type")
            if metric is None and isinstance(description.get("params"), dict):
                metric = description["params"].get("metric_type")
            reported.setdefault(field_name, []).append(
                (str(index_name), _metric_name(metric))
            )

        missing_indexes = sorted(set(expected) - set(reported))
        for field_name, indexes in reported.items():
            wanted = expected[field_name]
            if len(indexes) != 1:
                incompatible.append(
                    f"{field_name}: expected exactly one index, found {len(indexes)}"
                )
            for index_name, metric in indexes:
                if metric != wanted:
                    incompatible.append(
                        f"{field_name}: metric_type {metric!r} != expected {wanted!r} "
                        f"(index {index_name!r})"
                    )

        if missing_indexes or incompatible:
            raise MilvusSchemaMismatchError(
                f"Milvus collection {collection_name!r} index drift: "
                f"missing={missing_indexes}, incompatible={incompatible}. "
                "The index is rebuildable from markdown; run "
                "`everos cascade rebuild`."
            )
