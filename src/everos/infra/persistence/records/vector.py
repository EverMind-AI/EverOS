"""The backend-neutral marker for a dense embedding column.

A record says *that* a column holds an embedding, never *how wide* it is: the
width is deployment configuration (``[embedding] dim``), handed to the index
backend by the application bootstrap. Each backend turns the marked column
into its own physical type at that width.
"""

from __future__ import annotations

__all__ = ["DenseVector", "VectorColumn", "is_vector_field"]

from dataclasses import dataclass
from typing import Annotated, Any, get_args, get_origin

from pydantic.fields import FieldInfo


@dataclass(frozen=True)
class VectorColumn:
    """Annotation marker: the field is a dense embedding column."""


DenseVector = Annotated[list[float], VectorColumn()]
"""A dense embedding. Declare it ``DenseVector | None = None`` on a record."""


def is_vector_field(field: FieldInfo) -> bool:
    """Whether ``field`` is declared as a :data:`DenseVector`.

    pydantic moves the metadata of a bare ``Annotated`` field onto
    ``field.metadata`` but leaves it inside the annotation when the field is a
    union (``DenseVector | None``), so both places are checked.
    """
    if any(isinstance(item, VectorColumn) for item in field.metadata):
        return True
    return any(_is_marked(arg) for arg in get_args(field.annotation))


def _is_marked(annotation: Any) -> bool:
    if get_origin(annotation) is not Annotated:
        return False
    return any(isinstance(item, VectorColumn) for item in annotation.__metadata__)
