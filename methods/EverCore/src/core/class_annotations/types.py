from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class AnnotationValueBase(ABC):
    """
    Base class for all class-annotation values.

    Implementations must be immutable or treated as value objects. They should
    provide a serializable representation via `to_data` for potential logging or
    transport, and a human-readable `__repr__`.
    """

    @abstractmethod
    def to_data(self) -> Any:
        """
        Return a JSON-serializable payload representing this value.
        """
        raise NotImplementedError


class AnnotationKeyBase(ABC):
    """
    Base class for annotation keys. Concrete implementations must provide a
    stable string key through `to_key()`.
    """

    @abstractmethod
    def to_key(self) -> str:
        raise NotImplementedError


class FreeformAnnotationValue(AnnotationValueBase):
    """
    Flexible annotation value that only carries the actual payload `data`.
    """

    def __init__(self, data: Any) -> None:
        self._data = data

    @property
    def data(self) -> Any:
        return self._data

    def to_data(self) -> Any:
        return {"type": "freeform", "data": self._data}

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"FreeformAnnotationValue(data={self._data!r})"


class StringEnumAnnotation(str, Enum):
    """
    Base class for string-backed enums that are directly usable as annotation values.

    Usage:
        class Role(StringEnumAnnotation):
            ADMIN = "admin"
            USER = "user"
    """

    def to_data(self) -> Any:
        # Ensure a stable, serializable shape
        return {
            "type": "enum",
            "enum": self.__class__.__name__,
            "name": self.name,
            "value": str(self.value),
        }


class StringEnumAnnotationKey(str, Enum):
    """
    Base class for string-backed enum keys.
    The enum's value (a string) is used as the canonical key.
    """

    def to_key(self) -> str:
        return str(self.value)


# Register virtual subclasses to satisfy isinstance/issubclass checks without metaclass conflicts
AnnotationValueBase.register(StringEnumAnnotation)
AnnotationKeyBase.register(StringEnumAnnotationKey)
