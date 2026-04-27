from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Type, Union

from .types import AnnotationKeyBase, AnnotationValueBase


_NOTES_ATTR = "__notes__"


def _ensure_notes_dict(cls: Type[Any]) -> MutableMapping[str, AnnotationValueBase]:
    notes = getattr(cls, _NOTES_ATTR, None)
    if notes is None:
        notes = {}
        setattr(cls, _NOTES_ATTR, notes)
    elif not isinstance(notes, dict):
        # If user provided a non-dict mistakenly, reset to a dict to keep behavior consistent.
        notes = dict(notes)
        setattr(cls, _NOTES_ATTR, notes)
    return notes


def _normalize_key(key: Union[str, AnnotationKeyBase]) -> str:
    if isinstance(key, AnnotationKeyBase):
        return key.to_key()
    if isinstance(key, str):
        return key
    raise TypeError(
        f"Annotation key must be str or AnnotationKeyBase, got {type(key).__name__}"
    )


def class_annotations(
    annotations: Optional[
        Mapping[Union[str, AnnotationKeyBase], AnnotationValueBase]
    ] = None,
    /,
    **kwargs: AnnotationValueBase,
):
    """
    Class decorator to attach strict annotation values to a class.

    Usage:
        @class_annotations(owner=FreeformAnnotationValue(...))
        class MyClass: ...

        @class_annotations({"role": EnumAnnotationValue(RoleEnum.ADMIN)})
        class MyClass: ...

    Rules:
    - All values must be instances of AnnotationValueBase.
    - Multiple decorators can be stacked; later ones override same keys.
    """

    provided: Dict[str, AnnotationValueBase] = {}
    if annotations:
        for k, v in annotations.items():
            provided[_normalize_key(k)] = v
    if kwargs:
        # kwargs keys are always strings by Python's syntax
        for k, v in kwargs.items():
            provided[_normalize_key(k)] = v

    # Validate value types early
    for k, v in provided.items():
        if not isinstance(v, AnnotationValueBase):
            raise TypeError(
                f"Annotation '{k}' must be an instance of AnnotationValueBase, got {type(v).__name__}"
            )

    def _wrapper(cls: Type[Any]):
        notes = _ensure_notes_dict(cls)
        notes.update(provided)
        return cls

    return _wrapper
