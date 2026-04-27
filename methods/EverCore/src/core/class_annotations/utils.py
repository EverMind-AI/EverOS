from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Type, Union

from .types import AnnotationKeyBase, AnnotationValueBase


_NOTES_ATTR = "__notes__"


def _collect_mro_notes(
    cls: Type[Any],
) -> Tuple[MutableMapping[str, AnnotationValueBase], ...]:
    """
    Collect `__notes__` from MRO (base classes first, subclass last).
    Only include those that are dict-like.
    """
    collected = []
    for base in reversed(cls.mro()):
        if base is object:
            continue
        notes = getattr(base, _NOTES_ATTR, None)
        if isinstance(notes, dict):
            collected.append(notes)
    return tuple(collected)


def _merged_notes(cls: Type[Any]) -> Dict[str, AnnotationValueBase]:
    merged: Dict[str, AnnotationValueBase] = {}
    for notes in _collect_mro_notes(cls):
        for k, v in notes.items():
            if not isinstance(v, AnnotationValueBase):
                # Skip invalid values quietly to be robust at runtime.
                continue
            merged[k] = v
    return merged


def get_annotations(
    target: Union[Type[Any], Any], *, include_inherited: bool = True
) -> Mapping[str, AnnotationValueBase]:
    """
    Get all annotations attached to a class or instance.

    - If `include_inherited` is True, the result is a merged view following MRO, where
      subclass values override base ones.
    - If False, only annotations defined directly on that class are returned.
    """
    cls: Type[Any] = target if isinstance(target, type) else type(target)
    if include_inherited:
        return _merged_notes(cls)
    notes = getattr(cls, _NOTES_ATTR, {})
    return {k: v for k, v in notes.items() if isinstance(v, AnnotationValueBase)}


def get_annotation(
    target: Union[Type[Any], Any],
    key: Union[str, AnnotationKeyBase],
    *,
    include_inherited: bool = True,
) -> Optional[AnnotationValueBase]:
    """
    Get a single annotation value by key from a class or instance.
    """
    normalized = key.to_key() if isinstance(key, AnnotationKeyBase) else key
    notes = get_annotations(target, include_inherited=include_inherited)
    return notes.get(normalized)


def has_annotation(
    target: Union[Type[Any], Any],
    key: Union[str, AnnotationKeyBase],
    *,
    include_inherited: bool = True,
) -> bool:
    return get_annotation(target, key, include_inherited=include_inherited) is not None
