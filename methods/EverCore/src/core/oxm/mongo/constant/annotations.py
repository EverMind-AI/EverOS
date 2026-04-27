from __future__ import annotations

from core.class_annotations.types import StringEnumAnnotationKey, StringEnumAnnotation


class ClassAnnotationKey(StringEnumAnnotationKey):
    """Infra-layer class annotation keys."""

    READONLY = "odm.readonly"


class Toggle(StringEnumAnnotation):
    """Simple toggle values for annotations."""

    ENABLED = "enabled"
    DISABLED = "disabled"
