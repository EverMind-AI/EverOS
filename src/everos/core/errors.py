"""Cross-cutting domain errors surfaced to API callers.

These live in ``core`` so the ``memory`` layer can raise them and the
``entrypoints`` layer can catch them without crossing the layered import
boundary — ``any -> core`` is the only edge both share (entrypoints must
not import ``memory`` directly).
"""

from __future__ import annotations


class MultimodalError(Exception):
    """Base for multimodal-parsing errors meant to reach the caller.

    The API layer maps any ``MultimodalError`` to an aligned
    ``{error: {code, message}}`` envelope (HTTP 415).
    """


class UnsupportedModalityError(MultimodalError):
    """everalgo cannot handle this modality (e.g. video stub, unknown type).

    Wraps everalgo's ``NotImplementedError`` / dispatch ``ValueError`` so the
    caller gets a stable, aligned error instead of a raw 500.
    """


class MultimodalNotEnabledError(MultimodalError):
    """Multimodal capability is not ready.

    Raised when the ``everos[multimodal]`` extra is not installed, or when a
    required system dependency (LibreOffice for Office documents) is absent.
    """
