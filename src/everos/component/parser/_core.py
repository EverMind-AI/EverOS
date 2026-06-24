"""Core parse dispatch — wraps everalgo-parser with LLM injection and error mapping.

``everalgo-parser`` is an optional dependency (``everos[multimodal]``).
All imports are deferred so this module is safe to import without the extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from everos.core.errors import MultimodalNotEnabledError, UnsupportedModalityError

if TYPE_CHECKING:
    from everalgo.types import ParsedContent, RawFile


def parser_available() -> bool:
    """Whether ``everalgo.parser`` is importable."""
    try:
        import everalgo.parser  # noqa: F401
    except ImportError:
        return False
    return True


def require_parser() -> None:
    """Raise when the parser extra is not installed.

    Raises:
        MultimodalNotEnabledError: When ``everalgo.parser`` cannot be imported.
    """
    if not parser_available():
        raise MultimodalNotEnabledError(
            "Multimodal parsing requires the parser extra. "
            "Install with: pip install 'everos[multimodal]'"
        )


async def aparse_file(raw_file: RawFile) -> ParsedContent:
    """Parse a file via everalgo-parser with the multimodal LLM client.

    Args:
        raw_file: Hydrated ``RawFile`` with ``content`` bytes or ``uri``.

    Returns:
        Parsed text content with modality metadata.

    Raises:
        MultimodalNotEnabledError: Parser not installed or system dep missing.
        UnsupportedModalityError: File format not supported by the parser.
    """
    from everalgo.llm import LLMError
    from everalgo.parser import aparse  # Deferred: optional dep

    from everos.component.llm import get_multimodal_llm_client
    from everos.core.errors import LLMServiceError

    try:
        return await aparse(raw_file, llm=get_multimodal_llm_client())
    except NotImplementedError as exc:
        raise UnsupportedModalityError(f"modality not supported: {exc}") from exc
    except LLMError as exc:
        raise LLMServiceError(str(exc)) from exc
    except ValueError as exc:
        raise UnsupportedModalityError(str(exc)) from exc
    except RuntimeError as exc:
        raise MultimodalNotEnabledError(str(exc)) from exc
