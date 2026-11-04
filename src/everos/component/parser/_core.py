"""Core parse dispatch — wraps the parser backend with LLM injection and error mapping.

The backend's third-party readers (``pillow``, ``beautifulsoup4``, ``cairosvg``) are
optional (``everos[multimodal]``). All imports are deferred so this module is safe
to import without the extra.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from everos.config import missing_config_error
from everos.core.errors import MultimodalNotEnabledError, UnsupportedModalityError

if TYPE_CHECKING:
    from everos.algo.llm import LLMClient
    from everos.algo.types import ParsedContent, RawFile


@lru_cache(maxsize=1)
def parser_available() -> bool:
    """Whether ``everos.component.parser.backend`` is importable.

    Memoised: the underlying ``import everos.component.parser.backend`` pulls in
    ``pillow`` and ``beautifulsoup4``, which would
    block the event loop for hundreds of ms — unacceptable inside
    ``async def health()`` on a liveness probe. First call at startup
    (via ``build_application``) pays the import cost off the
    request path; subsequent calls hit the cache instantly.

    The cache is process-wide. Tests that patch ``sys.modules`` to
    swap the ``everos.component.parser.backend`` module in or out must invoke
    :meth:`parser_available.cache_clear` between assertions to avoid
    cross-test contamination.
    """
    try:
        import everos.component.parser.backend  # noqa: F401
    except ImportError:
        return False
    return True


def require_parser() -> None:
    """Raise when the parser extra is not installed.

    Raises:
        MultimodalNotEnabledError: When ``everos.component.parser.backend``
            cannot be imported.
    """
    if not parser_available():
        raise MultimodalNotEnabledError(
            "Multimodal parsing requires the parser extra. "
            "Install with: pip install 'everos[multimodal]'"
        )


async def aparse_file(raw_file: RawFile, *, llm: LLMClient | None) -> ParsedContent:
    """Parse a file via the parser backend with the multimodal LLM client.

    Args:
        raw_file: Hydrated ``RawFile`` with ``content`` bytes or ``uri``.
        llm: The multimodal LLM; ``None`` when ``[multimodal]`` is not
            configured.

    Returns:
        Parsed text content with modality metadata.

    Raises:
        LLMNotConfiguredError: ``llm`` is ``None``.
        MultimodalNotEnabledError: Parser not installed or system dep missing.
        UnsupportedModalityError: File format not supported by the parser.
    """
    from everos.component.llm import LLMCallError, LLMNotConfiguredError
    from everos.component.parser.backend import aparse  # Deferred: optional dep
    from everos.core.errors import LLMServiceError

    if llm is None:
        raise LLMNotConfiguredError(
            missing_config_error("Multimodal LLM api_key and base_url", "multimodal")
        )
    try:
        return await aparse(raw_file, llm=llm)
    except NotImplementedError as exc:
        raise UnsupportedModalityError(f"modality not supported: {exc}") from exc
    except LLMCallError as exc:
        raise LLMServiceError(str(exc)) from exc
    except ValueError as exc:
        raise UnsupportedModalityError(str(exc)) from exc
    except RuntimeError as exc:
        raise MultimodalNotEnabledError(str(exc)) from exc
