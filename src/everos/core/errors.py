"""Cross-cutting exception hierarchy for EverOS.

All application exceptions derive from ``AppError``, split into four branches:

- ``DomainError`` — business-rule violations (not-found, conflict, invalid
  input, path traversal, unsupported format).
- ``InfrastructureError`` — transient storage and external-service failures
  (retryable).
- ``CapabilityError`` — permanent server-side capability gaps (not retryable).
- ``ConfigurationError`` — misconfiguration detected at runtime.

Any layer may raise ``AppError`` subclasses; the entrypoints layer catches
them and maps them to aligned HTTP responses.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCode(StrEnum):
    """Machine-readable error codes returned in the API error envelope.

    Each code maps to exactly one HTTP status code. Clients can switch on
    this value to decide retry / display / routing behaviour without parsing
    the human-readable ``message`` field.
    """

    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    INVALID_INPUT = "INVALID_INPUT"
    EXTRACTION_EMPTY = "EXTRACTION_EMPTY"
    BAD_REQUEST = "BAD_REQUEST"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    EXTERNAL_SERVICE_UNAVAILABLE = "EXTERNAL_SERVICE_UNAVAILABLE"
    CAPABILITY_UNAVAILABLE = "CAPABILITY_UNAVAILABLE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    PROVIDER_NOT_CONFIGURED = "PROVIDER_NOT_CONFIGURED"


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class AppError(Exception):
    """Root for all EverOS application exceptions."""


# ---------------------------------------------------------------------------
# Domain branch — client-side / business-rule errors
# ---------------------------------------------------------------------------


class DomainError(AppError):
    """Business-rule violation originating in the domain or service layer."""


class NotFoundError(DomainError):
    """A requested resource does not exist."""


class DocumentNotFoundError(NotFoundError):
    """A document with the given identifier was not found."""


class TopicNotFoundError(NotFoundError):
    """A knowledge topic with the given identifier was not found."""


class ConflictError(DomainError):
    """An operation conflicts with existing state (e.g. duplicate resource)."""


class DuplicateDocumentError(ConflictError):
    """A document with the same identifier already exists."""


class InvalidInputError(DomainError):
    """Input does not meet domain rules."""


class ExtractionEmptyError(InvalidInputError):
    """An extraction pipeline produced no output when output was required."""


class FilterError(InvalidInputError):
    """A caller-supplied filter expression is invalid or malformed."""


class PathTraversalError(DomainError):
    """A write target resolved outside the configured memory root.

    Raised by the markdown writer as a defense-in-depth backstop: any
    caller-supplied identifier that becomes a path segment (``app_id`` /
    ``project_id`` / ``sender_id`` -> ``owner_id``) is validated at the DTO
    layer, but this containment check does not depend on every such id being
    sanitised upstream. The API layer maps it to HTTP 400.
    """


class UnsupportedModalityError(DomainError):
    """The uploaded file format is not supported (e.g. video, unknown type).

    Wraps everalgo's ``NotImplementedError`` / dispatch ``ValueError`` so
    the caller gets a stable 415 instead of a raw 500.
    """


# ---------------------------------------------------------------------------
# Infrastructure branch — transient failures (retryable)
# ---------------------------------------------------------------------------


class InfrastructureError(AppError):
    """Transient failure in a storage adapter or external service."""


class StorageError(InfrastructureError):
    """A markdown or SQLite persistence operation failed."""


class VectorStoreError(InfrastructureError):
    """A LanceDB vector-store operation failed."""


class ExternalServiceError(InfrastructureError):
    """An external service (LLM, embedding, rerank) returned an error or timed out."""


class VectorStoreBusyError(ExternalServiceError):
    """A LanceDB table's write lock could not be taken, or the critical
    section it guards overran its deadline.

    Deliberately under :class:`ExternalServiceError` rather than
    :class:`VectorStoreError`: the cascade worker retries that branch, and a
    contended or slow table is exactly the transient condition retrying is
    for. Under :class:`VectorStoreError` the row would be marked permanently
    failed and need a manual ``cascade fix``.
    """


class LLMServiceError(ExternalServiceError):
    """The configured LLM provider returned an error or timed out."""


class EmbeddingServiceError(ExternalServiceError):
    """The configured embedding provider returned an error or timed out."""


class RerankServiceError(ExternalServiceError):
    """The configured rerank provider returned an error or timed out."""


# ---------------------------------------------------------------------------
# Capability branch — permanent server-side gaps (not retryable)
# ---------------------------------------------------------------------------


class CapabilityError(AppError):
    """A required server-side capability is not available.

    Unlike ``InfrastructureError`` (transient — retry may help),
    ``CapabilityError`` signals a permanent gap that requires admin
    action (install a dependency, enable a feature).
    """


class MultimodalNotEnabledError(CapabilityError):
    """Multimodal parsing capability is not available.

    Raised when the ``everos[multimodal]`` extra is not installed, or when
    a required system dependency (LibreOffice for Office documents) is absent.
    """


# ---------------------------------------------------------------------------
# Configuration branch — misconfiguration detected at runtime
# ---------------------------------------------------------------------------


class ConfigurationError(AppError):
    """A required configuration is missing or invalid.

    Raised when a mandatory setting (e.g. embedding model, rerank provider)
    is not configured but the code path requires it.
    """


# TOML section name for each provider kind, keyed by the `provider` argument
# passed to `ProviderNotConfiguredError`.
_PROVIDER_SECTIONS: dict[str, str] = {
    "llm": "llm",
    "embedding": "embedding",
    "rerank": "rerank",
    "multimodal_llm": "multimodal",
}


class ProviderNotConfiguredError(ConfigurationError):
    """Raised when a runtime path requires a provider that is not configured.

    Maps to HTTP 422 via the FastAPI exception handler; the message is
    directly user-facing and points at the toml file for remediation.

    Args:
        provider: Which provider is missing. One of ``"llm"``,
            ``"embedding"``, ``"rerank"``, ``"multimodal_llm"``.
        feature: Optional user-facing feature that required this
            provider (e.g. ``"knowledge"``, ``"agent_hybrid"``).
        alternative_hint: Optional alternative workaround the user can
            take without configuring the missing provider (e.g. flip
            ``enable_llm_rerank=true`` to use the LLM lane).
    """

    def __init__(
        self,
        provider: str,
        feature: str | None = None,
        alternative_hint: str | None = None,
    ) -> None:
        # Lazy import: `config_hints` -> `core.persistence` -> `core.persistence
        # .markdown.writer` imports `PathTraversalError` from this module, so a
        # module-level import here would be circular.
        from everos.component.utils.config_hints import missing_config_error

        self.provider = provider
        self.feature = feature
        self.alternative_hint = alternative_hint
        section = _PROVIDER_SECTIONS.get(provider, provider)
        field_label = f"Provider '{provider}'"
        if feature:
            field_label += f" (required by {feature})"
        message = missing_config_error(field_label, section)
        if alternative_hint:
            message += f" Alternative: {alternative_hint}"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Backward compatibility aliases
# ---------------------------------------------------------------------------

# Renamed in v0.2 — old names kept for external consumers.
DocumentAlreadyExistsError = DuplicateDocumentError
ValidationError = InvalidInputError
