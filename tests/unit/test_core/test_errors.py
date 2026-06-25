"""Tests for the DDD-aligned exception hierarchy."""

from __future__ import annotations

from everos.core.errors import (
    AppError,
    CapabilityError,
    ConfigurationError,
    ConflictError,
    DocumentNotFoundError,
    DomainError,
    DuplicateDocumentError,
    EmbeddingServiceError,
    ExternalServiceError,
    ExtractionEmptyError,
    FilterError,
    InfrastructureError,
    InvalidInputError,
    LLMServiceError,
    MultimodalNotEnabledError,
    NotFoundError,
    PathTraversalError,
    RerankServiceError,
    StorageError,
    TopicNotFoundError,
    UnsupportedModalityError,
    VectorStoreError,
)


class TestDomainBranch:
    def test_domain_errors_are_app_errors(self) -> None:
        assert issubclass(DomainError, AppError)
        assert issubclass(NotFoundError, DomainError)
        assert issubclass(ConflictError, DomainError)
        assert issubclass(InvalidInputError, DomainError)

    def test_not_found_subtypes(self) -> None:
        assert issubclass(DocumentNotFoundError, NotFoundError)
        assert issubclass(TopicNotFoundError, NotFoundError)

    def test_conflict_subtypes(self) -> None:
        assert issubclass(DuplicateDocumentError, ConflictError)

    def test_invalid_input_subtypes(self) -> None:
        assert issubclass(ExtractionEmptyError, InvalidInputError)
        assert issubclass(FilterError, InvalidInputError)

    def test_path_traversal_is_domain_not_invalid_input(self) -> None:
        assert issubclass(PathTraversalError, DomainError)
        assert not issubclass(PathTraversalError, InvalidInputError)

    def test_unsupported_modality_is_domain(self) -> None:
        assert issubclass(UnsupportedModalityError, DomainError)
        assert not issubclass(UnsupportedModalityError, InfrastructureError)


class TestInfrastructureBranch:
    def test_infrastructure_errors_are_app_errors(self) -> None:
        assert issubclass(InfrastructureError, AppError)
        assert issubclass(StorageError, InfrastructureError)
        assert issubclass(VectorStoreError, InfrastructureError)
        assert issubclass(ExternalServiceError, InfrastructureError)

    def test_external_service_subtypes(self) -> None:
        assert issubclass(LLMServiceError, ExternalServiceError)
        assert issubclass(EmbeddingServiceError, ExternalServiceError)
        assert issubclass(RerankServiceError, ExternalServiceError)


class TestCapabilityBranch:
    def test_capability_is_app_error(self) -> None:
        assert issubclass(CapabilityError, AppError)

    def test_multimodal_not_enabled(self) -> None:
        assert issubclass(MultimodalNotEnabledError, CapabilityError)
        assert not issubclass(MultimodalNotEnabledError, InfrastructureError)
        assert not issubclass(MultimodalNotEnabledError, DomainError)


class TestConfigurationBranch:
    def test_configuration_is_app_error(self) -> None:
        assert issubclass(ConfigurationError, AppError)
        assert not issubclass(ConfigurationError, DomainError)
        assert not issubclass(ConfigurationError, InfrastructureError)


class TestBackwardCompat:
    def test_old_document_already_exists_alias(self) -> None:
        from everos.core.errors import DocumentAlreadyExistsError

        assert DocumentAlreadyExistsError is DuplicateDocumentError

    def test_old_validation_error_alias(self) -> None:
        from everos.core.errors import ValidationError

        assert ValidationError is InvalidInputError


class TestMRODispatch:
    def test_infrastructure_catches_embedding(self) -> None:
        exc = EmbeddingServiceError("provider down")
        assert isinstance(exc, InfrastructureError)
        assert isinstance(exc, AppError)

    def test_instantiation_with_message(self) -> None:
        exc = DocumentNotFoundError("d_abc123")
        assert str(exc) == "d_abc123"
        exc2 = LLMServiceError("timeout after 30s")
        assert str(exc2) == "timeout after 30s"
