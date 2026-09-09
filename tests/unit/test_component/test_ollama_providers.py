"""Tests for Ollama LLM and Embedding providers."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, patch

import pytest

# ── LLM Provider ────────────────────────────────────────────────────


class TestOllamaProvider:
    """Ollama LLM provider tests."""

    def test_defaults(self):
        """OllamaProvider applies localhost defaults when omitted."""
        from everos.component.llm.ollama_provider import OllamaProvider

        with patch("openai.AsyncOpenAI"):
            p = OllamaProvider(model="llama3.1")

        assert p._model == "llama3.1"

    def test_custom_base_url(self):
        """Custom base_url and api_key are respected."""
        from everos.component.llm.ollama_provider import OllamaProvider

        with patch("openai.AsyncOpenAI"):
            p = OllamaProvider(
                model="qwen2.5",
                api_key="custom-key",
                base_url="http://192.168.1.100:11434/v1",
            )

        assert p._model == "qwen2.5"

    def test_extends_openai_provider(self):
        """OllamaProvider inherits from OpenAIProvider."""
        from everos.component.llm.ollama_provider import OllamaProvider
        from everos.component.llm.openai_provider import OpenAIProvider

        assert issubclass(OllamaProvider, OpenAIProvider)


# ── Embedding Provider ──────────────────────────────────────────────


class TestOllamaEmbeddingProvider:
    """Ollama Embedding provider tests."""

    def test_defaults(self):
        """OllamaEmbeddingProvider applies localhost + dim=768 defaults."""
        from everos.component.embedding.ollama_provider import (
            OllamaEmbeddingProvider,
        )

        with patch("openai.AsyncOpenAI"):
            p = OllamaEmbeddingProvider(model="nomic-embed-text")

        assert p._model == "nomic-embed-text"
        assert p.dim == 768

    def test_custom_dim(self):
        """Custom dim overrides the default 768."""
        from everos.component.embedding.ollama_provider import (
            OllamaEmbeddingProvider,
        )

        with patch("openai.AsyncOpenAI"):
            p = OllamaEmbeddingProvider(
                model="mxbai-embed-large",
                dim=1024,
            )

        assert p.dim == 1024

    def test_extends_openai_provider(self):
        """OllamaEmbeddingProvider inherits from OpenAIEmbeddingProvider."""
        from everos.component.embedding.ollama_provider import (
            OllamaEmbeddingProvider,
        )
        from everos.component.embedding.openai_provider import (
            OpenAIEmbeddingProvider,
        )

        assert issubclass(OllamaEmbeddingProvider, OpenAIEmbeddingProvider)


# ── Import surface ──────────────────────────────────────────────────


class TestImportSurface:
    """Verify public imports work from the package level."""

    def test_llm_ollama_importable(self):
        """OllamaProvider is importable from everos.component.llm."""
        from everos.component.llm import OllamaProvider

        assert OllamaProvider is not None

    def test_embedding_ollama_importable(self):
        """OllamaEmbeddingProvider is importable from everos.component.embedding."""
        from everos.component.embedding import OllamaEmbeddingProvider

        assert OllamaEmbeddingProvider is not None
