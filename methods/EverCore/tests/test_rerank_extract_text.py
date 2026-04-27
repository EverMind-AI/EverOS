"""Unit tests for extract_text_from_hit

Usage:
    PYTHONPATH=src pytest tests/test_rerank_extract_text.py -v
"""

import time
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agentic_layer.rerank_interface import extract_text_from_hit


class TestExtractTextFromHit:
    """Test extract_text_from_hit with various memory types"""

    def test_episodic_memory(self):
        hit = {
            "memory_type": "episodic_memory",
            "_source": {"episode": "User likes coffee"},
        }
        assert extract_text_from_hit(hit) == "Episode Memory: User likes coffee"

    def test_foresight_with_evidence(self):
        hit = {
            "memory_type": "foresight",
            "_source": {
                "foresight": "Will need more storage",
                "evidence": "Usage growing 10% monthly",
            },
        }
        result = extract_text_from_hit(hit)
        assert (
            result
            == "Foresight: Will need more storage (Evidence: Usage growing 10% monthly)"
        )

    def test_foresight_without_evidence(self):
        hit = {
            "memory_type": "foresight",
            "_source": {"foresight": "Will need more storage"},
        }
        assert extract_text_from_hit(hit) == "Foresight: Will need more storage"

    def test_atomic_fact(self):
        hit = {
            "memory_type": "atomic_fact",
            "_source": {"atomic_fact": "User is 30 years old"},
        }
        assert extract_text_from_hit(hit) == "Atomic Fact: User is 30 years old"

    def test_fallback_episode_no_type(self):
        hit = {"memory_type": "", "_source": {"episode": "Some episode"}}
        assert extract_text_from_hit(hit) == "Some episode"

    def test_fallback_content(self):
        hit = {"memory_type": "", "_source": {"content": "Some content"}}
        assert extract_text_from_hit(hit) == "Some content"

    def test_fallback_to_str(self):
        hit = {"memory_type": "", "_source": {}}
        result = extract_text_from_hit(hit)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_source_key_uses_hit_directly(self):
        hit = {"memory_type": "", "episode": "Direct episode"}
        assert extract_text_from_hit(hit) == "Direct episode"

    def test_foresight_content_fallback(self):
        hit = {
            "memory_type": "foresight",
            "_source": {"content": "Foresight via content field"},
        }
        assert extract_text_from_hit(hit) == "Foresight: Foresight via content field"

    def test_episodic_memory_empty_episode_falls_to_fallback(self):
        hit = {
            "memory_type": "episodic_memory",
            "_source": {"episode": "", "content": "fallback content"},
        }
        assert extract_text_from_hit(hit) == "fallback content"

    def test_foresight_both_empty_falls_to_generic_summary(self):
        """foresight type with both foresight and content empty falls through to generic fallback (summary)."""
        hit = {
            "memory_type": "foresight",
            "_source": {"foresight": "", "content": "", "summary": "via summary"},
        }
        assert extract_text_from_hit(hit) == "via summary"

    def test_atomic_fact_empty_falls_to_generic_subject(self):
        """atomic_fact type with empty atomic_fact falls through to generic fallback (subject)."""
        hit = {
            "memory_type": "atomic_fact",
            "_source": {"atomic_fact": "", "subject": "via subject"},
        }
        assert extract_text_from_hit(hit) == "via subject"

    def test_no_type_has_atomic_fact(self):
        """No type with atomic_fact field uses the generic atomic_fact fallback."""
        hit = {"memory_type": "", "_source": {"atomic_fact": "Some fact"}}
        assert extract_text_from_hit(hit) == "Some fact"

    def test_no_type_has_foresight(self):
        """No type with foresight field uses the generic foresight fallback."""
        hit = {"memory_type": "", "_source": {"foresight": "Some foresight"}}
        assert extract_text_from_hit(hit) == "Some foresight"

    def test_no_type_has_summary(self):
        """No type with summary field uses the generic summary fallback."""
        hit = {"memory_type": "", "_source": {"summary": "Some summary"}}
        assert extract_text_from_hit(hit) == "Some summary"

    def test_no_type_has_subject(self):
        """No type with subject field uses the generic subject fallback."""
        hit = {"memory_type": "", "_source": {"subject": "Some subject"}}
        assert extract_text_from_hit(hit) == "Some subject"


class TestRerankInputLogging:
    """Test rerank input logging in HybridRerankService"""

    @pytest.fixture
    def mock_tokenizer_factory(self):
        factory = MagicMock()
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [1] * 100  # 100 tokens
        factory.get_tokenizer_from_tiktoken.return_value = tokenizer
        return factory

    @pytest.fixture
    def hybrid_service(self):
        with patch(
            "agentic_layer.rerank_service._create_service_from_config"
        ) as mock_create:
            mock_primary = AsyncMock()
            mock_primary.rerank_documents = AsyncMock(
                return_value={
                    "results": [
                        {"index": 0, "score": 0.9, "rank": 0},
                        {"index": 1, "score": 0.5, "rank": 1},
                    ]
                }
            )
            mock_create.return_value = mock_primary

            from agentic_layer.rerank_service import (
                HybridRerankService,
                HybridRerankConfig,
            )

            config = HybridRerankConfig()
            config.enable_fallback = False
            service = HybridRerankService(config)
            yield service

    @pytest.mark.asyncio
    async def test_logs_doc_count_and_tokens(
        self, hybrid_service, mock_tokenizer_factory
    ):
        hits = [
            {
                "memory_type": "episodic_memory",
                "_source": {"episode": "User likes coffee"},
            },
            {"memory_type": "atomic_fact", "_source": {"atomic_fact": "User is 30"}},
        ]

        with (
            patch(
                "agentic_layer.rerank_service.get_bean_by_type",
                return_value=mock_tokenizer_factory,
            ),
            patch("agentic_layer.rerank_service.logger") as mock_logger,
        ):
            await hybrid_service.rerank_memories("coffee", hits)

            mock_logger.info.assert_any_call("Rerank input: %d docs, %d tokens", 2, 100)

    @pytest.mark.asyncio
    async def test_logs_fallback_on_tokenizer_error(self, hybrid_service):
        hits = [
            {
                "memory_type": "episodic_memory",
                "_source": {"episode": "User likes coffee"},
            }
        ]

        with (
            patch(
                "agentic_layer.rerank_service.get_bean_by_type",
                side_effect=Exception("no tokenizer"),
            ),
            patch("agentic_layer.rerank_service.logger") as mock_logger,
        ):
            await hybrid_service.rerank_memories("coffee", hits)

            mock_logger.info.assert_any_call(
                "Rerank input: %d docs (token count unavailable)", 1
            )
            mock_logger.debug.assert_any_call("Token count failed", exc_info=True)

    @pytest.mark.asyncio
    async def test_error_path_records_error_metric_and_reraises(self, hybrid_service):
        """When execute_with_fallback raises, record_rerank_request is called with status='error'."""
        hits = [{"memory_type": "episodic_memory", "_source": {"episode": "A hit"}}]

        with (
            patch(
                "agentic_layer.rerank_service.get_bean_by_type",
                side_effect=Exception("no tokenizer"),
            ),
            patch.object(
                hybrid_service.primary_service,
                "rerank_memories",
                new_callable=AsyncMock,
                side_effect=RuntimeError("rerank failed"),
            ),
            patch("agentic_layer.rerank_service.record_rerank_request") as mock_record,
        ):
            with pytest.raises(Exception):
                await hybrid_service.rerank_memories("query", hits)

        # Verify error metric recorded
        mock_record.assert_called_once()
        call_kwargs = mock_record.call_args.kwargs
        assert call_kwargs["status"] == "error"


class TestCreateServiceFromConfig:
    """Tests for _create_service_from_config factory function (L154-177)"""

    def test_creates_vllm_service(self):
        from agentic_layer.rerank_service import _create_service_from_config
        from agentic_layer.rerank_vllm import VllmRerankService

        service = _create_service_from_config(
            "vllm", "key", "http://localhost", "model", 3, 2, 10, 5
        )
        assert isinstance(service, VllmRerankService)

    def test_creates_deepinfra_service(self):
        from agentic_layer.rerank_service import _create_service_from_config
        from agentic_layer.rerank_deepinfra import DeepInfraRerankService

        service = _create_service_from_config(
            "deepinfra", "key", "http://localhost", "model", 3, 2, 10, 5
        )
        assert isinstance(service, DeepInfraRerankService)

    def test_raises_for_unsupported_provider(self):
        from agentic_layer.rerank_service import _create_service_from_config
        from agentic_layer.rerank_interface import RerankError

        with pytest.raises(RerankError, match="Unsupported provider"):
            _create_service_from_config(
                "unknown", "key", "http://localhost", "model", 3, 2, 10, 5
            )


class TestHybridRerankServiceInit:
    """Tests for HybridRerankService.__init__ (L203, L222)"""

    def test_init_with_default_config(self):
        """L203: config=None causes HybridRerankConfig() to be created internally"""
        with patch(
            "agentic_layer.rerank_service._create_service_from_config"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            from agentic_layer.rerank_service import HybridRerankService

            service = HybridRerankService(config=None)
            assert service.config is not None

    def test_init_with_fallback_enabled(self):
        """L222: fallback service is created when enable_fallback is True"""
        with patch(
            "agentic_layer.rerank_service._create_service_from_config"
        ) as mock_create:
            mock_primary = MagicMock()
            mock_fallback = MagicMock()
            mock_create.side_effect = [mock_primary, mock_fallback]

            from agentic_layer.rerank_service import (
                HybridRerankService,
                HybridRerankConfig,
            )

            config = HybridRerankConfig.__new__(HybridRerankConfig)
            # Manually set all fields, bypassing __post_init__ env reads
            config.primary_provider = "vllm"
            config.fallback_provider = "deepinfra"
            config.primary_api_key = ""
            config.primary_base_url = "http://primary"
            config.fallback_api_key = "key"
            config.fallback_base_url = "http://fallback"
            config.model = "test-model"
            config.timeout = 3
            config.max_retries = 2
            config.batch_size = 10
            config.max_concurrent_requests = 5
            config.enable_fallback = True
            config.max_primary_failures = 3
            config.failure_reset_interval = 300
            config._primary_failure_count = 0
            config._last_failure_time = 0.0

            service = HybridRerankService(config)
            assert service.fallback_service is not None
            assert mock_create.call_count == 2


class TestHybridRerankServiceMethods:
    """Tests for get_service (L250), get_model_name (L351), rerank_documents (L367-368)"""

    @pytest.fixture
    def service_with_mock(self):
        with patch(
            "agentic_layer.rerank_service._create_service_from_config"
        ) as mock_create:
            mock_primary = AsyncMock()
            mock_primary.get_model_name = MagicMock(return_value="test-model")
            mock_primary.close = AsyncMock()
            mock_create.return_value = mock_primary
            from agentic_layer.rerank_service import (
                HybridRerankService,
                HybridRerankConfig,
            )

            config = HybridRerankConfig.__new__(HybridRerankConfig)
            config.primary_provider = "vllm"
            config.fallback_provider = "deepinfra"
            config.primary_api_key = ""
            config.primary_base_url = ""
            config.fallback_api_key = ""
            config.fallback_base_url = ""
            config.model = "test-model"
            config.timeout = 3
            config.max_retries = 2
            config.batch_size = 10
            config.max_concurrent_requests = 5
            config.enable_fallback = False
            config.max_primary_failures = 3
            config.failure_reset_interval = 300
            config._primary_failure_count = 0
            config._last_failure_time = 0.0
            yield HybridRerankService(config)

    def test_get_service(self, service_with_mock):
        result = service_with_mock.get_service()
        assert result == service_with_mock.primary_service

    def test_get_model_name(self, service_with_mock):
        assert service_with_mock.get_model_name() == "test-model"

    @pytest.mark.asyncio
    async def test_rerank_documents(self, service_with_mock):
        """L367-368: rerank_documents delegates via execute_with_fallback"""
        service_with_mock.primary_service.rerank_documents = AsyncMock(
            return_value={"results": [{"index": 0, "score": 0.9, "rank": 0}]}
        )
        result = await service_with_mock.rerank_documents("query", ["doc1"])
        assert "results" in result


class TestExecuteWithFallback:
    """Tests for execute_with_fallback (L406-512)"""

    @pytest.fixture
    def service_with_fallback(self):
        with patch(
            "agentic_layer.rerank_service._create_service_from_config"
        ) as mock_create:
            mock_primary = AsyncMock()
            mock_fallback = AsyncMock()
            mock_create.side_effect = [mock_primary, mock_fallback]
            from agentic_layer.rerank_service import (
                HybridRerankService,
                HybridRerankConfig,
            )

            config = HybridRerankConfig.__new__(HybridRerankConfig)
            config.primary_provider = "vllm"
            config.fallback_provider = "deepinfra"
            config.primary_api_key = ""
            config.primary_base_url = "http://primary"
            config.fallback_api_key = "key"
            config.fallback_base_url = "http://fallback"
            config.model = "test-model"
            config.timeout = 3
            config.max_retries = 2
            config.batch_size = 10
            config.max_concurrent_requests = 5
            config.enable_fallback = True
            config.max_primary_failures = 3
            config.failure_reset_interval = 300
            config._primary_failure_count = 0
            config._last_failure_time = 0.0
            service = HybridRerankService(config)
            service.fallback_service = mock_fallback
            yield service

    @pytest.mark.asyncio
    async def test_failure_count_reset_after_timeout(self, service_with_fallback):
        """L406-411: Reset failure count when timeout interval has expired"""
        service_with_fallback.config._primary_failure_count = 5
        service_with_fallback.config._last_failure_time = (
            time.time() - 600
        )  # 10 min ago
        service_with_fallback.config.failure_reset_interval = 300

        primary_func = AsyncMock(return_value="result")
        fallback_func = AsyncMock()
        result = await service_with_fallback.execute_with_fallback(
            "op", primary_func, fallback_func
        )
        assert result == "result"
        assert service_with_fallback.config._primary_failure_count == 0

    @pytest.mark.asyncio
    async def test_skip_primary_use_fallback_on_max_failures(
        self, service_with_fallback
    ):
        """L419-433: Skip primary and use fallback when max failures exceeded"""
        service_with_fallback.config._primary_failure_count = 5
        service_with_fallback.config.max_primary_failures = 3
        service_with_fallback.config._last_failure_time = (
            time.time()
        )  # recent, no reset

        fallback_func = AsyncMock(return_value="fallback_result")
        with patch("agentic_layer.rerank_service.record_rerank_fallback"):
            result = await service_with_fallback.execute_with_fallback(
                "op", AsyncMock(), fallback_func
            )
        assert result == "fallback_result"

    @pytest.mark.asyncio
    async def test_skip_primary_fallback_also_fails(self, service_with_fallback):
        """L435-445: Skip primary, fallback also fails -> RerankError"""
        service_with_fallback.config._primary_failure_count = 5
        service_with_fallback.config.max_primary_failures = 3
        service_with_fallback.config._last_failure_time = time.time()

        fallback_func = AsyncMock(side_effect=Exception("fallback died"))
        with (
            patch("agentic_layer.rerank_service.record_rerank_fallback"),
            patch("agentic_layer.rerank_service.record_rerank_error"),
        ):
            from agentic_layer.rerank_interface import RerankError

            with pytest.raises(RerankError, match="Fallback service failed"):
                await service_with_fallback.execute_with_fallback(
                    "op", AsyncMock(), fallback_func
                )

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_succeeds(self, service_with_fallback):
        """L478-500: Primary fails, fallback succeeds"""
        primary_func = AsyncMock(side_effect=Exception("primary died"))
        fallback_func = AsyncMock(return_value="fallback_ok")

        with (
            patch("agentic_layer.rerank_service.record_rerank_error"),
            patch("agentic_layer.rerank_service.record_rerank_fallback"),
        ):
            result = await service_with_fallback.execute_with_fallback(
                "op", primary_func, fallback_func
            )
        assert result == "fallback_ok"

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_also_fails(self, service_with_fallback):
        """L502-516: Both primary and fallback fail -> RerankError"""
        primary_func = AsyncMock(side_effect=Exception("primary died"))
        fallback_func = AsyncMock(side_effect=Exception("fallback died"))

        with (
            patch("agentic_layer.rerank_service.record_rerank_error"),
            patch("agentic_layer.rerank_service.record_rerank_fallback"),
        ):
            from agentic_layer.rerank_interface import RerankError

            with pytest.raises(RerankError, match="Both primary and fallback"):
                await service_with_fallback.execute_with_fallback(
                    "op", primary_func, fallback_func
                )

    @pytest.mark.asyncio
    async def test_primary_fails_no_fallback_raises(self, service_with_fallback):
        """L471-475: Primary fails, fallback disabled -> RerankError"""
        service_with_fallback.config.enable_fallback = False
        primary_func = AsyncMock(side_effect=Exception("primary died"))

        with patch("agentic_layer.rerank_service.record_rerank_error"):
            from agentic_layer.rerank_interface import RerankError

            with pytest.raises(RerankError, match="fallback is disabled"):
                await service_with_fallback.execute_with_fallback(
                    "op", primary_func, None
                )

    @pytest.mark.asyncio
    async def test_max_failures_reason_on_threshold(self, service_with_fallback):
        """L478-484: fallback_reason = 'max_failures_exceeded' when count reaches max"""
        service_with_fallback.config.max_primary_failures = 1  # next failure hits max
        service_with_fallback.config._primary_failure_count = 0

        primary_func = AsyncMock(side_effect=Exception("fail"))
        fallback_func = AsyncMock(return_value="ok")

        with (
            patch("agentic_layer.rerank_service.record_rerank_error"),
            patch("agentic_layer.rerank_service.record_rerank_fallback") as mock_fb,
        ):
            result = await service_with_fallback.execute_with_fallback(
                "op", primary_func, fallback_func
            )
        assert result == "ok"
        mock_fb.assert_called_once_with(
            primary_provider=service_with_fallback.config.primary_provider,
            fallback_provider=service_with_fallback.config.fallback_provider,
            reason='max_failures_exceeded',
        )


class TestClassifyError:
    """Tests for _classify_error (L524-534)"""

    @pytest.fixture
    def service(self):
        with patch(
            "agentic_layer.rerank_service._create_service_from_config"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            from agentic_layer.rerank_service import (
                HybridRerankService,
                HybridRerankConfig,
            )

            config = HybridRerankConfig.__new__(HybridRerankConfig)
            config.primary_provider = "vllm"
            config.fallback_provider = "deepinfra"
            config.primary_api_key = ""
            config.primary_base_url = ""
            config.fallback_api_key = ""
            config.fallback_base_url = ""
            config.model = "test-model"
            config.timeout = 3
            config.max_retries = 2
            config.batch_size = 10
            config.max_concurrent_requests = 5
            config.enable_fallback = False
            config.max_primary_failures = 3
            config.failure_reset_interval = 300
            config._primary_failure_count = 0
            config._last_failure_time = 0.0
            yield HybridRerankService(config)

    def test_timeout_by_message(self, service):
        assert service._classify_error(Exception("request timeout")) == 'timeout'

    def test_timeout_by_type(self, service):
        import asyncio

        assert service._classify_error(asyncio.TimeoutError()) == 'timeout'

    def test_rate_limit(self, service):
        assert service._classify_error(Exception("rate limit exceeded")) == 'rate_limit'

    def test_validation_error(self, service):
        assert (
            service._classify_error(Exception("validation failed"))
            == 'validation_error'
        )

    def test_connection_error(self, service):
        assert (
            service._classify_error(Exception("connection refused"))
            == 'connection_error'
        )

    def test_api_error(self, service):
        assert service._classify_error(Exception("api error 500")) == 'api_error'

    def test_unknown_error(self, service):
        assert service._classify_error(Exception("something weird")) == 'unknown'


class TestUtilityMethods:
    """Tests for get_failure_count, reset_failure_count, close (L538, L542-543, L547-549)"""

    @pytest.fixture
    def service(self):
        with patch(
            "agentic_layer.rerank_service._create_service_from_config"
        ) as mock_create:
            mock_primary = AsyncMock()
            mock_primary.close = AsyncMock()
            mock_create.return_value = mock_primary
            from agentic_layer.rerank_service import (
                HybridRerankService,
                HybridRerankConfig,
            )

            config = HybridRerankConfig.__new__(HybridRerankConfig)
            config.primary_provider = "vllm"
            config.fallback_provider = "deepinfra"
            config.primary_api_key = ""
            config.primary_base_url = ""
            config.fallback_api_key = ""
            config.fallback_base_url = ""
            config.model = "test-model"
            config.timeout = 3
            config.max_retries = 2
            config.batch_size = 10
            config.max_concurrent_requests = 5
            config.enable_fallback = False
            config.max_primary_failures = 3
            config.failure_reset_interval = 300
            config._primary_failure_count = 0
            config._last_failure_time = 0.0
            yield HybridRerankService(config)

    def test_get_failure_count(self, service):
        service.config._primary_failure_count = 7
        assert service.get_failure_count() == 7

    def test_reset_failure_count(self, service):
        service.config._primary_failure_count = 5
        service.reset_failure_count()
        assert service.config._primary_failure_count == 0

    @pytest.mark.asyncio
    async def test_close_without_fallback(self, service):
        await service.close()
        service.primary_service.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_with_fallback(self, service):
        mock_fallback = AsyncMock()
        mock_fallback.close = AsyncMock()
        service.fallback_service = mock_fallback
        await service.close()
        service.primary_service.close.assert_awaited_once()
        mock_fallback.close.assert_awaited_once()


class TestSingletonAndDI:
    """Tests for get_hybrid_service singleton (L564-566) and get_rerank_service DI (L590)"""

    def test_get_hybrid_service_returns_singleton(self):
        with patch(
            "agentic_layer.rerank_service._create_service_from_config"
        ) as mock_create:
            mock_create.return_value = MagicMock()
            import agentic_layer.rerank_service as mod

            mod._service_instance = None  # Reset singleton
            s1 = mod.get_hybrid_service()
            s2 = mod.get_hybrid_service()
            assert s1 is s2
            mod._service_instance = None  # Cleanup

    def test_get_rerank_service_returns_hybrid(self):
        with patch("agentic_layer.rerank_service.get_hybrid_service") as mock_get:
            mock_instance = MagicMock()
            mock_get.return_value = mock_instance
            from agentic_layer.rerank_service import get_rerank_service

            result = get_rerank_service()
            assert result == mock_instance
