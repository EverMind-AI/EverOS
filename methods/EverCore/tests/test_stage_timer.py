"""Unit tests for StageTimer JSON output format.

Tests cover:
- Sequential, nested, and parallel stage dict structure
- Empty timer omits 'stages' key
- total_ms accuracy
- log_summary JSON format
- JSON roundtrip serialization
- Slow threshold flag
- Summary keys completeness
- timed() / timed_parallel() convenience wrappers
- get_current_timer() / start_timer() / log_timer() context resolution
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import patch

from core.context.context import clear_current_app_info, set_current_app_info
from core.observation.stage_timer import (
    StageTimer,
    get_current_timer,
    log_timer,
    start_timer,
    timed,
    timed_parallel,
)


class TestStageTimer:
    """Tests for StageTimer core functionality with JSON output."""

    def test_sequential_stages(self):
        """Sequential stages produce two entries in stages list."""
        timer = StageTimer("/test")
        with timer.stage("a"):
            time.sleep(0.01)
        with timer.stage("b"):
            time.sleep(0.01)
        s = timer.summary()
        assert "stages" in s
        assert s["stages"][0]["name"] == "a"
        assert s["stages"][1]["name"] == "b"
        assert isinstance(s["stages"][0]["duration_ms"], int)
        assert isinstance(s["stages"][1]["duration_ms"], int)

    def test_nested_stages(self):
        """Nested stages produce child stages list."""
        timer = StageTimer("/test")
        with timer.stage("parent"):
            with timer.stage("child"):
                time.sleep(0.01)
        s = timer.summary()
        parent = s["stages"][0]
        assert parent["name"] == "parent"
        assert "stages" in parent
        assert parent["stages"][0]["name"] == "child"

    def test_parallel_stages_with_gather(self):
        """Parallel via asyncio.gather produces parallel group with children."""

        async def run():
            timer = StageTimer("/test")

            async def branch_a():
                with timer.stage("a"):
                    await asyncio.sleep(0.01)

            async def branch_b():
                with timer.stage("b"):
                    await asyncio.sleep(0.01)

            with timer.parallel("group"):
                await asyncio.gather(branch_a(), branch_b())
            return timer.summary()

        s = asyncio.run(run())
        group = s["stages"][0]
        assert group["name"] == "group"
        assert group.get("parallel") is True
        child_names = {child["name"] for child in group["stages"]}
        assert child_names == {"a", "b"}

    def test_nested_sequential_inside_parallel(self):
        """Sequential stages nested inside parallel branches."""

        async def run():
            timer = StageTimer("/test")

            async def episode_branch():
                with timer.stage("episode"):
                    with timer.stage("es"):
                        await asyncio.sleep(0.005)
                    with timer.stage("milvus"):
                        await asyncio.sleep(0.005)

            async def profile_branch():
                with timer.stage("profile"):
                    await asyncio.sleep(0.005)

            with timer.parallel("retrieval"):
                await asyncio.gather(episode_branch(), profile_branch())
            return timer.summary()

        s = asyncio.run(run())
        retrieval = s["stages"][0]
        assert retrieval["name"] == "retrieval"
        assert retrieval.get("parallel") is True

        child_names = {child["name"] for child in retrieval["stages"]}
        assert "episode" in child_names
        assert "profile" in child_names

        # Find episode branch and verify nested sequential stages
        episode = next(c for c in retrieval["stages"] if c["name"] == "episode")
        assert "stages" in episode
        nested_names = [c["name"] for c in episode["stages"]]
        assert nested_names == ["es", "milvus"]

    def test_single_branch_parallel(self):
        """Single child parallel still has parallel flag."""
        timer = StageTimer("/test")
        with timer.parallel("group"):
            with timer.stage("only"):
                time.sleep(0.01)
        s = timer.summary()
        group = s["stages"][0]
        assert group.get("parallel") is True
        assert group["stages"][0]["name"] == "only"

    def test_empty_timer(self):
        """No stages -> 'stages' key absent from summary."""
        timer = StageTimer("/test")
        s = timer.summary()
        assert "stages" not in s

    def test_total_ms_accuracy(self):
        """total_ms should approximate the sum of sequential stage durations."""
        timer = StageTimer("/test")
        with timer.stage("a"):
            time.sleep(0.05)
        with timer.stage("b"):
            time.sleep(0.05)
        s = timer.summary()
        assert s["total_ms"] >= 90, f"Expected >= 90ms, got {s['total_ms']}ms"

    def test_log_summary_format(self):
        """Verify log outputs [stage_timer] followed by valid JSON."""
        timer = StageTimer("/api/v1/memories/search")
        with timer.stage("retrieval"):
            time.sleep(0.01)

        with patch("core.observation.stage_timer.logger") as mock_logger:
            timer.log_summary()
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            fmt = call_args[0][0]
            assert "[stage_timer]" in fmt
            # Second positional arg is the JSON string
            json_str = call_args[0][1]
            parsed = json.loads(json_str)
            assert parsed["endpoint"] == "/api/v1/memories/search"
            assert "total_ms" in parsed
            assert parsed["stages"][0]["name"] == "retrieval"

    def test_json_roundtrip(self):
        """summary() output survives JSON roundtrip."""

        async def run():
            timer = StageTimer("/test")
            with timer.stage("outer"):
                with timer.parallel("par"):

                    async def x():
                        with timer.stage("x"):
                            await asyncio.sleep(0.005)

                    async def y():
                        with timer.stage("y"):
                            await asyncio.sleep(0.005)

                    await asyncio.gather(x(), y())
                with timer.stage("final"):
                    await asyncio.sleep(0.005)
            return timer.summary()

        s = asyncio.run(run())
        json_str = json.dumps(s, ensure_ascii=False, separators=(",", ":"))
        restored = json.loads(json_str)
        assert restored == s

    def test_slow_threshold_exceeded(self):
        """When total_ms >= threshold, 'slow' flag appears in log output."""
        timer = StageTimer("/test")
        with timer.stage("work"):
            time.sleep(0.1)

        with patch("core.observation.stage_timer._SLOW_THRESHOLD_MS", 50):
            with patch("core.observation.stage_timer.logger") as mock_logger:
                timer.log_summary()
                json_str = mock_logger.info.call_args[0][1]
                parsed = json.loads(json_str)
                assert parsed.get("slow") is True

    def test_slow_threshold_not_exceeded(self):
        """When total_ms < threshold, 'slow' key is absent."""
        timer = StageTimer("/test")
        with timer.stage("work"):
            time.sleep(0.01)

        with patch("core.observation.stage_timer._SLOW_THRESHOLD_MS", 1000):
            with patch("core.observation.stage_timer.logger") as mock_logger:
                timer.log_summary()
                json_str = mock_logger.info.call_args[0][1]
                parsed = json.loads(json_str)
                assert "slow" not in parsed

    def test_summary_keys_completeness_with_stages(self):
        """Non-empty timer has exactly {endpoint, total_ms, stages} keys."""
        timer = StageTimer("/test")
        with timer.stage("a"):
            time.sleep(0.005)
        s = timer.summary()
        assert set(s.keys()) == {"endpoint", "total_ms", "stages"}

    def test_summary_keys_completeness_empty(self):
        """Empty timer has no 'stages' or 'trace' keys."""
        timer = StageTimer("/test")
        s = timer.summary()
        assert "stages" not in s
        assert "trace" not in s
        assert set(s.keys()) == {"endpoint", "total_ms"}


class TestTimedConvenience:
    """Tests for timed(), timed_parallel(), and get_current_timer()."""

    def test_timed_no_op_without_timer(self):
        """Silent no-op when no timer in context."""
        with timed("anything"):
            pass

    def test_timed_records_stage(self):
        """Records stage when timer is in app_info_context."""
        timer = StageTimer("/test")
        token = set_current_app_info({"stage_timer": timer})
        try:
            with timed("work"):
                time.sleep(0.01)
            s = timer.summary()
            assert s["stages"][0]["name"] == "work"
        finally:
            clear_current_app_info(token)

    def test_timed_parallel_marks_parallel(self):
        """timed_parallel() produces parallel group in dict."""
        timer = StageTimer("/test")
        token = set_current_app_info({"stage_timer": timer})
        try:
            with timed_parallel("sources"):
                with timed("es"):
                    time.sleep(0.005)
                with timed("milvus"):
                    time.sleep(0.005)
            s = timer.summary()
            group = s["stages"][0]
            assert group["name"] == "sources"
            assert group.get("parallel") is True
            child_names = [c["name"] for c in group["stages"]]
            assert "es" in child_names
            assert "milvus" in child_names
        finally:
            clear_current_app_info(token)

    def test_timed_nesting_across_functions(self):
        """Nesting works across function boundaries."""
        timer = StageTimer("/test")
        token = set_current_app_info({"stage_timer": timer})
        try:

            def inner_work():
                with timed("inner"):
                    time.sleep(0.005)

            with timed("outer"):
                inner_work()

            s = timer.summary()
            outer = s["stages"][0]
            assert outer["name"] == "outer"
            assert outer["stages"][0]["name"] == "inner"
        finally:
            clear_current_app_info(token)

    def test_get_current_timer(self):
        """Returns timer from context or None."""
        assert get_current_timer() is None

        timer = StageTimer("/test")
        token = set_current_app_info({"stage_timer": timer})
        try:
            assert get_current_timer() is timer
        finally:
            clear_current_app_info(token)

        assert get_current_timer() is None

    def test_start_timer_creates_timer_in_context(self):
        """start_timer() stores a StageTimer in app_info context."""
        token = set_current_app_info({"request_id": "test-123"})
        try:
            start_timer("add")
            timer = get_current_timer()
            assert timer is not None
            assert timer._endpoint == "add"
        finally:
            clear_current_app_info(token)

    def test_start_timer_no_op_without_context(self):
        """start_timer() is a no-op when app_info is not set."""
        start_timer("add")
        assert get_current_timer() is None

    def test_log_timer_logs_summary(self):
        """log_timer() calls log_summary with JSON on the current timer."""
        timer = StageTimer("/test")
        with timer.stage("a"):
            time.sleep(0.005)
        token = set_current_app_info({"stage_timer": timer})
        try:
            with patch("core.observation.stage_timer.logger") as mock_logger:
                log_timer()
                mock_logger.info.assert_called_once()
                fmt = mock_logger.info.call_args[0][0]
                assert "[stage_timer]" in fmt
                json_str = mock_logger.info.call_args[0][1]
                parsed = json.loads(json_str)
                assert "endpoint" in parsed
        finally:
            clear_current_app_info(token)

    def test_log_timer_no_op_without_timer(self):
        """log_timer() is a no-op when no timer in context."""
        log_timer()

    def test_timed_parallel_no_op_without_timer(self):
        """timed_parallel() is a no-op when no timer in context."""
        with timed_parallel("group"):
            pass

    def test_call_llm_nests_under_parent(self):
        """call_llm auto-instrumentation nests correctly under business stage."""
        timer = StageTimer("/test")
        token = set_current_app_info({"stage_timer": timer})
        try:
            with timed("detect_boundaries"):
                with timed("call_llm"):
                    time.sleep(0.01)
            s = timer.summary()
            parent = s["stages"][0]
            assert parent["name"] == "detect_boundaries"
            assert parent["stages"][0]["name"] == "call_llm"
        finally:
            clear_current_app_info(token)
