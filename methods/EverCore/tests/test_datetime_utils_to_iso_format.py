"""
测试 datetime_utils.to_iso_format 函数的 None 处理逻辑
"""

import datetime
import pytest
from zoneinfo import ZoneInfo

from common_utils.datetime_utils import to_iso_format, get_timezone


class TestToIsoFormatNoneHandling:
    """测试 to_iso_format 函数对 None 值的处理"""

    def test_none_with_allow_none_true_returns_none(self):
        """当 allow_none=True（默认）且传入 None 时，应返回 None"""
        result = to_iso_format(None)
        assert result is None

    def test_none_with_allow_none_true_explicit_returns_none(self):
        """当显式设置 allow_none=True 且传入 None 时，应返回 None"""
        result = to_iso_format(None, allow_none=True)
        assert result is None

    def test_none_with_allow_none_false_raises_value_error(self):
        """当 allow_none=False 且传入 None 时，应抛出 ValueError"""
        with pytest.raises(ValueError) as exc_info:
            to_iso_format(None, allow_none=False)

        # 验证错误信息
        assert "time_value cannot be None" in str(exc_info.value)
        assert "allow_none=False" in str(exc_info.value)


class TestToIsoFormatNormalCases:
    """测试 to_iso_format 函数的正常输入情况（确保修改没有破坏原有功能）"""

    def test_datetime_input(self):
        """测试 datetime 对象输入"""
        tz = get_timezone()
        dt = datetime.datetime(2025, 12, 5, 10, 30, 0, tzinfo=tz)
        result = to_iso_format(dt)

        assert result is not None
        assert "2025-12-05" in result
        assert "10:30:00" in result

    def test_datetime_input_with_allow_none_false(self):
        """测试 datetime 对象输入且 allow_none=False（不应影响正常输入）"""
        tz = get_timezone()
        dt = datetime.datetime(2025, 12, 5, 10, 30, 0, tzinfo=tz)
        result = to_iso_format(dt, allow_none=False)

        assert result is not None
        assert "2025-12-05" in result

    def test_timestamp_seconds_input(self):
        """测试秒级时间戳输入"""
        # 2024-12-05 10:30:00 UTC 的时间戳
        timestamp = 1733394600
        result = to_iso_format(timestamp)

        assert result is not None
        assert "2024-12-05" in result

    def test_timestamp_milliseconds_input(self):
        """测试毫秒级时间戳输入"""
        # 2024-12-05 10:30:00 UTC 的毫秒级时间戳
        timestamp_ms = 1733394600000
        result = to_iso_format(timestamp_ms)

        assert result is not None
        assert "2024-12-05" in result

    def test_string_input_passthrough(self):
        """测试字符串输入直接返回"""
        iso_str = "2025-12-05T10:30:00+00:00"
        result = to_iso_format(iso_str)

        assert result == iso_str

    def test_string_input_with_allow_none_false(self):
        """测试字符串输入且 allow_none=False"""
        iso_str = "2025-12-05T10:30:00+00:00"
        result = to_iso_format(iso_str, allow_none=False)

        assert result == iso_str

    def test_empty_string_returns_none(self):
        """测试空字符串返回 None"""
        result = to_iso_format("")
        assert result is None

    def test_negative_timestamp_returns_none(self):
        """测试负数时间戳返回 None"""
        result = to_iso_format(-1)
        assert result is None

    def test_zero_timestamp_returns_none(self):
        """测试零时间戳返回 None"""
        result = to_iso_format(0)
        assert result is None


class TestToIsoFormatEdgeCases:
    """测试 to_iso_format 函数的边界情况"""

    def test_float_timestamp(self):
        """测试浮点数时间戳"""
        # 2024-12-05 10:30:00.123 UTC 的浮点数时间戳
        timestamp = 1733394600.123
        result = to_iso_format(timestamp)

        assert result is not None
        assert "2024-12-05" in result

    def test_datetime_without_timezone(self):
        """测试不带时区的 datetime 对象（应自动添加时区）"""
        dt = datetime.datetime(2025, 12, 5, 10, 30, 0)
        result = to_iso_format(dt)

        assert result is not None
        # 应该包含时区信息
        assert "+" in result or "-" in result

    def test_unsupported_type_returns_none(self):
        """测试不支持的类型返回 None"""
        result = to_iso_format([1, 2, 3])  # type: ignore
        assert result is None
