import datetime
from zoneinfo import ZoneInfo
import os
from core.observation.logger import get_logger

logger = get_logger(__name__)


def get_timezone() -> ZoneInfo:
    """Get timezone from TZ env var (default: UTC)."""
    tz = os.getenv("TZ", "UTC")
    return ZoneInfo(tz)


timezone = get_timezone()


def get_now_with_timezone(tz: ZoneInfo = None) -> datetime.datetime:
    """Get current time with local timezone."""
    return datetime.datetime.now(tz=tz or timezone)


def to_timezone(dt: datetime.datetime, tz: ZoneInfo = None) -> datetime.datetime:
    """Convert datetime to specified timezone."""
    if tz is None:
        tz = timezone
    return dt.astimezone(tz)


def to_date_str(dt: datetime.datetime | None) -> str | None:
    """Convert datetime to ISO date string (YYYY-MM-DD format).

    Args:
        dt: Datetime object to convert.

    Returns:
        ISO date string (e.g. "2025-01-07"), or None if input is None.

    Example:
        >>> to_date_str(datetime.datetime(2025, 1, 7, 9, 15, 33))
        "2025-01-07"
    """
    if dt is None:
        return None
    return dt.date().isoformat()


def to_iso_format(
    time_value: datetime.datetime | int | float | str | None,
) -> str | None:
    """Convert time value to ISO format string with timezone.

    Supports: datetime, int/float (unix timestamp), str, None.

    Args:
        time_value: Time value to convert.

    Returns:
        ISO format string (e.g. 2025-09-16T20:20:06Z), or None.

    Raises:
        TypeError: If time_value is not a supported type.
        ValueError: If timestamp is invalid.
    """

    if time_value is None:
        return None

    value_type = type(time_value)

    if value_type is str:
        if not time_value:
            return None
        # Validate and parse ISO format string
        time_str = (
            time_value.replace("Z", "+00:00")
            if time_value.endswith("Z")
            else time_value
        )
        dt = datetime.datetime.fromisoformat(time_str)
    elif value_type in (int, float):
        if time_value <= 0:
            raise ValueError(f"Invalid timestamp: {time_value}. Must be positive.")
        dt = from_timestamp(time_value)
    elif value_type is datetime.datetime:
        dt = time_value
    else:
        raise TypeError(
            f"Unsupported type: {value_type.__name__}. "
            f"Expected: datetime, int, float, str, or None."
        )

    # Ensure timezone and convert to local
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone)
    return dt.astimezone(timezone).isoformat()


def from_timestamp(timestamp: int | float) -> datetime.datetime:
    """Convert unix timestamp to datetime. Auto-detects seconds vs milliseconds."""
    # >= 1e12 is milliseconds, < 1e12 is seconds
    if timestamp >= 1e12:
        timestamp_seconds = timestamp / 1000.0
    else:
        timestamp_seconds = timestamp
    return datetime.datetime.fromtimestamp(timestamp_seconds, tz=timezone)


def to_timestamp(dt: datetime.datetime) -> int:
    """Convert datetime to unix timestamp (seconds)."""
    return int(dt.timestamp())


def to_timestamp_ms(dt: datetime.datetime) -> int:
    """Convert datetime to unix timestamp (milliseconds)."""
    return int(dt.timestamp() * 1000)


def to_timestamp_ms_universal(time_value) -> int:
    """Convert any time format to milliseconds timestamp.

    Supports: int/float (timestamp), str (ISO format), datetime, None.
    Returns 0 on failure or None input.
    """
    try:
        if time_value is None:
            return 0

        if isinstance(time_value, (int, float)):
            # Auto-detect: >= 1e12 is ms, otherwise seconds
            if time_value >= 1e12:
                return int(time_value)
            return int(time_value * 1000)

        if isinstance(time_value, str):
            try:
                return to_timestamp_ms_universal(float(time_value))
            except ValueError:
                return to_timestamp_ms(from_iso_format(time_value))

        if isinstance(time_value, datetime.datetime):
            return to_timestamp_ms(time_value)

        return to_timestamp_ms_universal(str(time_value))

    except Exception as e:
        logger.error(
            "[DateTimeUtils] to_timestamp_ms_universal - Error converting %s: %s",
            time_value,
            str(e),
        )
        return 0


def _parse_datetime_core(
    time_value, target_timezone: ZoneInfo = None
) -> datetime.datetime:
    """
    Core datetime parsing logic. Raises exception on failure.

    Supported inputs:
        - datetime object (passed through)
        - ISO format string: "2025-09-15T13:11:15.588000", "2025-09-15T13:11:15.588000Z"
        - Space-separated string: "2025-01-07 09:15:33" (Python 3.11+)
        - With timezone offset: "2025-09-15T13:11:15+00:00"

    Args:
        time_value: datetime object or time string
        target_timezone: Timezone for naive datetime (default: TZ env variable)

    Returns:
        Timezone-aware datetime object

    Raises:
        ValueError: If parsing fails
    """
    # Handle datetime object
    if isinstance(time_value, datetime.datetime):
        dt = time_value
    elif isinstance(time_value, (int, float)):
        # Handle numeric timestamps (auto-detects seconds vs milliseconds)
        dt = from_timestamp(time_value)
    elif isinstance(time_value, str):
        time_str = time_value.strip()
        # Handle "Z" suffix (UTC)
        if time_str.endswith("Z"):
            time_str = time_str[:-1] + "+00:00"
        # Python 3.11+ fromisoformat supports space-separated format
        dt = datetime.datetime.fromisoformat(time_str)
    else:
        # Other types: convert to string first
        time_str = str(time_value).strip()
        if time_str.endswith("Z"):
            time_str = time_str[:-1] + "+00:00"
        dt = datetime.datetime.fromisoformat(time_str)

    # Add timezone if naive
    if dt.tzinfo is None:
        tz = target_timezone or get_timezone()
        dt_localized = dt.replace(tzinfo=tz)
    else:
        dt_localized = dt

    # Convert to system timezone
    return dt_localized.astimezone(get_timezone())


def from_iso_format(
    create_time, target_timezone: ZoneInfo = None, strict: bool = False
) -> datetime.datetime:
    """
    Parse datetime string or object to timezone-aware datetime.

    Args:
        create_time: datetime object or time string
        target_timezone: Timezone for naive datetime (default: TZ env variable)
        strict: If True, raises ValueError on failure (for data import).
                If False (default), returns current time on failure (for runtime conversion).

    Supported formats:
        - datetime object (passed through)
        - "2025-01-07 09:15:33" (space-separated)
        - "2025-01-07T09:15:33" (ISO T-separated)
        - "2025-01-07 09:15:33.123456" (with microseconds)
        - "2025-01-07T09:15:33+00:00" (with timezone)
        - "2025-01-07T09:15:33Z" (UTC)

    Returns:
        Timezone-aware datetime object. Returns current time if parsing fails (when strict=False).

    Raises:
        ValueError: If strict=True and parsing fails

    Example:
        >>> from_iso_format("2025-01-07 09:15:33")
        datetime.datetime(2025, 1, 7, 9, 15, 33, tzinfo=ZoneInfo('UTC'))

        >>> from_iso_format("invalid", strict=True)
        ValueError: ...
    """
    if strict:
        # Strict mode: raise exception on failure
        return _parse_datetime_core(create_time, target_timezone)
    else:
        # Lenient mode: return current time on failure
        try:
            return _parse_datetime_core(create_time, target_timezone)
        except Exception as e:
            logger.error(
                "[DateTimeUtils] from_iso_format - Error converting time: %s", str(e)
            )
            return get_now_with_timezone()
