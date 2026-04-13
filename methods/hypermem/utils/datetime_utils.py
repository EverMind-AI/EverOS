"""
DateTime utilities for HyperMem
Simplified version without external dependencies
"""

import datetime
from zoneinfo import ZoneInfo
import os
import logging

logger = logging.getLogger(__name__)


def get_timezone() -> ZoneInfo:
    """
    Get the timezone
    """
    tz = os.getenv("TZ", "Asia/Shanghai")
    return ZoneInfo(tz)


timezone = get_timezone()


def get_now_with_timezone() -> datetime.datetime:
    """
    Get the current time using the local timezone
    return datetime.datetime(2025, 9, 16, 20, 17, 41, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))
    """
    return datetime.datetime.now(tz=timezone)


def to_timezone(dt: datetime.datetime, tz: ZoneInfo = None) -> datetime.datetime:
    """
    Convert a datetime object to a specified timezone
    """
    if tz is None:
        tz = timezone
    return dt.astimezone(tz)


def to_iso_format(dt: datetime.datetime) -> str:
    """
    Convert a datetime object to an ISO format string (with timezone)
    return 2025-09-16T20:20:06.517301+08:00
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        # If missing, since the default uses the TZ environment variable, manually set the timezone
        dt = dt.replace(tzinfo=timezone)
    # If it's UTC or similar, convert to local timezone
    return dt.astimezone(timezone).isoformat()


def from_timestamp(timestamp: int | float) -> datetime.datetime:
    """
    Convert a timestamp to a datetime object, automatically detecting second or millisecond precision

    Args:
        timestamp: Timestamp, supports second-level (10 digits) and millisecond-level (13 digits)

    Returns:
        datetime.datetime(2025, 9, 16, 20, 17, 41, tzinfo=zoneinfo.ZoneInfo(key='Asia/Shanghai'))
    """
    # Automatically detect timestamp precision
    # Millisecond timestamps are typically >= 1e12 (1000000000000), about 13 digits
    # Second timestamps are typically < 1e12, about 10 digits
    if timestamp >= 1e12:
        # Millisecond timestamp, convert to seconds
        timestamp_seconds = timestamp / 1000.0
    else:
        # Second-level timestamp, use directly
        timestamp_seconds = timestamp
    
    return datetime.datetime.fromtimestamp(timestamp_seconds, tz=timezone)


def to_timestamp(dt: datetime.datetime) -> int:
    """
    Convert a datetime object to a timestamp in seconds
    return 1758025061
    """
    return int(dt.timestamp())


def to_timestamp_ms(dt: datetime.datetime) -> int:
    """
    Convert a datetime object to a millisecond-level timestamp
    return 1758025061123
    """
    return int(dt.timestamp() * 1000)


def to_timestamp_ms_universal(time_value) -> int:
    """
    Universal time format to millisecond timestamp conversion function
    Supports multiple input formats:
    - int/float: timestamp (automatically detects second or millisecond precision)
    - str: ISO format time string
    - datetime object
    - None: returns 0

    Args:
        time_value: Time value in various formats

    Returns:
        int: Millisecond-level timestamp, returns 0 on failure
    """
    try:
        if time_value is None:
            return 0
            
        # Handle numeric types (timestamps)
        if isinstance(time_value, (int, float)):
            # Automatically detect timestamp precision
            if time_value >= 1e12:
                # Millisecond timestamp, return directly
                return int(time_value)
            else:
                # Second-level timestamp, convert to milliseconds
                return int(time_value * 1000)
        
        # Handle string types
        if isinstance(time_value, str):
            # First try to parse as a number
            try:
                numeric_value = float(time_value)
                return to_timestamp_ms_universal(numeric_value)
            except ValueError:
                # Not a number, try to parse as an ISO format time string
                dt = from_iso_format(time_value)
                return to_timestamp_ms(dt)
        
        # Handle datetime objects
        if isinstance(time_value, datetime.datetime):
            return to_timestamp_ms(time_value)
            
        # Other types, try converting to string and then parsing
        return to_timestamp_ms_universal(str(time_value))
        
    except Exception as e:
        logger.error("[DateTimeUtils] to_timestamp_ms_universal - Error converting time value %s: %s", time_value, str(e))
        return 0


def from_iso_format(create_time, target_timezone: ZoneInfo = None) -> datetime.datetime:
    """
    Convert a time value to a timezone-aware datetime object

    Args:
        create_time: Time object or string, e.g., a datetime object or "2025-09-15T13:11:15.588000"
        target_timezone: Timezone object; if None, uses the TZ environment variable

    Returns:
        A timezone-aware datetime object, defaults to the configured timezone
    """
    try:
        # Handle different input types
        if isinstance(create_time, datetime.datetime):
            # If already a datetime object, use directly
            dt = create_time
        elif isinstance(create_time, str):
            # If it's a string, parse it into a datetime object
            dt = datetime.datetime.fromisoformat(create_time)
        else:
            # Other types, try converting to string and then parsing
            dt = datetime.datetime.fromisoformat(str(create_time))
        
        # If the datetime object has no timezone info, default to the specified timezone
        if dt.tzinfo is None:
            # Use the specified timezone, defaults to the configured timezone
            tz = target_timezone or get_timezone()
            dt_localized = dt.replace(tzinfo=tz)
        else:
            # If timezone info already exists, use it directly
            dt_localized = dt
        
        # Uniformly convert to the timezone consistent with get_timezone()
        return dt_localized.astimezone(get_timezone())
        
    except Exception as e:
        # If conversion fails, return the current time with the configured timezone
        logger.error("[DateTimeUtils] from_iso_format - Error converting time: %s", str(e))
        return get_now_with_timezone()
