"""
Application metadata management module

Provides read and write operations for application metadata, including service name and other information.
"""

from typing import Dict, Optional

# Application metadata storage
_APP_META_DATA: Dict = {}


def set_service_name(name: str) -> None:
    """
    Set service name

    Args:
        name: service name
    """
    _APP_META_DATA['service_name'] = name


def get_service_name() -> Optional[str]:
    """
    Get service name

    Returns:
        str: service name, returns None if not set
    """
    return _APP_META_DATA.get('service_name')


def set_meta_data(key: str, value: any) -> None:
    """
    Set metadata

    Args:
        key: metadata key
        value: metadata value
    """
    _APP_META_DATA[key] = value


def get_meta_data(key: str) -> Optional[any]:
    """
    Get metadata

    Args:
        key: metadata key

    Returns:
        any: metadata value, returns None if not exists
    """
    return _APP_META_DATA.get(key)


def get_all_meta_data() -> Dict:
    """
    Get all metadata

    Returns:
        Dict: a copy of all metadata
    """
    return _APP_META_DATA.copy()
