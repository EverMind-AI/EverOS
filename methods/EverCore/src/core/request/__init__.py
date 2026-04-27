# -*- coding: utf-8 -*-
"""
Request module

Provides request-related utilities:
- AppLogicProvider: Application logic provider interface for request lifecycle hooks
- log_request: Decorator for logging request information as events for replay
- RequestHistoryEvent: Event class containing complete request information
- RequestHistoryConfig: Configuration interface for enabling/disabling request history
- is_request_history_enabled: Utility function to check if request history is enabled (cached)
"""

from core.request.app_logic_provider import AppLogicProvider, AppLogicProviderImpl
from core.request.request_history_config import (
    RequestHistoryConfig,
    DefaultRequestHistoryConfig,
    is_request_history_enabled,
    clear_request_history_cache,
    get_request_history_config,
)
from core.request.request_history_decorator import log_request, log_request_default
from core.request.request_history_event import RequestHistoryEvent

__all__ = [
    # App logic provider
    'AppLogicProvider',
    'AppLogicProviderImpl',
    # Request history config
    'RequestHistoryConfig',
    'DefaultRequestHistoryConfig',
    'is_request_history_enabled',
    'clear_request_history_cache',
    'get_request_history_config',
    # Request history decorator and event
    'log_request',
    'log_request_default',
    'RequestHistoryEvent',
]

