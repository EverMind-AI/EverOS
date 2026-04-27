# -*- coding: utf-8 -*-
"""
Request history event module

Defines the event class for recording HTTP request information.
Used for request replay functionality.

Note: This event contains raw request data without parsing.
Enterprise version is responsible for parsing and processing the data.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Type, TypeVar

from core.events.base_event import BaseEvent


T = TypeVar('T', bound='RequestHistoryEvent')


@dataclass
class RequestHistoryEvent(BaseEvent):
    """
    Request history event

    Records raw HTTP request information for replay functionality.
    Contains minimal parsed data - enterprise code handles further processing.

    Attributes:
        version: Code version from project_meta.py
        endpoint_name: Name of the endpoint function
        controller_name: Name of the controller class
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        url: Full URL including scheme, host, path, and query string
        headers: Raw request headers as dictionary (unfiltered)
        body: Raw request body as string
        client_host: Client IP address
        client_port: Client port number

    Example:
        >>> event = RequestHistoryEvent(
        ...     version="1.0.0",
        ...     endpoint_name="create_user",
        ...     controller_name="UserController",
        ...     method="POST",
        ...     url="http://localhost:8000/api/v0/users?page=1",
        ...     headers={"Content-Type": "application/json"},
        ...     body='{"name": "John"}',
        ... )
    """

    # Version info
    version: str = ""

    # Endpoint info
    endpoint_name: Optional[str] = None
    controller_name: Optional[str] = None

    # Raw request data (unprocessed)
    method: str = ""
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[str] = None

    # Client info
    client_host: Optional[str] = None
    client_port: Optional[int] = None

    @classmethod
    def event_type(cls) -> str:
        """Get event type name"""
        return "RequestHistoryEvent"

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """
        Create RequestHistoryEvent from dictionary

        Args:
            data: Dictionary containing event data

        Returns:
            RequestHistoryEvent instance
        """
        return cls(
            # Base event fields
            event_id=data.get("event_id", ""),
            created_at=data.get("created_at", ""),
            # Version info
            version=data.get("version", ""),
            # Endpoint info
            endpoint_name=data.get("endpoint_name"),
            controller_name=data.get("controller_name"),
            # Raw request data
            method=data.get("method", ""),
            url=data.get("url", ""),
            headers=data.get("headers", {}),
            body=data.get("body"),
            # Client info
            client_host=data.get("client_host"),
            client_port=data.get("client_port"),
        )

    def __repr__(self) -> str:
        """Return string representation"""
        return (
            f"RequestHistoryEvent("
            f"event_id={self.event_id!r}, "
            f"version={self.version!r}, "
            f"method={self.method!r}, "
            f"endpoint_name={self.endpoint_name!r}"
            f")"
        )
