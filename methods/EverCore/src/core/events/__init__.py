# -*- coding: utf-8 -*-
"""
Event module

Provides application-level event publish/subscribe mechanism, supporting:
- Base event (BaseEvent): Base class for all business events, supports JSON/BSON serialization
- Event listener (EventListener): Abstract base class for event listeners
- Event publisher (ApplicationEventPublisher): Global event publisher

Usage examples:

1. Define an event:
    >>> from dataclasses import dataclass
    >>> from core.events import BaseEvent
    >>>
    >>> @dataclass
    ... class UserCreatedEvent(BaseEvent):
    ...     user_id: str
    ...     username: str
    ...
    ...     @classmethod
    ...     def from_dict(cls, data):
    ...         return cls(
    ...             event_id=data.get("event_id"),
    ...             created_at=data.get("created_at"),
    ...             user_id=data["user_id"],
    ...             username=data["username"],
    ...         )

2. Define a listener:
    >>> from core.di import component
    >>> from core.events import EventListener, BaseEvent
    >>>
    >>> @component("user_event_listener")
    ... class UserEventListener(EventListener):
    ...     def get_event_types(self):
    ...         return [UserCreatedEvent]
    ...
    ...     async def on_event(self, event: BaseEvent):
    ...         print(f"User created: {event.user_id}")

3. Publish an event:
    >>> from core.di import get_bean_by_type
    >>> from core.events import ApplicationEventPublisher
    >>>
    >>> publisher = get_bean_by_type(ApplicationEventPublisher)
    >>> await publisher.publish(UserCreatedEvent(user_id="123", username="alice"))
"""

from core.events.base_event import BaseEvent
from core.events.event_listener import EventListener
from core.events.event_publisher import ApplicationEventPublisher

__all__ = ['BaseEvent', 'EventListener', 'ApplicationEventPublisher']
