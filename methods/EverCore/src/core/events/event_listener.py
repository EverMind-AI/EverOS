# -*- coding: utf-8 -*-
"""
Event listener module

Provides an abstract base class for event listeners, supporting declarative registration of event types to listen for.
Business listeners should inherit from this base class and implement the `get_event_types` and `on_event` methods.
"""

from abc import ABC, abstractmethod
from typing import List, Set, Type

from core.events.base_event import BaseEvent


class EventListener(ABC):
    """
    Abstract base class for event listeners

    Business listeners should inherit from this class and implement the following methods:
    - `get_event_types()`: Returns a list of event types to listen for
    - `on_event(event)`: Handles the event-specific logic (asynchronous method)

    Listeners will be automatically discovered and registered by ApplicationEventPublisher.
    It is recommended to use the @component or @service decorator to register the listener into the DI container.

    Example:
        >>> from core.di import component
        >>>
        >>> @component("user_event_listener")
        ... class UserEventListener(EventListener):
        ...     def get_event_types(self) -> List[Type[BaseEvent]]:
        ...         return [UserCreatedEvent, UserUpdatedEvent]
        ...
        ...     async def on_event(self, event: BaseEvent) -> None:
        ...         if isinstance(event, UserCreatedEvent):
        ...             await self._handle_user_created(event)
        ...         elif isinstance(event, UserUpdatedEvent):
        ...             await self._handle_user_updated(event)
    """

    @abstractmethod
    def get_event_types(self) -> List[Type[BaseEvent]]:
        """
        Get the list of event types to listen for

        Returns a list of event types that this listener is interested in. When events of these types are published,
        the listener's `on_event` method will be called.

        Returns:
            List[Type[BaseEvent]]: List of event types to listen for

        Example:
            >>> def get_event_types(self) -> List[Type[BaseEvent]]:
            ...     return [UserCreatedEvent, OrderCreatedEvent]
        """
        pass

    @abstractmethod
    async def on_event(self, event: BaseEvent) -> None:
        """
        Handle the event

        This method is called asynchronously when a listened event is published.
        Implement this method to handle specific business logic.

        Note:
        - This method is asynchronous and can perform IO operations
        - Multiple listeners execute concurrently without blocking each other
        - It is recommended to catch exceptions within this method to avoid affecting other listeners

        Args:
            event: The received event object
        """
        pass

    def get_listener_name(self) -> str:
        """
        Get the listener name

        Returns the class name by default. Subclasses can override this method to customize the name.

        Returns:
            str: Listener name
        """
        return self.__class__.__name__

    def get_event_type_set(self) -> Set[Type[BaseEvent]]:
        """
        Get the set of event types to listen for (for internal use)

        Returns a set of event types for fast lookup.

        Returns:
            Set[Type[BaseEvent]]: Set of event types
        """
        return set(self.get_event_types())
