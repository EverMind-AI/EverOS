# -*- coding: utf-8 -*-
"""
Application event publisher module

Provides a global event publishing mechanism, supporting asynchronous concurrent dispatch of events to multiple listeners.
Automatically discovers and registers all EventListener implementations through the DI container.
"""

import asyncio
from typing import Dict, List, Set, Type, Optional

from core.di import service, get_beans_by_type
from core.events.base_event import BaseEvent
from core.events.event_listener import EventListener
from core.observation.logger import get_logger


logger = get_logger(__name__)


@service("application_event_publisher")
class ApplicationEventPublisher:
    """
    Application event publisher

    Global singleton service responsible for dispatching events to corresponding listeners.
    Automatically discovers all EventListener implementations through the DI container and builds a mapping from event types to listeners.

    Features:
    - Lazy loading: builds the listener mapping only when the first event is published
    - Asynchronous concurrency: uses asyncio.gather to concurrently invoke all matching listeners
    - Error isolation: exceptions from individual listeners do not affect others
    - Refreshable: supports dynamic refreshing of listener mappings

    Usage:
    1. Obtain instance via DI:
        >>> from core.di import get_bean_by_type
        >>> publisher = get_bean_by_type(ApplicationEventPublisher)

    2. Publish event:
        >>> await publisher.publish(UserCreatedEvent(user_id="123"))

    3. Synchronous publishing (in non-async contexts):
        >>> publisher.publish_sync(UserCreatedEvent(user_id="123"))
    """

    def __init__(self):
        """Initialize event publisher"""
        # Mapping from event type to list of listeners
        self._event_listeners_map: Dict[Type[BaseEvent], List[EventListener]] = {}
        # Whether initialized
        self._initialized: bool = False
        # All listener instances
        self._listeners: List[EventListener] = []

    def _ensure_initialized(self) -> None:
        """
        Ensure listener mapping is initialized

        Lazy loading mechanism: obtains all listeners from DI container and builds mapping only on first call.
        """
        if self._initialized:
            return

        self._build_listener_mapping()
        self._initialized = True

    def _build_listener_mapping(self) -> None:
        """
        Build mapping from event types to listeners

        Retrieve all EventListener instances from the DI container,
        then build a mapping table based on the event types declared by each listener.
        """
        # Clear existing mapping
        self._event_listeners_map.clear()
        self._listeners.clear()

        # Get all EventListener implementations from DI container
        try:
            listeners = get_beans_by_type(EventListener)
        except Exception as e:
            logger.warning(f"Failed to get EventListener instances: {e}")
            listeners = []

        self._listeners = listeners

        # Build mapping from event type to listeners
        for listener in listeners:
            listener_name = listener.get_listener_name()
            event_types = listener.get_event_types()

            logger.debug(
                f"Registering listener [{listener_name}], listening to event types: {[et.__name__ for et in event_types]}"
            )

            for event_type in event_types:
                if event_type not in self._event_listeners_map:
                    self._event_listeners_map[event_type] = []
                self._event_listeners_map[event_type].append(listener)

        # Log initialization completion
        total_listeners = len(listeners)
        total_event_types = len(self._event_listeners_map)
        logger.info(
            f"Event publisher initialization completed: {total_listeners} listeners, {total_event_types} event types"
        )

    def refresh(self) -> None:
        """
        Refresh listener mapping

        Call this method to refresh the mapping after new listeners are registered into the DI container.
        """
        self._initialized = False
        self._ensure_initialized()
        logger.info("Event publisher listener mapping has been refreshed")

    def get_listeners_for_event(
        self, event_type: Type[BaseEvent]
    ) -> List[EventListener]:
        """
        Get all listeners for a specified event type

        Args:
            event_type: Event type

        Returns:
            List[EventListener]: List of all listeners for this event type
        """
        self._ensure_initialized()
        return self._event_listeners_map.get(event_type, [])

    def get_all_listeners(self) -> List[EventListener]:
        """
        Get all registered listeners

        Returns:
            List[EventListener]: List of all listeners
        """
        self._ensure_initialized()
        return self._listeners.copy()

    def get_registered_event_types(self) -> Set[Type[BaseEvent]]:
        """
        Get all event types that have registered listeners

        Returns:
            Set[Type[BaseEvent]]: Set of event types
        """
        self._ensure_initialized()
        return set(self._event_listeners_map.keys())

    async def publish(self, event: BaseEvent) -> None:
        """
        Asynchronously publish event

        Dispatch the event to all listeners that listen to this event type.
        Uses asyncio.gather to concurrently invoke all listeners, improving efficiency for IO-intensive operations.

        Exceptions from individual listeners do not affect execution of others,
        and all exceptions are logged.

        Args:
            event: Event object to publish
        """
        self._ensure_initialized()

        event_type = type(event)
        event_type_name = event.event_type()
        listeners = self._event_listeners_map.get(event_type, [])

        if not listeners:
            logger.debug(
                f"No listeners for event [{event_type_name}], skipping publish"
            )
            return

        logger.debug(
            f"Publishing event [{event_type_name}] (id={event.event_id}), {len(listeners)} listeners"
        )

        # Create coroutine tasks for all listeners
        async def safe_invoke(listener: EventListener) -> Optional[Exception]:
            """
            Safely invoke listener, catch exceptions to avoid affecting others

            Returns:
                Exception object if occurred, otherwise None
            """
            try:
                await listener.on_event(event)
                return None
            except Exception as e:
                listener_name = listener.get_listener_name()
                logger.error(
                    f"Listener [{listener_name}] encountered exception when processing event [{event_type_name}]: {e}",
                    exc_info=True,
                )
                return e

        # Concurrently execute all listeners
        tasks = [safe_invoke(listener) for listener in listeners]
        results = await asyncio.gather(*tasks)

        # Count execution results
        errors = [r for r in results if r is not None]
        if errors:
            logger.warning(
                f"Event [{event_type_name}] publishing completed, "
                f"success: {len(listeners) - len(errors)}, failure: {len(errors)}"
            )
        else:
            logger.debug(
                f"Event [{event_type_name}] publishing completed, all {len(listeners)} listeners executed successfully"
            )

    def publish_sync(self, event: BaseEvent) -> None:
        """
        Synchronously publish event

        Use this method to publish events in non-async contexts.
        Internally creates or uses an existing event loop to execute asynchronous publishing.

        Note: If already in an async context, prefer using the `publish()` method.

        Args:
            event: Event object to publish
        """
        try:
            # Try to get the currently running event loop
            loop = asyncio.get_running_loop()
            # If in async context, create a task
            loop.create_task(self.publish(event))
        except RuntimeError:
            # No running event loop, create and run a new one
            asyncio.run(self.publish(event))

    async def publish_batch(self, events: List[BaseEvent]) -> None:
        """
        Publish multiple events in batch

        Concurrently publish multiple events to improve efficiency of batch operations.

        Args:
            events: List of events to publish
        """
        if not events:
            return

        logger.debug(f"Batch publishing {len(events)} events")

        # Concurrently publish all events
        tasks = [self.publish(event) for event in events]
        await asyncio.gather(*tasks)

        logger.debug(f"Batch publishing completed, total {len(events)} events")

    def __repr__(self) -> str:
        """Return string representation of the object"""
        self._ensure_initialized()
        return (
            f"ApplicationEventPublisher("
            f"listeners={len(self._listeners)}, "
            f"event_types={len(self._event_listeners_map)}"
            f")"
        )
