"""
Message Group Queue Manager

Provides fixed-number queue management with hash-based routing to solve blocking issues in Kafka message processing.
Supports message delivery, consumption, statistics, monitoring, and other features.
"""

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from core.observation.logger import get_logger
from common_utils.datetime_utils import get_now_with_timezone, to_iso_format

logger = get_logger(__name__)


class ShutdownMode(Enum):
    """Shutdown mode enumeration"""

    SOFT = "soft"  # Soft shutdown: check for remaining messages with delay time control
    HARD = "hard"  # Hard shutdown: shut down immediately, record number of unprocessed messages


@dataclass
class ShutdownState:
    """Shutdown state"""

    is_shutting_down: bool = False
    first_soft_shutdown_time: Optional[float] = None
    max_delay_seconds: Optional[float] = None

    def reset(self):
        """Reset shutdown state"""
        self.is_shutting_down = False
        self.first_soft_shutdown_time = None
        self.max_delay_seconds = None


@dataclass
class TimeWindowStats:
    """Time window statistics"""

    delivered_1min: int = 0
    consumed_1min: int = 0
    delivered_1hour: int = 0
    consumed_1hour: int = 0


@dataclass
class QueueStats:
    """Queue statistics"""

    queue_id: int
    current_size: int
    total_delivered: int = 0
    total_consumed: int = 0
    last_deliver_time: Optional[str] = None
    last_consume_time: Optional[str] = None
    # Time window statistics
    time_window_stats: TimeWindowStats = field(default_factory=TimeWindowStats)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "queue_id": self.queue_id,
            "current_size": self.current_size,
            "total_delivered": self.total_delivered,
            "total_consumed": self.total_consumed,
            "last_deliver_time": self.last_deliver_time,
            "last_consume_time": self.last_consume_time,
            "delivered_1min": self.time_window_stats.delivered_1min,
            "consumed_1min": self.time_window_stats.consumed_1min,
            "delivered_1hour": self.time_window_stats.delivered_1hour,
            "consumed_1hour": self.time_window_stats.consumed_1hour,
        }


@dataclass
class ManagerStats:
    """Manager overall statistics"""

    total_queues: int
    total_current_messages: int
    total_delivered_messages: int = 0
    total_consumed_messages: int = 0
    total_rejected_messages: int = 0
    start_time: str = field(
        default_factory=lambda: to_iso_format(get_now_with_timezone())
    )
    uptime_seconds: float = 0
    # Time window statistics
    time_window_stats: TimeWindowStats = field(default_factory=TimeWindowStats)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "total_queues": self.total_queues,
            "total_current_messages": self.total_current_messages,
            "total_delivered_messages": self.total_delivered_messages,
            "total_consumed_messages": self.total_consumed_messages,
            "total_rejected_messages": self.total_rejected_messages,
            "start_time": self.start_time,
            "uptime_seconds": self.uptime_seconds,
            "delivered_1min": self.time_window_stats.delivered_1min,
            "consumed_1min": self.time_window_stats.consumed_1min,
            "delivered_1hour": self.time_window_stats.delivered_1hour,
            "consumed_1hour": self.time_window_stats.consumed_1hour,
        }


class MsgGroupQueueManager:
    """
    Message Group Queue Manager

    Features:
    1. Fixed number of queues (default 10, configurable)
    2. Hash-based routing to fixed groups based on group_key
    3. Supports maximum message count limit (default 100)
    4. Empty queue priority delivery strategy
    5. Supports wait/no-wait mode for message retrieval
    6. Provides detailed metrics and logging
    """

    def __init__(
        self,
        name: str = "default",
        num_queues: int = 10,
        max_total_messages: int = 100,
        enable_metrics: bool = True,
        log_interval_seconds: int = 30,
    ):
        """
        Initialize the message group queue manager

        Args:
            name: Manager name
            num_queues: Number of queues
            max_total_messages: Maximum total message count limit
            enable_metrics: Whether to enable statistics
            log_interval_seconds: Logging interval in seconds
        """
        self.name = name
        self.num_queues = num_queues
        self.max_total_messages = max_total_messages
        self.enable_metrics = enable_metrics
        self.log_interval_seconds = log_interval_seconds

        # Initialize queues - using asyncio.Queue
        self._queues: List[asyncio.Queue] = [asyncio.Queue() for _ in range(num_queues)]

        # Queue statistics
        self._queue_stats: List[QueueStats] = [
            QueueStats(queue_id=i, current_size=0) for i in range(num_queues)
        ]

        # Manager statistics
        self._manager_stats = ManagerStats(
            total_queues=num_queues, total_current_messages=0
        )

        # Async lock to protect statistics
        self._stats_lock = asyncio.Lock()

        # Start time
        self._start_time = time.time()

        # Periodic logging task
        self._log_task: Optional[asyncio.Task] = None
        self._running = False

        # Shutdown state
        self._shutdown_state = ShutdownState()

        # Time window event tracking - using deque to store timestamped events
        self._delivery_events: List[deque] = [
            deque() for _ in range(num_queues)
        ]  # Delivery events for each queue
        self._consume_events: List[deque] = [
            deque() for _ in range(num_queues)
        ]  # Consumption events for each queue
        self._manager_delivery_events = deque()  # Manager total delivery events
        self._manager_consume_events = deque()  # Manager total consumption events

        logger.info(
            "🚀 MsgGroupQueueManager[%s] initialization completed: num_queues=%d, max_total_messages=%d",
            self.name,
            self.num_queues,
            self.max_total_messages,
        )

    def _hash_route(self, group_key: str) -> int:
        """
        Calculate hash-based routing to queue ID based on group_key

        Args:
            group_key: Group key

        Returns:
            int: Queue ID (0 to num_queues-1)
        """
        # Use MD5 hash to ensure even distribution
        hash_obj = hashlib.md5(group_key.encode('utf-8'))
        hash_int = int(hash_obj.hexdigest(), 16)
        return hash_int % self.num_queues

    async def deliver_message(self, group_key: str, message_data: Any) -> bool:
        """
        Deliver message to the specified group queue

        Args:
            group_key: Group key for hash routing
            message_data: Message data

        Returns:
            bool: Whether delivery was successful
        """
        try:
            # Calculate target queue
            target_queue_id = self._hash_route(group_key)
            target_queue = self._queues[target_queue_id]

            # Check delivery conditions
            can_deliver, reject_reason = self._can_deliver_message()
            if not can_deliver:
                # Reject delivery
                async with self._stats_lock:
                    self._manager_stats.total_rejected_messages += 1

                logger.warning(
                    "❌ MsgGroupQueueManager[%s] Delivery rejected: group_key=%s, reason=%s",
                    self.name,
                    group_key,
                    reject_reason,
                )
                return False

            # Perform delivery
            message_tuple = (group_key, message_data)
            await target_queue.put(message_tuple)

            # Update statistics
            current_time = to_iso_format(get_now_with_timezone())
            timestamp = time.time()

            async with self._stats_lock:
                self._queue_stats[target_queue_id].current_size = target_queue.qsize()
                self._queue_stats[target_queue_id].total_delivered += 1
                self._queue_stats[target_queue_id].last_deliver_time = current_time

                self._manager_stats.total_delivered_messages += 1
                self._manager_stats.total_current_messages = (
                    self._get_total_current_messages()
                )

                # Record time window events
                self._delivery_events[target_queue_id].append(timestamp)
                self._manager_delivery_events.append(timestamp)

            logger.debug(
                "✅ MsgGroupQueueManager[%s] Message delivered successfully: group_key=%s -> queue_id=%d, queue current size=%d, total remaining=%d",
                self.name,
                group_key,
                target_queue_id,
                target_queue.qsize(),
                self._get_total_current_messages(),
            )

            return True

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "❌ MsgGroupQueueManager[%s] Failed to deliver message: group_key=%s, error=%s",
                self.name,
                group_key,
                e,
            )
            return False

    async def get_by_queue(
        self, queue_id: int, wait: bool = True, timeout: Optional[float] = None
    ) -> Optional[Tuple[str, Any]]:
        """
        Get message from specified queue

        Args:
            queue_id: Queue ID
            wait: Whether to wait for message (True=blocking wait, False=return immediately)
            timeout: Wait timeout in seconds, only effective when wait=True

        Returns:
            Optional[Tuple[str, Any]]: Message tuple (group_key, message_data), None means no message
        """
        if queue_id < 0 or queue_id >= self.num_queues:
            raise ValueError(
                f"Queue ID out of range: {queue_id}, valid range: 0-{self.num_queues-1}"
            )

        target_queue = self._queues[queue_id]

        try:
            if wait:
                # Blocking wait mode
                if timeout is not None:
                    message_tuple = await asyncio.wait_for(
                        target_queue.get(), timeout=timeout
                    )
                else:
                    message_tuple = await target_queue.get()
            else:
                # Immediate return mode
                try:
                    message_tuple = target_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return None

            # Update statistics
            current_time = to_iso_format(get_now_with_timezone())
            timestamp = time.time()

            async with self._stats_lock:
                self._queue_stats[queue_id].current_size = target_queue.qsize()
                self._queue_stats[queue_id].total_consumed += 1
                self._queue_stats[queue_id].last_consume_time = current_time

                self._manager_stats.total_consumed_messages += 1
                self._manager_stats.total_current_messages = (
                    self._get_total_current_messages()
                )

                # Record time window events
                self._consume_events[queue_id].append(timestamp)
                self._manager_consume_events.append(timestamp)

            group_key, _ = message_tuple
            logger.debug(
                "📤 MsgGroupQueueManager[%s] Message consumed successfully: queue_id=%d, group_key=%s, queue remaining=%d",
                self.name,
                queue_id,
                group_key,
                target_queue.qsize(),
            )

            return message_tuple

        except asyncio.TimeoutError:
            logger.debug(
                "⏰ MsgGroupQueueManager[%s] Message retrieval timed out: queue_id=%d, timeout=%s",
                self.name,
                queue_id,
                timeout,
            )
            return None
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "❌ MsgGroupQueueManager[%s] Failed to retrieve message: queue_id=%d, error=%s",
                self.name,
                queue_id,
                e,
            )
            return None

    async def get_queue_info(
        self, queue_id: Optional[int] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get queue information

        Args:
            queue_id: Queue ID, None means get all queue information

        Returns:
            Union[Dict, List[Dict]]: Queue info dictionary or list of queue info
        """
        async with self._stats_lock:
            # Update current queue sizes
            for i, queue in enumerate(self._queues):
                self._queue_stats[i].current_size = queue.qsize()

            # Update time window statistics
            self._update_time_window_stats()

            if queue_id is not None:
                if queue_id < 0 or queue_id >= self.num_queues:
                    raise ValueError(
                        f"Queue ID out of range: {queue_id}, valid range: 0-{self.num_queues-1}"
                    )
                return self._queue_stats[queue_id].to_dict()
            else:
                return [stat.to_dict() for stat in self._queue_stats]

    async def get_manager_stats(self) -> Dict[str, Any]:
        """
        Get manager overall statistics

        Returns:
            Dict[str, Any]: Manager statistics
        """
        async with self._stats_lock:
            # Update uptime and total current message count
            self._manager_stats.uptime_seconds = time.time() - self._start_time
            self._manager_stats.total_current_messages = (
                self._get_total_current_messages()
            )

            # Update time window statistics
            self._update_time_window_stats()

            return self._manager_stats.to_dict()

    async def get_summary(self) -> Dict[str, Any]:
        """
        Get complete summary information

        Returns:
            Dict[str, Any]: Complete information including manager stats and queue details
        """
        return {
            "manager": await self.get_manager_stats(),
            "queues": await self.get_queue_info(),
        }

    def _get_total_current_messages(self) -> int:
        """Get current total message count"""
        return sum(queue.qsize() for queue in self._queues)

    def _can_deliver_message(self) -> Tuple[bool, str]:
        """
        Check if message can be delivered

        Returns:
            Tuple[bool, str]: (can deliver, rejection reason)
        """
        current_total = self._get_total_current_messages()

        # Allow delivery if there's an empty queue (regardless of total limit)
        has_empty_queue = any(q.qsize() == 0 for q in self._queues)

        # Delivery condition: total not exceeded OR has empty queue
        if current_total >= self.max_total_messages and not has_empty_queue:
            return (
                False,
                f"Current total messages={current_total}, limit={self.max_total_messages}, no empty queue",
            )

        return True, ""

    def _clean_old_events(self, events: deque, max_age_seconds: float):
        """Clean events older than specified time"""
        current_time = time.time()
        while events and current_time - events[0] > max_age_seconds:
            events.popleft()

    def _count_events_in_window(self, events: deque, window_seconds: float) -> int:
        """Count number of events within specified time window"""
        # First clean old events
        self._clean_old_events(events, window_seconds)
        # Return remaining event count
        return len(events)

    def _update_time_window_stats(self):
        """Update time window statistics for all queues and manager"""
        # Update time window stats for each queue
        for i in range(self.num_queues):
            # Clean old events and count
            self._queue_stats[i].time_window_stats.delivered_1min = (
                self._count_events_in_window(self._delivery_events[i], 60.0)
            )
            self._queue_stats[i].time_window_stats.consumed_1min = (
                self._count_events_in_window(self._consume_events[i], 60.0)
            )
            self._queue_stats[i].time_window_stats.delivered_1hour = (
                self._count_events_in_window(self._delivery_events[i], 3600.0)
            )
            self._queue_stats[i].time_window_stats.consumed_1hour = (
                self._count_events_in_window(self._consume_events[i], 3600.0)
            )

        # Update manager time window statistics
        self._manager_stats.time_window_stats.delivered_1min = (
            self._count_events_in_window(self._manager_delivery_events, 60.0)
        )
        self._manager_stats.time_window_stats.consumed_1min = (
            self._count_events_in_window(self._manager_consume_events, 60.0)
        )
        self._manager_stats.time_window_stats.delivered_1hour = (
            self._count_events_in_window(self._manager_delivery_events, 3600.0)
        )
        self._manager_stats.time_window_stats.consumed_1hour = (
            self._count_events_in_window(self._manager_consume_events, 3600.0)
        )

    async def start_periodic_logging(self):
        """Start periodic logging task"""
        if self._running:
            return

        self._running = True
        self._log_task = asyncio.create_task(self._periodic_log_worker())
        logger.info(
            "📊 MsgGroupQueueManager[%s] Periodic logging task started", self.name
        )

    async def stop_periodic_logging(self):
        """Stop periodic logging task"""
        if not self._running:
            return

        self._running = False
        if self._log_task and not self._log_task.done():
            self._log_task.cancel()
            try:
                await self._log_task
            except asyncio.CancelledError:
                pass

        logger.info(
            "📊 MsgGroupQueueManager[%s] Periodic logging task stopped", self.name
        )

    async def _periodic_log_worker(self):
        """Periodic logging worker coroutine"""
        try:
            while self._running:
                await asyncio.sleep(self.log_interval_seconds)
                if self._running:
                    await self._log_queue_details()
        except asyncio.CancelledError:
            logger.debug(
                "📊 MsgGroupQueueManager[%s] Periodic logging task cancelled", self.name
            )
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "📊 MsgGroupQueueManager[%s] Periodic logging task error: %s",
                self.name,
                e,
            )

    async def _log_queue_details(self):
        """Print queue details"""
        try:
            manager_stats = await self.get_manager_stats()
            queue_infos = await self.get_queue_info()

            # Print manager overall status summary
            logger.info(
                "📊 MsgGroupQueueManager[%s] Overall status: "
                "total messages=%d, total delivered=%d, total consumed=%d, total rejected=%d, uptime=%.1f seconds",
                self.name,
                manager_stats["total_current_messages"],
                manager_stats["total_delivered_messages"],
                manager_stats["total_consumed_messages"],
                manager_stats["total_rejected_messages"],
                manager_stats["uptime_seconds"],
            )

            # Print manager time window statistics
            logger.info(
                "⏱️  MsgGroupQueueManager[%s] Time window stats: "
                "last 1 minute (delivered=%d, consumed=%d), last 1 hour (delivered=%d, consumed=%d)",
                self.name,
                manager_stats["delivered_1min"],
                manager_stats["consumed_1min"],
                manager_stats["delivered_1hour"],
                manager_stats["consumed_1hour"],
            )

            # Print detailed information for each queue
            active_queues = []
            idle_queues = []
            empty_queues = []  # Currently empty queues

            for queue_info in queue_infos:
                queue_id = queue_info["queue_id"]
                current_size = queue_info["current_size"]
                total_delivered = queue_info["total_delivered"]
                total_consumed = queue_info["total_consumed"]
                last_deliver_time = queue_info["last_deliver_time"]
                last_consume_time = queue_info["last_consume_time"]

                # Get time window statistics
                delivered_1min = queue_info.get("delivered_1min", 0)
                consumed_1min = queue_info.get("consumed_1min", 0)
                delivered_1hour = queue_info.get("delivered_1hour", 0)
                consumed_1hour = queue_info.get("consumed_1hour", 0)

                # Record empty queues
                if current_size == 0:
                    empty_queues.append(queue_id)

                # Calculate queue activity level: based on time window activity and current queue state
                has_recent_activity = delivered_1min > 0 or consumed_1min > 0
                has_messages = current_size > 0
                has_historical_activity = total_delivered > 0 or total_consumed > 0

                # Active judgment: recent activity OR has messages OR historical activity
                is_active = (
                    has_recent_activity or has_messages or has_historical_activity
                )

                if is_active:
                    # Calculate delivery and consumption ratios
                    delivery_rate = (
                        total_delivered / max(1, total_delivered + total_consumed) * 100
                    )
                    consume_rate = (
                        total_consumed / max(1, total_delivered + total_consumed) * 100
                    )

                    # Format time display
                    last_deliver_display = (
                        last_deliver_time[-8:] if last_deliver_time else "none"
                    )
                    last_consume_display = (
                        last_consume_time[-8:] if last_consume_time else "none"
                    )

                    active_queues.append(
                        {
                            "id": queue_id,
                            "current": current_size,
                            "delivered": total_delivered,
                            "consumed": total_consumed,
                            "delivery_rate": delivery_rate,
                            "consume_rate": consume_rate,
                            "last_deliver": last_deliver_display,
                            "last_consume": last_consume_display,
                            "delivered_1min": delivered_1min,
                            "consumed_1min": consumed_1min,
                            "delivered_1hour": delivered_1hour,
                            "consumed_1hour": consumed_1hour,
                        }
                    )
                else:
                    idle_queues.append(queue_id)

            # Print active queue details
            if active_queues:
                logger.info("🔥 Active queue details (%d queues):", len(active_queues))
                for queue in active_queues:
                    # Queue status indicators
                    status_indicators = []
                    if queue["current"] == 0:
                        status_indicators.append("empty")
                    elif queue["current"] > self.max_total_messages * 0.3:
                        status_indicators.append("backlogged")

                    if queue["delivered_1min"] > 0:
                        status_indicators.append("recent delivery")
                    if queue["consumed_1min"] > 0:
                        status_indicators.append("recent consumption")

                    status_text = (
                        f"[{', '.join(status_indicators)}]" if status_indicators else ""
                    )

                    logger.info(
                        "   Queue[%d]%s: current=%d, total delivered=%d(%.1f%%), total consumed=%d(%.1f%%), "
                        "last delivery=%s, last consumption=%s",
                        queue["id"],
                        status_text,
                        queue["current"],
                        queue["delivered"],
                        queue["delivery_rate"],
                        queue["consumed"],
                        queue["consume_rate"],
                        queue["last_deliver"],
                        queue["last_consume"],
                    )

                    # Print time window statistics
                    logger.info(
                        "      ⏱️  Last 1 minute (delivered=%d, consumed=%d), last 1 hour (delivered=%d, consumed=%d)",
                        queue["delivered_1min"],
                        queue["consumed_1min"],
                        queue["delivered_1hour"],
                        queue["consumed_1hour"],
                    )

            # Print idle queue information
            if idle_queues:
                logger.info(
                    "💤 Idle queues: %s (total %d)",
                    ", ".join([f"Queue[{qid}]" for qid in idle_queues]),
                    len(idle_queues),
                )

            # Print empty queue information (queues with no messages currently)
            if empty_queues:
                logger.info(
                    "📭 Empty queues: %s (total %d, can accept new messages)",
                    ", ".join([f"Queue[{qid}]" for qid in empty_queues]),
                    len(empty_queues),
                )

            # Print queue load analysis
            if active_queues:
                # Find busiest and most backlogged queues
                busiest_queue = max(
                    active_queues, key=lambda q: q["delivered"] + q["consumed"]
                )
                most_backlogged = max(active_queues, key=lambda q: q["current"])

                logger.info(
                    "📈 Queue load analysis: busiest=Queue[%d](processed %d messages), most backlogged=Queue[%d](backlog %d messages)",
                    busiest_queue["id"],
                    busiest_queue["delivered"] + busiest_queue["consumed"],
                    most_backlogged["id"],
                    most_backlogged["current"],
                )

            # If any queues are backlogged, issue warning
            high_backlog_queues = [
                q for q in active_queues if q["current"] > self.max_total_messages * 0.3
            ]
            if high_backlog_queues:
                logger.warning(
                    "⚠️ Queue backlog warning: %s",
                    ", ".join(
                        [f"Queue[{q['id']}](%d messages)" for q in high_backlog_queues]
                    ),
                )

        except (OSError, ValueError, RuntimeError) as e:
            logger.error(
                "📊 MsgGroupQueueManager[%s] Failed to print queue details: %s",
                self.name,
                e,
            )

    async def shutdown(
        self,
        mode: ShutdownMode = ShutdownMode.HARD,
        max_delay_seconds: Optional[float] = None,
    ) -> bool:
        """
        Shutdown manager, supports hard and soft shutdown modes

        Args:
            mode: Shutdown mode (HARD: hard shutdown, SOFT: soft shutdown)
            max_delay_seconds: Maximum delay time in seconds for soft shutdown, only set on first soft shutdown

        Returns:
            bool: True means successfully shut down, False means still messages to process (only for soft shutdown)
        """
        current_time = time.time()

        if mode == ShutdownMode.SOFT:
            # Soft shutdown logic
            if not self._shutdown_state.is_shutting_down:
                # First soft shutdown
                self._shutdown_state.is_shutting_down = True
                self._shutdown_state.first_soft_shutdown_time = current_time
                self._shutdown_state.max_delay_seconds = max_delay_seconds

                logger.info(
                    "🔄 MsgGroupQueueManager[%s] Soft shutdown started, maximum delay time: %s seconds",
                    self.name,
                    max_delay_seconds,
                )

            # Check if there are messages
            total_remaining = self._get_total_current_messages()

            if total_remaining == 0:
                # No messages, can safely shut down
                await self._perform_hard_shutdown()
                self._shutdown_state.reset()
                return True

            # Check if delay time has been exceeded
            if (
                self._shutdown_state.max_delay_seconds is not None
                and current_time - self._shutdown_state.first_soft_shutdown_time
                >= self._shutdown_state.max_delay_seconds
            ):
                # Exceeded delay time, force shutdown
                logger.warning(
                    "⏰ MsgGroupQueueManager[%s] Soft shutdown timeout, forcing shutdown. Remaining messages: %d",
                    self.name,
                    total_remaining,
                )
                await self._perform_hard_shutdown()
                self._shutdown_state.reset()
                return True

            # Still messages and not timed out, return False
            elapsed_time = current_time - self._shutdown_state.first_soft_shutdown_time
            logger.info(
                "📋 MsgGroupQueueManager[%s] Soft shutdown check: remaining messages=%d, elapsed time=%.1f seconds",
                self.name,
                total_remaining,
                elapsed_time,
            )
            return False

        else:
            # Hard shutdown
            await self._perform_hard_shutdown()
            self._shutdown_state.reset()
            return True

    async def _perform_hard_shutdown(self):
        """Perform hard shutdown"""
        await self.stop_periodic_logging()

        # Count remaining messages
        total_remaining = 0
        queue_details = []

        for i, queue in enumerate(self._queues):
            queue_size = queue.qsize()
            total_remaining += queue_size

            if queue_size > 0:
                queue_details.append(f"Queue[{i}]: {queue_size} messages")

            # Clear queue
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        if total_remaining > 0:
            logger.warning(
                "⚠️ MsgGroupQueueManager[%s] Hard shutdown, discarded %d unprocessed messages. Details: %s",
                self.name,
                total_remaining,
                ", ".join(queue_details),
            )
        else:
            logger.info(
                "🔌 MsgGroupQueueManager[%s] Shut down safely, no unprocessed messages",
                self.name,
            )

    def __repr__(self) -> str:
        return (
            f"MsgGroupQueueManager(name={self.name}, "
            f"num_queues={self.num_queues}, "
            f"max_total_messages={self.max_total_messages})"
        )
