"""
Gauge Wrapper

Provides a unified Gauge interface with built-in auto-refresh capability.
"""
from prometheus_client import Gauge as PrometheusGauge
from typing import Sequence, Optional, Callable, Any, Dict, Tuple
import asyncio
import logging
import inspect
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseGauge(ABC):
    """
    Gauge base class
    
    Features:
    - Can increase or decrease (instantaneous value)
    - Built-in auto-refresh capability (default 5 seconds)
    - Must inherit and override refresh() method
    - Each instance manages its own refresh tasks independently
    - Supports manual set() method
    
    Usage - inherit and override refresh method:
        class KafkaPendingMessagesGauge(BaseGauge):
            def __init__(self, kafka_consumer):
                super().__init__(
                    name='kafka_pending_messages',
                    description='Number of pending messages',
                    labelnames=['job_name']
                )
                self.kafka_consumer = kafka_consumer
            
            def refresh(self, labels: dict) -> float:
                '''Return current value'''
                return len(self.kafka_consumer.pending_messages)
        
        # Usage 1: Auto-refresh (default 5 seconds)
        gauge = KafkaPendingMessagesGauge(kafka_consumer)
        gauge.labels(job_name='tanka').start_refresh()
        
        # Usage 2: Custom refresh interval
        gauge.labels(job_name='tanka').start_refresh(interval_seconds=10)
        
        # Usage 3: Manual set (without auto-refresh)
        gauge.labels(job_name='tanka').set(42)
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        labelnames: Sequence[str] = (),
        namespace: str = '',
        subsystem: str = '',
        unit: str = '',
    ):
        """
        Args:
            name: Metric name
            description: Metric description
            labelnames: List of label names
            namespace: Namespace (optional)
            subsystem: Subsystem (optional)
            unit: Unit (optional)
        """
        from .registry import get_metrics_registry
        registry = get_metrics_registry()
        
        self._gauge = PrometheusGauge(
            name=name,
            documentation=description,
            labelnames=labelnames,
            namespace=namespace,
            subsystem=subsystem,
            unit=unit,
            registry=registry,
        )
        
        self._name = name
        self._labelnames = labelnames
        
        # Store refresh tasks for each label combination
        # key: tuple of label values, value: RefreshTask
        self._refresh_tasks: Dict[Tuple, 'RefreshTask'] = {}
    
    def labels(self, **labels) -> 'LabeledGauge':
        """
        Return a Gauge with labels
        
        Returns:
            LabeledGauge instance
        """
        labeled_gauge = self._gauge.labels(**labels)
        label_key = self._make_label_key(**labels)
        
        return LabeledGauge(
            base_gauge=self,
            labeled_gauge=labeled_gauge,
            label_key=label_key,
            label_dict=labels,
        )
    
    def set(self, value: float) -> None:
        """Set value (no labels version)"""
        self._gauge.set(value)
    
    def inc(self, amount: float = 1) -> None:
        """Increment value (no labels version)"""
        self._gauge.inc(amount)
    
    def dec(self, amount: float = 1) -> None:
        """Decrement value (no labels version)"""
        self._gauge.dec(amount)
    
    @abstractmethod
    def refresh(self, labels: dict) -> float:
        """
        Refresh method (subclass must override)
        
        Args:
            labels: Label dictionary
        
        Returns:
            Current Gauge value
        
        Notes:
            - Subclass must override this method to implement custom refresh logic
            - This method is called periodically by auto-refresh task (default 5 seconds)
            - Can return any float value, will be automatically updated to Gauge
        
        Example:
            class QueueSizeGauge(BaseGauge):
                def __init__(self, queue):
                    super().__init__('queue_size', 'Queue size')
                    self.queue = queue
                
                def refresh(self, labels: dict) -> float:
                    return self.queue.qsize()
        """
        pass
    
    def _make_label_key(self, **labels) -> Tuple:
        """Generate label key"""
        if self._labelnames:
            return tuple(labels.get(name, '') for name in self._labelnames)
        return ()
    
    async def _stop_all_refresh_tasks(self) -> None:
        """Stop all refresh tasks"""
        for task in self._refresh_tasks.values():
            await task.stop()
        self._refresh_tasks.clear()


class LabeledGauge:
    """
    Gauge with labels
    
    Provides the same interface as native Gauge, with auto-refresh support.
    """
    
    def __init__(
        self,
        base_gauge: BaseGauge,
        labeled_gauge: Any,
        label_key: Tuple,
        label_dict: dict,
    ):
        self._base_gauge = base_gauge
        self._labeled_gauge = labeled_gauge
        self._label_key = label_key
        self._label_dict = label_dict
    
    def set(self, value: float) -> None:
        """Set value"""
        self._labeled_gauge.set(value)
    
    def inc(self, amount: float = 1) -> None:
        """Increment value"""
        self._labeled_gauge.inc(amount)
    
    def dec(self, amount: float = 1) -> None:
        """Decrement value"""
        self._labeled_gauge.dec(amount)
    
    def set_to_current_time(self) -> None:
        """Set to current timestamp"""
        self._labeled_gauge.set_to_current_time()
    
    def start_refresh(
        self,
        interval_seconds: int = 5,
        enable_async: bool = True,
    ) -> 'LabeledGauge':
        """
        Start auto-refresh
        
        Args:
            interval_seconds: Refresh interval (seconds), default 5 seconds
            enable_async: Whether to support async refresh method, default True
        
        Returns:
            self (supports chaining)
        
        Example:
            # Default 5 second refresh
            gauge.labels(job='tanka').start_refresh()
            
            # Custom refresh interval
            gauge.labels(job='tanka').start_refresh(interval_seconds=10)
            
            # Async refresh method
            class AsyncGauge(BaseGauge):
                async def refresh(self, labels: dict) -> float:
                    return await self.get_value_async()
            
            gauge.labels(type='A').start_refresh(enable_async=True)
        """
        # Stop existing task if any (prevent task leak)
        existing_task = self._base_gauge._refresh_tasks.get(self._label_key)
        if existing_task and existing_task._running:
            logger.warning(
                f"Replacing existing refresh task for {self._label_key}"
            )
            # Schedule stop in background to avoid blocking
            asyncio.create_task(existing_task.stop())
        
        # Create wrapper function that calls base_gauge.refresh()
        def refresh_wrapper():
            return self._base_gauge.refresh(self._label_dict)
        
        # Create refresh task
        task = RefreshTask(
            refresh_func=refresh_wrapper,
            labeled_gauge=self._labeled_gauge,
            interval_seconds=interval_seconds,
            enable_async=enable_async,
            label_key=self._label_key,
        )
        
        # Store task
        self._base_gauge._refresh_tasks[self._label_key] = task
        
        # Start task
        task.start()
        
        return self
    
    async def stop_refresh(self) -> None:
        """Stop auto-refresh"""
        task = self._base_gauge._refresh_tasks.get(self._label_key)
        if task:
            await task.stop()
            del self._base_gauge._refresh_tasks[self._label_key]


class RefreshTask:
    """
    Refresh task
    
    Each label combination has an independent refresh task.
    """
    
    def __init__(
        self,
        refresh_func: Callable[[], float],
        labeled_gauge: Any,
        interval_seconds: int,
        enable_async: bool,
        label_key: Tuple,
    ):
        self.refresh_func = refresh_func
        self.labeled_gauge = labeled_gauge
        self.interval_seconds = interval_seconds
        self.enable_async = enable_async
        self.label_key = label_key
        
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._error_count = 0
    
    def start(self) -> None:
        """Start refresh task"""
        if self._running:
            logger.warning(f"Refresh task already running for {self.label_key}")
            return
        
        self._running = True
        self._task = asyncio.create_task(self._refresh_loop())
        logger.info(
            f"Started refresh task: label_key={self.label_key}, "
            f"interval={self.interval_seconds}s"
        )
    
    async def stop(self) -> None:
        """Stop refresh task"""
        if not self._running:
            return
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        
        logger.info(f"Stopped refresh task: label_key={self.label_key}")
    
    async def _refresh_loop(self) -> None:
        """Refresh loop"""
        while self._running:
            try:
                # Check if it's an async function
                if self.enable_async and (
                    asyncio.iscoroutinefunction(self.refresh_func) or 
                    inspect.iscoroutinefunction(self.refresh_func)
                ):
                    value = await self.refresh_func()
                else:
                    value = self.refresh_func()
                
                # Update Gauge
                self.labeled_gauge.set(value)
                
                # Reset error count
                self._error_count = 0
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._error_count += 1
                logger.error(
                    f"Refresh failed for {self.label_key}: {e} "
                    f"(error_count={self._error_count})",
                    exc_info=True
                )
            
            # Wait for next refresh
            try:
                await asyncio.sleep(self.interval_seconds)
            except asyncio.CancelledError:
                break
