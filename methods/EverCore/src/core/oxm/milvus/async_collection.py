import asyncio
import concurrent.futures
import contextvars
import os
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union, List, Dict

from pymilvus import Collection, SearchResult
from pymilvus.orm.mutation import MutationResult
from pymilvus.client.types import CompactionPlans, CompactionState, Replica

T = TypeVar('T')

# Dedicated thread pool for pymilvus sync calls.
# Pre-warms threads at import time so concurrent requests don't pay
# thread-creation latency. Sized for burst concurrency: pymilvus is
# I/O-bound (gRPC network waits), so idle threads consume near-zero CPU.
_MILVUS_POOL_SIZE = int(os.getenv("MILVUS_THREAD_POOL_SIZE", "25"))
_milvus_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=_MILVUS_POOL_SIZE, thread_name_prefix="milvus"
)
# Pre-warm: force all threads to be created immediately
concurrent.futures.wait(
    [_milvus_executor.submit(lambda: None) for _ in range(_MILVUS_POOL_SIZE)]
)


def async_wrap(func: Callable[..., T]) -> Callable[..., asyncio.Future[T]]:
    """Decorator that wraps a synchronous method into an asynchronous one.

    Uses a dedicated pre-warmed thread pool (_milvus_executor) instead of
    the default executor, so pymilvus blocking calls don't compete with
    other run_in_executor users and threads are ready immediately.

    Note: Use contextvars.copy_context() to ensure that threads in the thread pool can access contextvars
    (such as tenant context), because run_in_executor does not pass the asyncio Context by default.
    """

    @wraps(func)
    async def run(*args, **kwargs) -> T:
        loop = asyncio.get_running_loop()
        # Copy current context to ensure contextvars are accessible in the thread pool
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(
            _milvus_executor, lambda: ctx.run(func, *args, **kwargs)
        )

    return run


class AsyncCollection:
    """Asynchronous version of the Collection class.

    This class wraps pymilvus's Collection class to provide asynchronous interfaces.
    All synchronous operations are executed in the event loop's default executor.
    """

    def __init__(self, collection: Collection):
        """Initialize AsyncCollection.

        Args:
            collection: pymilvus Collection instance
        """
        self._collection = collection

    def __getattr__(self, name: str) -> Any:
        """Intercept all attribute access to the original collection.

        If it's a method call, wrap it into an asynchronous method.
        If it's an attribute access, return directly.
        """
        attr = getattr(self._collection, name)
        if callable(attr):
            return async_wrap(attr)
        return attr

    @property
    def collection(self) -> Collection:
        """Return the original Collection instance."""
        return self._collection

    # Explicit asynchronous implementations of some commonly used methods.
    # Although __getattr__ can handle these methods, explicit definitions provide better type hints.

    async def insert(
        self,
        data: Union[List, Dict],
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> MutationResult:
        """Asynchronously insert data."""
        return await async_wrap(self._collection.insert)(
            data, partition_name, timeout, **kwargs
        )

    async def search(
        self,
        data: List,
        anns_field: str,
        param: Dict,
        limit: int,
        expr: Optional[str] = None,
        partition_names: Optional[List[str]] = None,
        output_fields: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        round_decimal: int = -1,
        **kwargs,
    ) -> SearchResult:
        """Asynchronously search."""
        return await async_wrap(self._collection.search)(
            data,
            anns_field,
            param,
            limit,
            expr,
            partition_names,
            output_fields,
            timeout,
            round_decimal,
            **kwargs,
        )

    async def query(
        self,
        expr: str,
        output_fields: Optional[List[str]] = None,
        partition_names: Optional[List[str]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List:
        """Asynchronously query."""
        return await async_wrap(self._collection.query)(
            expr, output_fields, partition_names, timeout, **kwargs
        )

    async def delete(
        self,
        expr: str,
        partition_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> MutationResult:
        """Asynchronously delete."""
        return await async_wrap(self._collection.delete)(
            expr, partition_name, timeout, **kwargs
        )

    async def flush(self, timeout: Optional[float] = None, **kwargs) -> None:
        """Asynchronously flush."""
        return await async_wrap(self._collection.flush)(timeout, **kwargs)

    async def load(
        self,
        partition_names: Optional[List[str]] = None,
        replica_number: Optional[int] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Asynchronously load."""
        return await async_wrap(self._collection.load)(
            partition_names, replica_number, timeout, **kwargs
        )

    async def release(self, timeout: Optional[float] = None, **kwargs) -> None:
        """Asynchronously release."""
        return await async_wrap(self._collection.release)(timeout, **kwargs)

    async def compact(
        self,
        is_clustering: Optional[bool] = False,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Asynchronously compact."""
        return await async_wrap(self._collection.compact)(
            is_clustering, timeout, **kwargs
        )

    async def get_compaction_state(
        self,
        timeout: Optional[float] = None,
        is_clustering: Optional[bool] = False,
        **kwargs,
    ) -> CompactionState:
        """Asynchronously get compaction state."""
        return await async_wrap(self._collection.get_compaction_state)(
            timeout, is_clustering, **kwargs
        )

    async def get_compaction_plans(
        self,
        timeout: Optional[float] = None,
        is_clustering: Optional[bool] = False,
        **kwargs,
    ) -> CompactionPlans:
        """Asynchronously get compaction plans."""
        return await async_wrap(self._collection.get_compaction_plans)(
            timeout, is_clustering, **kwargs
        )

    async def get_replicas(self, timeout: Optional[float] = None, **kwargs) -> Replica:
        """Asynchronously get replica information."""
        return await async_wrap(self._collection.get_replicas)(timeout, **kwargs)
