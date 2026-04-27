"""
Redis distributed lock implementation

Distributed lock service supporting coroutine-level reentrancy based on Redis
Using contextvar to manage coroutine context, ensuring thread safety and coroutine safety
"""

import asyncio
from typing import Optional, Union
from contextlib import asynccontextmanager

from core.di.decorators import component
from core.observation.logger import get_logger
from core.component.redis_provider import RedisProvider
from core.di.utils import get_bean_by_type

logger = get_logger(__name__)

# Default configuration constants
DEFAULT_LOCK_TIMEOUT = 60.0  # Default lock timeout (seconds)
DEFAULT_BLOCKING_TIMEOUT = 80.0  # Default blocking timeout for acquiring lock (seconds)
DEFAULT_RETRY_INTERVAL = 3  # Default retry interval (seconds)


class DistributedLockError(Exception):
    """Exception related to distributed lock"""


class RedisDistributedLock:
    """
    Redis distributed reentrant lock

    A single lock instance responsible for lock operations on a specific resource
    """

    def __init__(self, resource: str, lock_manager: 'RedisDistributedLockManager'):
        """
        Initialize distributed lock

        Args:
            resource: Name of the lock resource
            lock_manager: Lock manager instance
        """
        self.resource = resource
        self.lock_manager = lock_manager
        self._acquired = False
        self._reentry_count = 0

    @asynccontextmanager
    async def acquire(
        self, timeout: Optional[float] = None, blocking_timeout: Optional[float] = None
    ):
        """
        Asynchronous context manager for acquiring lock

        Args:
            timeout: Lock timeout (seconds)
            blocking_timeout: Blocking timeout for acquiring lock (seconds)

        Yields:
            bool: Whether the lock was successfully acquired
        """
        timeout = timeout or DEFAULT_LOCK_TIMEOUT
        blocking_timeout = blocking_timeout or DEFAULT_BLOCKING_TIMEOUT

        acquired = False
        try:
            # Call the lock manager's internal method to acquire the lock
            acquired = await self.lock_manager._acquire_lock(  # pylint: disable=protected-access
                self.resource, timeout, blocking_timeout
            )
            if acquired:
                self._acquired = True

            yield acquired

        finally:
            if acquired:
                try:
                    # Call the lock manager's internal method to release the lock
                    await self.lock_manager._release_lock(
                        self.resource
                    )  # pylint: disable=protected-access
                    self._acquired = False
                except (ConnectionError, TimeoutError, OSError) as e:
                    logger.error(
                        "Failed to release lock: %s, error: %s", self.resource, e
                    )

    async def is_locked(self) -> bool:
        """Check if the lock is held"""
        return await self.lock_manager.is_locked(self.resource)

    async def is_owned_by_current_coroutine(self) -> bool:
        """Check if the lock is held by the current coroutine"""
        return await self.lock_manager.is_owned_by_current_coroutine(self.resource)

    async def get_reentry_count(self) -> int:
        """Get the reentry count of the current coroutine"""
        return await self.lock_manager.get_reentry_count(self.resource)


@component(name="redis_distributed_lock_manager")
class RedisDistributedLockManager:
    """
    Redis distributed lock manager

    Responsible for managing multiple lock instances, providing lock creation and global operations
    """

    # Lock key template
    LOCK_KEY_TEMPLATE = "reentrant_lock:{resource}"

    # Lua script: Acquire reentrant lock
    LUA_ACQUIRE_SCRIPT = """
        local lock_key = KEYS[1]
        local owner_id = ARGV[1]
        local timeout_ms = tonumber(ARGV[2])
        
        -- Get current lock information
        -- Note: When lock_key does not exist, HMGET returns {false, false}
        local lock_info = redis.call('HMGET', lock_key, 'owner', 'count')
        local current_owner = lock_info[1]  -- false when not exists
        local current_count = tonumber(lock_info[2]) or 0  -- tonumber(false) is nil, use 0 as default
        
        if current_owner == false or current_owner == owner_id then
            -- Lock is not occupied (current_owner == false) or held by current coroutine, can acquire/reenter
            local new_count = current_count + 1
            redis.call('HMSET', lock_key, 'owner', owner_id, 'count', new_count)
            if timeout_ms > 0 then
                redis.call('PEXPIRE', lock_key, timeout_ms)
            end
            return new_count
        else
            -- Lock is held by another coroutine
            return 0
        end
    """

    # Lua script: Release reentrant lock
    LUA_RELEASE_SCRIPT = """
        local lock_key = KEYS[1]
        local owner_id = ARGV[1]
        
        -- Get current lock information
        -- Note: When lock_key does not exist, HMGET returns {false, false}
        local lock_info = redis.call('HMGET', lock_key, 'owner', 'count')
        local current_owner = lock_info[1]  -- false when not exists
        local current_count = tonumber(lock_info[2]) or 0  -- tonumber(false) is nil, use 0 as default
        
        if current_owner ~= owner_id then
            -- Not the lock holder, cannot release or lock does not exist
            return 0
        end
        
        local new_count = current_count - 1
        if new_count <= 0 then
            -- Reentry count reaches zero, completely release the lock
            redis.call('DEL', lock_key)
            return -1
        else
            -- Decrease reentry count but keep the lock
            redis.call('HSET', lock_key, 'count', new_count)
            return new_count
        end
    """

    # Lua script: Check lock status
    LUA_STATUS_SCRIPT = """
        local lock_key = KEYS[1]
        local owner_id = ARGV[1]
        
        -- Get current lock information
        -- Note: When lock_key does not exist, HMGET returns {false, false}
        local lock_info = redis.call('HMGET', lock_key, 'owner', 'count')
        local current_owner = lock_info[1]  -- false when not exists
        local current_count = tonumber(lock_info[2]) or 0  -- tonumber(false) is nil, use 0 as default
        
        if current_owner == false then
            return {0, 0}  -- Not locked
        elseif current_owner == owner_id then
            return {1, current_count}  -- Held by current coroutine
        else
            return {2, current_count}  -- Held by other coroutine
        end
    """

    def __init__(self, redis_provider: RedisProvider):
        """
        Initialize Redis distributed lock manager

        Args:
            redis_provider: Redis provider
        """
        self.redis_provider = redis_provider

        # Lua script cache
        self._lua_acquire = None
        self._lua_release = None
        self._lua_status = None

    def get_lock(self, resource: str) -> RedisDistributedLock:
        """
        Get lock instance for specified resource

        Args:
            resource: Name of the lock resource

        Returns:
            RedisDistributedLock: Lock instance
        """
        return RedisDistributedLock(resource, self)

    async def _ensure_scripts(self):
        """Ensure Lua scripts are registered"""
        if self._lua_acquire is None:
            redis_client = await self.redis_provider.get_client()
            self._lua_acquire = redis_client.register_script(self.LUA_ACQUIRE_SCRIPT)
            self._lua_release = redis_client.register_script(self.LUA_RELEASE_SCRIPT)
            self._lua_status = redis_client.register_script(self.LUA_STATUS_SCRIPT)

    def _get_owner_id(self) -> str:
        """
        Get unique identifier for current coroutine

        Returns:
            str: Unique identifier for coroutine

        Raises:
            DistributedLockError: If not in coroutine environment
        """
        # First try to get from context variable
        try:
            current_task = asyncio.current_task()
            if current_task is None:
                raise DistributedLockError(
                    "Distributed lock must be used in coroutine environment, no running coroutine task"
                )

            # Use task id as coroutine identifier
            task_id = id(current_task)
            owner_id = f"task_{task_id}"
            return owner_id

        except RuntimeError as e:
            raise DistributedLockError(
                f"Distributed lock must be used in coroutine environment: {e}"
            ) from e

    async def _acquire_lock(
        self, resource: str, timeout: float, blocking_timeout: float
    ) -> bool:
        """
        Internal method: Acquire lock

        Args:
            resource: Resource name
            timeout: Lock timeout
            blocking_timeout: Blocking timeout for acquiring lock

        Returns:
            bool: Whether the lock was successfully acquired
        """
        await self._ensure_scripts()

        lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
        owner_id = self._get_owner_id()
        timeout_ms = int(timeout * 1000) if timeout > 0 else 0

        # Calculate retry count
        retry_count = max(1, int(blocking_timeout / DEFAULT_RETRY_INTERVAL))

        for attempt in range(retry_count):
            try:
                redis_client = await self.redis_provider.get_client()
                result = await self._lua_acquire(
                    keys=[lock_key], args=[owner_id, timeout_ms], client=redis_client
                )

                if result > 0:
                    logger.debug(
                        "Successfully acquired reentrant lock: %s, coroutine: %s, reentry count: %s (attempt %s)",
                        resource,
                        owner_id,
                        result,
                        attempt + 1,
                    )
                    return True
                else:
                    if attempt < retry_count - 1:
                        await asyncio.sleep(DEFAULT_RETRY_INTERVAL)

            except (ConnectionError, TimeoutError, OSError) as e:
                logger.debug(
                    "Failed to acquire lock (attempt %s): %s, error: %s",
                    attempt + 1,
                    resource,
                    e,
                )
                if attempt < retry_count - 1:
                    await asyncio.sleep(DEFAULT_RETRY_INTERVAL)

        logger.warning(
            "Timed out acquiring reentrant distributed lock: %s, coroutine: %s",
            resource,
            owner_id,
        )
        return False

    async def _release_lock(self, resource: str):
        """
        Internal method: Release lock

        Args:
            resource: Resource name
        """
        lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
        owner_id = self._get_owner_id()

        try:
            redis_client = await self.redis_provider.get_client()
            result = await self._lua_release(
                keys=[lock_key], args=[owner_id], client=redis_client
            )

            if result == -1:
                logger.debug(
                    "Completely released reentrant lock: %s, coroutine: %s",
                    resource,
                    owner_id,
                )
            elif result > 0:
                logger.debug(
                    "Reduced reentrant lock count: %s, coroutine: %s, remaining count: %s",
                    resource,
                    owner_id,
                    result,
                )
            else:
                logger.warning(
                    "Cannot release lock not owned by current coroutine or lock does not exist: %s, coroutine: %s",
                    resource,
                    owner_id,
                )

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(
                "Exception occurred while releasing reentrant lock: %s, coroutine: %s, error: %s",
                resource,
                owner_id,
                e,
            )

    async def is_locked(self, resource: str) -> bool:
        """
        Check if resource is locked

        Args:
            resource: Name of the lock resource

        Returns:
            bool: Whether it is locked
        """
        try:
            await self._ensure_scripts()
            redis_client = await self.redis_provider.get_client()
            lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
            owner_id = self._get_owner_id()

            result = await self._lua_status(
                keys=[lock_key], args=[owner_id], client=redis_client
            )

            status_code = result[0] if result else 0
            return status_code > 0  # 1 or 2 both indicate locked

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(
                "Failed to check reentrant lock status: %s, error: %s", resource, e
            )
            return False

    async def is_owned_by_current_coroutine(self, resource: str) -> bool:
        """
        Check if the lock is held by the current coroutine

        Args:
            resource: Name of the lock resource

        Returns:
            bool: Whether it is held by the current coroutine
        """
        try:
            await self._ensure_scripts()
            redis_client = await self.redis_provider.get_client()
            lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
            owner_id = self._get_owner_id()

            result = await self._lua_status(
                keys=[lock_key], args=[owner_id], client=redis_client
            )

            status_code = result[0] if result else 0
            return status_code == 1  # 1 indicates held by current coroutine

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(
                "Failed to check reentrant lock ownership: %s, error: %s", resource, e
            )
            return False

    async def get_reentry_count(self, resource: str) -> int:
        """
        Get the reentry count of current coroutine for specified resource

        Args:
            resource: Name of the lock resource

        Returns:
            int: Reentry count, 0 means lock is not held
        """
        try:
            await self._ensure_scripts()
            redis_client = await self.redis_provider.get_client()
            lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
            owner_id = self._get_owner_id()

            result = await self._lua_status(
                keys=[lock_key], args=[owner_id], client=redis_client
            )

            if result and result[0] == 1:  # Held by current coroutine
                return result[1]
            else:
                return 0

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error("Failed to get reentry count: %s, error: %s", resource, e)
            return 0

    async def force_unlock(self, resource: str) -> bool:
        """
        Force release lock (use with caution)

        Args:
            resource: Name of the lock resource

        Returns:
            bool: Whether the release was successful
        """
        try:
            redis_client = await self.redis_provider.get_client()
            lock_key = self.LOCK_KEY_TEMPLATE.format(resource=resource)
            result = await redis_client.delete(lock_key)

            logger.warning(
                "Forcibly released reentrant lock: %s, result: %s", resource, result
            )
            return result > 0

        except (ConnectionError, TimeoutError, OSError) as e:
            logger.error(
                "Failed to forcibly release reentrant lock: %s, error: %s", resource, e
            )
            return False

    @asynccontextmanager
    async def acquire_lock(
        self,
        resource: str,
        timeout: Optional[float] = None,
        blocking_timeout: Optional[float] = None,
    ):
        """
        Asynchronous context manager for acquiring reentrant distributed lock (compatible with old interface)

        Args:
            resource: Name of the lock resource (key name)
            timeout: Lock timeout (seconds)
            blocking_timeout: Blocking timeout for acquiring lock (seconds)

        Yields:
            bool: Whether the lock was successfully acquired
        """
        lock = self.get_lock(resource)
        async with lock.acquire(timeout, blocking_timeout) as acquired:
            yield acquired

    async def close(self):
        """Close service and clean up resources"""
        logger.info("Redis distributed lock manager closed")


# Convenient context manager function
@asynccontextmanager
async def distributed_lock(
    resource: str,
    timeout: Optional[float] = None,
    blocking_timeout: Optional[float] = None,
):
    """
    Convenient distributed lock context manager, used within functions

    Args:
        resource: Name of the lock resource
        timeout: Lock timeout (seconds)
        blocking_timeout: Blocking timeout for acquiring lock (seconds)

    Yields:
        bool: Whether the lock was successfully acquired

    Example:
        async def some_function():
            async with distributed_lock("user:balance:123") as acquired:
                if acquired:
                    # Execute code that requires locking
                    print("Lock acquired, executing business logic")
                else:
                    print("Failed to acquire lock")

        # Supports reentrancy
        async def reentrant_function():
            async with distributed_lock("resource:123") as acquired1:
                if acquired1:
                    print("First level acquired lock")
                    async with distributed_lock("resource:123") as acquired2:
                        if acquired2:
                            print("Second level acquired lock (reentrant)")
    """

    # Get lock manager
    lock_manager = get_bean_by_type(RedisDistributedLockManager)

    # Acquire lock and execute
    lock = lock_manager.get_lock(resource)
    async with lock.acquire(
        timeout=timeout, blocking_timeout=blocking_timeout
    ) as acquired:
        yield acquired


# Convenient decorator function
def with_distributed_lock(
    resource_key: Union[str, callable],
    timeout: float = DEFAULT_LOCK_TIMEOUT,
    blocking_timeout: float = DEFAULT_BLOCKING_TIMEOUT,
):
    """
    Distributed lock decorator (supports reentrancy)

    Args:
        resource_key: Lock resource key, can be string or function returning string
        timeout: Lock timeout
        blocking_timeout: Blocking timeout for acquiring lock

    Example:
        @with_distributed_lock("user:balance:{user_id}")
        async def update_user_balance(user_id: int, amount: float):
            # This function can be recursively called within the same coroutine without deadlock
            if amount > 100:
                await update_user_balance(user_id, amount / 2)
                await update_user_balance(user_id, amount / 2)
    """

    def decorator(func):
        async def wrapper(*args, **kwargs):

            # Get lock manager
            lock_manager = get_bean_by_type(RedisDistributedLockManager)

            # Calculate resource key
            if callable(resource_key):
                resource = resource_key(*args, **kwargs)
            else:
                # Support formatted string
                try:
                    resource = resource_key.format(*args, **kwargs)
                except (IndexError, KeyError):
                    resource = resource_key

            # Acquire lock and execute function
            lock = lock_manager.get_lock(resource)
            async with lock.acquire(
                timeout=timeout, blocking_timeout=blocking_timeout
            ) as acquired:
                if acquired:
                    return await func(*args, **kwargs)
                else:
                    raise RuntimeError(
                        f"Failed to acquire distributed lock: {resource}"
                    )

        return wrapper

    return decorator
