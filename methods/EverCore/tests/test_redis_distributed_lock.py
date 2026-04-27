"""
Redis Distributed Lock Test

Test scenarios include:
1. Basic lock acquisition and release
2. Lock reentrancy
3. Timeout mechanism
4. Concurrent competition
5. Decorator usage
"""

import asyncio
from core.lock.redis_distributed_lock import with_distributed_lock, distributed_lock


async def test_basic_lock_operations(redis_distributed_lock_manager):
    """Test basic lock operations"""
    resource = "test_resource"
    lock = redis_distributed_lock_manager.get_lock(resource)

    # Test acquiring lock
    async with lock.acquire() as acquired:
        assert acquired, "Should successfully acquire the lock"
        assert await lock.is_locked(), "Resource should be in locked state"
        assert await lock.is_owned_by_current_coroutine(), "The lock should be held by the current coroutine"

    # Test lock release
    assert not await lock.is_locked(), "The lock should have been released"
    assert not await lock.is_owned_by_current_coroutine(), "The lock should not be held by the current coroutine"


async def test_lock_reentrant(redis_distributed_lock_manager):
    """Test lock reentrancy"""
    resource = "test_reentrant"
    lock = redis_distributed_lock_manager.get_lock(resource)

    async with lock.acquire() as acquired:
        assert acquired, "First acquisition should succeed"
        count1 = await lock.get_reentry_count()
        assert count1 == 1, "After first acquisition, reentry count should be 1"

        # Re-enter to acquire the lock
        async with lock.acquire() as reacquired:
            assert reacquired, "Second acquisition should succeed (reentrant)"
            count2 = await lock.get_reentry_count()
            assert count2 == 2, "After second acquisition, reentry count should be 2"

        # After one release, the lock should still exist
        count3 = await lock.get_reentry_count()
        assert count3 == 1, "After one release, reentry count should be 1"
        assert await lock.is_locked(), "After one release, the lock should still exist"

    # After full release, the lock should disappear
    assert not await lock.is_locked(), "After full release, the lock should disappear"
    assert await lock.get_reentry_count() == 0, "After full release, reentry count should be 0"


async def test_lock_expiration(redis_distributed_lock_manager):
    """Test lock expiration mechanism"""
    resource = "test_expiration"

    # Test case 1: Basic expiration
    lock1 = redis_distributed_lock_manager.get_lock(resource)
    async with lock1.acquire(timeout=1) as acquired:  # Expire in 1 second
        assert acquired, "Should successfully acquire the lock"
        assert await lock1.is_locked(), "Resource should be in locked state"
        assert await lock1.is_owned_by_current_coroutine(), "The lock should be held by the current coroutine"

        # Wait less than expiration time, lock should still exist
        await asyncio.sleep(0.5)
        assert await lock1.is_locked(), "Before expiration, the lock should still exist"

        # Wait until expiration
        await asyncio.sleep(1)
        assert not await lock1.is_locked(), "The lock should have expired and been released"

    # Test case 2: Another coroutine can acquire the lock after expiration
    async def try_acquire_expired_lock():
        lock2 = redis_distributed_lock_manager.get_lock(resource)
        async with lock2.acquire(timeout=5) as acquired:  # Set a long enough expiration time
            assert acquired, "After the original lock expires, a new coroutine should be able to acquire the lock"
            assert await lock2.is_locked(), "The newly acquired lock should be in locked state"
            assert (
                await lock2.is_owned_by_current_coroutine()
            ), "The new lock should be held by the current coroutine"
            return True
        return False

    success = await try_acquire_expired_lock()
    assert success, "Should be able to acquire a new lock after the original lock expires"

    # Test case 3: Different expiration times
    test_times = [0.5, 1, 2]  # Test different expiration times
    for expire_time in test_times:
        lock = redis_distributed_lock_manager.get_lock(f"{resource}_{expire_time}")
        async with lock.acquire(timeout=expire_time) as acquired:
            assert acquired, f"Should successfully acquire the lock (expiration time: {expire_time} seconds)"

            # Wait half the time, lock should still exist
            await asyncio.sleep(expire_time / 2)
            assert (
                await lock.is_locked()
            ), f"For expiration time {expire_time} seconds, the lock should still exist after {expire_time/2} seconds"

            # Wait the remaining time plus a margin, lock should have expired
            await asyncio.sleep(expire_time / 2 + 0.1)
            assert (
                not await lock.is_locked()
            ), f"For expiration time {expire_time} seconds, the lock should have expired after {expire_time+0.1} seconds"

    # Test case 4: Update expiration time during re-entry
    lock3 = redis_distributed_lock_manager.get_lock("test_reentry_expiration")
    async with lock3.acquire(timeout=1) as acquired1:  # Expire in 1 second
        assert acquired1, "First acquisition should succeed"

        # Wait 0.8 seconds (close to expiration)
        await asyncio.sleep(0.8)
        assert await lock3.is_locked(), "The lock should still exist when close to expiration"

        # Re-enter and set a new expiration time
        async with lock3.acquire(timeout=2) as acquired2:  # Expire in 2 seconds
            assert acquired2, "Should be able to re-enter and acquire the lock"
            assert await lock3.get_reentry_count() == 2, "Reentry count should be 2"

            # Wait 1.2 seconds (exceeding the original 1-second expiration)
            await asyncio.sleep(1.2)
            assert await lock3.is_locked(), "After re-entry with a new expiration time, the lock should still exist"
            assert await lock3.get_reentry_count() == 2, "Reentry count should remain 2"


async def test_concurrent_lock_competition(redis_distributed_lock_manager):
    """Test concurrent competition scenario"""
    resource = "test_concurrent"
    results = []

    async def compete_for_lock(task_id):
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire(blocking_timeout=1) as acquired:
            if acquired:
                results.append(task_id)
                await asyncio.sleep(0.1)  # Simulate workload

    # Create multiple concurrent tasks
    tasks = [compete_for_lock(i) for i in range(5)]
    await asyncio.gather(*tasks)

    # Verify results
    assert len(results) > 0, "At least one task should acquire the lock"
    assert len(results) == len(set(results)), "Each task ID should appear only once"


@with_distributed_lock("test_decorator")
async def decorated_function(value):
    return value * 2


@with_distributed_lock("test_decorator_{value}")
async def decorated_function_with_format(value):
    return value * 2


async def test_lock_decorator(_redis_distributed_lock_manager):
    """Test decorator functionality"""
    # Test basic decorator
    result1 = await decorated_function(21)
    assert result1 == 42, "Decorator should not affect function return value"

    # Test decorator with formatted string
    result2 = await decorated_function_with_format(21)
    assert result2 == 42, "Decorator with formatted string should not affect function return value"


async def test_force_unlock(redis_distributed_lock_manager):
    """Test force unlock functionality"""
    resource = "test_force_unlock"
    lock = redis_distributed_lock_manager.get_lock(resource)

    async with lock.acquire() as acquired:
        assert acquired, "Should successfully acquire the lock"

        # Force unlock
        success = await redis_distributed_lock_manager.force_unlock(resource)
        assert success, "Force unlock should succeed"
        assert not await lock.is_locked(), "After force unlock, the lock should be released"


async def test_blocking_timeout_and_reentry(redis_distributed_lock_manager):
    """Test blocking timeout and reentrancy (using asyncio tasks)"""
    resource = "test_blocking"

    # Test case 1: Reentrancy within the same task
    async def reentry_test():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire() as acquired1:
            assert acquired1, "First lock acquisition should succeed"
            assert await lock.get_reentry_count() == 1, "Reentry count should be 1 after first acquisition"

            # Re-enter within the same task
            async with lock.acquire() as acquired2:
                assert acquired2, "Reentrancy within the same task should succeed"
                assert await lock.get_reentry_count() == 2, "Reentry count should be 2 after re-entry"

                # Re-enter again
                async with lock.acquire() as acquired3:
                    assert acquired3, "Third re-entry within the same task should succeed"
                    assert (
                        await lock.get_reentry_count() == 3
                    ), "Reentry count should be 3 after third re-entry"
                    await asyncio.sleep(0.1)  # Ensure task switching

    # Create and run task
    task1 = asyncio.create_task(reentry_test())
    await task1

    # Test case 2: Blocking and reentrancy between different tasks
    async def blocking_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire() as acquired:
            assert acquired, "First task should be able to acquire the lock"
            assert await lock.get_reentry_count() == 1, "Reentry count for first task should be 1"
            await asyncio.sleep(1)  # Hold the lock for a while

    async def competing_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire(blocking_timeout=0.5) as acquired:
            assert not acquired, "Second task should not be able to acquire the lock"
            assert await lock.get_reentry_count() == 0, "Reentry count should be 0 when acquisition fails"

    # Create two competing tasks
    task2 = asyncio.create_task(blocking_task())
    await asyncio.sleep(0.1)  # Ensure first task acquires the lock first
    task3 = asyncio.create_task(competing_task())

    # Wait for tasks to complete
    await asyncio.gather(task2, task3)

    # Test case 3: Reentrancy during nested tasks
    async def parent_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire() as acquired:
            assert acquired, "Parent task should be able to acquire the lock"
            assert await lock.get_reentry_count() == 1, "Reentry count for parent task should be 1"

            # Create child task
            child = asyncio.create_task(child_task())
            await asyncio.sleep(0.1)  # Ensure child task has a chance to run

            # Parent task re-enters
            async with lock.acquire() as reentry:
                assert reentry, "Parent task re-entry should succeed"
                assert await lock.get_reentry_count() == 2, "Reentry count should be 2 after parent re-entry"
                await asyncio.sleep(0.1)  # Give child task another chance

            await child  # Wait for child task to complete

    async def child_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire(blocking_timeout=0.5) as acquired:
            assert not acquired, "Child task should not be able to acquire the lock held by parent task"
            assert await lock.get_reentry_count() == 0, "Reentry count should be 0 when child task fails to acquire"

    # Run parent-child task test
    await parent_task()

    # Test case 4: New task acquiring lock after release
    async def final_task():
        lock = redis_distributed_lock_manager.get_lock(resource)
        async with lock.acquire(blocking_timeout=0.5) as acquired:
            assert acquired, "New task should be able to acquire the lock after it is released"
            assert await lock.get_reentry_count() == 1, "Reentry count should be 1 after new task acquires the lock"

    # Ensure lock is released
    assert not await redis_distributed_lock_manager.get_lock(
        resource
    ).is_locked(), "The lock should be released after all tasks complete"

    # Run final test
    final = asyncio.create_task(final_task())
    await final


async def run_all_tests():
    """Run all tests"""
    from core.di.utils import get_bean_by_type
    from core.lock.redis_distributed_lock import RedisDistributedLockManager

    print("Starting Redis distributed lock tests...")

    # Get lock manager instance
    lock_manager = get_bean_by_type(RedisDistributedLockManager)

    # Define all test functions
    tests = [
        test_basic_lock_operations,
        test_lock_reentrant,
        test_lock_expiration,  # New expiration test
        test_concurrent_lock_competition,
        test_lock_decorator,
        test_force_unlock,
        test_blocking_timeout_and_reentry,  # Updated blocking and reentrancy test
        test_convenient_context_manager,
        test_context_manager_with_timeout,
        test_context_manager_concurrent,
    ]

    # Run all tests
    for test_func in tests:
        print(f"\nRunning test: {test_func.__name__}")
        print("-" * 50)
        try:
            await test_func(lock_manager)
            print(f"✅ {test_func.__name__} passed")
        except AssertionError as e:
            print(f"❌ {test_func.__name__} failed: {str(e)}")
        except (ConnectionError, TimeoutError, OSError) as e:
            print(f"❌ {test_func.__name__} error: {str(e)}")

    print("\nTests completed!")


async def test_convenient_context_manager(_redis_distributed_lock_manager):
    """Test convenient context manager function"""
    resource = "test_context_manager"

    # Test basic usage
    async with distributed_lock(resource) as acquired:
        assert acquired, "Should successfully acquire the lock"

        # Test reentrancy
        async with distributed_lock(resource) as reacquired:
            assert reacquired, "Should support reentrancy"

    # Test locks for different resources
    async with distributed_lock("resource1") as acquired1:
        assert acquired1, "Should successfully acquire lock for resource1"

        async with distributed_lock("resource2") as acquired2:
            assert acquired2, "Should successfully acquire lock for resource2"

    print("✅ Convenient context manager test passed")


async def test_context_manager_with_timeout(_redis_distributed_lock_manager):
    """Test context manager timeout functionality"""
    resource = "test_timeout_context"

    # Test custom timeout parameters
    async with distributed_lock(
        resource, timeout=30.0, blocking_timeout=5.0
    ) as acquired:
        assert acquired, "Should successfully acquire the lock"

    print("✅ Context manager timeout test passed")


async def test_context_manager_concurrent(_redis_distributed_lock_manager):
    """Test context manager concurrency"""
    resource = "test_concurrent_context"
    results = []

    async def worker(worker_id: int):
        async with distributed_lock(resource, blocking_timeout=0.2) as acquired:
            if acquired:
                results.append(f"worker_{worker_id}")
                # Hold the lock long enough to ensure other workers time out
                await asyncio.sleep(0.5)

    # Truly run multiple workers concurrently
    tasks = [worker(i) for i in range(3)]
    await asyncio.gather(*tasks)

    # Due to the lock and short blocking_timeout, only one worker should succeed
    assert len(results) == 1, f"Only one worker should succeed, but got: {results}"
    print(f"✅ Context manager concurrency test passed, successful worker: {results[0]}")


if __name__ == "__main__":
    asyncio.run(run_all_tests())