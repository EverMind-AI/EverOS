#!/usr/bin/env python3
"""
Refactored complete test suite for MsgGroupQueueManager
"""

import asyncio
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from core.queue.msg_group_queue.msg_group_queue_manager import (
    MsgGroupQueueManager,
    QueueStats,
    ManagerStats,
    ShutdownMode,
    TimeWindowStats,
)
from core.queue.msg_group_queue.msg_group_queue_manager_factory import (
    MsgGroupQueueManagerFactory,
    MsgGroupQueueConfig,
)

# ============ Basic test functions ============


async def test_basic_functionality():
    """Basic functionality test"""
    manager = MsgGroupQueueManager("basic_test", num_queues=3, max_total_messages=10)

    # Delivery test
    success = await manager.deliver_message("test_user", {"msg": "hello"})
    assert success, "Basic delivery failed"

    # Consumption test
    target_queue = manager._hash_route("test_user")  # pylint: disable=protected-access
    message = await manager.get_by_queue(target_queue, wait=False)
    assert message is not None, "Basic consumption failed"

    # Statistics test
    stats = await manager.get_manager_stats()
    assert stats["total_delivered_messages"] == 1, "Statistics error"

    await manager.shutdown(ShutdownMode.HARD)


async def test_queue_full_scenarios():
    """Test scenarios when queue is full"""
    manager = MsgGroupQueueManager("full_test", num_queues=3, max_total_messages=10)

    # Attempt to deliver a large number of messages
    delivered_count = 0
    rejected_count = 0

    for i in range(15):
        success = await manager.deliver_message(f"user_{i}", f"msg_{i}")
        if success:
            delivered_count += 1
        else:
            rejected_count += 1

    print(f"✅ Queue full test passed (delivered {delivered_count}, rejected {rejected_count})")
    await manager.shutdown(ShutdownMode.HARD)


async def test_time_window_stats():
    """Time window statistics test"""
    manager = MsgGroupQueueManager("time_test", num_queues=3, max_total_messages=20)

    # Deliver messages
    for i in range(5):
        await manager.deliver_message(f"time_user_{i}", f"time_msg_{i}")

    # Consume some messages
    consumed_count = 0
    for i in range(3):
        target_queue = manager._hash_route(
            f"time_user_{i}"
        )  # pylint: disable=protected-access
        message = await manager.get_by_queue(target_queue, wait=False)
        if message:
            consumed_count += 1

    # Check time window statistics
    stats = await manager.get_manager_stats()
    assert stats["delivered_1min"] >= 5, "1-minute delivery statistics error"
    assert stats["consumed_1min"] >= consumed_count, "1-minute consumption statistics error"

    print(f"✅ Time window statistics test passed (delivered 5, consumed {consumed_count})")
    await manager.shutdown(ShutdownMode.HARD)


async def test_basic_concurrent_operations():
    """Basic concurrent operations test"""
    manager = MsgGroupQueueManager(
        "concurrent_test", num_queues=5, max_total_messages=50
    )

    async def producer(producer_id: int):
        for i in range(10):
            await manager.deliver_message(
                f"producer_{producer_id}_msg_{i}", f"data_{i}"
            )

    async def consumer():
        consumed = 0
        for queue_id in range(manager.num_queues):
            for _ in range(5):  # Try to consume 5 times per queue
                message = await manager.get_by_queue(queue_id, wait=False)
                if message:
                    consumed += 1
        return consumed

    # Start 3 producers
    producers = [producer(i) for i in range(3)]
    await asyncio.gather(*producers)

    # Start consumer
    consumed_count = await consumer()

    stats = await manager.get_manager_stats()
    assert (
        stats["total_delivered_messages"] == 30
    ), f"Concurrent delivery statistics error: {stats['total_delivered_messages']}"

    print(f"✅ Concurrent operations test passed (produced 30, consumed {consumed_count})")
    await manager.shutdown(ShutdownMode.HARD)


async def test_basic_shutdown_modes():
    """Basic Shutdown mode test"""
    manager = MsgGroupQueueManager(
        "basic_shutdown_test", num_queues=3, max_total_messages=10
    )

    # Ensure shutdown state is clean
    manager._shutdown_state.reset()  # pylint: disable=protected-access

    # Add some messages
    for i in range(5):
        await manager.deliver_message(f"user_{i}", f"msg_{i}")

    # Test soft shutdown (requires polling until success)
    start_time = time.time()

    # First call to soft shutdown should return False immediately
    result = await manager.shutdown(ShutdownMode.SOFT, max_delay_seconds=1.0)
    first_call_time = time.time()

    assert result is False, "First soft shutdown should return False immediately (unprocessed messages)"
    assert first_call_time - start_time < 0.1, "First soft shutdown should return immediately"
    print("  First soft shutdown correctly returned False (immediate return)")

    # Poll until soft shutdown succeeds or times out
    max_poll_time = start_time + 2.0  # Poll for up to 2 seconds
    final_result = False

    while time.time() < max_poll_time:
        await asyncio.sleep(0.1)  # Wait 100ms and check again
        result = await manager.shutdown(ShutdownMode.SOFT)  # No need to set delay again
        if result is True:
            final_result = True
            break

    end_time = time.time()
    total_elapsed = end_time - start_time

    # Verify final result and timing
    assert final_result is True, "Soft shutdown should eventually succeed"
    assert (
        0.8 <= total_elapsed <= 1.5
    ), f"Soft shutdown total time abnormal: {total_elapsed:.2f}s, expected ~1.0s"

    print(f"  Soft shutdown polling succeeded, total time: {total_elapsed:.2f}s")


async def test_edge_cases():
    """Edge case test"""
    manager = MsgGroupQueueManager("edge_test", num_queues=3, max_total_messages=10)

    # Test 1: Empty string group_key
    success = await manager.deliver_message("", "empty_key_msg")
    assert success, "Delivery with empty string key failed"

    empty_queue = manager._hash_route("")  # pylint: disable=protected-access
    result = await manager.get_by_queue(empty_queue, wait=False)
    assert result is not None, "Consumption with empty string key failed"
    key, data = result
    assert key == "", f"Empty string key mismatch: {key}"
    assert data == "empty_key_msg", f"Empty string key data mismatch: {data}"

    # Test 2: None message data
    success = await manager.deliver_message("none_test", None)
    assert success, "Delivery of None message failed"

    none_queue = manager._hash_route("none_test")  # pylint: disable=protected-access
    result = await manager.get_by_queue(none_queue, wait=False)
    assert result is not None, "Consumption of None message failed"
    key, data = result
    assert key == "none_test", f"None message key mismatch: {key}"
    assert data is None, f"None message data mismatch: {data}"

    # Test 3: Complex data
    complex_data = {"nested": {"list": [1, 2, 3]}, "unicode": "测试🎉"}
    success = await manager.deliver_message("complex_test", complex_data)
    assert success, "Delivery of complex data failed"

    complex_queue = manager._hash_route(
        "complex_test"
    )  # pylint: disable=protected-access
    result = await manager.get_by_queue(complex_queue, wait=False)
    assert result is not None, "Consumption of complex data failed"
    key, data = result
    assert key == "complex_test", f"Complex data key mismatch: {key}"
    assert data == complex_data, f"Complex data mismatch: expected={complex_data}, actual={data}"

    await manager.shutdown(ShutdownMode.HARD)


async def test_factory_pattern():
    """Factory pattern test"""
    factory = MsgGroupQueueManagerFactory()

    # Default manager
    manager1 = await factory.get_default_manager(auto_start=False)
    manager2 = await factory.get_default_manager(auto_start=False)
    assert manager1 is manager2, "Default manager should be singleton"

    # Custom configuration
    config = MsgGroupQueueConfig(name="custom", num_queues=5, max_total_messages=25)
    manager3 = await factory.get_manager(config, auto_start=False)
    assert manager3.name == "custom", "Custom configuration error"

    # Named manager
    manager4 = await factory.get_named_manager("test_named", auto_start=False)
    assert manager4.name == "test_named", "Named manager error"

    await factory.stop_all_managers()


async def test_timeout_mechanism():
    """Timeout mechanism test"""
    manager = MsgGroupQueueManager("timeout_test", num_queues=3, max_total_messages=10)

    try:
        # Test timeout get (empty queue)
        start_time = time.time()
        try:
            result = await manager.get_by_queue(0, wait=True, timeout=0.5)
            end_time = time.time()

            # Should timeout and return None or raise TimeoutError
            if result is not None:
                raise AssertionError("Timeout should return None or raise exception")
            assert (
                0.4 <= end_time - start_time <= 0.6
            ), f"Timeout time inaccurate: {end_time - start_time}"

        except asyncio.TimeoutError:
            end_time = time.time()
            assert (
                0.4 <= end_time - start_time <= 0.6
            ), f"Timeout time inaccurate: {end_time - start_time}"

        # Test get when message exists
        await manager.deliver_message("timeout_user", "timeout_msg")
        target_queue = manager._hash_route(
            "timeout_user"
        )  # pylint: disable=protected-access

        start_time = time.time()
        result = await manager.get_by_queue(target_queue, wait=True, timeout=1.0)
        end_time = time.time()

        # Should return message immediately
        assert result is not None, "Should return immediately when message exists"
        assert (
            end_time - start_time < 0.1
        ), f"Should not wait when message exists: {end_time - start_time}"

        key, data = result
        assert key == "timeout_user", "Timeout test message key error"
        assert data == "timeout_msg", "Timeout test message data error"

    finally:
        await manager.shutdown(ShutdownMode.HARD)


async def test_routing_uniformity():
    """Routing uniformity test - Use random UUIDs to verify hash distribution"""
    manager = MsgGroupQueueManager(
        "routing_test", num_queues=10, max_total_messages=2000
    )

    try:
        # Generate a large number of random UUIDs as group_key
        test_count = 1000
        uuid_keys = [str(uuid.uuid4()) for _ in range(test_count)]

        # Count messages received by each queue
        queue_counts = defaultdict(int)

        # Deliver all messages and count routing distribution
        for i, group_key in enumerate(uuid_keys):
            success = await manager.deliver_message(group_key, f"message_{i}")
            assert success, f"UUID message delivery failed: {group_key}"

            # Calculate which queue this key routes to
            target_queue = manager._hash_route(
                group_key
            )  # pylint: disable=protected-access
            queue_counts[target_queue] += 1

        # Analyze distribution uniformity
        print(f"📊 Routing distribution statistics ({test_count} UUIDs):")
        expected_per_queue = test_count / manager.num_queues

        total_deviation = 0
        max_count = 0
        min_count = test_count

        for queue_id in range(manager.num_queues):
            count = queue_counts[queue_id]
            percentage = (count / test_count) * 100
            deviation = abs(count - expected_per_queue)
            deviation_percent = (deviation / expected_per_queue) * 100

            print(
                f"   Queue[{queue_id}]: {count:3d} messages ({percentage:5.1f}%) - Deviation: {deviation_percent:5.1f}%"
            )

            total_deviation += deviation
            max_count = max(max_count, count)
            min_count = min(min_count, count)

        # Calculate distribution quality metrics
        avg_count = test_count / manager.num_queues
        variance = (
            sum((queue_counts[i] - avg_count) ** 2 for i in range(manager.num_queues))
            / manager.num_queues
        )
        std_dev = variance**0.5
        coefficient_of_variation = (std_dev / avg_count) * 100

        print(f"\n📈 Distribution quality analysis:")
        print(f"   Expected per queue: {expected_per_queue:.1f} messages")
        print(f"   Actual range: {min_count}-{max_count} messages")
        print(f"   Standard deviation: {std_dev:.2f}")
        print(f"   Coefficient of variation: {coefficient_of_variation:.1f}%")

        # Verify distribution quality
        # 1. Coefficient of variation should be less than 15% (good uniformity)
        assert (
            coefficient_of_variation < 15.0
        ), f"Distribution not uniform enough, coefficient of variation: {coefficient_of_variation:.1f}%"

        # 2. Difference between max and min should not be too large
        max_min_ratio = max_count / min_count if min_count > 0 else float('inf')
        assert (
            max_min_ratio < 2.0
        ), f"Queue load difference too large, max/min ratio: {max_min_ratio:.2f}"

        # 3. Every queue should have messages
        assert min_count > 0, "Empty queue exists, distribution issue"

        # Verify routing consistency - Same UUID always routes to same queue
        print(f"\n🔍 Verifying routing consistency...")
        consistency_test_keys = uuid_keys[:50]  # Use first 50 UUIDs to test consistency

        for test_key in consistency_test_keys:
            # Calculate routing for same key multiple times, should always be same
            routes = [
                manager._hash_route(test_key) for _ in range(10)
            ]  # pylint: disable=protected-access
            assert (
                len(set(routes)) == 1
            ), f"Routing inconsistent: key={test_key}, routes={set(routes)}"

        print(f"   ✅ {len(consistency_test_keys)} UUID routing consistency verified")

        # Verify statistics
        stats = await manager.get_manager_stats()
        assert stats["total_delivered_messages"] == test_count, "Delivery statistics error"

        print(f"\n✅ Routing uniformity test passed:")
        print(f"   - Coefficient of variation: {coefficient_of_variation:.1f}% (< 15%)")
        print(f"   - Load ratio: {max_min_ratio:.2f} (< 2.0)")
        print(f"   - Routing consistency: 100%")

    finally:
        await manager.shutdown(ShutdownMode.HARD)


# ============ Extended test functions ============


async def test_concurrent_operations():
    """Detailed concurrent operations test"""
    manager = MsgGroupQueueManager(
        "concurrent_test", num_queues=5, max_total_messages=100
    )

    # Define producer and consumer tasks
    async def producer(producer_id: int, message_count: int):
        for i in range(message_count):
            group_key = f"producer_{producer_id}_user_{i % 10}"  # 10 different users
            message_data = {"producer": producer_id, "seq": i, "data": f"message_{i}"}
            success = await manager.deliver_message(group_key, message_data)
            if not success:
                print(f"Producer {producer_id} message {i} delivery failed")
            await asyncio.sleep(0.01)  # Small delay to simulate real scenario

    async def consumer(consumer_id: int, target_queues: List[int]):
        consumed = 0
        for queue_id in target_queues:
            while True:
                try:
                    message = await manager.get_by_queue(
                        queue_id, wait=True, timeout=0.1
                    )
                    if message is None:
                        break
                    consumed += 1
                    await asyncio.sleep(0.005)  # Simulate processing time
                except asyncio.TimeoutError:
                    break
        return consumed

    # Start 3 producers, each producing 5 messages
    producers = [producer(i, 5) for i in range(3)]
    await asyncio.gather(*producers)

    # Start 2 consumers, processing different queues
    consumer_tasks = [
        consumer(0, [0, 1, 2]),  # Consumer 0 processes queues 0,1,2
        consumer(1, [3, 4]),  # Consumer 1 processes queues 3,4
    ]
    consumed_counts = await asyncio.gather(*consumer_tasks)

    # Verify results
    total_consumed = sum(consumed_counts)
    stats = await manager.get_manager_stats()

    assert (
        stats["total_delivered_messages"] == 15
    ), f"Delivery count error: {stats['total_delivered_messages']}"
    assert total_consumed <= 15, f"Consumption count abnormal: {total_consumed}"

    await manager.shutdown(ShutdownMode.HARD)


async def test_real_world_scenario():
    """Real-world scenario test (simulating Kafka message processing)"""
    print("🚀 Simulating message burst...")
    manager = MsgGroupQueueManager(
        "kafka_simulator", num_queues=10, max_total_messages=200
    )

    # Simulate message burst
    delivered = 0
    for i in range(100):
        user_id = f"user_{i % 20}"  # 20 different users
        message = {
            "user_id": user_id,
            "timestamp": time.time(),
            "event_type": "click" if i % 3 == 0 else "view",
            "data": {"page": f"page_{i % 5}", "session": f"session_{i % 10}"},
        }
        success = await manager.deliver_message(user_id, message)
        if success:
            delivered += 1

    print(f"📊 Successfully delivered {delivered} messages")

    # Simulate consumer processing
    print("🔄 Simulating consumer processing...")
    total_consumed = 0

    # Start multiple consumers to process concurrently
    async def consumer_worker(queue_ids: List[int]):
        consumed = 0
        for queue_id in queue_ids:
            while True:
                try:
                    message = await manager.get_by_queue(
                        queue_id, wait=True, timeout=0.05
                    )
                    if message is None:
                        break
                    consumed += 1
                    await asyncio.sleep(0.001)  # Fast processing
                except asyncio.TimeoutError:
                    break
        return consumed

    # Start 5 consumer worker threads
    consumer_tasks = [
        consumer_worker([0, 1]),
        consumer_worker([2, 3]),
        consumer_worker([4, 5]),
        consumer_worker([6, 7]),
        consumer_worker([8, 9]),
    ]

    consumed_counts = await asyncio.gather(*consumer_tasks)
    total_consumed = sum(consumed_counts)

    print(f"📊 Total consumed {total_consumed} messages")

    # Check queue load distribution
    queue_info = await manager.get_queue_info()
    print("📈 Queue load distribution:")
    for info in queue_info:
        if info["current_size"] > 0:
            print(f"   Queue[{info['queue_id']}]: {info['current_size']} messages")

    # Test delivery rejection under high load
    print("⚠️ Testing delivery rejection under high load...")
    rejected = 0
    for i in range(100, 200):
        success = await manager.deliver_message(f"flood_user_{i}", f"flood_msg_{i}")
        if not success:
            rejected += 1

    # Cleanup
    print("🧹 Cleaning up remaining messages...")
    cleaned = 0
    for queue_id in range(manager.num_queues):
        while True:
            message = await manager.get_by_queue(queue_id, wait=False)
            if message is None:
                break
            cleaned += 1

    print(f"🧹 Cleaned up {cleaned} remaining messages")
    print("✅ Real-world scenario test completed!")

    await manager.shutdown(ShutdownMode.HARD)


async def test_queue_overflow_and_recovery():
    """Queue overflow recovery test"""
    manager = MsgGroupQueueManager("overflow_test", num_queues=3, max_total_messages=15)

    # Fill the queue
    print("📦 Filling queue...")
    delivered = 0
    rejected = 0

    for i in range(30):
        success = await manager.deliver_message(f"user_{i % 5}", f"msg_{i}")
        if success:
            delivered += 1
        else:
            rejected += 1

    print(f"📊 Delivery statistics: Success={delivered}, Rejected={rejected}")

    # Partial consumption to restore delivery capability
    print("🔄 Partial consumption to restore delivery capability...")
    consumed = 0
    for queue_id in range(manager.num_queues):
        # Consume a few messages from each queue
        for _ in range(2):
            message = await manager.get_by_queue(queue_id, wait=False)
            if message:
                consumed += 1

    print(f"📤 Consumed {consumed} messages for recovery")

    # Try to deliver again
    recovery_delivered = 0
    for i in range(5):
        success = await manager.deliver_message(
            f"recovery_user_{i}", f"recovery_msg_{i}"
        )
        if success:
            recovery_delivered += 1

    print(f"✅ Recovered delivery of {recovery_delivered} messages")

    await manager.shutdown(ShutdownMode.HARD)


async def test_shutdown_modes_integration():
    """Integration test for shutdown modes"""
    manager = MsgGroupQueueManager("shutdown_test", num_queues=3, max_total_messages=20)

    try:
        # Add some messages
        for i in range(5):
            await manager.deliver_message(f"user_{i}", f"msg_{i}")

        # Start a consumer task (simulating waiting consumption)
        async def consumer_task():
            consumed = 0
            for _ in range(10):  # Try to consume multiple times
                try:
                    message = await manager.get_by_queue(1, wait=True, timeout=0.5)
                    if message:
                        consumed += 1
                        await asyncio.sleep(0.1)  # Simulate processing time
                except asyncio.TimeoutError:
                    break
            return consumed

        # Start shutdown task
        async def shutdown_task():
            await asyncio.sleep(1.0)  # Wait for consumer to start
            result = await manager.shutdown(ShutdownMode.SOFT, max_delay_seconds=3.0)
            return result

        # Wait for tasks to complete
        results = await asyncio.gather(
            shutdown_task(), consumer_task(), return_exceptions=True
        )

        shutdown_result, consumed_count = results[0], results[1]

        # Verify results
        if isinstance(shutdown_result, Exception):
            raise shutdown_result
        if isinstance(consumed_count, Exception):
            raise consumed_count

        # shutdown_result could be bool or dict
        assert isinstance(
            shutdown_result, (bool, dict)
        ), f"Shutdown result type error: {type(shutdown_result)}"
        assert isinstance(
            consumed_count, int
        ), f"Consumption count type error: {type(consumed_count)}"
        assert consumed_count >= 0, f"Consumption count abnormal: {consumed_count}"

    except Exception:
        # Ensure resources are cleaned up even in case of exception
        try:
            await manager.shutdown(ShutdownMode.HARD)
        except Exception:
            pass  # Ignore errors during cleanup
        raise


# ============ Main test runner ============

if __name__ == "__main__":

    async def run_all_tests():
        print("🚀 Starting MsgGroupQueueManager refactored test suite...")
        print("=" * 60)

        test_results = []

        # Basic test suite
        basic_tests = [
            ("Basic functionality test", test_basic_functionality),
            ("Queue full test", test_queue_full_scenarios),
            ("Time window statistics test", test_time_window_stats),
            ("Basic concurrent operations test", test_basic_concurrent_operations),
            ("Basic Shutdown mode test", test_basic_shutdown_modes),
            ("Edge case test", test_edge_cases),
            ("Factory pattern test", test_factory_pattern),
            ("Timeout mechanism test", test_timeout_mechanism),
            ("Routing uniformity test", test_routing_uniformity),
        ]

        # Extended test suite
        extended_tests = [
            ("Detailed concurrent operations test", test_concurrent_operations),
            ("Real-world scenario test", test_real_world_scenario),
            ("Queue overflow recovery test", test_queue_overflow_and_recovery),
            ("Shutdown mode integration test", test_shutdown_modes_integration),
        ]

        # Run all tests
        all_tests = basic_tests + extended_tests

        for i, (test_name, test_func) in enumerate(all_tests, 1):
            print(f"\n{i}️⃣ {test_name}...")
            try:
                await test_func()
                test_results.append((test_name, "✅ Passed"))
                print(f"✅ {test_name} passed")
            except Exception as e:
                test_results.append((test_name, f"❌ Failed: {e}"))
                print(f"❌ {test_name} failed: {e}")

        # Test results summary
        print("\n" + "=" * 60)
        print("📊 Test results summary:")
        print("=" * 60)

        passed_count = 0
        total_count = len(test_results)

        for test_name, result in test_results:
            print(f"{result:<20} {test_name}")
            if "✅" in result:
                passed_count += 1

        print("=" * 60)
        print(f"📈 Total: {passed_count}/{total_count} tests passed")

        if passed_count == total_count:
            print("🎉 All tests passed! MsgGroupQueueManager is working correctly!")
        else:
            print(
                f"⚠️  {total_count - passed_count} tests failed, please check the error messages above"
            )

        return passed_count == total_count

    # Run all tests
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)