"""
Rate limiter decorator quick test

Test cases specifically for quickly verifying the functionality of the rate limiter decorator, avoiding long waiting times.
"""

import asyncio
import time
import pytest

from core.rate_limit.rate_limiter import rate_limit


class TestRateLimiterQuick:
    """Quick rate limiting test to avoid long waiting times"""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic functionality with a short time window"""
        call_times = []

        @rate_limit(max_rate=2, time_period=1)  # 2 times per second
        async def quick_test_func():
            call_times.append(time.time())
            return "success"

        start_time = time.time()

        # Call 3 times consecutively
        results = await asyncio.gather(
            quick_test_func(), quick_test_func(), quick_test_func()
        )

        total_time = time.time() - start_time

        assert len(results) == 3
        assert all(r == "success" for r in results)
        assert len(call_times) == 3

        # The third call should wait approximately 0.5 seconds (leaky bucket algorithm)
        assert total_time >= 0.4, f"Rate limiting wait time too short: {total_time} seconds"
        assert total_time < 0.8, f"Rate limiting wait time too long: {total_time} seconds"

        print(f"Basic functionality test completed, duration: {total_time:.3f} seconds")

    @pytest.mark.asyncio
    async def test_high_frequency_short_period(self):
        """Test high-frequency rate limiting (10 times per second)"""
        call_count = 0

        @rate_limit(max_rate=10, time_period=1)
        async def high_freq_func():
            nonlocal call_count
            call_count += 1
            return call_count

        start_time = time.time()

        # Attempt 15 calls, first 10 should execute immediately, last 5 need to wait
        tasks = [high_freq_func() for _ in range(15)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        assert len(results) == 15
        assert results == list(range(1, 16))

        # 15 calls at 10 per second should take about 0.5 seconds (leaky bucket algorithm)
        assert total_time >= 0.4, f"High-frequency rate limiting time too short: {total_time} seconds"
        assert total_time < 0.8, f"High-frequency rate limiting time too long: {total_time} seconds"

        print(f"High-frequency rate limiting test completed, duration: {total_time:.3f} seconds")

    @pytest.mark.asyncio
    async def test_concurrent_different_keys(self):
        """Test concurrent calls with different keys"""
        results = {}

        @rate_limit(max_rate=1, time_period=1, key_func=lambda user: f"user_{user}")
        async def user_func(user):
            if user not in results:
                results[user] = []
            results[user].append(time.time())
            return f"result_{user}"

        start_time = time.time()

        # 3 users each call once, should execute concurrently
        tasks = [
            user_func("A"),  # User A's call
            user_func("B"),  # User B's call
            user_func("C"),  # User C's call
        ]

        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Verify each user was called once
        assert len(results["A"]) == 1
        assert len(results["B"]) == 1
        assert len(results["C"]) == 1

        # Calls from different users should execute concurrently, total time should be very short
        assert total_time < 0.1, f"Concurrent calls with different keys took too long: {total_time} seconds"

        print(f"Concurrent different keys test completed, duration: {total_time:.3f} seconds")

    @pytest.mark.asyncio
    async def test_decorator_performance_quick(self):
        """Quick performance test"""

        # Function without decorator
        async def plain_func():
            await asyncio.sleep(0.001)  # Add small delay for more accurate measurement
            return 42

        # Function with decorator (loose limit)
        @rate_limit(max_rate=1000, time_period=1)
        async def decorated_func():
            await asyncio.sleep(0.001)  # Add small delay for more accurate measurement
            return 42

        # Test performance difference over 50 calls
        iterations = 50

        start_time = time.time()
        for _ in range(iterations):
            await plain_func()
        plain_time = time.time() - start_time

        start_time = time.time()
        for _ in range(iterations):
            await decorated_func()
        decorated_time = time.time() - start_time

        overhead = decorated_time - plain_time
        overhead_percent = (overhead / plain_time) * 100 if plain_time > 0 else 0

        print(
            f"Performance test - Without decorator: {plain_time:.4f} seconds, With decorator: {decorated_time:.4f} seconds"
        )
        print(f"Overhead: {overhead:.4f} seconds ({overhead_percent:.1f}%)")

        # Decorator overhead should be less than 200% (relatively loose limit)
        assert overhead_percent < 200, f"Decorator overhead too high: {overhead_percent:.1f}%"

    @pytest.mark.asyncio
    async def test_error_handling_quick(self):
        """Quick error handling test"""
        call_count = 0

        @rate_limit(max_rate=2, time_period=1)
        async def error_func(should_fail=False):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise RuntimeError("Test error")
            return "success"

        # Normal call
        result1 = await error_func(False)
        assert result1 == "success"

        # Error call (still consumes quota)
        with pytest.raises(RuntimeError):
            await error_func(True)

        # Third call should be rate-limited
        start_time = time.time()
        result3 = await error_func(False)
        elapsed = time.time() - start_time

        assert result3 == "success"
        assert elapsed >= 0.4, f"Rate limiting wait time too short: {elapsed} seconds"
        assert call_count == 3

        print(f"Error handling test completed, wait time: {elapsed:.3f} seconds")

    @pytest.mark.asyncio
    async def test_parameter_validation(self):
        """Test parameter validation"""

        # Test invalid max_rate
        with pytest.raises(ValueError, match="max_rate must be positive"):

            @rate_limit(max_rate=0, time_period=1)
            async def invalid_rate_func():
                pass

        with pytest.raises(ValueError, match="max_rate must be positive"):

            @rate_limit(max_rate=-1, time_period=1)
            async def negative_rate_func():
                pass

        # Test invalid time_period
        with pytest.raises(ValueError, match="time_period must be positive"):

            @rate_limit(max_rate=1, time_period=0)
            async def invalid_period_func():
                pass

        with pytest.raises(ValueError, match="time_period must be positive"):

            @rate_limit(max_rate=1, time_period=-1)
            async def negative_period_func():
                pass

        print("Parameter validation test completed")

    @pytest.mark.asyncio
    async def test_1_second_10_requests(self):
        """Test rate limiting for 10 requests within 1 second"""
        call_times = []

        @rate_limit(max_rate=10, time_period=1)
        async def high_freq_func():
            call_times.append(time.time())
            return len(call_times)

        start_time = time.time()

        # Call 12 times consecutively, first 10 should execute immediately, last 2 need to wait
        tasks = [high_freq_func() for _ in range(12)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        assert len(results) == 12
        assert results == list(range(1, 13))
        assert len(call_times) == 12

        # First 10 should complete relatively quickly, though some delay may occur due to leaky bucket algorithm
        first_10_time = call_times[9] - call_times[0]
        assert first_10_time < 0.5, f"First 10 calls took too long: {first_10_time:.3f} seconds"

        # 11th and 12th calls need to wait
        wait_time_11 = call_times[10] - call_times[9]
        wait_time_12 = call_times[11] - call_times[10]

        assert wait_time_11 >= 0.08, f"11th call wait time too short: {wait_time_11:.3f} seconds"
        assert wait_time_12 >= 0.08, f"12th call wait time too short: {wait_time_12:.3f} seconds"

        print(f"1 second 10 requests test completed, total duration: {total_time:.3f} seconds")
        print(f"First 10 duration: {first_10_time:.3f} seconds")
        print(f"11th call wait: {wait_time_11:.3f} seconds, 12th call wait: {wait_time_12:.3f} seconds")

    @pytest.mark.asyncio
    async def test_10_seconds_1_request(self):
        """Test rate limiting for 1 request within 10 seconds (quick version, using 0.5 seconds for simulation)"""
        call_times = []

        @rate_limit(max_rate=1, time_period=0.5)  # Use 0.5 seconds for quick testing
        async def low_freq_func():
            call_times.append(time.time())
            return len(call_times)

        start_time = time.time()

        # Call 3 times consecutively
        results = []
        for i in range(3):
            result = await low_freq_func()
            results.append(result)
            elapsed = time.time() - start_time
            print(f"Call {i+1} completed, duration: {elapsed:.3f} seconds")

        total_time = time.time() - start_time

        assert len(results) == 3
        assert results == [1, 2, 3]
        assert len(call_times) == 3

        # Check time intervals
        if len(call_times) >= 2:
            interval1 = call_times[1] - call_times[0]
            assert interval1 >= 0.4, f"Second call interval too short: {interval1:.3f} seconds"

        if len(call_times) >= 3:
            interval2 = call_times[2] - call_times[1]
            assert interval2 >= 0.4, f"Third call interval too short: {interval2:.3f} seconds"

        # Total time should be approximately 1 second (two 0.5-second intervals)
        assert total_time >= 0.9, f"Total time too short: {total_time:.3f} seconds"
        assert total_time < 1.5, f"Total time too long: {total_time:.3f} seconds"

        print(f"10 seconds 1 request test completed (simulated), total duration: {total_time:.3f} seconds")

    @pytest.mark.asyncio
    async def test_concurrent_performance_stress(self):
        """Concurrent performance stress test"""
        import asyncio

        call_count = 0
        lock = asyncio.Lock()

        @rate_limit(max_rate=20, time_period=1)  # Reduce limit to make effect more noticeable
        async def stress_func(task_id):
            nonlocal call_count
            async with lock:  # Use lock to ensure counter atomicity
                call_count += 1
                current_count = call_count
            await asyncio.sleep(0.001)  # Simulate some processing time
            return current_count

        start_time = time.time()

        # Create 50 concurrent tasks, 20 per second, should take about 2.5 seconds
        tasks = [stress_func(i) for i in range(50)]
        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        assert len(results) == 50
        # Due to concurrency and rate limiting, cannot guarantee completely unique results, but should have reasonable distribution
        unique_results = len(set(results))
        assert unique_results >= 25, f"Too few unique results: {unique_results}"
        assert max(results) == 50

        # 50 requests at 20 per second, leaky bucket algorithm may behave differently
        # Mainly verify that rate limiting is working, no need for exact timing
        assert total_time >= 1.0, f"Stress test completed too quickly: {total_time:.3f} seconds"
        assert total_time <= 4.0, f"Stress test completed too slowly: {total_time:.3f} seconds"

        # Calculate throughput
        throughput = 50 / total_time
        print(
            f"Concurrent stress test: 50 requests completed in {total_time:.3f} seconds, throughput: {throughput:.1f} req/s"
        )
        print(f"Number of unique results: {unique_results}/50")

        # Throughput verification (considering characteristics of leaky bucket algorithm, allow some margin)
        assert throughput <= 40, f"Throughput significantly exceeds limit: {throughput:.1f} req/s"
        assert throughput >= 10, f"Throughput too low: {throughput:.1f} req/s"

        # Main purpose is to verify the rate limiter works correctly and controls execution speed
        print("✓ Rate limiter is working properly, controlling execution speed")

    @pytest.mark.asyncio
    async def test_multiple_limiters_isolation(self):
        """Test isolation between multiple rate limiters"""
        results = {"fast": [], "slow": []}

        @rate_limit(
            max_rate=5, time_period=1, key_func=lambda service: f"service_{service}"
        )
        async def service_call(service: str):
            results[service].append(time.time())
            return f"{service}_result_{len(results[service])}"

        start_time = time.time()

        # Test fast and slow services simultaneously
        tasks = []

        # Fast service called 6 times (should have 1 wait)
        for i in range(6):
            tasks.append(service_call("fast"))

        # Slow service called 3 times (should all execute immediately due to independent rate limiter)
        for i in range(3):
            tasks.append(service_call("slow"))

        await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Verify call counts
        assert len(results["fast"]) == 6
        assert len(results["slow"]) == 3

        # Fast service should have wait time
        fast_total_time = results["fast"][-1] - results["fast"][0]
        slow_total_time = results["slow"][-1] - results["slow"][0]

        assert (
            fast_total_time >= 0.15
        ), f"Fast service rate limiting wait time too short: {fast_total_time:.3f} seconds"
        assert slow_total_time < 0.1, f"Slow service should not have noticeable wait: {slow_total_time:.3f} seconds"

        print(f"Multiple rate limiters isolation test completed, total duration: {total_time:.3f} seconds")
        print(
            f"Fast service total duration: {fast_total_time:.3f} seconds, slow service total duration: {slow_total_time:.3f} seconds"
        )


if __name__ == "__main__":
    # Run quick tests directly
    pytest.main([__file__, "-v", "-s"])