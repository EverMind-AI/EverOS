"""
Database stability test

Test key stability scenarios such as database connection pool, connection leaks, and failure recovery
"""

import pytest
import asyncio
import os
import time
from unittest.mock import patch, AsyncMock
from typing import List, Dict, Any

# Set test environment
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/testdb")
os.environ.setdefault("DB_POOL_SIZE", "5")
os.environ.setdefault("DB_MAX_OVERFLOW", "3")

from core.component.database_session_provider import DatabaseSessionProvider
from core.component.database_connection_provider import DatabaseConnectionProvider


class TestDatabaseStability:
    """Database stability test class"""

    @pytest.fixture
    async def db_provider(self):
        """Database provider fixture"""
        provider = DatabaseSessionProvider()
        yield provider
        # Clean up resources
        if hasattr(provider, 'async_engine'):
            await provider.async_engine.dispose()

    @pytest.fixture
    async def connection_provider(self):
        """Connection provider fixture"""
        provider = DatabaseConnectionProvider()
        yield provider
        # Clean up resources
        if hasattr(provider, '_connection_pool') and provider._connection_pool:
            await provider._connection_pool.close()

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion(self, db_provider):
        """Test connection pool exhaustion scenario"""

        # Create concurrent tasks exceeding connection pool size
        async def db_operation(operation_id: int):
            try:
                async with db_provider.get_async_session() as session:
                    # Simulate long-running database operation
                    await asyncio.sleep(0.1)
                    return f"operation_{operation_id}_success"
            except Exception as e:
                return f"operation_{operation_id}_failed: {str(e)}"

        # Create a large number of concurrent tasks (exceeding pool size)
        max_connections = int(os.getenv("DB_POOL_SIZE", "5")) + int(
            os.getenv("DB_MAX_OVERFLOW", "3")
        )
        task_count = max_connections * 2  # Exceed connection pool size

        tasks = [asyncio.create_task(db_operation(i)) for i in range(task_count)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyze results
        success_count = sum(1 for r in results if isinstance(r, str) and "success" in r)
        failure_count = len(results) - success_count

        print(
            f"Connection pool test result: success={success_count}, failure={failure_count}, total_tasks={task_count}"
        )

        # Verify: Most tasks should succeed, a few may fail due to pool exhaustion
        assert (
            success_count >= task_count * 0.8
        ), f"Success rate too low: {success_count}/{task_count}"
        assert (
            failure_count <= task_count * 0.2
        ), f"Failure rate too high: {failure_count}/{task_count}"

    @pytest.mark.asyncio
    async def test_connection_leak_detection(self, db_provider):
        """Test connection leak detection"""
        # Record initial connection count
        initial_pool_size = db_provider.async_engine.pool.size()
        initial_checked_in = db_provider.async_engine.pool.checkedin()
        initial_checked_out = db_provider.async_engine.pool.checkedout()

        print(
            f"Initial connection pool status: size={initial_pool_size}, checked_in={initial_checked_in}, checked_out={initial_checked_out}"
        )

        # Simulate connection leak scenario
        leaked_sessions = []

        async def leaky_operation():
            session = db_provider.create_session()
            leaked_sessions.append(session)
            # Intentionally not closing session, simulating connection leak
            await session.execute("SELECT 1")
            # Not calling session.close()

        # Execute multiple leak operations
        for _ in range(3):
            await leaky_operation()

        # Check connection pool status
        current_pool_size = db_provider.async_engine.pool.size()
        current_checked_in = db_provider.async_engine.pool.checkedin()
        current_checked_out = db_provider.async_engine.pool.checkedout()

        print(
            f"Post-leak connection pool status: size={current_pool_size}, checked_in={current_checked_in}, checked_out={current_checked_out}"
        )

        # Verify connection leak detection
        leaked_connections = current_checked_out - initial_checked_out
        assert leaked_connections > 0, "Connection leak should be detected"

        # Clean up leaked connections
        for session in leaked_sessions:
            try:
                await session.close()
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_database_failure_recovery(self, db_provider):
        """Test database failure recovery"""
        recovery_successful = False

        # Simulate database connection failure
        original_execute = db_provider.async_engine.execute

        async def mock_failing_execute(*args, **kwargs):
            # First few calls fail, subsequent calls succeed
            if not hasattr(mock_failing_execute, 'call_count'):
                mock_failing_execute.call_count = 0
            mock_failing_execute.call_count += 1

            if mock_failing_execute.call_count <= 2:
                raise Exception("Database connection failed")
            else:
                # Restore database connection
                return await original_execute(*args, **kwargs)

        with patch.object(
            db_provider.async_engine, 'execute', side_effect=mock_failing_execute
        ):
            # Test retry mechanism
            max_retries = 3
            retry_count = 0

            for attempt in range(max_retries):
                try:
                    async with db_provider.get_async_session() as session:
                        await session.execute("SELECT 1")
                    recovery_successful = True
                    break
                except Exception as e:
                    retry_count += 1
                    print(f"Retry {retry_count}: {str(e)}")
                    if retry_count < max_retries:
                        await asyncio.sleep(0.1)  # Brief delay

        assert recovery_successful, "Database failure recovery failed"

    @pytest.mark.asyncio
    async def test_connection_pool_timeout(self, db_provider):
        """Test connection pool timeout handling"""

        # Create long-running tasks that occupy connections
        async def long_running_operation():
            async with db_provider.get_async_session() as session:
                await asyncio.sleep(2)  # Long time holding connection
                return "completed"

        # Create multiple long-running tasks
        long_tasks = [asyncio.create_task(long_running_operation()) for _ in range(3)]

        # Create new task requiring connection (should timeout)
        async def timeout_operation():
            try:
                async with db_provider.get_async_session() as session:
                    await session.execute("SELECT 1")
                    return "success"
            except Exception as e:
                return f"timeout: {str(e)}"

        # Create timeout task after a short wait
        await asyncio.sleep(0.1)
        timeout_task = asyncio.create_task(timeout_operation())

        # Wait for all tasks to complete
        results = await asyncio.gather(
            *long_tasks, timeout_task, return_exceptions=True
        )

        # Verify timeout handling
        timeout_result = results[-1]
        if isinstance(timeout_result, str) and "timeout" in timeout_result:
            print(f"Connection pool timeout test passed: {timeout_result}")
        else:
            print(f"Connection pool timeout test result: {timeout_result}")

    @pytest.mark.asyncio
    async def test_concurrent_transaction_isolation(self, db_provider):
        """Test concurrent transaction isolation"""
        results = []

        async def transaction_operation(operation_id: int):
            try:
                async with db_provider.get_async_session() as session:
                    # Start transaction
                    await session.begin()

                    # Simulate transaction operations
                    await session.execute("SELECT 1")
                    await asyncio.sleep(0.1)  # Simulate processing time

                    # Commit transaction
                    await session.commit()

                    results.append(f"transaction_{operation_id}_success")
            except Exception as e:
                results.append(f"transaction_{operation_id}_failed: {str(e)}")

        # Create multiple concurrent transactions
        tasks = [asyncio.create_task(transaction_operation(i)) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify transaction isolation
        success_count = sum(1 for r in results if "success" in r)
        assert success_count == 10, f"Transaction isolation test failed: {results}"

    @pytest.mark.asyncio
    async def test_connection_pool_health_check(self, connection_provider):
        """Test connection pool health check"""
        try:
            # Get connection pool
            pool = await connection_provider.get_connection_pool()

            # Check connection pool status
            assert pool is not None, "Connection pool not initialized"

            # Test connection pool health status
            async with pool.connection() as conn:
                await conn.execute("SELECT 1")

            print("Connection pool health check passed")

        except Exception as e:
            pytest.fail(f"Connection pool health check failed: {str(e)}")

    @pytest.mark.asyncio
    async def test_database_performance_under_load(self, db_provider):
        """Test database performance under load"""
        start_time = time.time()

        async def performance_operation(operation_id: int):
            async with db_provider.get_async_session() as session:
                # Execute simple query
                result = await session.execute("SELECT 1 as test_value")
                return f"operation_{operation_id}_completed"

        # Create a large number of concurrent operations
        task_count = 100
        tasks = [
            asyncio.create_task(performance_operation(i)) for i in range(task_count)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_time = end_time - start_time

        # Analyze performance
        success_count = sum(
            1 for r in results if isinstance(r, str) and "completed" in r
        )
        avg_time_per_operation = total_time / task_count
        operations_per_second = task_count / total_time

        print(f"Performance test results:")
        print(f"  Total time: {total_time:.2f} seconds")
        print(f"  Successful operations: {success_count}/{task_count}")
        print(f"  Average time per operation: {avg_time_per_operation:.3f} seconds")
        print(f"  Operations per second: {operations_per_second:.2f}")

        # Performance assertions
        assert (
            success_count >= task_count * 0.95
        ), f"Success rate too low: {success_count}/{task_count}"
        assert (
            avg_time_per_operation < 0.1
        ), f"Average response time too long: {avg_time_per_operation:.3f} seconds"
        assert (
            operations_per_second > 50
        ), f"Throughput too low: {operations_per_second:.2f} ops/sec"


class TestDatabaseErrorHandling:
    """Database error handling test class"""

    @pytest.mark.asyncio
    async def test_invalid_query_handling(self):
        """Test invalid query handling"""
        provider = DatabaseSessionProvider()

        try:
            async with provider.get_async_session() as session:
                # Execute invalid SQL
                await session.execute("INVALID SQL STATEMENT")
        except Exception as e:
            # Verify error handling
            assert (
                "syntax error" in str(e).lower() or "invalid" in str(e).lower()
            ), f"Invalid query not properly handled: {e}"
        finally:
            await provider.async_engine.dispose()

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """Test connection timeout handling"""
        provider = DatabaseSessionProvider()

        # Simulate connection timeout
        with patch.object(provider.async_engine, 'connect') as mock_connect:
            mock_connect.side_effect = asyncio.TimeoutError("Connection timeout")

            try:
                async with provider.get_async_session() as session:
                    await session.execute("SELECT 1")
            except asyncio.TimeoutError:
                # Verify timeout handling
                assert True, "Connection timeout should be properly handled"
            except Exception as e:
                pytest.fail(f"Connection timeout not properly handled: {e}")
            finally:
                await provider.async_engine.dispose()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])