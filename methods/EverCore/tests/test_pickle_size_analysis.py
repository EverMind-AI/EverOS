"""
Pickle Serialization Size Analysis Test

Test the binary size of various types and sizes of objects after Pickle serialization
Including:
1. Basic data type size analysis
2. Complex object size analysis
3. Large data structure size analysis
4. Function and class object size analysis
5. Nested structure size analysis
"""

import asyncio
import pickle
import sys
import time
import pytest
from datetime import timedelta
from core.di.utils import get_bean
from core.observation.logger import get_logger
from core.cache.redis_cache_queue.redis_data_processor import RedisDataProcessor
from common_utils.datetime_utils import get_now_with_timezone

logger = get_logger(__name__)


def format_size(size_bytes: int) -> str:
    """Format byte size into human-readable format"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


class ComplexTestObject:
    """Complex test object"""

    def __init__(self, name: str, data_size: int):
        self.name = name
        self.created_at = get_now_with_timezone()
        self.id = id(self)
        self.data = list(range(data_size))
        self.metadata = {
            "version": "1.0",
            "type": "test_object",
            "size": data_size,
            "nested": {
                "level1": {"level2": {"level3": "deep_data"}},
                "timestamps": [
                    get_now_with_timezone() + timedelta(seconds=i) for i in range(10)
                ],
            },
        }
        # Use data types that cannot be JSON serialized
        self.complex_types = {
            "set_data": {1, 2, 3, 4, 5},
            "bytes_data": b"binary_data_example",
            "tuple_data": (1, 2, 3, 4, 5),
        }

    def multiply_value(self, value, multiplier=2):
        return value * multiplier

    def power_value(self, value, power=2):
        return value**power

    def custom_process(self, value):
        return f"processed_{value}_{self.name}_{len(self.data)}"

    def get_summary(self):
        return f"ComplexTestObject(name={self.name}, data_len={len(self.data)})"


class LargeDataContainer:
    """Large data container"""

    def __init__(self, size_mb: float):
        # Create data of specified size
        target_size = int(size_mb * 1024 * 1024 / 8)  # Assume each number is 8 bytes
        self.large_list = list(range(target_size))
        self.large_dict = {
            f"key_{i}": f"value_{i}_{'x' * 100}" for i in range(target_size // 100)
        }
        self.metadata = {
            "size_mb": size_mb,
            "created": get_now_with_timezone(),
            "type": "large_container",
        }


@pytest.mark.asyncio
async def test_basic_types_size():
    """Test serialization size of basic data types"""
    logger.info("Starting test for basic data type serialization size...")

    test_data = {
        "Empty string": "",
        "Short string": "hello",
        "Medium string": "x" * 100,
        "Long string": "x" * 1000,
        "Very long string": "x" * 10000,
        "Small integer": 42,
        "Large integer": 123456789012345,
        "Float": 3.14159265359,
        "Boolean True": True,
        "Boolean False": False,
        "None": None,
        "Empty list": [],
        "Small list": [1, 2, 3],
        "Medium list": list(range(100)),
        "Large list": list(range(1000)),
        "Empty dict": {},
        "Small dict": {"a": 1, "b": 2, "c": 3},
        "Medium dict": {f"key_{i}": i for i in range(100)},
        "Large dict": {f"key_{i}": f"value_{i}" for i in range(1000)},
    }

    logger.info("=" * 60)
    logger.info("Basic Data Type Pickle Serialization Size Analysis")
    logger.info("=" * 60)

    for name, data in test_data.items():
        # JSON serialization size (if possible)
        json_size = "N/A"
        try:
            import json

            json_data = json.dumps(data, ensure_ascii=False)
            json_size = format_size(len(json_data.encode('utf-8')))
        except (TypeError, ValueError):
            json_size = "Cannot be JSON serialized"

        # Pickle serialization size
        pickle_data = pickle.dumps(data)
        pickle_size = format_size(len(pickle_data))

        # Process using RedisDataProcessor
        processed_data = RedisDataProcessor.serialize_data(data)
        if isinstance(processed_data, bytes):
            processed_size = format_size(len(processed_data))
            serialization_type = "Pickle"
        else:
            processed_size = format_size(len(processed_data.encode('utf-8')))
            serialization_type = "JSON"

        logger.info(
            "%-15s | JSON: %-12s | Pickle: %-12s | Processor: %-12s (%s)",
            name,
            json_size,
            pickle_size,
            processed_size,
            serialization_type,
        )

    logger.info("✅ Basic data type size analysis completed")


@pytest.mark.asyncio
async def test_complex_objects_size():
    """Test serialization size of complex objects"""
    logger.info("Starting test for complex object serialization size...")

    test_objects = [
        ("Small complex object", ComplexTestObject("small", 10)),
        ("Medium complex object", ComplexTestObject("medium", 100)),
        ("Large complex object", ComplexTestObject("large", 1000)),
        ("Extra large complex object", ComplexTestObject("xlarge", 10000)),
    ]

    logger.info("=" * 60)
    logger.info("Complex Object Pickle Serialization Size Analysis")
    logger.info("=" * 60)

    for name, obj in test_objects:
        # Pickle serialization
        pickle_data = pickle.dumps(obj)
        pickle_size = format_size(len(pickle_data))

        # Process using RedisDataProcessor
        processed_data = RedisDataProcessor.serialize_data(obj)
        processed_size = format_size(len(processed_data))

        # Estimate object memory usage
        obj_memory = format_size(
            sys.getsizeof(obj) + sys.getsizeof(obj.data) + sys.getsizeof(obj.metadata)
        )

        logger.info(
            "%-15s | Memory: %-12s | Pickle: %-12s | Processor: %-12s",
            name,
            obj_memory,
            pickle_size,
            processed_size,
        )

    logger.info("✅ Complex object size analysis completed")


@pytest.mark.asyncio
async def test_large_data_structures():
    """Test serialization size of large data structures"""
    logger.info("Starting test for large data structure serialization size...")

    logger.info("=" * 60)
    logger.info("Large Data Structure Pickle Serialization Size Analysis")
    logger.info("=" * 60)

    # Test data structures of different sizes
    sizes = [0.1, 0.5, 1.0, 2.0, 5.0]  # MB

    for size_mb in sizes:
        logger.info(f"Testing {size_mb} MB data container...")

        try:
            # Create large data container
            container = LargeDataContainer(size_mb)

            # Pickle serialization
            start_time = time.time()
            pickle_data = pickle.dumps(container)
            pickle_time = time.time() - start_time
            pickle_size = format_size(len(pickle_data))

            # Process using RedisDataProcessor
            start_time = time.time()
            processed_data = RedisDataProcessor.serialize_data(container)
            process_time = time.time() - start_time
            processed_size = format_size(len(processed_data))

            # Compression ratio calculation
            original_estimate = size_mb * 1024 * 1024
            compression_ratio = len(pickle_data) / original_estimate

            logger.info(
                "%-8s MB | Pickle: %-12s (%.2fs) | Processor: %-12s (%.2fs) | Compression ratio: %.2f",
                f"{size_mb:.1f}",
                pickle_size,
                pickle_time,
                processed_size,
                process_time,
                compression_ratio,
            )

        except MemoryError:
            logger.warning(
                "%-8s MB | Insufficient memory, skipping test", f"{size_mb:.1f}"
            )
        except Exception as e:
            logger.error("%-8s MB | Test failed: %s", f"{size_mb:.1f}", str(e))

    logger.info("✅ Large data structure size analysis completed")


@pytest.mark.asyncio
async def test_function_and_class_objects():
    """Test serialization size of function and class objects"""
    logger.info("Starting test for function and class object serialization size...")

    # Various functions and class objects
    def simple_function(x):
        return x * 2

    def complex_function(x, y, z=10):
        """Complex function with docstring"""
        result = x + y + z
        for i in range(100):
            result += i
        return result

    class SimpleClass:
        def __init__(self, value):
            self.value = value

        def method(self):
            return self.value * 2

    class ComplexClass:
        """Complex class with multiple methods"""

        class_var = "shared_data"

        def __init__(self, name, data):
            self.name = name
            self.data = data
            self.timestamp = get_now_with_timezone()

        def method1(self):
            return len(self.data)

        def method2(self, multiplier=2):
            return [x * multiplier for x in self.data]

        @staticmethod
        def static_method():
            return "static_result"

        @classmethod
        def class_method(cls):
            return cls.class_var

    test_objects = [
        ("Simple function", simple_function),
        ("Complex function", complex_function),
        ("Simple class instance", SimpleClass(42)),
        ("Complex class instance", ComplexClass("test", list(range(100)))),
        (
            "Dictionary containing set",
            {
                "set_data": {1, 2, 3, 4, 5},
                "bytes_data": b"function_test_binary",
                "tuple_data": (1, 2, 3),
            },
        ),
        (
            "Mixed object",
            {
                "functions": [simple_function, complex_function],
                "objects": [SimpleClass(i) for i in range(10)],
                "data": list(range(1000)),
                "complex_types": {
                    "set_data": {10, 20, 30},
                    "bytes_data": b"mixed_binary_data",
                },
                "metadata": {"type": "mixed", "count": 10},
            },
        ),
    ]

    logger.info("=" * 60)
    logger.info("Function and Class Object Pickle Serialization Size Analysis")
    logger.info("=" * 60)

    for name, obj in test_objects:
        try:
            # Pickle serialization
            pickle_data = pickle.dumps(obj)
            pickle_size = format_size(len(pickle_data))

            # Process using RedisDataProcessor
            processed_data = RedisDataProcessor.serialize_data(obj)
            processed_size = format_size(len(processed_data))

            # Object memory usage
            obj_memory = format_size(sys.getsizeof(obj))

            logger.info(
                "%-15s | Memory: %-12s | Pickle: %-12s | Processor: %-12s",
                name,
                obj_memory,
                pickle_size,
                processed_size,
            )

        except Exception as e:
            logger.error("%-15s | Serialization failed: %s", name, str(e))

    logger.info("✅ Function and class object size analysis completed")


@pytest.mark.asyncio
async def test_nested_structures():
    """Test serialization size of nested structures"""
    logger.info("Starting test for nested structure serialization size...")

    # Create nested structures of different depths
    def create_nested_dict(depth: int, width: int = 3):
        """Create nested dictionary with specified depth and width"""
        if depth == 0:
            return f"leaf_value_{width}"

        return {f"key_{i}": create_nested_dict(depth - 1, width) for i in range(width)}

    def create_nested_list(depth: int, width: int = 3):
        """Create nested list with specified depth and width"""
        if depth == 0:
            return f"leaf_{width}"

        return [create_nested_list(depth - 1, width) for _ in range(width)]

    test_structures = [
        ("Nested dict - depth 2", create_nested_dict(2, 3)),
        ("Nested dict - depth 3", create_nested_dict(3, 3)),
        ("Nested dict - depth 4", create_nested_dict(4, 2)),
        ("Nested list - depth 2", create_nested_list(2, 3)),
        ("Nested list - depth 3", create_nested_list(3, 3)),
        ("Nested list - depth 4", create_nested_list(4, 2)),
        (
            "Mixed nesting",
            {
                "dict_part": create_nested_dict(3, 2),
                "list_part": create_nested_list(3, 2),
                "objects": [ComplexTestObject(f"nested_{i}", 50) for i in range(5)],
                "complex_types": {
                    "nested_set": {frozenset({1, 2}), frozenset({3, 4})},
                    "nested_bytes": b"nested_binary_data",
                    "nested_tuple": ((1, 2), (3, 4), (5, 6)),
                },
            },
        ),
    ]

    logger.info("=" * 60)
    logger.info("Nested Structure Pickle Serialization Size Analysis")
    logger.info("=" * 60)

    for name, structure in test_structures:
        try:
            # Pickle serialization
            start_time = time.time()
            pickle_data = pickle.dumps(structure)
            pickle_time = time.time() - start_time
            pickle_size = format_size(len(pickle_data))

            # Process using RedisDataProcessor
            start_time = time.time()
            processed_data = RedisDataProcessor.serialize_data(structure)
            process_time = time.time() - start_time
            processed_size = format_size(len(processed_data))

            logger.info(
                "%-20s | Pickle: %-12s (%.3fs) | Processor: %-12s (%.3fs)",
                name,
                pickle_size,
                pickle_time,
                processed_size,
                process_time,
            )

        except Exception as e:
            logger.error("%-20s | Serialization failed: %s", name, str(e))

    logger.info("✅ Nested structure size analysis completed")


@pytest.mark.asyncio
async def test_redis_storage_efficiency():
    """Test Redis storage efficiency"""
    logger.info("Starting test for Redis storage efficiency...")

    factory = get_bean("redis_length_cache_factory")
    cache = await factory.create_cache_manager(max_length=1000, expire_minutes=10)

    test_key = "pickle_size_test"
    await cache.clear_queue(test_key)

    # Test data
    test_data = [
        ("Small JSON object", {"name": "test", "value": 123}),
        (
            "Large JSON object",
            {"data": list(range(1000)), "metadata": {"type": "large"}},
        ),
        ("Small Pickle object", ComplexTestObject("small", 10)),
        ("Large Pickle object", ComplexTestObject("large", 1000)),
    ]

    logger.info("=" * 60)
    logger.info("Redis Storage Efficiency Analysis")
    logger.info("=" * 60)

    for name, data in test_data:
        # Serialization size
        processed_data = RedisDataProcessor.process_data_for_storage(data)
        storage_size = (
            len(processed_data)
            if isinstance(processed_data, bytes)
            else len(processed_data.encode('utf-8'))
        )

        # Store to Redis
        start_time = time.time()
        success = await cache.append(test_key, data)
        store_time = time.time() - start_time

        # Read from Redis
        start_time = time.time()
        retrieved_data = await cache.get_by_timestamp_range(test_key, limit=1)
        read_time = time.time() - start_time

        if success and retrieved_data:
            logger.info(
                "%-15s | Storage: %-12s | Write: %.3fs | Read: %.3fs | Status: ✅",
                name,
                format_size(storage_size),
                store_time,
                read_time,
            )
        else:
            logger.error(
                "%-15s | Storage: %-12s | Status: ❌", name, format_size(storage_size)
            )

        # Clean up individual test data
        await cache.clear_queue(test_key)

    logger.info("✅ Redis storage efficiency analysis completed")


async def main():
    """Main test function"""
    logger.info("=" * 60)
    logger.info("Pickle Serialization Size Analysis Test Started")
    logger.info("=" * 60)

    try:
        await test_basic_types_size()
        await test_complex_objects_size()
        await test_large_data_structures()
        await test_function_and_class_objects()
        await test_nested_structures()
        await test_redis_storage_efficiency()

        logger.info("=" * 60)
        logger.info("✅ All Pickle size analysis tests passed")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("❌ Error occurred during test: %s", str(e))
        raise


if __name__ == "__main__":
    asyncio.run(main())
