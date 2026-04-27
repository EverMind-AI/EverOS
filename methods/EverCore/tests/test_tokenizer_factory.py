"""
TokenizerFactory Test

Test tokenizer factory functionality, including:
- Tokenizer loading and caching
- Default encodings preload
- Cache management

Usage:
    python src/bootstrap.py tests/test_tokenizer_factory.py
"""

from core.di.utils import get_bean_by_type
from core.observation.logger import get_logger
from core.component.llm.tokenizer.tokenizer_factory import (
    TokenizerFactory,
    DEFAULT_TIKTOKEN_ENCODINGS,
)

logger = get_logger(__name__)


class TestTokenizerFactory:
    """TokenizerFactory Test Class"""

    def test_get_tokenizer_factory_from_di(self):
        """Test getting TokenizerFactory from DI container"""
        print("\n🧪 Test getting TokenizerFactory from DI container")

        factory = get_bean_by_type(TokenizerFactory)

        assert (
            factory is not None
        ), "TokenizerFactory should be available in DI container"
        assert isinstance(
            factory, TokenizerFactory
        ), "Should be TokenizerFactory instance"

        print("✅ Successfully got TokenizerFactory from DI container")
        print(f"   - Factory instance: {factory}")
        print(f"   - Cached tokenizer count: {factory.get_cached_tokenizer_count()}")

    def test_get_tokenizer_from_tiktoken(self):
        """Test getting tokenizer from tiktoken"""
        print("\n🧪 Test getting tokenizer from tiktoken")

        factory: TokenizerFactory = get_bean_by_type(TokenizerFactory)

        # Test getting o200k_base encoding
        tokenizer = factory.get_tokenizer_from_tiktoken("o200k_base")

        assert tokenizer is not None, "Tokenizer should not be None"

        # Test tokenization
        test_text = "Hello, world! 你好，世界！"
        tokens = tokenizer.encode(test_text)

        print("✅ Successfully got tokenizer from tiktoken")
        print(f"   - Encoding: o200k_base")
        print(f"   - Test text: {test_text}")
        print(f"   - Token count: {len(tokens)}")
        print(
            f"   - Tokens: {tokens[:10]}..."
            if len(tokens) > 10
            else f"   - Tokens: {tokens}"
        )  # noqa: G004

        # Verify decoding
        decoded_text = tokenizer.decode(tokens)
        assert decoded_text == test_text, "Decoded text should match original"
        print(f"   - Decoded text: {decoded_text}")

    def test_tokenizer_caching(self):
        """Test tokenizer caching functionality"""
        print("\n🧪 Test tokenizer caching functionality")

        factory: TokenizerFactory = get_bean_by_type(TokenizerFactory)

        # Clear cache first
        factory.clear_cache()
        assert (
            factory.get_cached_tokenizer_count() == 0
        ), "Cache should be empty after clear"
        print("   - Cache cleared")

        # Get tokenizer first time
        tokenizer1 = factory.get_tokenizer_from_tiktoken("o200k_base")
        count_after_first = factory.get_cached_tokenizer_count()
        print(f"   - After first load: {count_after_first} tokenizer(s) cached")

        # Get tokenizer second time (should be from cache)
        tokenizer2 = factory.get_tokenizer_from_tiktoken("o200k_base")
        count_after_second = factory.get_cached_tokenizer_count()
        print(f"   - After second load: {count_after_second} tokenizer(s) cached")

        # Verify same instance (from cache)
        assert tokenizer1 is tokenizer2, "Should return same cached instance"
        assert (
            count_after_first == count_after_second
        ), "Cache count should not increase"

        print("✅ Tokenizer caching works correctly")
        print(f"   - Same instance returned: {tokenizer1 is tokenizer2}")

    def test_load_default_encodings(self):
        """Test preloading default encodings"""
        print("\n🧪 Test preloading default encodings")

        factory: TokenizerFactory = get_bean_by_type(TokenizerFactory)

        # Clear cache first
        factory.clear_cache()
        print(f"   - Cache cleared, count: {factory.get_cached_tokenizer_count()}")

        # Load default encodings
        print(f"   - Default encodings to load: {DEFAULT_TIKTOKEN_ENCODINGS}")
        factory.load_default_encodings()

        # Verify all default encodings are loaded
        cached_count = factory.get_cached_tokenizer_count()
        expected_count = len(DEFAULT_TIKTOKEN_ENCODINGS)

        print(f"   - Expected count: {expected_count}")
        print(f"   - Actual cached count: {cached_count}")

        assert (
            cached_count >= expected_count
        ), f"Should have at least {expected_count} tokenizers cached"

        print("✅ Default encodings preloaded successfully")

    def test_multiple_encodings(self):
        """Test loading multiple different encodings"""
        print("\n🧪 Test loading multiple different encodings")

        factory: TokenizerFactory = get_bean_by_type(TokenizerFactory)

        # Clear cache first
        factory.clear_cache()

        encodings_to_test = ["o200k_base", "cl100k_base"]
        test_text = "Hello, world!"

        for encoding in encodings_to_test:
            tokenizer = factory.get_tokenizer_from_tiktoken(encoding)
            tokens = tokenizer.encode(test_text)
            print(f"   - {encoding}: {len(tokens)} tokens for '{test_text}'")

        # Verify all are cached
        cached_count = factory.get_cached_tokenizer_count()
        assert cached_count == len(
            encodings_to_test
        ), f"Should have {len(encodings_to_test)} tokenizers cached"

        print("✅ Multiple encodings loaded and cached")
        print(f"   - Total cached: {cached_count}")

    def test_tokenizer_consistency_with_conv_memcell_extractor(self):
        """Test that tokenizer usage is consistent with ConvMemCellExtractor"""
        print("\n🧪 Test tokenizer consistency with ConvMemCellExtractor")

        factory: TokenizerFactory = get_bean_by_type(TokenizerFactory)

        # This is the same way ConvMemCellExtractor gets tokenizer
        tokenizer = factory.get_tokenizer_from_tiktoken("o200k_base")

        # Test with conversation-like content
        messages = [
            {"sender_name": "Alice", "content": "Hello, how are you?"},
            {"sender_name": "Bob", "content": "I'm fine, thanks! How about you?"},
            {"sender_name": "Alice", "content": "Great! Let's discuss the project."},
        ]

        total_tokens = 0
        for msg in messages:
            speaker = msg.get('sender_name', '')
            content = msg.get('content', '')
            text = f"{speaker}: {content}" if speaker else content
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
            print(f"   - '{text}' -> {len(tokens)} tokens")

        print("✅ Tokenizer works for conversation content")
        print(f"   - Total tokens: {total_tokens}")


def run_all_tests():
    """Run all tests"""
    print("🚀 Starting TokenizerFactory tests")
    print("=" * 60)

    test_instance = TestTokenizerFactory()

    try:
        test_instance.test_get_tokenizer_factory_from_di()
        test_instance.test_get_tokenizer_from_tiktoken()
        test_instance.test_tokenizer_caching()
        test_instance.test_load_default_encodings()
        test_instance.test_multiple_encodings()
        test_instance.test_tokenizer_consistency_with_conv_memcell_extractor()

        print("\n" + "=" * 60)
        print("🎉 All TokenizerFactory tests completed!")

    except Exception as e:
        logger.error("❌ Test execution failed: %s", e)
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
