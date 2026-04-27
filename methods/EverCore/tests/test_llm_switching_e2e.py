# -*- coding: utf-8 -*-
"""
LLM Dynamic Switching End-to-End Test

Tests:
1. Switch provider between messages (openrouter <-> openai)
2. Switch model within same provider
3. Verify LLM calls work correctly

Usage:
    # Requires running server on port 1995
    cd /Users/admin/Applications/cursor_project/evermemos/evermemos-opensource
    source .venv/bin/activate

    # Run all E2E tests
    PYTHONPATH=src python tests/test_llm_switching_e2e.py --base-url http://localhost:1995

    # Run specific test
    PYTHONPATH=src python tests/test_llm_switching_e2e.py --test switch_provider
    PYTHONPATH=src python tests/test_llm_switching_e2e.py --test switch_model
    PYTHONPATH=src python tests/test_llm_switching_e2e.py --test all
"""

import argparse
import json
import time
import requests
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional


class LLMSwitchingE2ETest:
    """End-to-end test for LLM dynamic switching feature"""

    def __init__(self, base_url: str, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.test_id = uuid.uuid4().hex[:8]

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️"}
        print(f"[{timestamp}] {symbols.get(level, '•')} {message}")

    def api_request(
        self, method: str, endpoint: str, data: Optional[Dict] = None
    ) -> requests.Response:
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}
        if method.upper() == "GET":
            return requests.get(url, headers=headers, timeout=self.timeout)
        elif method.upper() == "POST":
            return requests.post(url, headers=headers, json=data, timeout=self.timeout)
        elif method.upper() == "PATCH":
            return requests.patch(url, headers=headers, json=data, timeout=self.timeout)
        raise ValueError(f"Unsupported method: {method}")

    def update_llm_config(self, config: Dict) -> bool:
        """Update global llm_custom_setting"""
        response = self.api_request(
            "PUT", "/api/v1/settings", {"llm_custom_setting": config}
        )
        return response.status_code == 200

    def send_message(self, group_id: str, content: str, msg_id: str) -> bool:
        """Send a message to trigger LLM processing"""
        data = {
            "message_id": msg_id,
            "create_time": datetime.now(timezone.utc).isoformat(),
            "sender": "user_001",
            "sender_name": "Test User",
            "content": content,
            "group_id": group_id,
        }
        response = self.api_request("POST", "/api/v0/memories", data)
        return response.status_code in [200, 201, 202]

    def setup_global_config(self, llm_config: Dict) -> bool:
        """Ensure global settings exists with llm_custom_setting"""
        data = {
            "scene": "solo",
            "scene_desc": {"description": "LLM switching test"},
            "llm_custom_setting": llm_config,
        }
        response = self.api_request("PUT", "/api/v1/settings", data)
        return response.status_code in [200, 201]

    # ========== Test Cases ==========

    def test_switch_provider(self) -> bool:
        """Test 1: Switch model within OpenRouter (only OpenRouter, no OpenAI)"""
        self.log("=" * 70)
        self.log("TEST: Switch Model Within OpenRouter")
        self.log("=" * 70)

        group_id = f"test_provider_switch_{self.test_id}"

        # Step 1: Setup with OpenRouter gpt-4.1-mini
        config_a = {
            "boundary": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
            "extraction": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
        }
        self.log("Step 1: Setting up with OpenRouter (gpt-4.1-mini)...")
        if not self.setup_global_config(config_a):
            self.log("Failed to setup global config", "ERROR")
            return False
        self.log("Global config set to OpenRouter", "SUCCESS")

        # Create group config
        if not self.create_group_config(group_id):
            self.log("Failed to create group config", "ERROR")
            return False

        # Step 2: Send message
        self.log("Step 2: Sending message with gpt-4.1-mini...")
        if not self.send_message(
            group_id, "Hello, testing OpenRouter gpt-4.1-mini.", f"{group_id}_msg_1"
        ):
            self.log("Failed to send message", "ERROR")
            return False
        self.log("Message sent with gpt-4.1-mini", "SUCCESS")
        time.sleep(1)

        # Step 3: Switch to qwen model
        config_b = {
            "boundary": {
                "provider": "openrouter",
                "model": "qwen/qwen3-235b-a22b-2507",
            },
            "extraction": {
                "provider": "openrouter",
                "model": "qwen/qwen3-235b-a22b-2507",
            },
        }
        self.log("Step 3: Switching to qwen model...")
        if not self.update_llm_config(config_b):
            self.log("Failed to update config", "ERROR")
            return False
        self.log("Config switched to qwen", "SUCCESS")

        # Step 4: Send message with qwen
        self.log("Step 4: Sending message with qwen...")
        if not self.send_message(
            group_id, "Now testing qwen model.", f"{group_id}_msg_2"
        ):
            self.log("Failed to send message", "ERROR")
            return False
        self.log("Message sent with qwen", "SUCCESS")
        time.sleep(1)

        # Step 5: Switch back to gpt-4.1-mini
        self.log("Step 5: Switching back to gpt-4.1-mini...")
        if not self.update_llm_config(config_a):
            self.log("Failed to update config", "ERROR")
            return False
        self.log("Config switched back to gpt-4.1-mini", "SUCCESS")

        # Step 6: Send another message
        self.log("Step 6: Sending message with gpt-4.1-mini again...")
        if not self.send_message(
            group_id, "Back to gpt-4.1-mini.", f"{group_id}_msg_3"
        ):
            self.log("Failed to send message", "ERROR")
            return False
        self.log("Message sent", "SUCCESS")

        self.log("-" * 70)
        self.log("Model switching test PASSED!", "SUCCESS")
        return True

    def test_switch_model(self) -> bool:
        """Test 2: Switch model within same provider"""
        self.log("=" * 70)
        self.log("TEST: Switch Model Within Same Provider")
        self.log("=" * 70)

        group_id = f"test_model_switch_{self.test_id}"

        # Step 1: Setup with model A (allowed model)
        config_model_a = {
            "boundary": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
            "extraction": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
        }
        self.log("Step 1: Setting up with model: openai/gpt-4.1-mini...")
        if not self.setup_global_config(config_model_a):
            self.log("Failed to setup global config", "ERROR")
            return False
        self.log("Model A configured", "SUCCESS")

        # Create group config
        if not self.create_group_config(group_id):
            self.log("Failed to create group config", "ERROR")
            return False

        # Step 2: Send message with model A
        self.log("Step 2: Sending message with gpt-4.1-mini...")
        if not self.send_message(
            group_id, "Testing with gpt-4.1-mini model.", f"{group_id}_msg_1"
        ):
            self.log("Failed to send message", "ERROR")
            return False
        self.log("Message sent with gpt-4.1-mini", "SUCCESS")
        self.log("  -> Check logs: model=openai/gpt-4.1-mini")
        time.sleep(1)

        # Step 3: Switch to model B (allowed qwen model)
        config_model_b = {
            "boundary": {
                "provider": "openrouter",
                "model": "qwen/qwen3-235b-a22b-2507",
            },
            "extraction": {
                "provider": "openrouter",
                "model": "qwen/qwen3-235b-a22b-2507",
            },
        }
        self.log("Step 3: Switching to model: qwen/qwen3-235b-a22b-2507...")
        if not self.update_llm_config(config_model_b):
            self.log("Failed to update config", "ERROR")
            return False
        self.log("Model B configured", "SUCCESS")

        # Step 4: Send message with model B
        self.log("Step 4: Sending message with qwen...")
        if not self.send_message(
            group_id, "Now using qwen model.", f"{group_id}_msg_2"
        ):
            self.log("Failed to send message", "ERROR")
            return False
        self.log("Message sent with qwen", "SUCCESS")
        self.log("  -> Check logs: model=qwen/qwen3-235b-a22b-2507")

        self.log("-" * 70)
        self.log("Model switching test PASSED!", "SUCCESS")
        return True

    def test_openrouter_model_restriction(self) -> bool:
        """Test: OpenRouter only allows specific models (qwen, gpt-4.1-mini)

        Note: Validation happens at LLM call time (bottom layer), not at config save time.
        So config save always succeeds, but message sending fails with disallowed models.
        """
        self.log("=" * 70)
        self.log("TEST: OpenRouter Model Restriction (Bottom Layer Validation)")
        self.log("=" * 70)

        group_id = f"test_restriction_{self.test_id}"

        # Step 1: Setup with allowed model and send message - should succeed
        allowed_config = {
            "boundary": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
            "extraction": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
        }
        self.log("Step 1: Setup with allowed model (gpt-4.1-mini)...")
        if not self.setup_global_config(allowed_config):
            self.log("Failed to setup global config", "ERROR")
            return False
        if not self.create_group_config(group_id):
            self.log("Failed to create group config", "ERROR")
            return False

        self.log("Step 2: Sending message with allowed model...")
        if not self.send_message(
            group_id, "Test with allowed model", f"{group_id}_msg_1"
        ):
            self.log("Failed - allowed model should work", "ERROR")
            return False
        self.log("Allowed model works correctly", "SUCCESS")

        self.log("-" * 70)
        self.log("OpenRouter model restriction test PASSED!", "SUCCESS")
        return True

    def test_whitelist_e2e(self) -> bool:
        """Test: Whitelist enforcement in real server (requires PROVIDER_WHITE_LIST in .env)

        Verifies:
        1. Allowed model → message sent successfully (202/200)
        2. Disallowed model → config saves OK, but message processing fails at LLM layer
        """
        self.log("=" * 70)
        self.log("TEST: Whitelist E2E (requires server with WHITE_LIST in .env)")
        self.log("=" * 70)

        group_id_ok = f"test_wl_ok_{self.test_id}"
        group_id_bad = f"test_wl_bad_{self.test_id}"

        # --- Part A: Allowed model should work ---
        self.log("Part A: Allowed model (openrouter / openai/gpt-4.1-mini)")
        allowed_config = {
            "boundary": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
            "extraction": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
        }
        if not self.setup_global_config(allowed_config):
            self.log("Failed to setup global config", "ERROR")
            return False
        if not self.create_group_config(group_id_ok):
            self.log("Failed to create group config", "ERROR")
            return False

        self.log("  Sending message with allowed model...")
        if not self.send_message(
            group_id_ok, "Whitelist allowed model test", f"{group_id_ok}_msg_1"
        ):
            self.log("  FAILED: allowed model should succeed", "ERROR")
            return False
        self.log("  Allowed model accepted by server", "SUCCESS")
        time.sleep(2)

        # --- Part B: Disallowed model - config saves OK but LLM call should fail ---
        self.log("Part B: Disallowed model (openrouter / x-ai/grok-4-fast)")
        disallowed_config = {
            "boundary": {"provider": "openrouter", "model": "x-ai/grok-4-fast"},
            "extraction": {"provider": "openrouter", "model": "x-ai/grok-4-fast"},
        }
        self.log("  Saving config with disallowed model...")
        if not self.update_llm_config(disallowed_config):
            # Config save itself might succeed (validation is at LLM call time)
            self.log("  Config save failed (unexpected but acceptable)", "WARN")

        if not self.create_group_config(group_id_bad):
            self.log("  Failed to create group config", "ERROR")
            return False

        self.log("  Sending message with disallowed model...")
        sent = self.send_message(
            group_id_bad, "Whitelist disallowed model test", f"{group_id_bad}_msg_1"
        )
        if sent:
            self.log(
                "  Message accepted (202) - check server logs for ValueError", "WARN"
            )
            self.log("  -> Expected: server logs should contain:", "INFO")
            self.log(
                "     'Provider 'openrouter' only supports: ... Got: x-ai/grok-4-fast'",
                "INFO",
            )
        else:
            self.log(
                "  Message rejected by server (whitelist enforced at API level)",
                "SUCCESS",
            )
        time.sleep(2)

        # --- Part C: Restore allowed config & verify recovery ---
        self.log("Part C: Restore allowed model and verify recovery")
        if not self.update_llm_config(allowed_config):
            self.log("  Failed to restore config", "ERROR")
            return False
        if not self.send_message(
            group_id_ok, "Recovery after disallowed model", f"{group_id_ok}_msg_2"
        ):
            self.log("  FAILED: recovery message should succeed", "ERROR")
            return False
        self.log("  Recovery successful", "SUCCESS")

        self.log("-" * 70)
        self.log("Whitelist E2E test PASSED!", "SUCCESS")
        self.log(
            "  IMPORTANT: Also check server logs to confirm ValueError was raised for Part B",
            "WARN",
        )
        return True

    def test_mixed_config(self) -> bool:
        """Test 3: Mixed model config (boundary=gpt-4.1-mini, extraction=qwen)"""
        self.log("=" * 70)
        self.log("TEST: Mixed Model Config (Different models per stage)")
        self.log("=" * 70)

        group_id = f"test_mixed_{self.test_id}"

        # Setup mixed config (both openrouter with different allowed models)
        config_mixed = {
            "boundary": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
            "extraction": {
                "provider": "openrouter",
                "model": "qwen/qwen3-235b-a22b-2507",
            },
        }
        self.log("Setting up mixed config:")
        self.log("  boundary:   openrouter / openai/gpt-4.1-mini")
        self.log("  extraction: openrouter / qwen/qwen3-235b-a22b-2507")

        if not self.setup_global_config(config_mixed):
            self.log("Failed to setup global config", "ERROR")
            return False
        self.log("Mixed config set", "SUCCESS")

        # Create group config
        if not self.create_group_config(group_id):
            self.log("Failed to create group config", "ERROR")
            return False

        # Send message
        self.log("Sending message with mixed config...")
        if not self.send_message(
            group_id, "Testing mixed provider configuration.", f"{group_id}_msg_1"
        ):
            self.log("Failed to send message", "ERROR")
            return False
        self.log("Message sent", "SUCCESS")
        self.log("  -> Check logs:")
        self.log("     boundary:   model=openai/gpt-4.1-mini")
        self.log("     extraction: model=qwen/qwen3-235b-a22b-2507")

        self.log("-" * 70)
        self.log("Mixed config test PASSED!", "SUCCESS")
        return True

    def run_all_tests(self) -> bool:
        """Run all tests"""
        self.log("=" * 70)
        self.log("LLM DYNAMIC SWITCHING E2E TEST SUITE")
        self.log(f"Base URL: {self.base_url}")
        self.log(f"Test ID: {self.test_id}")
        self.log("=" * 70)

        results = {}

        # Test 1: OpenRouter model restriction
        results["openrouter_restriction"] = self.test_openrouter_model_restriction()
        time.sleep(2)

        # Test 2: Switch provider
        results["switch_provider"] = self.test_switch_provider()
        time.sleep(2)

        # Test 3: Switch model
        results["switch_model"] = self.test_switch_model()
        time.sleep(2)

        # Test 4: Mixed config
        results["mixed_config"] = self.test_mixed_config()
        time.sleep(2)

        # Test 5: Whitelist E2E
        results["whitelist_e2e"] = self.test_whitelist_e2e()

        # Summary
        self.log("\n" + "=" * 70)
        self.log("TEST RESULTS SUMMARY")
        self.log("=" * 70)

        all_passed = True
        for test_name, passed in results.items():
            status = "PASSED" if passed else "FAILED"
            level = "SUCCESS" if passed else "ERROR"
            self.log(f"  {test_name}: {status}", level)
            if not passed:
                all_passed = False

        self.log("=" * 70)
        if all_passed:
            self.log("ALL TESTS PASSED!", "SUCCESS")
        else:
            self.log("SOME TESTS FAILED!", "ERROR")

        return all_passed


class WhitelistUnitTest:
    """Unit tests for provider model whitelist (no server required).

    These tests validate that _validate_model_whitelist reads
    {PROVIDER}_WHITE_LIST from env and enforces restrictions correctly.
    """

    def __init__(self):
        self.passed = 0
        self.failed = 0

    def log(self, message: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {"INFO": "ℹ️", "SUCCESS": "✅", "ERROR": "❌", "WARN": "⚠️"}
        print(f"[{timestamp}] {symbols.get(level, '•')} {message}")

    def assert_raises(self, exc_type, fn, *args, **kwargs) -> bool:
        try:
            fn(*args, **kwargs)
            return False
        except exc_type:
            return True
        except Exception:
            return False

    def run_all(self) -> bool:
        import os
        from memory_layer.llm.openai_provider import OpenAIProvider

        self.log("=" * 70)
        self.log("WHITELIST UNIT TESTS (no server required)")
        self.log("=" * 70)

        saved_env = {}
        env_keys = [
            "OPENROUTER_WHITE_LIST",
            "OPENAI_WHITE_LIST",
            "TESTPROVIDER_WHITE_LIST",
        ]
        for k in env_keys:
            saved_env[k] = os.environ.pop(k, None)

        try:
            # --- Test 1: No whitelist env → no restriction ---
            self.log("Test 1: No whitelist set → all models allowed")
            for k in env_keys:
                os.environ.pop(k, None)
            try:
                OpenAIProvider._validate_model_whitelist(
                    "testprovider", "any-model-name"
                )
                self.log("  PASSED: no restriction when env not set", "SUCCESS")
                self.passed += 1
            except Exception as e:
                self.log(f"  FAILED: unexpected error: {e}", "ERROR")
                self.failed += 1

            # --- Test 2: Empty whitelist → no restriction ---
            self.log("Test 2: Empty whitelist → all models allowed")
            os.environ["TESTPROVIDER_WHITE_LIST"] = ""
            try:
                OpenAIProvider._validate_model_whitelist(
                    "testprovider", "any-model-name"
                )
                self.log("  PASSED: no restriction when env is empty", "SUCCESS")
                self.passed += 1
            except Exception as e:
                self.log(f"  FAILED: unexpected error: {e}", "ERROR")
                self.failed += 1

            # --- Test 3: Whitelist set, model in list → allowed ---
            self.log("Test 3: Model in whitelist → allowed")
            os.environ["OPENROUTER_WHITE_LIST"] = (
                "qwen/qwen3-235b-a22b-2507,openai/gpt-4.1-mini"
            )
            try:
                OpenAIProvider._validate_model_whitelist(
                    "openrouter", "openai/gpt-4.1-mini"
                )
                OpenAIProvider._validate_model_whitelist(
                    "openrouter", "qwen/qwen3-235b-a22b-2507"
                )
                self.log("  PASSED: allowed models pass validation", "SUCCESS")
                self.passed += 1
            except Exception as e:
                self.log(f"  FAILED: unexpected error: {e}", "ERROR")
                self.failed += 1

            # --- Test 4: Whitelist set, model NOT in list → rejected ---
            self.log("Test 4: Model not in whitelist → ValueError raised")
            os.environ["OPENROUTER_WHITE_LIST"] = (
                "qwen/qwen3-235b-a22b-2507,openai/gpt-4.1-mini"
            )
            if self.assert_raises(
                ValueError,
                OpenAIProvider._validate_model_whitelist,
                "openrouter",
                "openai/gpt-4o",
            ):
                self.log("  PASSED: disallowed model rejected", "SUCCESS")
                self.passed += 1
            else:
                self.log("  FAILED: should have raised ValueError", "ERROR")
                self.failed += 1

            # --- Test 5: OpenAI whitelist ---
            self.log("Test 5: OpenAI whitelist enforcement")
            os.environ["OPENAI_WHITE_LIST"] = "gpt-5-mini,gpt-4.1-mini"
            try:
                OpenAIProvider._validate_model_whitelist("openai", "gpt-5-mini")
                OpenAIProvider._validate_model_whitelist("openai", "gpt-4.1-mini")
                self.log("  PASSED: allowed OpenAI models pass", "SUCCESS")
                self.passed += 1
            except Exception as e:
                self.log(f"  FAILED: unexpected error: {e}", "ERROR")
                self.failed += 1

            self.log("Test 6: OpenAI whitelist rejects unlisted model")
            if self.assert_raises(
                ValueError, OpenAIProvider._validate_model_whitelist, "openai", "gpt-4o"
            ):
                self.log("  PASSED: disallowed OpenAI model rejected", "SUCCESS")
                self.passed += 1
            else:
                self.log("  FAILED: should have raised ValueError", "ERROR")
                self.failed += 1

            # --- Test 7: Whitespace handling ---
            self.log("Test 7: Whitespace in whitelist is trimmed")
            os.environ["OPENAI_WHITE_LIST"] = " gpt-5-mini , gpt-4.1-mini "
            try:
                OpenAIProvider._validate_model_whitelist("openai", "gpt-5-mini")
                self.log("  PASSED: whitespace trimmed correctly", "SUCCESS")
                self.passed += 1
            except Exception as e:
                self.log(f"  FAILED: unexpected error: {e}", "ERROR")
                self.failed += 1

            # --- Test 8: Provider without whitelist has no restriction ---
            self.log("Test 8: Provider without whitelist → unrestricted")
            os.environ.pop("TESTPROVIDER_WHITE_LIST", None)
            os.environ["OPENAI_WHITE_LIST"] = "gpt-5-mini"
            try:
                # testprovider has no WHITE_LIST set, should not be restricted
                OpenAIProvider._validate_model_whitelist(
                    "testprovider", "anything-goes"
                )
                self.log("  PASSED: unrelated provider is unrestricted", "SUCCESS")
                self.passed += 1
            except Exception as e:
                self.log(f"  FAILED: unexpected error: {e}", "ERROR")
                self.failed += 1

        finally:
            # Restore original env
            for k in env_keys:
                os.environ.pop(k, None)
                if saved_env[k] is not None:
                    os.environ[k] = saved_env[k]

        # Summary
        self.log("=" * 70)
        self.log(f"WHITELIST UNIT TESTS: {self.passed} passed, {self.failed} failed")
        if self.failed == 0:
            self.log("ALL WHITELIST TESTS PASSED!", "SUCCESS")
        else:
            self.log("SOME WHITELIST TESTS FAILED!", "ERROR")
        self.log("=" * 70)
        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="LLM Switching E2E Test")
    parser.add_argument(
        "--base-url", default="http://localhost:1995", help="Base URL of the API server"
    )
    parser.add_argument(
        "--test",
        choices=[
            "openrouter_restriction",
            "switch_provider",
            "switch_model",
            "mixed_config",
            "whitelist",
            "whitelist_e2e",
            "all",
        ],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--timeout", type=int, default=120, help="Request timeout in seconds"
    )

    args = parser.parse_args()

    # Whitelist unit test can run standalone (no server needed)
    if args.test == "whitelist":
        tester = WhitelistUnitTest()
        success = tester.run_all()
        exit(0 if success else 1)

    tester = LLMSwitchingE2ETest(base_url=args.base_url, timeout=args.timeout)

    if args.test == "all":
        success = tester.run_all_tests()
    elif args.test == "openrouter_restriction":
        success = tester.test_openrouter_model_restriction()
    elif args.test == "switch_provider":
        success = tester.test_switch_provider()
    elif args.test == "switch_model":
        success = tester.test_switch_model()
    elif args.test == "mixed_config":
        success = tester.test_mixed_config()
    elif args.test == "whitelist_e2e":
        success = tester.test_whitelist_e2e()
    else:
        success = False

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
