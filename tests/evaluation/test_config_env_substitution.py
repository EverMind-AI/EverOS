"""
Regression test for YAML env-var substitution.

The regex used by evaluation/src/utils/config.py must accept
- ${VAR}              required, no default
- ${VAR:default}      with default
- ${VAR:}             empty default (was broken: the regex required 1+
                      chars after ':' so the whole pattern failed to
                      match and the literal string was kept as-is, which
                      in turn sent '${SOPH_API_KEY:}' to OpenClaw as the
                      API key and got rejected with 422)
"""
import os

from evaluation.src.utils.config import _replace_env_vars


def test_replace_env_vars_required(monkeypatch):
    monkeypatch.setenv("MY_TEST_VAR", "hello")
    assert _replace_env_vars("${MY_TEST_VAR}") == "hello"


def test_replace_env_vars_default_when_unset(monkeypatch):
    monkeypatch.delenv("MY_TEST_VAR", raising=False)
    assert _replace_env_vars("${MY_TEST_VAR:fallback}") == "fallback"


def test_replace_env_vars_env_wins_over_default(monkeypatch):
    monkeypatch.setenv("MY_TEST_VAR", "winner")
    assert _replace_env_vars("${MY_TEST_VAR:loser}") == "winner"


def test_replace_env_vars_empty_default_when_unset(monkeypatch):
    monkeypatch.delenv("MY_TEST_VAR", raising=False)
    # This is the regression case.
    assert _replace_env_vars("${MY_TEST_VAR:}") == ""


def test_replace_env_vars_empty_default_with_env(monkeypatch):
    monkeypatch.setenv("MY_TEST_VAR", "from-env")
    assert _replace_env_vars("${MY_TEST_VAR:}") == "from-env"


def test_replace_env_vars_nested(monkeypatch):
    monkeypatch.setenv("A", "x")
    monkeypatch.setenv("B", "y")
    assert _replace_env_vars({"k": ["${A}", "${B:default}", "${MISSING:}"]}) == {
        "k": ["x", "y", ""]
    }
