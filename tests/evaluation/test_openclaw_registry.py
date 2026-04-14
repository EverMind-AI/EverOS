"""
Task 1: Verify openclaw adapter registration and basic instantiation.
"""
from evaluation.src.adapters.registry import create_adapter, list_adapters


def test_openclaw_adapter_is_registered(tmp_path):
    config = {
        "adapter": "openclaw",
        "llm": {"provider": "openai", "model": "gpt-4o-mini"},
        "openclaw": {"repo_path": "/tmp/openclaw"},
    }
    adapter = create_adapter("openclaw", config, output_dir=tmp_path)
    assert adapter.get_system_info()["name"] == "OpenClaw"


def test_openclaw_in_adapter_list():
    assert "openclaw" in list_adapters()
