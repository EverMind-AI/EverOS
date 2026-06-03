import pytest

from demo.utils.simple_memory_manager import SimpleMemoryManager


class _AcceptedResponse:
    status_code = 202

    def raise_for_status(self):
        return None


class _AsyncClient:
    def __init__(self):
        self.posts = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, json):
        self.posts.append((url, json))
        return _AcceptedResponse()


@pytest.mark.asyncio
async def test_store_treats_accepted_background_response_as_success(monkeypatch):
    client = _AsyncClient()
    monkeypatch.setattr(
        'demo.utils.simple_memory_manager.httpx.AsyncClient',
        lambda *args, **kwargs: client,
    )

    manager = SimpleMemoryManager(user_id='user-1')
    manager._settings_initialized = True

    assert await manager.store('background extraction') is True
    assert len(client.posts) == 1
