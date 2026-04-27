"""
Hello World task

Provides a simple Hello World task
"""

from core.asynctasks.task_manager import task
from typing import Any


@task()
async def hello_world(data: Any) -> Any:
    return f"hello world: {data}"
