import os
import uuid
import importlib
import pkgutil
from pathlib import Path
from typing import Any, Dict, Optional, List, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from arq import create_pool, ArqRedis
from arq.connections import RedisSettings
from arq.jobs import Job
from arq.worker import Worker, Function, func as arq_func

from core.asynctasks.task_scan_registry import TaskScanDirectoriesRegistry
from core.context.context_manager import ContextManager
from core.context.context import get_current_user_info
from core.di.decorators import component
from core.observation.logger import get_logger
from core.authorize.enums import Role

logger = get_logger(__name__)


class TaskStatus(Enum):
    """Task status enumeration"""

    PENDING = "pending"  # Pending execution
    RUNNING = "running"  # Running
    SUCCESS = "success"  # Execution succeeded
    FAILED = "failed"  # Execution failed
    CANCELLED = "cancelled"  # Cancelled


@dataclass
class TaskResult:
    """Task result"""

    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    retry_count: int = 0
    user_id: Optional[int] = None
    user_context: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Retry configuration"""

    max_retries: int = 1
    retry_delay: float = 1.0  # seconds
    exponential_backoff: bool = True
    max_retry_delay: float = 60.0  # seconds


@dataclass
class TaskFunction:
    """Task function"""

    name: str
    coroutine: Callable  # Function wrapped with user context handling
    original_func: Callable  # Original function
    timeout: Optional[float] = None
    retry_config: Optional[RetryConfig] = None

    def to_arq_function(self) -> Function:
        """Convert to arq function"""
        return arq_func(
            self.coroutine,
            name=self.name,
            max_tries=self.retry_config.max_retries,
            timeout=self.timeout,
        )

    def __call__(self, *args, **kwargs) -> Any:
        """Call task function"""
        return self.original_func(*args, **kwargs)


@component(name="task_manager")
class TaskManager:
    """
    Asynchronous task manager

    Implements asynchronous task management based on the arq framework, providing functions such as task addition, result retrieval, and task deletion.
    Uses ContextManager to automatically inject database sessions and user context.
    """

    def __init__(self, context_manager: ContextManager):
        """Initialize task manager"""
        self._pool: Optional[ArqRedis] = None
        self._worker: Optional[Worker] = None
        self._redis_settings = self._get_redis_settings()
        self._context_manager = context_manager

        # Task function registry
        self._task_registry: Dict[str, TaskFunction] = {}

        # Default retry configuration
        self._default_retry_config = RetryConfig()

        logger.info("Task manager initialization completed")

    def _get_current_user_info(self) -> Optional[Dict[str, Any]]:
        """Get current user information"""
        return get_current_user_info()

    def _get_current_user_id(self) -> Optional[int]:
        """Get current user ID"""
        user_info = self._get_current_user_info()
        return user_info.get("user_id") if user_info else None

    def _get_redis_settings(self) -> RedisSettings:
        """
        Get Redis configuration from environment variables

        Returns:
            RedisSettings: Redis connection configuration
        """
        return RedisSettings(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            database=int(os.getenv("REDIS_DB", "0")),
            password=os.getenv("REDIS_PASSWORD"),
            ssl=os.getenv("REDIS_SSL", "false").lower() == "true",
            username=os.getenv("REDIS_USERNAME"),
        )

    async def _get_pool(self) -> ArqRedis:
        """
        Get Redis connection pool

        Returns:
            ArqRedis: Redis connection pool
        """
        if self._pool is None:
            self._pool = await create_pool(self._redis_settings)
        return self._pool

    async def close(self) -> None:
        """Close connection pool"""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
        logger.info("Task manager connection closed")

    def register_task(self, task_function: TaskFunction) -> None:
        """
        Register task function

        Args:
            task_function: Task function to register
        """
        self._task_registry[task_function.name] = task_function
        logger.info(f"Task registered: {task_function.name}")

    def scan_and_register_tasks(self, registry: TaskScanDirectoriesRegistry) -> None:
        """
        Scan task directories and automatically register tasks

        Args:
            registry: Task scan directory registry
        """
        for directory in registry.get_scan_directories():
            self._scan_directory_for_tasks(directory)

    def _scan_directory_for_tasks(self, directory: str) -> None:
        """
        Scan a single directory for tasks

        Args:
            directory: Directory path to scan
        """
        try:
            # Convert to absolute path
            from common_utils.project_path import src_dir

            relative_path = Path(directory).resolve().relative_to(src_dir)
            package_name = ".".join(relative_path.parts)

            logger.info(f"Scanning task package: {package_name}")

            # Import package and scan
            try:
                package = importlib.import_module(package_name)

                # Scan all modules in the package
                if hasattr(package, '__path__'):
                    # This is a package, recursively scan all submodules
                    for _, module_name, _ in pkgutil.walk_packages(
                        package.__path__, prefix=f"{package_name}."
                    ):
                        try:
                            module = importlib.import_module(module_name)
                            self._scan_module_for_tasks(module)
                        except Exception as e:
                            logger.error(
                                f"Failed to import module: {module_name}, error: {e}"
                            )
                else:
                    # This is a module, scan directly
                    self._scan_module_for_tasks(package)

            except Exception as e:
                logger.error(f"Failed to import package: {package_name}, error: {e}")

        except Exception as e:
            logger.error(f"Failed to scan directory: {directory}, error: {e}")

    def _scan_module_for_tasks(self, module: Any) -> None:
        """
        Scan module for task functions

        Args:
            module: Module object to scan
        """
        try:
            # Get all attributes in the module
            for attr_name in dir(module):
                # Skip private and special attributes
                if attr_name.startswith('_'):
                    continue

                try:
                    attr = getattr(module, attr_name)

                    # Check if it's a TaskFunction instance
                    if isinstance(attr, TaskFunction):
                        self.register_task(attr)
                        logger.info(
                            f"Task found in module {module.__name__}: {attr.name}"
                        )

                except Exception as e:
                    logger.debug(
                        f"Failed to get module attribute: {module.__name__}.{attr_name}, error: {e}"
                    )

        except Exception as e:
            logger.error(f"Failed to scan module tasks: {module.__name__}, error: {e}")

    async def enqueue_task(
        self,
        task_name: Union[str, TaskFunction, Any],
        *args,
        task_id: Optional[str] = None,
        delay: Optional[float] = None,
        retry_config: Optional[RetryConfig] = None,
        user_id: Optional[int] = None,
        user_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Add task to queue

        Args:
            task_name: Task name
            *args: Task arguments
            task_id: Task ID (optional, auto-generated if not provided)
            delay: Delay execution time (seconds)
            retry_config: Retry configuration (optional)
            user_id: User ID (optional, obtained from current context if not provided)
            user_data: User data (optional, obtained from current context if not provided)
            metadata: Task metadata (optional)
            **kwargs: Task keyword arguments

        Returns:
            str: Task ID
        """
        if isinstance(task_name, TaskFunction):
            task_name = task_name.name
        elif isinstance(task_name, str):
            pass
        else:
            raise ValueError(f"Type error: {type(task_name)}")

        assert task_name in self._task_registry, f"Task not found: {task_name}"
        task_function = self._task_registry[task_name]
        current_retry_config = (
            retry_config or task_function.retry_config or self._default_retry_config
        )

        # Generate task ID
        if task_id is None:
            task_id = str(uuid.uuid4())

        # Get user context (if not provided)
        if user_data is None:
            current_user_context = self._get_current_user_info()
            if current_user_context is not None:
                user_data = current_user_context.copy()
            elif user_id is not None:
                user_data = {"user_id": user_id, "role": Role.USER}

        if user_data is None and user_id is None:
            # Try to get user_id from current context
            current_user_id = self._get_current_user_id()
            if current_user_id is not None:
                user_data = {"user_id": current_user_id, "role": Role.USER}

        # 🔧 Get current app_info_context (containing task_id, etc.)
        from core.context.context import get_current_app_info

        current_app_info = get_current_app_info()

        # Prepare task context
        task_context = {
            "user_data": user_data,
            "app_info": current_app_info,  # 🔧 Copy app_info_context
            "metadata": metadata or {},
            "task_id": task_id,
            "retry_config": current_retry_config,
        }

        # Get connection pool
        pool = await self._get_pool()

        # Calculate delay time
        defer_until = None
        if delay is not None:
            from common_utils.datetime_utils import get_now_with_timezone

            defer_until = get_now_with_timezone() + timedelta(seconds=delay)

        # Enqueue task
        job = await pool.enqueue_job(
            task_name,
            task_context,
            *args,
            _job_id=task_id,
            _defer_until=defer_until,
            **kwargs,
        )

        user_id_for_log = user_data.get("user_id") if user_data else "unknown"
        logger.info(
            f"Task added to queue: {task_id}, task name: {task_name}, user: {user_id_for_log}"
        )
        return task_id

    async def execute_task_with_context(
        self,
        task_func: Callable,
        context: Dict[str, Any],
        task_context: Dict[str, Any],
        *args,
        force_new_session: bool = False,
        **kwargs,
    ) -> Any:
        """
        Execute task within context

        Args:
            task_func: Task function
            context: Task execution context (redis, job_id, job_try, score, enqueue_time)
            task_context: Business context (user data, task ID, etc.)
            *args: Task arguments
            force_new_session: Whether to force creation of a new session (default False, to avoid unnecessary session creation)
            **kwargs: Task keyword arguments

        Returns:
            Any: Task execution result
        """
        user_data = task_context.get("user_data")
        app_info = task_context.get("app_info")  # 🔧 Get saved app_info_context

        # 🔧 Restore app_info_context (containing task_id, etc.)
        if app_info:
            from core.context.context import set_current_app_info

            set_current_app_info(app_info)
            logger.debug(f"🔧 app_info_context restored: {app_info}")

        # Use ContextManager to execute task, automatically injecting user context and database session
        # 🔧 Configurable session isolation: Only force new session when explicitly needed
        result = await self._context_manager.run_with_full_context(
            task_func,
            *args,
            user_data=user_data,
            auto_inherit_user=False,
            auto_commit=True,
            force_new_session=force_new_session,  # 🔑 Key: Configurable session isolation
            **kwargs,
        )

        task_id = task_context.get("task_id")
        user_id = user_data.get("user_id") if user_data else "unknown"
        logger.info(
            f"Task execution completed (independent session): {task_id}, user: {user_id}"
        )
        return result

    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Get task result

        Args:
            task_id: Task ID

        Returns:
            Optional[TaskResult]: Task result, returns None if task does not exist
        """
        pool = await self._get_pool()

        try:
            job = Job(task_id, pool)

            # Get task information
            info = await job.info()
            if info is None:
                return None

            # Construct task result
            status = self._map_arq_status_to_task_status(info.job_status)

            # Try to get user information from task context (this may not be available as arq does not save our custom context)
            user_id = None
            user_context_data = None

            result = TaskResult(
                task_id=task_id,
                status=status,
                result=info.result if status == TaskStatus.SUCCESS else None,
                error=str(info.result) if status == TaskStatus.FAILED else None,
                created_at=info.enqueue_time,
                started_at=info.start_time,
                finished_at=info.finish_time,
                retry_count=info.job_try or 0,
                user_id=user_id,
                user_context=user_context_data,
            )

            return result

        except Exception as e:
            logger.error(f"Failed to get task result: {task_id}, error: {str(e)}")
            return None

    def _map_arq_status_to_task_status(self, arq_status: str) -> TaskStatus:
        """
        Map arq status to task status

        Args:
            arq_status: arq task status

        Returns:
            TaskStatus: Task status
        """
        mapping = {
            "queued": TaskStatus.PENDING,
            "deferred": TaskStatus.PENDING,
            "in_progress": TaskStatus.RUNNING,
            "complete": TaskStatus.SUCCESS,
            "failed": TaskStatus.FAILED,
            "cancelled": TaskStatus.CANCELLED,
        }
        return mapping.get(arq_status, TaskStatus.PENDING)

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel task

        Args:
            task_id: Task ID

        Returns:
            bool: Whether cancellation was successful
        """
        pool = await self._get_pool()

        try:
            job = Job(task_id, pool)
            await job.abort()
            logger.info(f"Task cancelled: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel task: {task_id}, error: {str(e)}")
            return False

    async def delete_task(self, task_id: str) -> bool:
        """
        Delete task record

        Args:
            task_id: Task ID

        Returns:
            bool: Whether deletion was successful
        """
        pool = await self._get_pool()

        try:
            # Delete task record
            await pool.delete(f"arq:job:{task_id}")
            logger.info(f"Task deleted: {task_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete task: {task_id}, error: {str(e)}")
            return False

    async def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[TaskResult]:
        """
        List tasks

        Note: Due to arq limitations, filtering tasks by user ID is not effective.
        This method returns all tasks and filters at the application layer.
        In production environments, it is recommended to use a dedicated task status storage system.

        Args:
            status: Task status filter (optional)
            user_id: User ID filter (optional, may be ineffective due to arq limitations)
            limit: Limit on number of returned items

        Returns:
            List[TaskResult]: List of tasks
        """
        pool = await self._get_pool()

        try:
            # Get all task keys
            keys = await pool.keys("arq:job:*")
            tasks = []

            for key in keys[:limit]:
                task_id = key.decode().split(":")[-1]
                task_result = await self.get_task_result(task_id)

                if task_result is not None:
                    # Apply filter conditions
                    if status is not None and task_result.status != status:
                        continue

                    # Note: Due to arq limitations, user_id filtering may not be accurate
                    if user_id is not None and task_result.user_id != user_id:
                        continue

                    tasks.append(task_result)

            return tasks

        except Exception as e:
            logger.error(f"Failed to list tasks: {str(e)}")
            return []

    async def get_task_count(self, status: Optional[TaskStatus] = None) -> int:
        """
        Get task count

        Args:
            status: Task status filter (optional)

        Returns:
            int: Number of tasks
        """
        tasks = await self.list_tasks(status=status)
        return len(tasks)

    def get_worker_functions(self) -> List[Function]:
        """
        Get worker function mappings

        Returns:
            Dict[str, Callable]: Worker function mappings
        """
        return [v.to_arq_function() for v in self._task_registry.values()]

    def list_registered_task_names(self) -> List[str]:
        """
        Get all registered task names

        Returns:
            List[str]: List of task names
        """
        return list(self._task_registry.keys())


def task(retry_config: Optional[RetryConfig] = None, timeout: Optional[float] = 300):
    """
    Task decorator

    Args:
        retry_config: Retry configuration (optional)

    Returns:
        Decorated function
    """

    if not retry_config:
        retry_config = RetryConfig()

    task_manager = get_task_manager()

    def decorator(func: Callable) -> Callable:

        async def _task_wrapper(*args, **kwargs):
            return await task_manager.execute_task_with_context(func, *args, **kwargs)

        function_name = func.__name__

        # Some attributes are required by the arq worker framework
        return TaskFunction(
            name=function_name,
            coroutine=_task_wrapper,
            original_func=func,
            timeout=timeout,
            retry_config=retry_config,
        )

    return decorator


def get_task_manager() -> TaskManager:
    """
    Get task manager instance

    Returns:
        TaskManager: Task manager instance
    """
    from core.di.utils import get_bean_by_type

    return get_bean_by_type(TaskManager)
