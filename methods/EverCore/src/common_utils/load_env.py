#!/usr/bin/env python3
"""
Unified environment loading tool

Provides Python path setup and .env file loading functionality to ensure environment variables are correctly loaded when the project runs from different locations.
"""

import logging
import os
import sys
from typing import Optional
from dotenv import load_dotenv
import time

from common_utils.app_meta import set_service_name
from common_utils.project_path import PROJECT_DIR
from common_utils.datetime_utils import get_timezone

# Environment variables are not loaded yet, so get_logger cannot be used here
logger = logging.getLogger(__name__)

# Project metadata has been moved to the app_meta module

"""
- setup_pythonpath is not needed.
  - For 'python run.py' and 'python src/run.py', this is not required as src will be added automatically.
  - If src is truly missing, setup_pythonpath would need to import load_env.py, which then depends on pythonpath and may not actually load correctly.
  - Not needed for VSCode launch, which can be configured via launch.json.
  - Not needed for online deployment using 'python run.py'.
    - The web entry point might indeed cause src to be missing; this needs to be addressed.
"""


def load_env_file(
    env_file_name: str = ".env", check_env_var: Optional[str] = None
) -> bool:
    """
    Load .env file

    Args:
        env_file_name: .env filename
        check_env_var: Environment variable name to check, used to determine if environment has been loaded

    Returns:
        bool: Whether environment variables were successfully loaded
    """
    # Calculate .env file path based on the location of load_env.py
    # .env file is located in the parent directory of src

    env_file_path = PROJECT_DIR / env_file_name

    if not env_file_path.exists():
        logger.warning(".env file does not exist: %s", env_file_path)
        return False

    try:
        load_dotenv(env_file_path)
        logger.debug("Successfully loaded .env file: %s", env_file_path)
    except (IOError, OSError) as e:
        logger.error("Failed to load .env file: %s", e)
        return False

    if check_env_var and os.getenv(check_env_var):
        logger.info("%s is set, environment variables have been loaded", check_env_var)
        return True
    else:
        if check_env_var:
            logger.error(
                "Please ensure that the %s environment variable is set", check_env_var
            )
        return False


def reset_timezone():
    """
    Reset timezone
    """
    timezone = get_timezone()
    os.environ["TZ"] = timezone.key
    # tzset() is not available on Windows, only call it if available
    if hasattr(time, 'tzset'):
        time.tzset()


def sync_pythonpath_with_syspath():
    """
    Synchronize PYTHONPATH and sys.path to ensure all paths in sys.path are included in PYTHONPATH

    Notes:
    1. Only synchronize non-standard library paths
    2. Exclude .venv and similar virtual environment paths
    3. Maintain the original priority of PYTHONPATH
    """
    import sys
    import os
    from pathlib import Path

    # Get current PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "").split(":")
    pythonpath = [p for p in pythonpath if p]  # Remove empty strings

    # Path patterns to exclude
    exclude_patterns = [
        ".venv",
        "site-packages",
        "dist-packages",
        "lib/python",
        "__pycache__",
    ]

    # Get paths from sys.path that need to be added
    new_paths = []
    for path in sys.path:
        # Skip empty paths
        if not path:
            continue

        # Convert to Path object for processing
        path_obj = Path(path)

        # Skip non-existent paths
        if not path_obj.exists():
            continue

        # Skip paths that should be excluded
        if any(pattern in str(path_obj) for pattern in exclude_patterns):
            continue

        # Convert to string and normalize
        path_str = str(path_obj.resolve())

        # If path is not in current PYTHONPATH, add to new paths list
        if path_str not in pythonpath:
            new_paths.append(path_str)

    # If there are new paths to add
    if new_paths:
        # Append new paths to existing PYTHONPATH
        all_paths = pythonpath + new_paths
        # Update environment variable
        os.environ["PYTHONPATH"] = ":".join(all_paths)
        logger.debug("Updated PYTHONPATH: %s", os.environ["PYTHONPATH"])


def setup_environment(
    load_env_file_name: str = ".env",
    check_env_var: Optional[str] = None,
    service_name: Optional[str] = None,
) -> bool:
    """
    Unified environment setup function

    Args:
        load_env_file_name: .env filename
        check_env_var: Environment variable name to check, used to determine if environment has been loaded
        service_name: Name of the current service being started, will be stored in APP_META_DATA

    Returns:
        bool: Whether environment was successfully set up
    """
    # Load .env file
    success = load_env_file(
        env_file_name=load_env_file_name, check_env_var=check_env_var
    )

    # Synchronize PYTHONPATH and sys.path
    sync_pythonpath_with_syspath()

    # Reset timezone
    reset_timezone()

    # Set service name
    if service_name:
        set_service_name(service_name)
        logger.debug("Service name set: %s", service_name)

    if not success:
        logger.error("Environment setup failed, exiting program")
        sys.exit(1)

    return success
