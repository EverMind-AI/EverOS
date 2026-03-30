import os

PROJECT_NAME = "EverMemOS"
PROJECT_VERSION = "1.0.0"


def get_env_project_name():
    """
    Get the project name from environment variables
    """
    project_name = os.getenv("project_name") or os.getenv("PROJECT_NAME")
    if project_name:
        return project_name
    else:
        return PROJECT_NAME
