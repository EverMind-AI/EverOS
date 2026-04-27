import os
import yaml
import json
from typing import Dict, Any, Optional

from core.di.decorators import component
from common_utils.project_path import CURRENT_DIR


@component(name="config_provider")
class ConfigProvider:
    """Configuration provider"""

    def __init__(self):
        """Initialize configuration provider"""
        self.config_dir = CURRENT_DIR / "config"
        self._cache: Dict[str, Any] = {}

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get configuration

        Args:
            config_name: Configuration file name (without extension)

        Returns:
            Dict[str, Any]: Configuration data
        """
        if config_name in self._cache:
            return self._cache[config_name]

        config_file = self.config_dir / f"{config_name}.yaml"
        if not config_file.exists():
            config_file = self.config_dir / f"{config_name}.yml"
        if not config_file.exists():
            config_file = self.config_dir / f"{config_name}.json"

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_name}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)

            self._cache[config_name] = config_data
            return config_data

        except Exception as e:
            raise RuntimeError(f"Failed to load configuration file {config_name}: {e}")

    def get_raw_config(self, config_name: str) -> str:
        """
        Get raw configuration text content

        Args:
            config_name: Configuration file name (with extension)

        Returns:
            str: Raw text content of the configuration file
        """
        # Check cache
        cache_key = f"raw_{config_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Build configuration file path (config_name already includes extension)
        config_file = self.config_dir / config_name

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_name}")

        try:
            # Directly read text file content
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # Cache raw text content
            self._cache[cache_key] = raw_content
            return raw_content

        except Exception as e:
            raise RuntimeError(f"Failed to read configuration file {config_name}: {e}")

    def get_available_configs(self) -> list:
        """
        Get list of all files in the config directory

        Returns:
            list: List of file names
        """
        configs = []
        for file in self.config_dir.iterdir():
            if file.is_file():
                configs.append(file.name)

        return sorted(configs)
