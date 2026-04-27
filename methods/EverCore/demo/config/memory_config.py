import os
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from typing import Optional

from api_specs.memory_types import ScenarioType


@dataclass
class LLMConfig:
    """LLM configuration - loaded from environment variables automatically"""

    provider: str = "openai"
    model: Optional[str] = field(default_factory=lambda: os.getenv("LLM_MODEL"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    temperature: float = 0.3
    max_tokens: int = 16384


@dataclass
class EmbeddingConfig:
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "EMB_BASE_URL", "http://0.0.0.0:11000/v1/embeddings"
        )
    )
    model: str = field(
        default_factory=lambda: os.getenv("EMB_MODEL", "Qwen3-Embedding-4B")
    )


@dataclass
class MongoDBConfig:
    """MongoDB Configuration - supports adding authentication information through URI parameters"""

    uri: Optional[str] = None
    host: str = "localhost"
    port: str = "27017"
    database: str = "memsys"
    username: Optional[str] = None
    password: Optional[str] = None

    def __post_init__(self):
        """Load configuration from environment variables and build URI"""
        if not os.getenv("MONGODB_URI"):
            self.host = os.getenv("MONGODB_HOST", self.host)
            self.port = os.getenv("MONGODB_PORT", self.port)
            self.database = os.getenv("MONGODB_DATABASE", self.database)
            self.username = os.getenv("MONGODB_USERNAME")
            self.password = os.getenv("MONGODB_PASSWORD")

            if self.username and self.password:
                from urllib.parse import quote_plus

                self.uri = f"mongodb://{quote_plus(self.username)}:{quote_plus(self.password)}@{self.host}:{self.port}/{self.database}"
            else:
                self.uri = f"mongodb://{self.host}:{self.port}/{self.database}"
            uri_params = os.getenv("MONGODB_URI_PARAMS", "").strip()
            if uri_params:
                separator = '&' if ('?' in self.uri) else '?'
                self.uri = f"{self.uri}{separator}{uri_params}"

        else:
            self.uri = os.getenv("MONGODB_URI")
            self.database = os.getenv("MONGODB_DATABASE", self.database)


@dataclass
class ExtractModeConfig:

    scenario_type: ScenarioType = ScenarioType.TEAM
    language: str = "zh"

    # Optional overrides
    data_file: Optional[Path] = None
    output_dir: Optional[Path] = None
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    enable_profile_extraction: bool = True
    enable_foresight_extraction: bool = False

    def __post_init__(self):
        # Automatically set output directory
        if self.output_dir is None:
            self.output_dir = (
                Path(__file__).parent.parent
                / "memcell_outputs"
                / f"{self.scenario_type.value}_{self.language}"
            )

        # Set default values based on scenario
        if self.scenario_type == ScenarioType.TEAM:
            self.group_id = self.group_id or "team_chat_001"
            self.group_name = self.group_name or "Project Discussion Group"
            self.enable_foresight_extraction = False
        else:
            self.group_id = self.group_id or "assistant"
            self.group_name = self.group_name or "Personal Assistant"
            self.enable_foresight_extraction = True

        # Backward compatibility
        self.prompt_language = self.language


@dataclass
class ChatModeConfig:
    """Chat system configuration - using reasonable default values"""

    # Core parameters (usually no need to modify)
    api_base_url: str = field(
        default_factory=lambda: os.getenv("API_BASE_URL", "http://localhost:1995")
    )
    top_k_memories: int = 10
    conversation_history_size: int = 10
    time_range_days: int = 365
    show_retrieved_memories: bool = True

    # Paths (automatically set)
    chat_history_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "chat_history"
    )
    memcell_output_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "memcell_outputs"
    )

    def __post_init__(self):
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)
