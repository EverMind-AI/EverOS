"""Environment-backed configuration for Cursor hooks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a dotenv file into os.environ (no override)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _find_dotenv() -> Path | None:
    """Search upward from cwd and hook install dir for .env."""
    candidates: list[Path] = []
    hook_root = Path(__file__).resolve().parent.parent
    candidates.append(hook_root / ".env")
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        candidates.append(parent / ".env")
        candidates.append(parent / "use-cases" / "cursor-agent-memory" / ".env")
    for path in candidates:
        if path.is_file():
            return path
    return None


@dataclass(frozen=True)
class EverOSHookConfig:
    base_url: str
    user_id: str
    app_id: str
    project_id: str
    session_prefix: str
    top_k: int
    min_score: float
    min_query_words: int
    debug: bool

    @classmethod
    def load(cls) -> EverOSHookConfig:
        dotenv = _find_dotenv()
        if dotenv is not None:
            _load_dotenv(dotenv)

        return cls(
            base_url=os.environ.get("EVEROS_BASE_URL", "http://127.0.0.1:8000").rstrip(
                "/"
            ),
            user_id=os.environ.get("EVEROS_USER_ID", "cursor-user"),
            app_id=os.environ.get("EVEROS_APP_ID", "default"),
            project_id=os.environ.get("EVEROS_PROJECT_ID", "default"),
            session_prefix=os.environ.get("EVEROS_SESSION_PREFIX", "cursor-"),
            top_k=int(os.environ.get("EVEROS_TOP_K", "5")),
            min_score=float(os.environ.get("EVEROS_MIN_SCORE", "0.1")),
            min_query_words=int(os.environ.get("EVEROS_MIN_QUERY_WORDS", "3")),
            debug=os.environ.get("EVEROS_DEBUG", "0") in {"1", "true", "yes"},
        )

    def is_configured(self) -> bool:
        return bool(self.base_url and self.user_id)

    def session_id_for(self, conversation_id: str) -> str:
        return f"{self.session_prefix}{conversation_id}"
