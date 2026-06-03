"""YAML config loader for category-organised file trees.

Concept: a project keeps several *categories* of YAML config files under
their own subdirectories — for example PromptSlot templates under
``config/prompt_slots/<name>.yaml``. The loader:

    1. registers a category → subdirectory mapping
    2. resolves ``find(category, name)`` to ``<root>/<subdir>/<name>.yaml``
    3. caches parsed contents until ``refresh`` is called

Uses ``yaml.safe_load`` (no arbitrary tags) — PyYAML is already a project
dependency for markdown frontmatter, so no extra cost.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


class YamlConfigLoader:
    """Load YAML files organised by category subdirectories.

    Usage:
        loader = YamlConfigLoader(root=Path("src/everos/config"))
        loader.register_category("prompt_slots")
        # → reads <root>/prompt_slots/episode.yaml
        meta = loader.find("prompt_slots", "episode")
        names = loader.list("prompt_slots")
        loader.refresh()  # next find() re-reads from disk

    Cache semantics:
        * ``find`` parses the file on first access and caches the dict.
        * ``refresh()`` empties the entire cache.
        * ``refresh(category)`` empties one category's entries.
        * ``refresh(category, name)`` empties a single entry.
    """

    def __init__(
        self,
        root: Path,
        categories: Mapping[str, str | None] | None = None,
    ) -> None:
        """
        Args:
            root: Base directory containing the category subdirectories.
            categories: Optional pre-registered category map (``name → subdir``).
                When ``subdir`` is ``None`` the category name is used as-is.
        """
        self._root = Path(root)
        self._subdirs: dict[str, str] = {}
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

        if categories:
            for name, subdir in categories.items():
                self.register_category(name, subdir)

    # ── Category management ────────────────────────────────────────────────

    def register_category(self, name: str, subdir: str | None = None) -> None:
        """Register a category. ``subdir`` defaults to ``name``."""
        self._subdirs[name] = subdir if subdir is not None else name

    def categories(self) -> list[str]:
        """Return registered category names (sorted)."""
        return sorted(self._subdirs)

    # ── Lookup ─────────────────────────────────────────────────────────────

    def find(self, category: str, name: str) -> dict[str, Any]:
        """Load ``<root>/<subdir>/<name>.yaml`` for ``category``.

        Raises:
            KeyError: if ``category`` was not registered.
            FileNotFoundError: if the yaml file does not exist.
            TypeError: if the parsed YAML is not a mapping.
        """
        cache_key = (category, name)
        if cache_key in self._cache:
            return self._cache[cache_key]

        path = self._path_for(category, name)
        if not path.is_file():
            raise FileNotFoundError(f"yaml not found: {path}")

        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise TypeError(
                f"yaml top-level must be a mapping, got {type(data).__name__}: {path}"
            )
        self._cache[cache_key] = data
        return data

    def list(self, category: str) -> list[str]:
        """Return sorted yaml stems available in ``category`` (no extension).

        Raises:
            KeyError: if ``category`` was not registered.
        """
        directory = self._dir_for(category)
        if not directory.is_dir():
            return []
        return sorted(p.stem for p in directory.glob("*.yaml"))

    # ── Cache control ──────────────────────────────────────────────────────

    def refresh(
        self,
        category: str | None = None,
        name: str | None = None,
    ) -> None:
        """Invalidate cached entries.

        - ``refresh()``                  → drop every cached entry
        - ``refresh(category)``          → drop everything in ``category``
        - ``refresh(category, name)``    → drop a single entry
        """
        if category is None:
            self._cache.clear()
            return
        if name is not None:
            self._cache.pop((category, name), None)
            return
        self._cache = {
            (cat, n): v for (cat, n), v in self._cache.items() if cat != category
        }

    # ── Internals ──────────────────────────────────────────────────────────

    def _dir_for(self, category: str) -> Path:
        try:
            subdir = self._subdirs[category]
        except KeyError as exc:
            raise KeyError(
                f"category not registered: {category!r}; known: {sorted(self._subdirs)}"
            ) from exc
        return self._root / subdir

    def _path_for(self, category: str, name: str) -> Path:
        return self._dir_for(category) / f"{name}.yaml"
