"""Unit tests for YamlConfigLoader."""

from __future__ import annotations

from pathlib import Path

import pytest

from everos.component.config import YamlConfigLoader


@pytest.fixture
def config_root(tmp_path: Path) -> Path:
    """Build a fixture config tree::

    tmp_path/
      prompt_slots/
        episode.yaml
        atomic_fact.yaml
      custom_dir/
        alpha.yaml
    """
    (tmp_path / "prompt_slots").mkdir()
    (tmp_path / "prompt_slots" / "episode.yaml").write_text(
        "template: extract episode\nvariables:\n  memcell: input memcell\n",
        encoding="utf-8",
    )
    (tmp_path / "prompt_slots" / "atomic_fact.yaml").write_text(
        "template: extract atomic fact\n", encoding="utf-8"
    )
    (tmp_path / "custom_dir").mkdir()
    (tmp_path / "custom_dir" / "alpha.yaml").write_text(
        "value: alpha\n", encoding="utf-8"
    )
    return tmp_path


def test_register_default_subdir(config_root: Path) -> None:
    loader = YamlConfigLoader(root=config_root)
    loader.register_category("prompt_slots")
    meta = loader.find("prompt_slots", "episode")
    assert meta == {
        "template": "extract episode",
        "variables": {"memcell": "input memcell"},
    }


def test_register_custom_subdir(config_root: Path) -> None:
    loader = YamlConfigLoader(root=config_root)
    loader.register_category("alphas", subdir="custom_dir")
    meta = loader.find("alphas", "alpha")
    assert meta == {"value": "alpha"}


def test_constructor_categories_dict(config_root: Path) -> None:
    loader = YamlConfigLoader(
        root=config_root,
        categories={"prompt_slots": None, "alphas": "custom_dir"},
    )
    assert sorted(loader.categories()) == ["alphas", "prompt_slots"]
    assert loader.find("alphas", "alpha") == {"value": "alpha"}


def test_find_unregistered_category_raises(config_root: Path) -> None:
    loader = YamlConfigLoader(root=config_root)
    with pytest.raises(KeyError, match="not registered"):
        loader.find("ghost", "x")


def test_find_missing_file_raises(config_root: Path) -> None:
    loader = YamlConfigLoader(root=config_root)
    loader.register_category("prompt_slots")
    with pytest.raises(FileNotFoundError):
        loader.find("prompt_slots", "no_such")


def test_find_non_mapping_top_level_raises(tmp_path: Path) -> None:
    (tmp_path / "prompt_slots").mkdir()
    # Top-level is a list, not a mapping — must be rejected.
    (tmp_path / "prompt_slots" / "bad.yaml").write_text(
        "- one\n- two\n", encoding="utf-8"
    )
    loader = YamlConfigLoader(root=tmp_path)
    loader.register_category("prompt_slots")
    with pytest.raises(TypeError, match="must be a mapping"):
        loader.find("prompt_slots", "bad")


def test_find_empty_file_yields_empty_dict(tmp_path: Path) -> None:
    (tmp_path / "prompt_slots").mkdir()
    (tmp_path / "prompt_slots" / "blank.yaml").write_text("", encoding="utf-8")
    loader = YamlConfigLoader(root=tmp_path)
    loader.register_category("prompt_slots")
    assert loader.find("prompt_slots", "blank") == {}


def test_list_returns_sorted_stems(config_root: Path) -> None:
    loader = YamlConfigLoader(root=config_root)
    loader.register_category("prompt_slots")
    assert loader.list("prompt_slots") == ["atomic_fact", "episode"]


def test_list_unregistered_category_raises(config_root: Path) -> None:
    loader = YamlConfigLoader(root=config_root)
    with pytest.raises(KeyError):
        loader.list("ghost")


def test_list_empty_directory(tmp_path: Path) -> None:
    loader = YamlConfigLoader(root=tmp_path)
    loader.register_category("nope")
    assert loader.list("nope") == []  # missing directory → empty


def test_cache_returns_same_object(config_root: Path) -> None:
    loader = YamlConfigLoader(root=config_root)
    loader.register_category("prompt_slots")
    a = loader.find("prompt_slots", "episode")
    b = loader.find("prompt_slots", "episode")
    assert a is b  # cached, same dict reference


def test_refresh_invalidates_cache_and_reloads(config_root: Path) -> None:
    loader = YamlConfigLoader(root=config_root)
    loader.register_category("prompt_slots")
    a = loader.find("prompt_slots", "episode")

    # Modify the file on disk; without refresh the loader still returns
    # the cached value.
    (config_root / "prompt_slots" / "episode.yaml").write_text(
        "template: MODIFIED\n", encoding="utf-8"
    )
    cached = loader.find("prompt_slots", "episode")
    assert cached is a  # still the cached object

    loader.refresh()
    fresh = loader.find("prompt_slots", "episode")
    assert fresh is not a
    assert fresh == {"template": "MODIFIED"}


def test_refresh_specific_entry(config_root: Path) -> None:
    loader = YamlConfigLoader(root=config_root)
    loader.register_category("prompt_slots")
    e = loader.find("prompt_slots", "episode")
    a = loader.find("prompt_slots", "atomic_fact")

    (config_root / "prompt_slots" / "episode.yaml").write_text(
        "template: NEW\n", encoding="utf-8"
    )
    loader.refresh("prompt_slots", "episode")

    assert loader.find("prompt_slots", "episode") != e  # reloaded
    assert loader.find("prompt_slots", "atomic_fact") is a  # untouched


def test_refresh_full_category(config_root: Path) -> None:
    loader = YamlConfigLoader(
        root=config_root,
        categories={"prompt_slots": None, "alphas": "custom_dir"},
    )
    loader.find("prompt_slots", "episode")
    a = loader.find("alphas", "alpha")

    loader.refresh("prompt_slots")
    # alphas cache survives the prompt_slots refresh
    assert loader.find("alphas", "alpha") is a
