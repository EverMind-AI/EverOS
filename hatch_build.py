"""Wheel build hook: ship tiktoken's ``o200k_base`` encoding inside the wheel.

everalgo counts prompt tokens with tiktoken's ``o200k_base`` encoding, which
tiktoken downloads on first use. The wheel carries a copy, fetched here and
checked against the sha256 tiktoken itself expects, so an installed EverOS
starts without the network (``everos.service._warmup`` hands it to tiktoken).
It is not in git: the repository caps files at 640 KB
(``scripts/check_file_sizes.py``) and the encoding is 3.6 MB.

An editable install (``uv sync``) gets no copy and downloads on first start,
as before. Building a wheel needs the network once per ``build/`` directory.
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

URL = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
WHEEL_PATH = "everos/service/o200k_base.tiktoken"


def _fetch(cached: Path) -> None:
    if cached.is_file() and hashlib.sha256(cached.read_bytes()).hexdigest() == SHA256:
        return
    with urllib.request.urlopen(URL, timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError(f"{URL} does not match tiktoken's sha256 {SHA256}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(data)


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        if version == "editable":
            return
        cached = Path(self.root) / "build" / "o200k_base.tiktoken"
        _fetch(cached)
        build_data["force_include"][str(cached)] = WHEEL_PATH
