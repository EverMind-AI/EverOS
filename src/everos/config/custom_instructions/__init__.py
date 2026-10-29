"""Shipped custom-instruction YAML files, one per operator.

Holds ``boundary_detection.yaml``, ``episode_extract.yaml``,
``profile_extract.yaml`` and ``reflect.yaml``. They are data, not code:
``everos.memory.custom_instructions`` reads them from
``<config_root>/custom_instructions/``. The wheel ships them because the build
packages the whole ``src/everos`` tree; this module only documents the
directory.
"""

from __future__ import annotations

__all__: list[str] = []
