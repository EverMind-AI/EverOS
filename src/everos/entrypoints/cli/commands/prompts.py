"""``everos prompts`` -- print which prompts this EverOS sends, as JSON.

The manifest of :func:`everos.service.prompt_manifest`: every registered
EverOS prompt with its version and fingerprint, and the everalgo prompts it
sends with their package version and fingerprint. A store's builder saves it
beside the store (``everos prompts > <store>/prompts.json``), so a reader can
tell which prompt versions built it. Needs no root, configuration or model.

Top-level leaf command, like ``everos init``.
"""

from __future__ import annotations

import json

import typer

from everos.service import prompt_manifest


def register(parent: typer.Typer) -> None:
    """Attach the ``prompts`` command to the root CLI app."""

    @parent.command("prompts")
    def prompts() -> None:
        """Print the prompt manifest (ids, versions, fingerprints) as JSON."""
        typer.echo(json.dumps(prompt_manifest(), indent=2, sort_keys=True))
