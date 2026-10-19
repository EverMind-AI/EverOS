"""``everos cascade`` — what an operator sees when the index is rejected.

Two contracts, both about the moment a startup guard fires in a CLI phase
rather than in the server:

- the message is the message, not a traceback; and
- the exit code is 3, the one this CLI already uses for "a precondition is not
  met, go and do something" (the OME lock, a running server blocking a drop).
  It cannot be 1: that already means "there is no everos.toml here", and a
  script could not tell the two apart.

``rebuild`` additionally has to *write* the attestation. Nothing else can: a
table created from a pydantic model carries no field metadata, so a rebuilt
index would otherwise come up unattested and the next start would recommend
the rebuild that had just finished.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from everos.core.errors import EmbeddingProvenanceError
from everos.entrypoints.cli.main import app

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("EVEROS_ROOT", str(tmp_path))
    (tmp_path / "everos.toml").write_text("[embedding]\ndim = 1024\n", encoding="utf-8")
    yield


@pytest.fixture
def rejecting_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the shared verification refuse, the way a changed model would."""

    async def _refuse(*, may_attest: bool = True) -> None:
        raise EmbeddingProvenanceError(
            "embedding configuration does not match this index:\n"
            "  - model: index built with 'nomic-embed-text', config uses "
            "'Qwen/Qwen3-Embedding-4B'\n"
            "    Recover with: everos cascade rebuild"
        )

    import everos.entrypoints.cli.commands.cascade as cascade_cmd

    monkeypatch.setattr(cascade_cmd, "verify_configuration", _refuse)


@pytest.mark.parametrize("command", ["status", "sync"])
def test_read_and_write_phases_are_both_refused(
    command: str, rejecting_index: None
) -> None:
    """A read-only phase is refused too: its similarities are meaningless."""
    result = runner.invoke(app, ["cascade", command], catch_exceptions=False)

    assert result.exit_code == 3, result.output
    assert "nomic-embed-text" in result.output
    assert "Qwen/Qwen3-Embedding-4B" in result.output
    assert "cascade rebuild" in result.output
    assert "Traceback" not in result.output


def test_refusal_is_not_exit_one(rejecting_index: None) -> None:
    """1 is taken: it is what a missing everos.toml exits with."""
    assert runner.invoke(app, ["cascade", "status"]).exit_code != 1
