"""``everos demo --plain`` runs as a process and prints the static preview.

The parity smoke for the ``demo`` command group: the console entry point
starts, the plain renderer needs no network, no TTY and no memory root, and
the command exits 0.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_demo_plain_exits_zero_with_the_preview(tmp_path: Path) -> None:
    env = {**os.environ, "EVEROS_ROOT": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-m", "everos.entrypoints.cli.main", "demo", "--plain"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "EverOS remembered" in done.stdout
    assert not any(tmp_path.iterdir()), "the plain preview must not touch the root"
