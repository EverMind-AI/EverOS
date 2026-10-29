"""``everos cascade`` without ``[llm]`` exits 3 with the toml hint, not a traceback.

Every cascade command builds the same application as the server, and
building it requires an extraction LLM. The CLI turns the build failure into
one actionable line naming ``everos.toml`` and the ``[llm]`` section, and a
distinct exit code a script can test for. Runs the real CLI in a subprocess
with every provider key blanked, as on a credential-free CI runner.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .conftest import hermetic_env


def test_cascade_status_without_an_llm_exits_3_with_the_toml_hint(
    tmp_path: Path,
) -> None:
    env = hermetic_env()
    env["EVEROS_LLM__API_KEY"] = ""

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "everos.entrypoints.cli.main",
            "cascade",
            "--root",
            str(tmp_path),
            "status",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 3, result.stderr
    # The hint prints the resolved root; on a Windows runner ``tmp_path`` may
    # still carry an 8.3 short name, so resolve it the same way.
    assert f"{tmp_path.resolve()}/everos.toml" in result.stderr
    assert "[llm]" in result.stderr
    assert "Traceback" not in result.stderr
