"""Checks that stop a run before it spends anything.

Each of these used to surface late: a missing dataset as a traceback after the fleet
was up, an empty answer key only once the servers were running, and a missing
optional provider never -- EverOS starts without embedding or rerank (both are
optional there), so ADD paid for extraction and the search stage then failed with
HTTP 422. Every problem is collected before any is reported, so a first run is told
everything it is missing at once instead of one item per attempt.

What a run needs follows from its retrieval methods, the same rule the EverOS search
manager enforces (``SearchManager._validate_components``): ``keyword`` needs only the
LLM, the vector-based methods need embedding, and ``agentic`` needs rerank as well.
The decider is not checked here: ``run.py`` resolves it the way the servers do and
asks it once before the first question (``decider_probe.py``).

Pure functions over the environment and the servers' ``/health`` payloads: nothing
here reads files other than the dataset path, starts a process or makes a call.
"""

from __future__ import annotations

__all__ = [
    "blocking_problems",
    "format_problems",
    "format_unready",
    "ready_line",
    "required_capabilities",
    "runs_server_stages",
    "servers_lacking",
]

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

_GUIDE = "benchmarks/README.md"

# Stages that run against the EverOS servers.
_SERVER_STAGES = frozenset({"add", "search"})

# Retrieval methods by the optional EverOS capability they need, in the vocabulary of
# `/health` `capabilities`.
_METHODS_NEEDING = {
    "embed": frozenset({"vector", "hybrid", "agentic", "llm_multiround"}),
    "rerank": frozenset({"agentic"}),
}

# (variable, why the run needs it) -- the reason is printed when it is missing.
_LLM_KEY = (
    "EVEROS_LLM__API_KEY",
    "the EverOS servers extract memories with it and refuse to start without it "
    "(an OpenRouter key)",
)
_CAPABILITY_KEYS = {
    "embed": (
        "EVEROS_EMBEDDING__API_KEY",
        "the retrieval method embeds every memory and query with it "
        "(a DeepInfra key for EverOS's default embedding; not needed for "
        "--methods keyword)",
    ),
    "rerank": (
        "EVEROS_RERANK__API_KEY",
        "agentic search reranks with it (a DeepInfra key for EverOS's default "
        "reranker)",
    ),
}
_EVAL_KEYS = (
    (
        "ANSWER_API_KEY",
        "the answer stage calls the model named in configs/<benchmark>.toml with it",
    ),
    (
        "JUDGE_API_KEY",
        "the judge stage calls the model named in configs/<benchmark>.toml with it",
    ),
)

# capability -> (line when all servers have it, problem when one lacks it, what to fix)
_CAPABILITY_TEXT = {
    "embed": (
        "Embedding answered",
        "The embedding service did not answer the startup probe",
        "EVEROS_EMBEDDING__BASE_URL / __MODEL / __API_KEY",
    ),
    "rerank": (
        "Rerank configured",
        "No rerank provider is configured",
        "EVEROS_RERANK__API_KEY",
    ),
}


def runs_server_stages(stages: Collection[str]) -> bool:
    """Whether any stage extracts into or searches the EverOS servers."""
    return bool(_SERVER_STAGES & set(stages))


def required_capabilities(methods: Collection[str]) -> list[str]:
    """The optional EverOS capabilities these retrieval methods need."""
    chosen = set(methods)
    return [cap for cap, users in _METHODS_NEEDING.items() if users & chosen]


def blocking_problems(
    env: Mapping[str, str],
    *,
    benchmark: str,
    data_path: str,
    stages: Collection[str],
    methods: Collection[str],
    starts_servers: bool,
    server_env: Mapping[str, str] | None = None,
) -> list[str]:
    """Everything that would make this run fail or waste money.

    Args:
        env: The process environment after ``benchmarks/.env`` was loaded; the answer
            and judge keys are read from it.
        benchmark: Adapter name, used in the download hint.
        data_path: Resolved dataset path for this run.
        stages: The stages this run executes.
        methods: The retrieval methods this run searches with.
        starts_servers: Whether the harness launches its own EverOS fleet.
        server_env: The environment the servers will be launched with, when it
            differs from ``env``; the provider keys are read from it.

    Returns:
        One human-readable entry per problem, each naming its fix; empty when the
        run can start.
    """
    server_keys = _required_server_keys(stages, methods) if starts_servers else []
    return (
        _missing_data(benchmark, data_path)
        + _missing_keys(env if server_env is None else server_env, server_keys)
        + _missing_keys(env, _EVAL_KEYS)
    )


def format_problems(problems: Collection[str], *, env_file: Path) -> str:
    """The message a run blocked before starting its servers exits with."""
    lines = ["Not started -- nothing has been spent yet. Fix these first:"]
    lines += [f"  - {p}" for p in problems]
    if not env_file.exists():
        lines.append(
            f"{env_file} does not exist:  cp benchmarks/.env.example benchmarks/.env"
        )
    lines.append(f'Keys go in benchmarks/.env -- see {_GUIDE}, "3. Configure".')
    return "\n".join(lines)


def servers_lacking(
    health: Mapping[str, Mapping[str, Any]], capability: str
) -> list[str]:
    """URLs whose ``/health`` reports ``capabilities.<capability>`` as false.

    ``embed`` is false both when no embedding key is configured and when the
    configured endpoint did not answer the startup dimension probe; ``rerank`` is
    false when no rerank provider could be built.
    """
    return [
        url
        for url, payload in health.items()
        if not (payload.get("capabilities") or {}).get(capability)
    ]


def ready_line(capability: str, servers: int) -> str:
    """The line a server check prints when every server has the capability."""
    return f"  {_CAPABILITY_TEXT[capability][0]} on {servers} server(s)"


def format_unready(
    lacking: Mapping[str, Collection[str]], *, logs: Mapping[str, Path]
) -> str:
    """The message a run blocked after starting its servers exits with.

    Args:
        lacking: Capability -> URLs of the servers that lack it.
        logs: URL -> that server's log file.
    """
    lines = ["Not started -- nothing has been spent yet."]
    for capability, urls in lacking.items():
        problem, check = _CAPABILITY_TEXT[capability][1:]
        lines.append(f"{problem} on:")
        lines += [f"  {url}  log: {logs.get(url, '?')}" for url in urls]
        lines.append(f"  check {check} in benchmarks/.env")
    lines.append(
        f'See {_GUIDE}, "3. Configure"; `grep -E "embedding_|rerank_" <log>` shows '
        "the server's own reason."
    )
    return "\n".join(lines)


def _is_unset(value: str | None) -> bool:
    """Empty, an unexpanded ``${VAR}``, or an older .env.example's ``sk-...``."""
    text = (value or "").strip()
    return not text or "..." in text or "${" in text


def _required_server_keys(
    stages: Collection[str], methods: Collection[str]
) -> list[tuple[str, str]]:
    # [llm] to start at all; the optional providers only for the methods that use them.
    keys: list[tuple[str, str]] = [_LLM_KEY]
    if runs_server_stages(stages):
        keys += [_CAPABILITY_KEYS[c] for c in required_capabilities(methods)]
    return keys


def _missing_data(benchmark: str, data_path: str) -> list[str]:
    if Path(data_path).exists():
        return []
    return [
        f"{benchmark} data not found at {data_path}\n"
        f"      fetch it:  bash benchmarks/download_data.sh {benchmark}"
    ]


def _missing_keys(
    env: Mapping[str, str], required: Collection[tuple[str, str]]
) -> list[str]:
    return [
        f"{name} is not set -- {why}"
        for name, why in required
        if _is_unset(env.get(name))
    ]
