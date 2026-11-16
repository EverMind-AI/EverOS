"""Run core MemTrace graph attribution on an exported EverOS QA graph."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from smartcomment.runtime import ExecNetwork

if __package__:
    from .graphtrace import (
        AttributionConfig,
        OpenAIChatProvider,
        attribute,
        collect_failed_queries,
    )
    from .graphtrace.cases import decode_snapshot
else:
    from graphtrace import (
        AttributionConfig,
        OpenAIChatProvider,
        attribute,
        collect_failed_queries,
    )
    from graphtrace.cases import decode_snapshot


def load_credentials(path: Path | None) -> tuple[str, str | None]:
    """Read one credential pair without copying secrets into the report."""
    if path is not None:
        config = json.loads(path.read_text(encoding="utf-8"))
        keys, urls = config.get("api_keys"), config.get("base_urls")
        if not isinstance(keys, list) or not isinstance(urls, list):
            raise ValueError("api_config must contain api_keys and base_urls lists.")
        if not keys or len(keys) != len(urls):
            raise ValueError(
                "Credential lists must be nonempty and have equal lengths."
            )
        key, url = keys[0], urls[0]
    else:
        key = os.environ.get("OPENAI_API_KEY")
        url = os.environ.get("OPENAI_BASE_URL")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("Set OPENAI_API_KEY or provide --api-config.")
    if url is not None and (not isinstance(url, str) or not url.strip()):
        raise ValueError("The base URL must be a nonempty string.")
    return key, url


async def _run(
    graph: ExecNetwork,
    case: Any,
    args: argparse.Namespace,
    config: AttributionConfig,
    key: str,
    url: str | None,
) -> dict[str, Any]:
    async with AsyncOpenAI(
        api_key=key, base_url=url, timeout=120, max_retries=1
    ) as client:
        report = await attribute(
            graph, case, OpenAIChatProvider(client, args.model), config
        )
    report["model"] = args.model
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--query-id", help="Exact full_node_id or query identity.")
    parser.add_argument("--api-config", type=Path)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "qwen3.8-flash"))
    parser.add_argument("--max-iters", type=int, default=200)
    parser.add_argument("--max-trace-nodes", type=int, default=16)
    parser.add_argument("--num-starting-points", type=int, default=8)
    parser.add_argument("--context-chars", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.graph.read_text(encoding="utf-8"))
        if payload.get("metadata", {}).get("merge_value_conflicts"):
            print(
                "The input graph has unresolved revision conflicts; "
                "recollect traces and use the safe merger.",
                file=sys.stderr,
            )
            return 2
        graph = ExecNetwork.import_graph(payload)
        cases, diagnostics = collect_failed_queries(graph)
        for diagnostic in diagnostics:
            print(diagnostic, file=sys.stderr)
        if args.list_cases:
            print(json.dumps([case.model_dump() for case in cases], indent=2))
            return 0
        if args.output is None:
            parser.error("--output is required when running attribution.")
        protected_paths = {args.graph.resolve()}
        if args.api_config:
            protected_paths.add(args.api_config.resolve())
        if args.output.resolve() in protected_paths:
            parser.error("The report must not overwrite the graph or credential file.")
        if args.query_id:
            cases = [
                case
                for case in cases
                if args.query_id
                in {
                    case.query_full_node_id,
                    case.query_full_node_id.rsplit("@", 1)[0],
                    case.query_full_node_id.rsplit("@", 1)[0].removeprefix("query:"),
                    _query_identity(graph, case.query_full_node_id),
                }
            ]
            if len(cases) != 1:
                parser.error("--query-id must match exactly one complete failed case.")
        if not cases:
            parser.error("No complete failed QA cases found in this graph.")
        config = AttributionConfig(
            max_iters=args.max_iters,
            max_trace_nodes=args.max_trace_nodes,
            num_starting_points=args.num_starting_points,
            context_chars=args.context_chars,
        )
        key, url = load_credentials(args.api_config)
        case = cases[0]
        print(f"Attributing {case.query_full_node_id} using {args.model}.", flush=True)
        report = asyncio.run(_run(graph, case, args, config, key, url))
        report["diagnostics"] = diagnostics
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        # Credential parsing exceptions must not print the supplied file contents.
        print(f"Cannot run attribution: {type(exc).__name__}.", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "attribution": report["attribution"],
                "failure": report["failure"],
                "iterations": report["iterations"],
                "usage": report["usage"],
                "report": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "success" else 1


def _query_identity(graph: ExecNetwork, full_node_id: str) -> str:
    node = graph.get_variable(full_node_id)
    snapshot = decode_snapshot(node.value)
    if isinstance(snapshot, dict) and snapshot.get("id"):
        return str(snapshot["id"])
    return node.full_name.removeprefix("query:").removeprefix(
        node.metadata.get("tenant_scope", "")
    )


if __name__ == "__main__":
    sys.exit(main())
