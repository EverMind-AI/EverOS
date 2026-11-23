"""Export MemTrace execution graphs (smartcomment JSON) to readable PNGs.

The default presentation is intentionally stable and report-friendly: white
nodes with black borders/edges, one image per query, and only the first two
items from each candidate pool.  The source JSON is never modified.

Usage:
    python examples/memtrace/export_png.py <graph_dir> [--out DIR]
    python examples/memtrace/export_png.py <graph_dir> --top-k 5
    python examples/memtrace/export_png.py <graph_dir> --combined
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from smartcomment.runtime import ExecNetwork

_ID_VERSION = re.compile(r"^(?P<base>.+)@(?P<version>\d+)$")
_COLOUR = re.compile(r'(?P<key>BGCOLOR|color|fontcolor)="#[0-9A-Fa-f]{6}"')


def _node_full_id(node: dict[str, Any]) -> str:
    """Return the smartcomment identity used by graph edges."""
    return str(node.get("full_node_id") or node.get("node_id") or "")


def _base_id(full_id: str) -> str:
    """Remove the version suffix and optional class-name prefix."""
    match = _ID_VERSION.match(full_id)
    value = match.group("base") if match else full_id
    return value.split(":", 1)[-1]


def _snapshot(node: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(str(node.get("value", "")))
    except (ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _belongs_to_query(node: dict[str, Any], query: dict[str, Any]) -> bool:
    scope = query.get("metadata", {}).get("tenant_scope", "")
    if node.get("metadata", {}).get("tenant_scope", "") != scope:
        return False
    identity = _snapshot(query).get("id") or str(query["name"]).removeprefix(scope)
    return _snapshot(node).get("query_id") == identity or str(
        node.get("name", "")
    ).removeprefix(scope).endswith(identity)


def _pool_info(node: dict[str, Any], top_k: int) -> tuple[list[str], bool]:
    """Truncate a candidate-pool value and return (kept ids, is_pool)."""
    try:
        value = json.loads(str(node.get("value", "")))
    except (TypeError, json.JSONDecodeError):
        return [], False
    if not isinstance(value, dict) or not isinstance(value.get("ids"), list):
        return [], False

    original_ids = [str(item) for item in value["ids"]]
    kept = original_ids if top_k == 0 else original_ids[:top_k]
    if top_k != 0:
        original_count = int(value.get("total_count", len(original_ids)))
        value["ids"] = kept
        value["scores"] = list(value.get("scores", []))[: len(kept)]
        value["sources"] = list(value.get("sources", []))[: len(kept)]
        value["count"] = len(kept)
        value["total_count"] = original_count
        value["display_only_top_k"] = top_k
        node["value"] = json.dumps(value, ensure_ascii=False, indent=2)
        metadata = node.setdefault("metadata", {})
        metadata["display_only_top_k"] = top_k
        metadata["total_count"] = original_count
    return kept, True


def _display_payload(
    payload: dict[str, Any], top_k: int, query_name: str | None = None
) -> dict[str, Any]:
    """Copy a graph payload and optionally select one query subgraph."""
    result = copy.deepcopy(payload)
    data = result.get("data", result)
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    candidate_ids = {
        _node_full_id(node): _snapshot(node).get("memory_identity")
        or _base_id(_node_full_id(node)).removeprefix(
            node.get("metadata", {}).get("tenant_scope", "")
        )
        for node in nodes
    }

    pool_ids: dict[str, set[str]] = {}
    pool_nodes: set[str] = set()
    query_nodes: list[dict[str, Any]] = []
    for node in nodes:
        full_id = _node_full_id(node)
        kept, is_pool = _pool_info(node, top_k)
        if is_pool:
            pool_ids[full_id] = set(kept)
            pool_nodes.add(full_id)
        if node.get("class_name") == "query" or str(node.get("name", "")).startswith(
            "query-"
        ):
            query_nodes.append(node)

    # Pool snapshots use raw entry ids; graph nodes are owner-prefixed.
    # Normalize only the selection keys, leaving the observed lists intact.
    for query in query_nodes:
        try:
            value = json.loads(str(query.get("value", "")))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        owner = value.get("user_id") or value.get("agent_id")
        if not owner:
            continue
        prefix = f"{owner}_"
        for node in nodes:
            pool_id = _node_full_id(node)
            if pool_id in pool_ids and _belongs_to_query(node, query):
                pool_ids[pool_id] = {
                    eid if eid.startswith(prefix) else f"{prefix}{eid}"
                    for eid in pool_ids[pool_id]
                }

    if query_name is None:
        included = {_node_full_id(node) for node in nodes}
    else:
        selected_query = next(
            (node for node in query_nodes if node.get("name") == query_name), None
        )
        if selected_query is None:
            raise ValueError(f"query node not found: {query_name}")
        included = {_node_full_id(selected_query)}
        for node in nodes:
            if node.get("class_name") != "retrieval_hit" and _belongs_to_query(
                node, selected_query
            ):
                included.add(_node_full_id(node))
        for pool_id, kept in pool_ids.items():
            pool_node = next(
                (node for node in nodes if _node_full_id(node) == pool_id), None
            )
            if pool_node is None or not _belongs_to_query(pool_node, selected_query):
                continue
            included.add(pool_id)
            for edge in edges:
                target = str(edge.get("target_full_node_id", ""))
                if (
                    edge.get("source_full_node_id") == pool_id
                    and candidate_ids.get(target) in kept
                ):
                    included.add(target)

    contexts = {
        _node_full_id(node)
        for node in nodes
        if node.get("class_name") == "context" and _node_full_id(node) in included
    }
    returned = {
        str(edge.get("source_full_node_id", ""))
        for edge in edges
        if edge.get("target_full_node_id") in contexts
    }
    included.update(returned)

    kept_edges: list[dict[str, Any]] = []
    used_operations: set[str] = set()
    for edge in edges:
        source = str(edge.get("source_full_node_id", ""))
        target = str(edge.get("target_full_node_id", ""))
        if source not in included or target not in included:
            continue
        if (
            source in pool_ids
            and target not in pool_nodes
            and target not in contexts
            and target not in returned
            and candidate_ids.get(target) not in pool_ids[source]
        ):
            continue
        kept_edges.append(edge)
        op_id = edge.get("op_id")
        if op_id:
            used_operations.add(str(op_id))

    data["nodes"] = [node for node in nodes if _node_full_id(node) in included]
    data["edges"] = kept_edges
    data["operations"] = [
        operation
        for operation in data.get("operations", [])
        if str(operation.get("op_id", "")) in used_operations
    ]
    return result


def _black_and_white(dot: str) -> str:
    """Normalize smartcomment's category palette to the report style."""
    return _COLOUR.sub(lambda match: f'{match.group("key")}="#000000"', dot).replace(
        'BGCOLOR="#000000"', 'BGCOLOR="#FFFFFF"'
    )


def export_graph_to_png(
    graph: ExecNetwork, png_path: Path, *, black_and_white: bool = True
) -> None:
    """Render one execution graph to PNG with the graphviz ``dot`` binary."""
    dot_str = graph.visualize(backend="dot")
    if black_and_white:
        dot_str = _black_and_white(dot_str)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".dot", delete=False) as f:
        f.write(dot_str)
        dot_file = f.name
    try:
        subprocess.run(
            ["dot", "-Tpng", dot_file, "-o", str(png_path)],
            check=True,
            capture_output=True,
        )
    finally:
        Path(dot_file).unlink(missing_ok=True)


def _query_names(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data", payload)
    return [
        str(node["name"])
        for node in data.get("nodes", [])
        if node.get("class_name") == "query"
        or str(node.get("name", "")).startswith("query-")
    ]


def _request_suffix(payload: dict[str, Any], query_name: str, index: int) -> str:
    data = payload.get("data", payload)
    for node in data.get("nodes", []):
        if node.get("name") != query_name:
            continue
        try:
            value = json.loads(str(node.get("value", "")))
        except (TypeError, json.JSONDecodeError):
            break
        request_id = value.get("request_id") if isinstance(value, dict) else None
        if request_id:
            return str(request_id)[:8]
    return f"query-{index:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_dir", type=Path, help="Directory with *.json graphs")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: <graph_dir>/png)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2,
        help="Candidate items shown per pool (default: 2; 0 shows all)",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Render each source JSON as one graph instead of one image per query",
    )
    args = parser.parse_args()
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")

    graph_dir: Path = args.graph_dir
    if not graph_dir.is_dir():
        print(f"error: {graph_dir} is not a directory", file=sys.stderr)
        return 1
    out_dir: Path = args.out or (graph_dir / "png")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_files = sorted(graph_dir.glob("*.json"))
    if not json_files:
        print(f"no *.json files in {graph_dir}", file=sys.stderr)
        return 1

    print(f"exporting {len(json_files)} graphs to {out_dir}")
    failed = 0
    for json_file in json_files:
        try:
            payload = json.loads(json_file.read_text(encoding="utf-8"))
            queries = _query_names(payload)
            selections = [None] if args.combined or not queries else queries
            for index, query_name in enumerate(selections, start=1):
                display = _display_payload(payload, args.top_k, query_name)
                graph = ExecNetwork.import_graph(display)
                if query_name is None:
                    filename = f"{json_file.stem}.png"
                else:
                    suffix = _request_suffix(payload, query_name, index)
                    filename = f"{json_file.stem}__search-{index:02d}-{suffix}.png"
                png_path = out_dir / filename
                export_graph_to_png(graph, png_path)
                print(f"  ok  {json_file.name} -> {png_path.name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL {json_file.name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
