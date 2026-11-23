"""Merge stopped EverOS traces, separating tenants and rejecting revision conflicts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from smartcomment.runtime import ExecNetwork


def merge_graphs(graphs: list[dict[str, Any]]) -> dict[str, Any]:
    """Namespace all identities by tenant; conflicting immutable values are errors."""
    if not graphs:
        raise ValueError("No execution graphs supplied.")
    result = copy.deepcopy(graphs[0])
    buckets: dict[str, dict[str, dict[str, Any]]] = {
        kind: {} for kind in ("nodes", "edges", "operations", "sessions")
    }
    for original in graphs:
        graph = copy.deepcopy(original)
        metadata = graph.get("metadata", {})
        scope = [
            metadata.get("app_id"),
            metadata.get("project_id", graph.get("project_id")),
        ]
        if any(value is None for value in scope):
            raise ValueError(
                "Every input graph must declare its app and project scope."
            )
        prefix = (
            "scope-" + hashlib.sha256(json.dumps(scope).encode()).hexdigest()[:16] + "/"
        )
        aliases = {}
        for node in graph["data"]["nodes"]:
            old_id = node["full_node_id"]
            node["name"] = prefix + node["name"]
            node["full_name"] = (
                (node["class_name"] + ":") if node.get("class_name") else ""
            ) + node["name"]
            node["full_node_id"] = f"{node['full_name']}@{node['version']}"
            aliases[old_id] = node["full_node_id"]
            node.setdefault("metadata", {})["tenant_scope"] = prefix
        for op in graph["data"]["operations"]:
            op.setdefault("metadata", {})["tenant_scope"] = prefix
        for edge in graph["data"]["edges"]:
            for field in ("source_full_node_id", "target_full_node_id"):
                edge[field] = aliases.get(edge[field], edge[field])
        for node in graph["data"]["nodes"]:
            try:
                snapshot = json.loads(node["value"])
            except (TypeError, ValueError):
                continue
            if isinstance(snapshot, dict) and snapshot.get("source_revision"):
                revision = snapshot["source_revision"]
                snapshot["source_revision"] = aliases.get(revision, revision)
                node["value"] = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
        for op in graph["data"]["operations"]:
            for field in ("source_node", "query_full_node_id"):
                reference = op["metadata"].get(field)
                if reference in aliases:
                    op["metadata"][field] = aliases[reference]
        for kind, id_field in (
            ("sessions", "session_id"),
            ("nodes", "full_node_id"),
            ("operations", "op_id"),
            ("edges", "edge_id"),
        ):
            for item in graph["data"][kind]:
                key = item[id_field]
                prior = buckets[kind].get(key)
                if (
                    kind == "nodes"
                    and prior is not None
                    and prior["value"] != item["value"]
                ):
                    raise ValueError(f"Conflicting memory revision: {key}")
                if (
                    kind in ("edges", "operations")
                    and prior is not None
                    and any(
                        prior.get(field) != item.get(field)
                        for field in (
                            "source_full_node_id",
                            "target_full_node_id",
                            "op_name",
                            "metadata",
                        )
                    )
                ):
                    raise ValueError(f"Conflicting trace record: {key}")
                buckets[kind].setdefault(key, item)
    result["data"] = {kind: list(items.values()) for kind, items in buckets.items()}
    result["data"]["annotations"] = []
    result["graph_id"] = (
        "merged-"
        + hashlib.sha256(
            json.dumps([graph["graph_id"] for graph in graphs]).encode()
        ).hexdigest()[:16]
    )
    result["metadata"] = {
        "merged_from": [g["graph_id"] for g in graphs],
        "identity_namespace": "app_project",
        "history_complete": False,
    }
    ExecNetwork.import_graph(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.graph_dir.glob("*.json"))
    if args.output.resolve() in {path.resolve() for path in paths}:
        parser.error("The merged graph must not overwrite a source graph.")
    graphs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    merged = merge_graphs(graphs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
