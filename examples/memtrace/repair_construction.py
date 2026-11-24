"""Repair repeated boundary observations in stopped EverOS experiment graphs.

SQLite supplies immutable memcell membership. Only sender-only duplicate exact
versions and repeated operations for that same boundary are merged. Other
operations and real membership changes remain intact. Applying writes backups
and an audit; dry runs never change source graphs.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import portalocker
from smartcomment.runtime import ExecNetwork


def repair_payload(payload: dict, boundaries: dict[str, dict]) -> tuple[dict, dict]:
    """Repair only boundaries whose exact graph snapshots agree with SQLite."""
    result = copy.deepcopy(payload)
    data = result["data"]
    ops = {op["op_id"]: op for op in data["operations"]}
    groups: dict[str, list[dict]] = {}
    for node in data["nodes"]:
        if node["category"] == "memory_entry" and node["name"].startswith("mc_"):
            groups.setdefault(node["name"], []).append(node)
    audit: dict[str, Any] = {"cells": [], "removed_nodes": 0, "removed_edges": 0}
    aliases = {}
    removed_ops = set()
    removed_nodes = set()
    extra_edges = []
    for identity, versions in groups.items():
        boundary = boundaries.get(identity)
        if boundary is None:
            continue
        versions.sort(key=lambda node: node["version"])
        canonical = versions[0]
        snapshots = [json.loads(node["value"]) for node in versions]
        expected = {
            key: boundary[key] for key in ("app_id", "project_id", "session_id")
        }
        if any(
            node["metadata"].get("membership") != "exact"
            or any(value.get(key) != item for key, item in expected.items())
            or value.get("message_ids") != boundary["message_ids"]
            or {key: item for key, item in value.items() if key != "sender_ids"}
            != {key: item for key, item in snapshots[0].items() if key != "sender_ids"}
            for node, value in zip(versions, snapshots, strict=True)
        ):
            continue
        boundary_ops = [
            op
            for op in ops.values()
            if op["category"] == "boundary_detection"
            and op["op_name"] == f"boundary_detection_{identity}"
        ]
        if not boundary_ops:
            continue
        boundary_ops.sort(key=lambda op: op["metadata"].get("event_seq", 0))
        first = boundary_ops[0]
        first_seq = first["metadata"].get("event_seq", 0)
        old_ids = {node["full_node_id"] for node in versions}
        for node in versions[1:]:
            aliases[node["full_node_id"]] = canonical["full_node_id"]
            removed_nodes.add(node["full_node_id"])
        removed_ops.update(op["op_id"] for op in boundary_ops[1:])
        value = snapshots[0]
        value["sender_ids"] = boundary["sender_ids"]
        encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        value_changed = canonical["value"] != encoded
        canonical["value"] = encoded

        buffer_inputs = []
        for node in data["nodes"]:
            if node["category"] != "message_buffer":
                continue
            buffered = json.loads(node["value"])
            if any(buffered.get(key) != item for key, item in expected.items()):
                continue
            if not boundary["message_ids"] or not set(boundary["message_ids"]).issubset(
                buffered["message_ids"]
            ):
                continue
            seqs = [
                ops[edge["op_id"]]["metadata"].get("event_seq", 0)
                for edge in data["edges"]
                if edge["target_full_node_id"] == node["full_node_id"]
                and ops[edge["op_id"]]["op_name"]
                == f"memory_add_{expected['session_id']}"
            ]
            if seqs and max(seqs) <= first_seq:
                buffer_inputs.append(node)
        buffer = max(buffer_inputs, key=lambda node: node["version"], default=None)
        added_buffer = False
        if buffer is not None and not any(
            edge["source_full_node_id"] == buffer["full_node_id"]
            and edge["target_full_node_id"] in old_ids
            for edge in data["edges"]
        ):
            template = next(
                edge for edge in data["edges"] if edge["op_id"] == first["op_id"]
            )
            edge = copy.deepcopy(template)
            edge["source_full_node_id"] = buffer["full_node_id"]
            edge["target_full_node_id"] = canonical["full_node_id"]
            digest = hashlib.sha256(
                f"{first['op_id']}:{buffer['full_node_id']}:{canonical['full_node_id']}".encode()
            ).hexdigest()[:32]
            edge["edge_id"] = f"edge-{digest}"
            edge.setdefault("metadata", {}).update(
                {"boundary_input": True, "construction_repaired": True}
            )
            extra_edges.append(edge)
            added_buffer = True
        if len(versions) > 1 or len(boundary_ops) > 1 or added_buffer or value_changed:
            audit["cells"].append(
                {
                    "memcell_id": identity,
                    "removed_versions": len(versions) - 1,
                    "removed_boundary_ops": len(boundary_ops) - 1,
                    "buffer_input": buffer["full_node_id"]
                    if buffer is not None
                    else None,
                    "buffer_edge_added": added_buffer,
                }
            )
    original_edges = len(data["edges"])
    retained = []
    seen = set()
    for edge in data["edges"]:
        if edge["op_id"] in removed_ops:
            continue
        for field in ("source_full_node_id", "target_full_node_id"):
            edge[field] = aliases.get(edge[field], edge[field])
        if edge["category"] == "boundary_detection":
            pair = (
                edge["source_full_node_id"],
                edge["target_full_node_id"],
                edge["op_id"],
            )
            if pair in seen or pair[0] == pair[1]:
                continue
            seen.add(pair)
        retained.append(edge)
    data["edges"] = retained + extra_edges
    data["nodes"] = [
        node for node in data["nodes"] if node["full_node_id"] not in removed_nodes
    ]
    data["operations"] = [
        op for op in data["operations"] if op["op_id"] not in removed_ops
    ]
    audit["removed_nodes"] = len(removed_nodes)
    audit["removed_edges"] = original_edges - len(retained)
    audit["added_buffer_edges"] = len(extra_edges)
    ExecNetwork.import_graph(result)
    return result, audit


def repair_root(root: Path, *, apply: bool) -> dict:
    """Use read-only SQLite boundary data and preserve every changed original."""
    database = root / ".index/sqlite/system.db"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        boundaries = {
            row["memcell_id"]: {
                key: row[key] for key in ("app_id", "project_id", "session_id")
            }
            | {
                "message_ids": json.loads(row["message_ids_json"]),
                "sender_ids": json.loads(row["sender_ids_json"]),
            }
            for row in connection.execute("SELECT * FROM memcell")
        }
    backup = root.parent / "construction_before_fix/memtrace"
    results = []
    for path in sorted((root / "memtrace").glob("*.json")):
        with portalocker.Lock(str(path) + ".lock", timeout=30):
            raw = path.read_bytes()
            original = json.loads(raw)
            repaired, audit = repair_payload(original, boundaries)
            if repaired == original:
                continue
            after = (json.dumps(repaired, ensure_ascii=False, indent=2) + "\n").encode()
            if apply:
                backup.mkdir(parents=True, exist_ok=True)
                saved = backup / path.name
                if saved.exists() and saved.read_bytes() != raw:
                    raise RuntimeError(f"A different backup already exists: {saved}")
                saved.write_bytes(raw)
                temporary = path.with_suffix(".construction.tmp")
                temporary.write_bytes(after)
                temporary.replace(path)
            results.append(
                {
                    "file": path.name,
                    "before_sha256": hashlib.sha256(raw).hexdigest(),
                    "after_sha256": hashlib.sha256(after).hexdigest(),
                    **audit,
                }
            )
    report = {"root": str(root), "applied": apply, "graphs": results}
    if apply and results:
        (root.parent / "construction_repair.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    for root in args.roots:
        report = repair_root(root, apply=args.apply)
        print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
