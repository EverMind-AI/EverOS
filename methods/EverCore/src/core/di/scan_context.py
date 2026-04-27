# -*- coding: utf-8 -*-
"""
Scan context manager
Used to pass context information during module scanning and importing

Implemented using a prefix tree (Trie) for efficient path matching lookup
"""

import sys

from typing import Dict, Any, Optional
from pathlib import Path
from threading import RLock


class _PathTrieNode:
    """
    Prefix tree node
    Used to store path segments and corresponding metadata
    """

    __slots__ = ['children', 'metadata', 'is_registered']

    def __init__(self):
        # Child node mapping {path_segment: _PathTrieNode}
        self.children: Dict[str, '_PathTrieNode'] = {}
        # Metadata corresponding to this node (only registered path nodes have it)
        self.metadata: Optional[Dict[str, Any]] = None
        # Flag indicating whether this node is a registered path endpoint
        self.is_registered: bool = False

    def print_tree(
        self, name: str = "(root)", prefix: str = "", is_last: bool = True
    ) -> str:
        """
        Recursively print tree structure

        Args:
            name: Current node name
            prefix: Prefix for current line (used for indentation and connecting lines)
            is_last: Whether this is the last child of the parent node

        Returns:
            String representation of the tree structure

        Example:
            (root)
            ├── Users
            │   └── admin
            │       └── project
            │           └── src [*] {"addon_tag": "core"}
        """
        lines = []

        # Build display content for current node
        connector = "└── " if is_last else "├── "
        node_display = name
        if self.is_registered:
            # Mark registered nodes and display metadata
            meta_str = str(self.metadata) if self.metadata else "{}"
            node_display = f"{name} [*] {meta_str}"

        # Root node does not need connector
        if prefix == "" and name == "(root)":
            lines.append(node_display)
        else:
            lines.append(f"{prefix}{connector}{node_display}")

        # Calculate prefix for children
        if prefix == "" and name == "(root)":
            child_prefix = ""
        else:
            child_prefix = prefix + ("    " if is_last else "│   ")

        # Recursively print child nodes
        children_items = sorted(self.children.items())
        for i, (child_name, child_node) in enumerate(children_items):
            is_last_child = i == len(children_items) - 1
            lines.append(child_node.print_tree(child_name, child_prefix, is_last_child))

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return string representation of tree structure"""
        return self.print_tree()


class ScanContextRegistry:
    """
    Scan context registry (singleton pattern)
    Uses a prefix tree (Trie) for efficient path matching lookup

    Paths are split by '/' or os.sep into segments and built into a tree structure:
    - Root node is an empty node
    - Each path segment becomes a child node
    - During lookup, traverse down the tree and return metadata from the longest matching path
    """

    # Singleton instance
    _instance: Optional['ScanContextRegistry'] = None
    _lock: RLock = RLock()

    # Instance attributes (initialized in __init__)
    _root: _PathTrieNode
    _path_context_mapping: Dict[str, Dict[str, Any]]
    _instance_lock: RLock
    _initialized: bool

    def __new__(cls) -> 'ScanContextRegistry':
        """Singleton pattern: ensure only one instance is created"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance (use _initialized flag to ensure initialization happens only once)"""
        # Check if already initialized to avoid re-initialization
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Prefix tree root node
        self._root: _PathTrieNode = _PathTrieNode()
        # Keep original path mapping for unregister and get_all_mappings
        self._path_context_mapping: Dict[str, Dict[str, Any]] = {}
        # Instance-level lock
        self._instance_lock: RLock = RLock()
        # Mark as initialized
        self._initialized: bool = True

    @classmethod
    def get_instance(cls) -> 'ScanContextRegistry':
        """
        Get singleton instance

        Returns:
            ScanContextRegistry singleton instance
        """
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset singleton instance (mainly for testing)

        Warning: This will clear all registered path contexts
        """
        with cls._lock:
            cls._instance = None

    def _split_path(self, path: str) -> list:
        """
        Split path into list of segments

        Args:
            path: File or directory path

        Returns:
            List of path segments
        """
        # Resolve to absolute path
        resolved = str(Path(path).resolve())
        # Split by path separator and filter out empty strings
        parts = [p for p in resolved.replace('\\', '/').split('/') if p]
        return parts

    def register(self, path: str, metadata: Dict[str, Any]) -> 'ScanContextRegistry':
        """
        Register context metadata for a scan path

        Args:
            path: Scan path
            metadata: Context metadata

        Returns:
            self, supports method chaining
        """
        with self._instance_lock:
            # Save original mapping
            resolved_path = str(Path(path).resolve())
            self._path_context_mapping[resolved_path] = metadata

            # Insert path into prefix tree
            parts = self._split_path(path)
            node = self._root
            for part in parts:
                if part not in node.children:
                    node.children[part] = _PathTrieNode()
                node = node.children[part]

            # Mark as registered node and store metadata
            node.is_registered = True
            node.metadata = metadata

        return self

    def unregister(self, path: str) -> 'ScanContextRegistry':
        """
        Unregister a scan path

        Args:
            path: Scan path

        Returns:
            self, supports method chaining
        """
        with self._instance_lock:
            resolved_path = str(Path(path).resolve())
            self._path_context_mapping.pop(resolved_path, None)

            # Find node in prefix tree and unregister
            parts = self._split_path(path)
            node = self._root
            for part in parts:
                if part not in node.children:
                    return self  # Path does not exist
                node = node.children[part]

            # Remove registration flag
            node.is_registered = False
            node.metadata = None

        return self

    def search_metadata_based_path(self, file_path: Path) -> Dict[str, Any]:
        """
        Search for corresponding context metadata based on file path (longest prefix match)

        Uses prefix tree for efficient lookup, time complexity O(path_depth)

        Args:
            file_path: File path

        Returns:
            Context metadata dictionary (returns empty dict if not found)
        """
        parts = self._split_path(str(file_path))

        # Traverse down the tree, recording the last matched metadata
        node = self._root
        matched_metadata: Dict[str, Any] = {}

        for part in parts:
            if part not in node.children:
                # Path does not match, return longest match found so far
                break
            node = node.children[part]
            # If current node is a registered path, update match result
            if node.is_registered and node.metadata is not None:
                matched_metadata = node.metadata

        return matched_metadata.copy()

    def clear(self) -> 'ScanContextRegistry':
        """
        Clear all registered path contexts

        Returns:
            self, supports method chaining
        """
        with self._instance_lock:
            self._root = _PathTrieNode()
            self._path_context_mapping.clear()
        return self

    def print_tree(self) -> str:
        """
        Print tree structure of the prefix tree

        Returns:
            String representation of the tree structure
        """
        return self._root.print_tree()

    @classmethod
    def search_metadata_for_type(cls, bean_type: type) -> Dict[str, Any]:
        """
        Get context metadata for the file containing the given type

        Uses bean_type.__module__ to get module name, then retrieves file path from sys.modules,
        and uses the prefix tree to find the context metadata corresponding to that file.

        Args:
            bean_type: Type of the bean

        Returns:
            Context metadata dictionary (returns empty dict if not found)
        """
        instance = cls.get_instance()

        # Get module name from bean_type
        module_name = bean_type.__module__
        module = sys.modules.get(module_name)

        # Get file path of the module
        if module and hasattr(module, '__file__') and module.__file__:
            return instance.search_metadata_based_path(Path(module.__file__))

        return {}


# Utility function: get singleton instance
def get_scan_context_registry() -> ScanContextRegistry:
    """
    Get singleton instance of the scan context registry

    Returns:
        ScanContextRegistry singleton instance
    """
    return ScanContextRegistry.get_instance()
