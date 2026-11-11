"""Every filesystem write in the markdown package is on a known list.

A purge refuses writes into the scope it clears through ``ensure_writable``;
a write that does not pass through a checked point would land behind its
back. So the package's write calls are frozen here, each with why it is
safe: it is behind a checked point, or it is exempt. A new write call fails
this test until it is classified.

Matched: ``os.replace / rename / remove / unlink``, ``shutil.move / rmtree /
copy*`` (called or passed to a thread), and ``.write_text / .write_bytes /
.rename / .mkdir / .unlink / .rmdir / .touch`` on any object. ``.replace`` is
not matched: it cannot be told apart from ``str.replace``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import everos.infra.persistence.markdown as markdown_pkg

_PKG = Path(markdown_pkg.__file__).parent
_METHODS = {"write_text", "write_bytes", "rename", "mkdir", "unlink", "rmdir", "touch"}
_OS = {"replace", "rename", "remove", "unlink"}

CHECKED = "checked"
WRITE = "behind MarkdownWriter.write, which checks"
KNOWLEDGE = "behind KnowledgeWriter.write, which checks"
PURGE = "exempt: the purge's own move / delete (runs with PURGING set)"
DELETE = "exempt: /delete's own trash handling; HTTP refuses /delete during a purge"
REMOVE = "exempt: only removes; a purge removes the same data, nothing comes back"
OME = "exempt: only removes; called from OME runs, which wait at the purge gate"
TAXONOMY = "exempt: .taxonomy.md is kept by a purge (R5)"

POINTS: dict[tuple[str, str, str], str] = {
    ("_base/writer.py", "MarkdownWriter.write", ".mkdir"): CHECKED,
    ("_base/writer.py", "_replace_with_retry", "os.replace"): WRITE,
    ("_base/writer.py", "_unlink_quiet", ".unlink"): WRITE,
    ("_readers/taxonomy_reader.py", "ensure_taxonomy", ".mkdir"): TAXONOMY,
    ("_readers/taxonomy_reader.py", "ensure_taxonomy", ".write_text"): TAXONOMY,
    (
        "_writers/agent_skill_writer.py",
        "AgentSkillWriter.delete_skill",
        "shutil.rmtree",
    ): OME,
    ("_writers/knowledge_writer.py", "KnowledgeWriter.write", ".mkdir"): CHECKED,
    ("_writers/knowledge_writer.py", "_remove_dir", "shutil.rmtree"): KNOWLEDGE,
    ("_writers/knowledge_writer.py", "_write_file", ".mkdir"): KNOWLEDGE,
    ("_writers/knowledge_writer.py", "_write_file", ".write_text"): KNOWLEDGE,
    ("knowledge_store.py", "_back_up", "shutil.move"): CHECKED,
    ("knowledge_store.py", "_drop", "shutil.rmtree"): REMOVE,
    ("knowledge_store.py", "_move_to_category", ".mkdir"): CHECKED,
    ("knowledge_store.py", "_move_to_category", "shutil.move"): CHECKED,
    ("knowledge_store.py", "_remove_dir", "shutil.rmtree"): REMOVE,
    ("knowledge_store.py", "_restore", "shutil.move"): CHECKED,
    ("knowledge_store.py", "_rewrite_frontmatter", ".write_text"): CHECKED,
    ("knowledge_store.py", "_write_original", ".mkdir"): CHECKED,
    ("knowledge_store.py", "_write_original", ".write_bytes"): CHECKED,
    ("owner_dir.py", "MarkdownOwnerDirStore._empty", "shutil.rmtree"): DELETE,
    ("owner_dir.py", "MarkdownOwnerDirStore._move", ".mkdir"): DELETE,
    ("owner_dir.py", "MarkdownOwnerDirStore._move", ".rename"): DELETE,
    ("owner_dir.py", "MarkdownOwnerDirStore._move", ".write_text"): DELETE,
    ("owner_dir.py", "MarkdownOwnerDirStore.sweep_trash", "shutil.rmtree"): DELETE,
}


def _operation(node: ast.Attribute) -> str | None:
    owner = node.value.id if isinstance(node.value, ast.Name) else None
    if owner == "os":
        return f"os.{node.attr}" if node.attr in _OS else None
    if owner == "shutil":
        is_write = node.attr in ("move", "rmtree") or node.attr.startswith("copy")
        return f"shutil.{node.attr}" if is_write else None
    return f".{node.attr}" if node.attr in _METHODS else None


class _Visitor(ast.NodeVisitor):
    def __init__(self, rel: str, found: set[tuple[str, str, str]]) -> None:
        self._rel, self._found, self._scopes = rel, found, list[str]()

    def _scoped(self, node: ast.AST, name: str) -> None:
        self._scopes.append(name)
        self.generic_visit(node)
        self._scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._scoped(node, node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._scoped(node, node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._scoped(node, node.name)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        op = _operation(node)
        if op is not None:
            self._found.add((self._rel, ".".join(self._scopes) or "<module>", op))
        self.generic_visit(node)


def write_points(package: Path) -> set[tuple[str, str, str]]:
    """``(file, enclosing function, operation)`` of every write call."""
    found: set[tuple[str, str, str]] = set()
    for source in sorted(package.rglob("*.py")):
        rel = source.relative_to(package).as_posix()
        _Visitor(rel, found).visit(ast.parse(source.read_text(encoding="utf-8")))
    return found


def test_write_points_pinned() -> None:
    found = write_points(_PKG)
    assert sorted(found - POINTS.keys()) == [], "unclassified write call"
    assert sorted(POINTS.keys() - found) == [], "listed write call is gone"


def test_scanner_sees_a_new_write(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        "import shutil\nfrom pathlib import Path\n"
        "def f(x):\n    Path(x).write_text('')\n    'a'.replace('a', 'b')\n"
        "async def g(run):\n    await run(shutil.move, 'a', 'b')\n",
        encoding="utf-8",
    )
    assert write_points(tmp_path) == {
        ("m.py", "f", ".write_text"),
        ("m.py", "g", "shutil.move"),
    }
