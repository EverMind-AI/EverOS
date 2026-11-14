"""The rewritten episode prompt cannot come back: every episode request is the original.

Every chat cell is sent everalgo-user-memory 0.3.1's original episode request
(``chat.episode``). The rewrite that replaced it in everalgo (2026-10: a
system message of rules that kept message times out of the content, and a
model-written summary) was deleted from everalgo and from EverOS on
2026-11-14, with GPT-4.1-mini dating 1% of LoCoMo episodes under it. These
checks fail when any of it returns:

- no EverOS module imports everalgo's episode operator (``EpisodeExtractor``),
  the summarizer that belonged to the rewrite (``EpisodeSummarizer``) or a
  rewrite prompt module, however it is spelled (a history note in a docstring
  is not an import);
- the vendored everalgo-user-memory carries none of them;
- every episode request, one-to-one or group, with pictures or without, is one
  user message that opens with the original's role sentence and carries none
  of the rewrite's text;
- a one-to-one text cell's request is byte for byte everalgo's own episode
  request, so the library's default and EverOS's request are the same text.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import everalgo.user_memory
import everalgo.user_memory.episode
import pytest
from everalgo.testing import FakeLLMClient
from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import OutputLanguage
from everalgo.user_memory.episode._render import _render_messages

from everos.memory.extract.pipeline.episode import build_messages, extract

_ROOT = Path(__file__).resolve().parents[5]
_SCANNED = ("src", "benchmarks", "scripts", "examples")

_GONE = frozenset(
    {
        "everalgo.user_memory.prompts.en.episode",
        "everalgo.user_memory.prompts.en.episode_summary",
        "everalgo.user_memory.episode._summarizer",
    }
)
"""The rewrite's two prompt modules and its summarizer, deleted from everalgo."""

_FORBIDDEN_MODULES = _GONE | {"everalgo.user_memory.episode._extractor"}
"""What EverOS never imports: the deleted modules, and the module of
everalgo's episode operator, whose request is not EverOS's to choose."""

_OPERATORS = frozenset({"EpisodeExtractor", "EpisodeSummarizer"})

_ORIGINAL_OPENING = (
    "\nYou are an episodic memory generation expert. "
    "Please convert the following conversation into an episodic memory.\n"
)
"""The first line of the original request, as 0.3.1 rendered it."""

_REWRITE_TEXT = (
    "You write episodic memories for a long-term memory system",
    "<what_to_write>",
    "<stay_faithful>",
    "<keep_details>",
    "<write_concisely>",
    "<time_references>",
    "<reply_format>",
    "<conversation",
)
"""The rewrite's role sentence and the blocks it alone carried."""

_T0 = 1_700_000_000_000
_PNG = b"\x89PNG\r\n\x1a\nnot really a picture"
_ROSTER = 'This conversation is a group chat in "Trip team".'


def _python_files() -> list[Path]:
    return sorted(
        path
        for folder in _SCANNED
        for path in (_ROOT / folder).rglob("*.py")
        if ".venv" not in path.parts
    )


def _rewrite_references(path: Path) -> list[str]:
    """Each import of a rewrite module or operator in ``path``, and each use of one.

    Covers ``import x``, ``from x import y`` (``y`` a module or a name), a
    module path written as a string (an ``importlib`` import), and an
    operator reached as a name or an attribute. Text inside a docstring or a
    comment is not code and is not read.
    """
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [
                alias.name for alias in node.names if alias.name in _FORBIDDEN_MODULES
            ]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            if node.module in _FORBIDDEN_MODULES:
                found.append(node.module)
            found += [
                f"{node.module}.{alias.name}"
                for alias in node.names
                if alias.name in _OPERATORS
                or f"{node.module}.{alias.name}" in _FORBIDDEN_MODULES
            ]
        elif isinstance(node, ast.Constant) and node.value in _FORBIDDEN_MODULES:
            found.append(f"string {node.value}")
        elif isinstance(node, ast.Name) and node.id in _OPERATORS:
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in _OPERATORS:
            found.append(f".{node.attr}")
    return found


def test_the_scan_reads_every_everos_module() -> None:
    """Guard the guard: a wrong root would let every file pass unread."""
    files = _python_files()

    assert _ROOT / "src/everos/memory/extract/pipeline/episode.py" in files
    assert len(files) > 400


def test_no_everos_module_imports_the_rewrite_or_its_operators() -> None:
    offenders = [
        f"{path.relative_to(_ROOT)}: {hit}"
        for path in _python_files()
        for hit in _rewrite_references(path)
    ]

    assert offenders == []


def test_a_rewrite_reference_is_found_however_it_is_spelled(tmp_path: Path) -> None:
    source = tmp_path / "copy.py"
    source.write_text(
        '"""A docstring may name EpisodeExtractor as history."""\n'
        "import everalgo.user_memory.prompts.en.episode\n"
        "from everalgo.user_memory.prompts.en import episode\n"
        "from everalgo.user_memory import EpisodeExtractor\n"
        "import everalgo.user_memory as um\n"
        "um.EpisodeSummarizer\n"
        'importlib.import_module("everalgo.user_memory.episode._extractor")\n'
        "from everalgo.user_memory.prompts.en import episode_original\n",
        encoding="utf-8",
    )
    hits = sorted(_rewrite_references(source))

    assert hits == [
        ".EpisodeSummarizer",
        "everalgo.user_memory.EpisodeExtractor",
        "everalgo.user_memory.prompts.en.episode",
        "everalgo.user_memory.prompts.en.episode",
        "string everalgo.user_memory.episode._extractor",
    ]


@pytest.mark.parametrize("module", sorted(_GONE))
def test_the_vendored_everalgo_carries_no_rewrite_module(module: str) -> None:
    assert importlib.util.find_spec(module) is None


def test_the_vendored_everalgo_carries_no_summarizer() -> None:
    assert not hasattr(everalgo.user_memory, "EpisodeSummarizer")
    assert not hasattr(everalgo.user_memory.episode, "EpisodeSummarizer")


def _cell() -> MemCell:
    items = [
        ChatMessage(
            id=f"m_{i}",
            role="user",
            content=text,
            timestamp=_T0 + i * 60_000,
            sender_id=sender,
            sender_name=name,
        )
        for i, (sender, name, text) in enumerate(
            [
                ("u1", "Caroline", "I went to a support group yesterday."),
                ("u2", "Melanie", "here is my latte [IMAGE: img-1]"),
                ("u1", "Caroline", "Looks great, let's go next Friday."),
            ]
        )
    ]
    return MemCell(items=items, timestamp=items[-1].timestamp)


def _texts(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    parts = [part if isinstance(part, dict) else part.model_dump() for part in content]
    return [part["text"] for part in parts if part.get("type") == "text"]


@pytest.mark.parametrize("roster", [None, _ROSTER], ids=["one-to-one", "group"])
@pytest.mark.parametrize("pictures", [0, 1], ids=["text", "pictures"])
@pytest.mark.parametrize("language", [None, OutputLanguage("Chinese")])
async def test_every_episode_request_is_the_original(
    roster: str | None, pictures: int, language: OutputLanguage | None
) -> None:
    llm = FakeLLMClient(responses=['{"title": "t", "content": "c"}'])

    await extract(
        llm,
        _cell(),
        pictures=[("img-1", _PNG, "image/png")] * pictures,
        custom_instructions=None,
        output_language=language,
        roster=roster,
    )

    (call,) = llm.calls
    (message,) = call.messages
    texts = _texts(message.content)
    assert message.role == "user"
    assert texts[0].startswith(_ORIGINAL_OPENING)
    assert [text for text in _REWRITE_TEXT if any(text in part for part in texts)] == []


@pytest.mark.parametrize(
    "language", [None, OutputLanguage("English"), OutputLanguage("Chinese")]
)
@pytest.mark.parametrize("custom", [None, "", "Name every product version."])
def test_a_one_to_one_text_cell_is_sent_everalgo_s_own_episode_request(
    language: OutputLanguage | None, custom: str | None
) -> None:
    ours = build_messages(
        _cell(), pictures=[], custom_instructions=custom, output_language=language
    )

    assert ours == _render_messages(
        _cell(), custom_instructions=custom, output_language=language
    )
