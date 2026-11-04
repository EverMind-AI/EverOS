"""Block splitting and token batching for ``KnowledgeExtractor``.

Pure-compute stage 1 of the extraction pipeline. No LLM calls, no I/O.
Normalizes parsed text, splits it into numbered atom blocks (paragraphs
with table / list merging), and groups atoms into token-bounded batches
ready for sequential LLM windows.

Higher-level normalization (HTML -> Markdown, table-marker injection) is
the parser's responsibility, not knowledge's. Callers that have already
emitted ``TABLE_START_MARKER`` / ``TABLE_END_MARKER`` around HTML-derived
tables get whole-table atoms; callers that have not pay no cost.

NOT exposed in the package ``__all__`` — these utilities are internal to
the knowledge extractor pipeline.
"""

from __future__ import annotations

import re

from everos.algo.utils import count_tokens

__all__ = [
    "DEFAULT_MAX_TOKENS_PER_ATOM",
    "DEFAULT_MAX_TOKENS_PER_BATCH",
    "TABLE_END_MARKER",
    "TABLE_START_MARKER",
    "format_numbered_paragraphs",
    "preprocess_content",
    "split_and_batch_content",
    "split_content_to_blocks",
]

# Input convention markers — callers (typically the parser) emit these around
# HTML-derived tables so that an entire table survives as a single atom.
TABLE_START_MARKER = "TABLE_START"
TABLE_END_MARKER = "TABLE_END"

# Default token budget per LLM-window batch. 80K leaves headroom on a 128K-context
# model after the topic-extraction prompt + JSON response. Tunable per call.
DEFAULT_MAX_TOKENS_PER_BATCH = 80_000

# Ceiling for a single atom. The table / list merge rules below are unbounded by
# design — a table is most useful to the segmenter as one unit — so a page that is
# mostly one table used to collapse into a single atom that no batch budget could
# contain, and the whole document went to the model in one over-budget prompt.
# This cap is deliberately loose: it never fires on an ordinary table and only
# rescues the pathological case.
DEFAULT_MAX_TOKENS_PER_ATOM = DEFAULT_MAX_TOKENS_PER_BATCH // 4


def preprocess_content(content: str) -> str:
    """Normalize parsed text before block splitting.

    Strips outer whitespace and collapses runs of three or more blank lines
    to a single blank-line separator. Returns an empty string for empty
    input.
    """
    if not content:
        return ""
    text = content.strip()
    return re.sub(r"\n{3,}", "\n\n", text)


_LIST_ITEM_RE = re.compile(r"^[-*]\s")


def _consume_table_marker_atom(lines: list[str], i: int) -> tuple[str, int]:
    """Consume an explicit ``TABLE_START`` ... ``TABLE_END`` block.

    Assumes ``lines[i].strip() == TABLE_START_MARKER`` on entry. Returns the
    merged atom text (may be empty if the bracket pair is empty) and the index
    just past the closing marker.
    """
    j = i + 1
    table_lines: list[str] = []
    while j < len(lines) and lines[j].strip() != TABLE_END_MARKER:
        stripped = lines[j].strip()
        if stripped:
            table_lines.append(stripped)
        j += 1
    if j < len(lines):
        j += 1  # consume TABLE_END
    return "\n".join(table_lines), j


def _consume_native_table_rows(lines: list[str], i: int) -> tuple[str, int]:
    """Consume a run of consecutive native markdown table rows.

    Assumes ``lines[i].strip().startswith('|')`` on entry. Returns merged text
    and the index of the first non-row line.
    """
    table_lines = [lines[i].strip()]
    j = i + 1
    while j < len(lines) and lines[j].strip().startswith("|"):
        table_lines.append(lines[j].strip())
        j += 1
    return "\n".join(table_lines), j


def _consume_list_items(lines: list[str], i: int) -> tuple[str, int]:
    """Consume a run of consecutive markdown list items (``-`` or ``*`` prefix).

    Assumes ``_LIST_ITEM_RE`` matches ``lines[i].strip()`` on entry. Returns
    merged text and the index of the first non-item line.
    """
    list_lines = [lines[i].strip()]
    j = i + 1
    while j < len(lines) and _LIST_ITEM_RE.match(lines[j].strip()):
        list_lines.append(lines[j].strip())
        j += 1
    return "\n".join(list_lines), j


_MAX_TOKENS_PER_CHAR = 4
"""Hard upper bound on tokens per character: 4 UTF-8 bytes max, 1 token per byte max."""

_TABLE_SEPARATOR_CHARS = set("-: |")


def _table_header(lines: list[str]) -> list[str]:
    """Return the ``[header, separator]`` rows when ``lines`` opens a markdown table."""
    if len(lines) >= 2 and lines[0].lstrip().startswith("|"):
        second = lines[1].strip()
        if second.startswith("|") and set(second) <= _TABLE_SEPARATOR_CHARS:
            return lines[:2]
    return []


def _split_long_line(text: str, max_tokens: int) -> list[str]:
    """Split a single line that alone exceeds ``max_tokens``, on CHARACTER boundaries.

    Slicing the *token* list instead would cut a multi-token character in half, and
    decoding the halves independently yields U+FFFD on both sides — measured on a
    66,000-character CJK line, which came back 66,006 characters with mojibake at
    every seam. Character slices concatenate back to the original exactly.
    """
    tokens_per_char = count_tokens(text) / len(text)
    # 0.9 leaves margin for a denser stretch than the whole-line average.
    window = max(1, int(max_tokens / tokens_per_char * 0.9))

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + window, len(text))
        while end - start > 1 and count_tokens(text[start:end]) > max_tokens:
            end = start + (end - start) * 3 // 4
        pieces.append(text[start:end])
        start = end
    return pieces


def _split_oversized_atom(text: str, max_tokens: int) -> list[str]:
    """Break one over-budget atom into pieces that each fit ``max_tokens``.

    Splits on line boundaries, and re-emits a markdown table's header rows at the
    head of every piece so a segmenter reading piece 2 still sees the column names.
    The pieces stay adjacent in the atom list, so a topic that claims all of them
    reassembles the original text through ``_topic_build._rebuild_content``.
    """
    lines = text.split("\n")
    header = _table_header(lines)
    header_text = "\n".join(header)
    header_tokens = count_tokens(header_text) if header else 0

    pieces: list[str] = []
    current: list[str] = []
    current_tokens = header_tokens

    def flush() -> None:
        if current:
            pieces.append("\n".join(header + current) if header else "\n".join(current))

    newline_tokens = count_tokens("\n")
    for line in lines[len(header) :]:
        # Count the "\n" that ``join`` will insert. Omitting it undercounts a piece by
        # roughly one token per line, which pushes long runs over the budget and drops
        # them into the character-boundary fallback — splitting list items mid-item.
        line_tokens = count_tokens(line) + newline_tokens
        if current and current_tokens + line_tokens > max_tokens:
            flush()
            current, current_tokens = [line], header_tokens + line_tokens
        else:
            current.append(line)
            current_tokens += line_tokens
    flush()
    if not pieces:  # pragma: no cover - defensive; ``lines`` is never empty here
        pieces = [text]

    # A line longer than the budget on its own cannot split on line boundaries, so fall
    # back to character boundaries. Without this the "every atom fits" invariant
    # silently fails on the shape that motivated the cap: CJK prose extracted from a
    # PDF often arrives as one very long unwrapped line.
    bounded: list[str] = []
    for piece in pieces:
        if count_tokens(piece) <= max_tokens:
            bounded.append(piece)
        else:
            bounded.extend(_split_long_line(piece, max_tokens))
    return bounded


def split_content_to_blocks(
    content: str,
    max_atom_tokens: int = DEFAULT_MAX_TOKENS_PER_ATOM,
) -> list[tuple[int, str]]:
    """Split ``content`` into indexed atom blocks ``[(id, text), ...]``.

    Merging rules (applied in this order):

    1. ``TABLE_START_MARKER`` / ``TABLE_END_MARKER`` enclose one atom — the
       entire bracketed body becomes a single block.
    2. Consecutive native markdown table rows (lines starting with ``|``) merge
       into one atom.
    3. Consecutive list items (lines starting with ``-`` or ``*`` followed by
       whitespace) merge into one atom.
    4. Otherwise each non-empty line becomes its own atom.
    5. A merged atom exceeding ``max_atom_tokens`` is broken on line boundaries
       into adjacent pieces that each fit, with a markdown table's header rows
       repeated at the head of every piece.

    Rule 5 exists because rules 1-3 are unbounded: a page that is mostly one table
    or one list otherwise collapses into a single atom that no batch budget can
    contain (``split_and_batch_content`` emits such an atom as its own over-budget
    batch rather than dropping it). Pass a larger ``max_atom_tokens`` to loosen it,
    or ``0`` to disable splitting entirely.

    Blank lines act only as separators and never become atoms. Ids stay dense and
    sequential over the returned list, which is the contract ``block_refs`` relies on.
    """
    if not content:
        return []

    lines = content.split("\n")
    blocks: list[tuple[int, str]] = []
    idx = 0
    i = 0

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line == TABLE_START_MARKER:
            merged, i = _consume_table_marker_atom(lines, i)
        elif line.startswith("|"):
            merged, i = _consume_native_table_rows(lines, i)
        elif _LIST_ITEM_RE.match(line):
            merged, i = _consume_list_items(lines, i)
        else:
            merged = line
            i += 1

        if merged:
            # Cheap pre-filter so the tokenizer stays off the path for ordinary atoms.
            # A character is at most 4 UTF-8 bytes and BPE emits at most one token per
            # byte, so ``4 * len(text)`` is a hard upper bound on the token count. Do
            # NOT tighten this to ``len(text)``: CJK and emoji exceed one token per
            # character (traditional Chinese ~1.05, emoji ~2.7), which would let an
            # over-budget atom through exactly on the corpora we care about.
            if (
                max_atom_tokens > 0
                and _MAX_TOKENS_PER_CHAR * len(merged) >= max_atom_tokens
            ):
                pieces = _split_oversized_atom(merged, max_atom_tokens)
            else:
                pieces = [merged]
            for piece in pieces:
                blocks.append((idx, piece))
                idx += 1

    return blocks


def split_and_batch_content(
    atoms: list[tuple[int, str]],
    max_tokens: int = DEFAULT_MAX_TOKENS_PER_BATCH,
) -> list[list[tuple[int, str]]]:
    """Greedy bin-packing of atoms into batches by ``o200k_base`` token count.

    Each returned batch fits within ``max_tokens`` tokens — except for the
    pathological case of a single atom that already exceeds the limit, which
    becomes its own (over-budget) batch rather than being silently dropped.
    Atoms are never split across batches; ordering is preserved.

    An empty ``atoms`` list returns an empty list (zero batches).
    """
    if not atoms:
        return []

    batches: list[list[tuple[int, str]]] = []
    current_batch: list[tuple[int, str]] = []
    current_tokens = 0

    for atom in atoms:
        atom_tokens = count_tokens(atom[1])
        if current_batch and current_tokens + atom_tokens > max_tokens:
            batches.append(current_batch)
            current_batch = [atom]
            current_tokens = atom_tokens
        else:
            current_batch.append(atom)
            current_tokens += atom_tokens

    if current_batch:
        batches.append(current_batch)

    return batches


def format_numbered_paragraphs(atoms: list[tuple[int, str]]) -> str:
    """Format atoms as ``'ID: content'`` lines for LLM prompt input."""
    return "\n".join(f"{idx}: {text}" for idx, text in atoms)
