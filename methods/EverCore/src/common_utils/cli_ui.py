"""CLI UI utilities for width-aware, emoji-aligned, optionally colored output.

This module provides a small, reusable toolkit for building terminal UI outputs
that adapt to the current terminal width, align text containing emoji/wide
characters, and optionally apply ANSI colors. It avoids external dependencies
and degrades gracefully if the environment does not support color.

Key features:
- Terminal width auto-detection
- Display-width aware wrapping and padding (emoji alignment)
- Box panels and section headings
- Simple tables with auto-fitting columns and truncation
- Optional ANSI color styling (toggle via constructor, env var NO_COLOR or
  CLI_UI_COLOR=0)

Design goals:
- Keep the API small and easy to extend
- Separate measurement (display width) from styling
- Keep ANSI escape sequences from affecting layout measurements

Usage example:

    ui = CLIUI()  # auto width, color enabled unless NO_COLOR set
    ui.banner("🧠 EverMem Memory Conversation Assistant", subtitle="Memory-Enhanced Chat")
    ui.section_heading("📊 Available group conversations")
    ui.table(headers=["#", "Group", "Name", "Count"], rows=[["1", "g001", "Team", "12"]])

"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import sys
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================================
# ANSI Color & Style Helpers
# ============================================================================


_ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI SGR sequences from text."""
    if not text:
        return ""
    return _ANSI_PATTERN.sub("", text)


def _supports_color() -> bool:
    """Return whether the current stream likely supports ANSI colors."""
    if os.getenv("NO_COLOR") is not None:
        return False
    if os.getenv("CLI_UI_COLOR", "1") in {"0", "false", "False"}:
        return False
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return False
    return stream.isatty()


class _Style:
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    DIM = "\x1b[2m"
    # Colors
    BLACK = "\x1b[30m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"
    MAGENTA = "\x1b[35m"
    CYAN = "\x1b[36m"
    WHITE = "\x1b[37m"


@dataclass
class ColorTheme:
    """Simple color theme for UI accents.

    Set any attribute to None to disable color for that role individually.
    """

    title: Optional[str] = _Style.CYAN
    subtitle: Optional[str] = _Style.DIM
    heading: Optional[str] = _Style.BOLD
    success: Optional[str] = _Style.GREEN
    warning: Optional[str] = _Style.YELLOW
    error: Optional[str] = _Style.RED
    info: Optional[str] = _Style.BLUE
    label: Optional[str] = _Style.MAGENTA


def _apply_style(enabled: bool, text: str, style: Optional[str]) -> str:
    if not enabled or not style:
        return text
    return f"{style}{text}{_Style.RESET}"


# ============================================================================
# Display Width Measurement (Emoji/Wide Char Aware)
# ============================================================================


def _char_display_width(ch: str) -> int:
    """Approximate display width of a single character.

    This handles common cases without external dependencies:
    - Combining marks => width 0
    - East Asian Wide/Fullwidth => width 2
    - Common emoji ranges => width 2
    - Otherwise => width 1
    """

    if not ch:
        return 0

    if unicodedata.combining(ch):
        return 0

    category = unicodedata.category(ch)
    if category in {"Mn", "Me"}:  # Nonspacing/Enclosing marks
        return 0

    eaw = unicodedata.east_asian_width(ch)
    if eaw in {"W", "F"}:
        return 2

    # Basic emoji heuristic (covers most common emoji)
    code = ord(ch)
    if (
        0x1F300
        <= code
        <= 0x1FAFF  # Misc symbols & pictographs, supplemental symbols & pictographs, etc.
        or 0x1F600 <= code <= 0x1F64F  # Emoticons
        or 0x1F680 <= code <= 0x1F6FF  # Transport & Map
        or 0x2600 <= code <= 0x26FF  # Misc symbols
        or 0x2700 <= code <= 0x27BF  # Dingbats
    ):
        return 2

    # Variation selectors should have zero width when standalone
    if 0xFE00 <= code <= 0xFE0F:
        return 0

    return 1


def visible_width(text: str) -> int:
    """Compute the on-screen width of a string (ignores ANSI SGR codes)."""
    if not text:
        return 0
    s = _strip_ansi(text)
    width = 0
    for ch in s:
        width += _char_display_width(ch)
    return width


def truncate_to_width(text: str, max_width: int, ellipsis: str = "…") -> str:
    """Truncate string so that display width <= max_width, appending ellipsis if needed."""
    if max_width <= 0:
        return ""
    if visible_width(text) <= max_width:
        return text
    # Reserve space for ellipsis
    target = max(0, max_width - visible_width(ellipsis))
    out: List[str] = []
    acc = 0
    for ch in text:
        w = _char_display_width(ch) if ch != "\x1b" else 0
        if acc + w > target:
            break
        out.append(ch)
        acc += w
    out.append(ellipsis)
    return "".join(out)


def wrap_text(text: str, max_width: int) -> List[str]:
    """Word-wrap text by visible width, preserving existing newlines and ANSI codes.

    - Breaks on whitespace where possible; falls back to character-level when needed
    - Does not count ANSI SGR sequences toward width
    """
    if max_width <= 0:
        return [""]
    if not text:
        return [""]

    lines: List[str] = []
    for raw_line in text.splitlines(False):
        if visible_width(raw_line) <= max_width:
            lines.append(raw_line)
            continue

        # Tokenize on whitespace to get words and spaces
        tokens = re.findall(r"\x1b\[[0-9;]*m|\S+|\s+", raw_line)
        current: List[str] = []
        current_w = 0

        def flush() -> None:
            nonlocal current, current_w
            lines.append("".join(current))
            current = []
            current_w = 0

        for tok in tokens:
            if _ANSI_PATTERN.fullmatch(tok):  # ANSI sequence has width 0
                current.append(tok)
                continue

            tok_w = visible_width(tok)
            if tok_w <= (max_width - current_w):
                current.append(tok)
                current_w += tok_w
            else:
                # If token is whitespace, flush line and skip leading spaces
                if tok.isspace():
                    flush()
                    continue
                # Split token to fit
                idx = 0
                while idx < len(tok):
                    # Find how many characters we can take
                    if current_w == max_width:
                        flush()
                    room = max_width - current_w
                    if room <= 0:
                        flush()
                        room = max_width
                    taken: List[str] = []
                    taken_w = 0
                    while (
                        idx < len(tok)
                        and taken_w + _char_display_width(tok[idx]) <= room
                    ):
                        taken.append(tok[idx])
                        taken_w += _char_display_width(tok[idx])
                        idx += 1
                    if taken:
                        current.append("".join(taken))
                        current_w += taken_w
                    if idx < len(tok):
                        flush()
                # done splitting token
        if current:
            lines.append("".join(current))
    return lines


def get_terminal_width(fallback: int = 80, min_width: int = 40) -> int:
    """Return the current terminal column width with reasonable bounds."""
    try:
        width = shutil.get_terminal_size((fallback, 24)).columns
    except Exception:
        width = fallback
    if width < min_width:
        width = min_width
    return width


# ============================================================================
# Box Styles
# ============================================================================


@dataclass
class BoxStyle:
    tl: str
    tr: str
    bl: str
    br: str
    h: str
    v: str
    t: str  # tee top (for separators)
    b: str  # tee bottom
    l: str  # tee left
    r: str  # tee right
    cross: str


HEAVY_BOX = BoxStyle(
    tl="╔", tr="╗", bl="╚", br="╝", h="═", v="║", t="╦", b="╩", l="╠", r="╣", cross="╬"
)
LIGHT_BOX = BoxStyle(
    tl="┌", tr="┐", bl="└", br="┘", h="─", v="│", t="┬", b="┴", l="├", r="┤", cross="┼"
)
ASCII_BOX = BoxStyle(
    tl="+", tr="+", bl="+", br="+", h="-", v="|", t="+", b="+", l="+", r="+", cross="+"
)


# ============================================================================
# Core UI Renderer
# ============================================================================


class CLIUI:
    """Width-aware, emoji-aligned terminal UI helper.

    Parameters:
        width: Override terminal width for layout. If None, auto-detect.
        color_enabled: Enable ANSI colors (default auto based on env/tty).
        box: Which box-drawing style to use ('heavy', 'light', 'ascii').
        theme: ColorTheme for accents.
        margin: Number of spaces to leave at left/right of the terminal.
        padding: Spaces inside panels at left/right.
    """

    def __init__(
        self,
        width: Optional[int] = None,
        color_enabled: Optional[bool] = None,
        box: str = "heavy",
        theme: Optional[ColorTheme] = None,
        margin: int = 0,
        padding: int = 1,
    ) -> None:
        if width is None:
            width = get_terminal_width()
        self.term_width: int = width
        self.margin: int = max(0, margin)
        self.padding: int = max(0, padding)
        self.color_enabled: bool = (
            _supports_color() if color_enabled is None else bool(color_enabled)
        )
        self.theme: ColorTheme = theme or ColorTheme()
        self.box: BoxStyle = {
            "heavy": HEAVY_BOX,
            "light": LIGHT_BOX,
            "ascii": ASCII_BOX,
        }.get(box, HEAVY_BOX)

    # ------------------------ Basic Styled Printing ------------------------
    def text(self, text: str = "") -> None:
        """Print plain text respecting margin."""
        line_prefix = " " * self.margin if self.margin else ""
        print(f"{line_prefix}{text}")

    def styled(self, text: str, role: Optional[str] = None) -> str:
        """Return text styled using the theme role."""
        if role is None:
            return text
        color = getattr(self.theme, role, None)
        return _apply_style(self.color_enabled, text, color)

    # ------------------------ Rules & Headings -----------------------------
    def rule(self, char: Optional[str] = None) -> None:
        """Print a horizontal rule spanning the content width."""
        h = char or self.box.h
        inner_width = max(1, self.term_width - self.margin * 2)
        self.text(h * inner_width)

    def section_heading(self, title: str) -> None:
        """Print a section heading as a single-line box-like rule with title centered."""
        inner_width = max(3, self.term_width - self.margin * 2)
        title_str = f" {title} "
        title_str = self.styled(title_str, "heading")
        h = self.box.h
        remaining = inner_width - visible_width(_strip_ansi(title_str))
        if remaining <= 2:
            self.text(truncate_to_width(title_str, inner_width))
            return
        left = remaining // 2
        right = remaining - left
        self.text(f"{h * left}{title_str}{h * right}")

    def banner(self, title: str, subtitle: Optional[str] = None) -> None:
        """Print a decorative banner with title and optional subtitle."""
        inner_w = self._panel_inner_width()

        # Center the title
        title_s = self.styled(title, "title")
        title_vis = visible_width(_strip_ansi(title_s))
        title_pad = max(0, (inner_w - title_vis) // 2)
        centered_title = (" " * title_pad) + title_s

        lines = [centered_title]

        # Center the subtitle if provided
        if subtitle:
            subtitle_s = self.styled(subtitle, "subtitle")
            subtitle_vis = visible_width(_strip_ansi(subtitle_s))
            subtitle_pad = max(0, (inner_w - subtitle_vis) // 2)
            centered_subtitle = (" " * subtitle_pad) + subtitle_s
            lines.append(centered_subtitle)

        self.panel(lines, title=None)

    # ------------------------ Panels --------------------------------------
    def _content_width(self) -> int:
        return max(10, self.term_width - self.margin * 2)

    def _panel_inner_width(self) -> int:
        # account for vertical borders and padding
        return max(1, self._content_width() - 2 - self.padding * 2)

    def panel(self, lines: Sequence[str], title: Optional[str] = None) -> None:
        """Render a bordered panel with wrapped content.

        Args:
            lines: Iterable of line strings (can contain ANSI/emoji). Will be wrapped.
            title: Optional title displayed in top border.
        """
        inner_w = self._panel_inner_width()
        content_lines: List[str] = []
        for raw in lines:
            content_lines.extend(wrap_text(raw, inner_w))

        # Build top border (with optional title)
        top: str
        if title:
            title_s = self.styled(f" {title} ", "heading")
            # compute visible width for capping
            tvis = visible_width(_strip_ansi(title_s))
            max_title = max(0, inner_w)
            if tvis > max_title:
                title_s = truncate_to_width(title_s, max_title)
                tvis = visible_width(_strip_ansi(title_s))
            filler_left = (inner_w - tvis) // 2
            filler_right = inner_w - tvis - filler_left
            top = (
                f"{self.box.tl}{self.box.h * (self.padding + filler_left)}"
                f"{title_s}"
                f"{self.box.h * (self.padding + filler_right)}{self.box.tr}"
            )
        else:
            top = (
                f"{self.box.tl}{self.box.h * (inner_w + self.padding * 2)}{self.box.tr}"
            )

        # Build bottom and sides
        bottom = (
            f"{self.box.bl}{self.box.h * (inner_w + self.padding * 2)}{self.box.br}"
        )
        left = f"{self.box.v}{' ' * self.padding}"
        right = f"{' ' * self.padding}{self.box.v}"

        # Print with margin
        prefix = " " * self.margin
        print(prefix + top)
        for line in content_lines:
            line = truncate_to_width(line, inner_w)
            pad = inner_w - visible_width(_strip_ansi(line))
            print(prefix + left + line + (" " * pad) + right)
        print(prefix + bottom)

    # ------------------------ Tables --------------------------------------
    def table(
        self,
        headers: Optional[Sequence[str]],
        rows: Sequence[Sequence[str]],
        aligns: Optional[Sequence[str]] = None,
        max_height: Optional[int] = None,
    ) -> None:
        """Render a simple table that auto-fits to the available width.

        - If total width exceeds content width, later columns are truncated.
        - Supports left/center/right alignment via aligns list.
        - If max_height is set and fewer rows can be shown, prints a notice panel.
        """
        if not rows and not headers:
            return

        content_w = self._content_width()

        # Compute column count
        num_cols = len(headers) if headers is not None else len(rows[0])
        aligns = aligns or ["left"] * num_cols

        # Measure column widths (max of header and content)
        col_widths = [0] * num_cols

        def measure(s: str) -> int:
            return visible_width(_strip_ansi(s))

        if headers is not None:
            for i, h in enumerate(headers):
                col_widths[i] = max(col_widths[i], measure(h))
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], measure(str(cell)))

        # Compute available width for a borderless table with single spaces between cols
        # We'll just pad with spaces; if it overflows, we reduce widths from the last column backward.
        spacing = num_cols - 1  # one space between columns
        total = sum(col_widths) + spacing
        max_total = content_w
        if total > max_total:
            # Reduce columns from the last backward, keeping a minimum width of 3
            overflow = total - max_total
            for idx in range(num_cols - 1, -1, -1):
                if overflow <= 0:
                    break
                available_reduce = max(0, col_widths[idx] - 3)
                reduce_by = min(available_reduce, overflow)
                col_widths[idx] -= reduce_by
                overflow -= reduce_by

        def align_text(s: str, width: int, how: str) -> str:
            s = truncate_to_width(s, width)
            pad = width - visible_width(_strip_ansi(s))
            if how == "right":
                return (" " * pad) + s
            if how == "center":
                left = pad // 2
                return (" " * left) + s + (" " * (pad - left))
            return s + (" " * pad)

        # Print header
        prefix = " " * self.margin
        if headers is not None:
            header_cells: List[str] = []
            for i, h in enumerate(headers):
                header_cells.append(
                    self.styled(align_text(h, col_widths[i], "center"), "heading")
                )
            print(prefix + " ".join(header_cells))
            # Use full content width for the separator line
            print(prefix + self.box.h * content_w)

        # Determine visible rows given max_height
        display_rows = rows
        total_count = len(rows)
        if max_height is not None and total_count > max_height:
            display_rows = rows[:max_height]

        for row in display_rows:
            cells: List[str] = []
            for i, cell in enumerate(row):
                cells.append(align_text(str(cell), col_widths[i], aligns[i]))
            print(prefix + " ".join(cells))

        # Notice for truncated rows
        if max_height is not None and total_count > max_height:
            remaining = total_count - max_height
            self.text(
                self.styled(f"... {remaining} more rows not displayed", "subtitle")
            )

    # ------------------------ Convenience ----------------------------------
    def note(self, text: str, icon: Optional[str] = None) -> None:
        msg = f"{icon} {text}" if icon else text
        self.text(self.styled(msg, "info"))

    def success(self, text: str) -> None:
        self.text(self.styled(text, "success"))

    def warning(self, text: str) -> None:
        self.text(self.styled(text, "warning"))

    def error(self, text: str) -> None:
        self.text(self.styled(text, "error"))
