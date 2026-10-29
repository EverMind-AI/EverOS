"""The Textual stylesheet of the demo app, built from the palette.

The CSS interpolates :mod:`everos.entrypoints.tui.demo.theme` colours, so it is
an f-string here rather than a static ``.tcss`` file; the palette stays the one
place a colour is defined.
"""

from __future__ import annotations

from everos.entrypoints.tui.demo.theme import (
    EVEROS_AMBER,
    EVEROS_AMBER_DIM,
    EVEROS_BLACK,
    EVEROS_BORDER,
    EVEROS_CYAN,
    EVEROS_GREEN,
    EVEROS_INK,
    EVEROS_MUTED,
    EVEROS_SURFACE,
    EVEROS_SURFACE_RAISED,
    EVEROS_YELLOW,
)

APP_CSS = f"""
    Screen {{
        background: {EVEROS_BLACK};
        color: {EVEROS_INK};
    }}

    #shell {{
        width: 100%;
        height: 100%;
        padding: 1 2;
        border: round {EVEROS_BORDER};
    }}

    #command-strip {{
        height: 1;
        padding: 0 1;
        color: {EVEROS_INK};
        content-align: left middle;
    }}

    #main {{
        height: 1fr;
        margin-top: 0;
    }}

    #memory-field {{
        width: 1fr;
        border: round {EVEROS_AMBER};
        background: {EVEROS_SURFACE};
        padding: 0 2;
    }}

    #field-header {{
        height: 2;
        content-align: left middle;
    }}

    #field-answer {{
        height: 2;
        border-top: hkey {EVEROS_AMBER_DIM};
        background: {EVEROS_SURFACE_RAISED};
        padding: 0 1;
        content-align: center middle;
    }}

    #right-rail {{
        width: 48;
        height: 100%;
        margin-left: 1;
    }}

    #capabilities {{
        height: 9;
        border: panel {EVEROS_YELLOW};
        border-title-color: {EVEROS_BLACK};
        border-title-background: {EVEROS_YELLOW};
        border-title-style: bold;
        background: {EVEROS_SURFACE_RAISED};
        padding: 0 2;
        margin-bottom: 1;
    }}

    #signal-rail {{
        width: 100%;
        height: 1fr;
        border: round {EVEROS_AMBER};
        background: {EVEROS_SURFACE};
        padding: 1 2;
    }}

    #provenance-strip {{
        height: 6;
        margin-top: 1;
    }}

    #source-lock {{
        width: 1fr;
        height: 100%;
        border: round {EVEROS_CYAN};
        background: {EVEROS_SURFACE};
        padding: 0 2;
        margin-right: 1;
    }}

    #recall-lock {{
        width: 54;
        height: 100%;
        border: round {EVEROS_GREEN};
        background: {EVEROS_SURFACE};
        padding: 0 2;
    }}

    #conversation {{
        height: 6;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        scrollbar-color: {EVEROS_AMBER};
        scrollbar-background: {EVEROS_SURFACE};
        border: round {EVEROS_YELLOW};
        background: {EVEROS_SURFACE};
        color: {EVEROS_INK};
        padding: 0 1;
        margin-top: 1;
    }}

    #conversation-log {{
        height: auto;
        width: 1fr;
        background: {EVEROS_SURFACE};
        color: {EVEROS_INK};
    }}

    #console {{
        height: auto;
        margin-top: 1;
    }}

    #console-prompt {{
        height: auto;
        padding: 0 1;
    }}

    #console-input {{
        border: round {EVEROS_AMBER};
        background: {EVEROS_SURFACE};
    }}

    Footer {{
        background: {EVEROS_BLACK};
        color: {EVEROS_MUTED};
    }}

    FooterKey {{
        background: {EVEROS_BLACK};
    }}

    FooterKey > .footer-key--key {{
        color: {EVEROS_BLACK};
        background: {EVEROS_YELLOW};
        text-style: bold;
    }}

    FooterKey > .footer-key--description {{
        color: {EVEROS_INK};
        background: {EVEROS_BLACK};
    }}
    """
