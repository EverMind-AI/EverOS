"""Boundary extractor prompts (English only).

Each prompt is a module-level Python string constant (no external .md / .yaml /
.toml prompt stores), split into a system template (the rules) and a user template (the data). Callers add to the
rules via the per-call ``custom_instructions=`` argument, appended as the system message's last block; the bundled
rules themselves change only by monkey-patching the constant at startup.
"""
