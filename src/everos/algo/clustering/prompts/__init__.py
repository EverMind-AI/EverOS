"""Clustering prompts (en + zh).

Each prompt is a module-level Python string constant. Callers add to the rules via the
per-call ``custom_instructions=`` argument, appended as the system message's last block; the bundled rules themselves
change only by monkey-patching the constant at startup.
"""
