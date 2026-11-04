"""User-memory extractor prompts.

Each prompt is a module-level Python string constant. Algorithm authors customize via
per-call ``prompt=`` argument or by monkey-patching the constant at startup.

English only. A parallel ``zh/`` tree used to ship translated prompts, and measurement retired it: prompt
language turns out to dictate output language almost completely — an English prompt carrying no language
rule wrote English for 100% of Chinese conversations, a Chinese one wrote Chinese for 100% of English ones —
so the translations were an implicit language switch maintained by hand. Naming the language through
``output_language`` does the same job from one prompt tree, at zero measured drift, for languages nobody
has to translate a prompt into.
"""
