"""Agent-memory extractor prompts — single multilingual prompt set.

Each prompt is a pair of module-level Python string constants: a static SYSTEM
message of rules and reply skeleton, and a USER template holding only the call's data in snake_case XML-style
elements. Prompts are language-agnostic (the extraction prompts carry a language instruction telling the LLM to
mirror the input language in its output). Algorithm authors customize the rules by monkey-patching a SYSTEM
constant at startup, through the operator module that re-exports it.
"""
