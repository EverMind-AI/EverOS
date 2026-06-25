# PromptSlot

PromptSlot is the layer between the algorithm code (`everalgo`) and
the prompts it sends to LLMs. Algorithm code receives a `PromptSlot`
parameter; the *project* (EverOS) supplies defaults and lets operators
override.

> **Status (2026-05-07)**: Layer 1 (bundled defaults under
> `config/prompt_slots/`) is live — `PromptLoader` is integrated into the
> memorize pipeline (`service/memorize.py`). Two slots ship today —
> `boundary_detection` and `episode_extract`; other extractors use their
> algo-bundled defaults. Layers 2-3 (app-level overlay from
> `~/.everos/prompt_slots/` and per-call runtime override) are still pending.

## Three-layer overlay

```
config/prompt_slots/<name>.yaml          (Layer 1: defaults shipped with the package)
       ↓
~/.everos/prompt_slots/<name>.yaml       (Layer 2: app-level override; per-deployment)
       ↓
runtime override                         (Layer 3: per-call override; e.g. "force model X")
```

Effective prompt = layer 3 wins → layer 2 → layer 1. Layer 1 is
loaded eagerly at startup; layer 2 is loaded on first reference (lazy);
layer 3 is supplied at the call site.

## Loader

The prompt-slots public entry point is
[`PromptLoader`](../src/everos/memory/prompt_slots/loader.py) (re-exported
from `everos.memory.prompt_slots`). Its public method is
`load(name: str) -> str | None` — returns the override template when the
slot is enabled and non-empty, or `None` to fall back to the algo default.

Internally, `PromptLoader` wraps the generic category loader
[`YamlConfigLoader`](../src/everos/component/config/loader.py):

```python
from everos.memory.prompt_slots import PromptLoader
from pathlib import Path

loader = PromptLoader(config_root=Path("src/everos/config"))

# Returns the template string, or None when disabled / empty.
template = loader.load("episode_extract")
```

The underlying `YamlConfigLoader` supports `find()`, `refresh()`, etc. —
but callers should use `PromptLoader.load()` rather than reaching into the
generic layer directly.

Top-level YAML is required to be a mapping; a list / scalar root
raises `TypeError` to fail-fast (loud, not silent).

## YAML format

Each slot file uses two keys: `enabled` (boolean) and `template` (string).

```yaml
# config/prompt_slots/episode_extract.yaml
enabled: false
template: ""
```

When `enabled` is `true` and `template` is a non-empty string,
`PromptLoader.load()` returns the template as-is. Otherwise it returns
`None`, and the pipeline falls back to the algo-bundled default prompt
(zero override cost).

## Why YAML (not TOML)

Two reasons:

1. **Multiline templates** — TOML's basic-string grammar fights
   prompt content (no easy `{{ jinja }}` variables, awkward escaping).
   YAML's literal block scalar (`|`) preserves prompts as-is.
2. **Comment + reference ergonomics** — operators frequently inherit
   slots, tweak a few keys, and leave inline notes. YAML is more
   forgiving for hand-editing.

The Pydantic Settings file (`config/default.toml`) stays TOML — it's
machine-managed and type-validated; YAML's flexibility costs more
than it pays for that case.

## Why a separate loader (not Pydantic Settings)

Settings = **one** structured tree, validated at load time, tied to a
single source of truth. PromptSlots = **many** separate templates
discovered by name, layered per-deployment. They're different shapes;
forcing one model on the other gets clunky.

## See also

- [`src/everos/memory/prompt_slots/`](../src/everos/memory/prompt_slots/) — `PromptLoader` (prompt-slots public API)
- [`src/everos/component/config/loader.py`](../src/everos/component/config/loader.py) — generic `YamlConfigLoader`
- [`tests/unit/test_component/test_config/test_loader.py`](../tests/unit/test_component/test_config/test_loader.py)
- [`docs/architecture.md`](architecture.md) — layer placement
