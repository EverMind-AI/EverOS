# PromptSlot

PromptSlot is the layer between the algorithm code (`everalgo`) and
the prompts it sends to LLMs. Algorithm code receives a `PromptSlot`
parameter; the *project* (EverOS) supplies defaults and lets operators
override.

> **Status (2026-05-07)**: the YAML loader is implemented; the higher-
> level `PromptSlot` model + sandbox dry-run + three-layer overlay
> resolution arrive when the memory layer ships (see Stage 2).

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
from `everos.memory.prompt_slots`); it wraps the generic category loader
[`YamlConfigLoader`](../src/everos/component/config/loader.py). The generic
loader is shown below — `PromptLoader` is the prompt-slots-specific wrapper
over the same mechanism:

```python
from pathlib import Path
from everos.component.config import YamlConfigLoader

loader = YamlConfigLoader(
    root=Path("src/everos/config"),
    categories={"prompt_slots": None},   # subdir == category name
)

# Reads <root>/prompt_slots/episode_extract.yaml → dict
slot = loader.find("prompt_slots", "episode_extract")

# Refresh after on-disk edits.
loader.refresh()                         # drop the entire cache
loader.refresh("prompt_slots")           # drop one category
loader.refresh("prompt_slots", "episode_extract")  # drop one entry
```

Top-level YAML is required to be a mapping; a list / scalar root
raises `TypeError` to fail-fast (loud, not silent).

## YAML format (proposed; subject to change)

```yaml
# config/prompt_slots/episode_extract.yaml
template: |
  Extract a single episode from this conversation:
  {{ memcell.text }}

variables:
  memcell: input memcell

output_schema:
  type: object
  properties:
    summary: { type: string }
    participants: { type: array }

llm:
  model: gpt-4.1-mini
  temperature: 0.3
  max_tokens: 2000

validation:
  test_cases:
    - input: { memcell: { text: "Hi" } }
      expected: { summary: "...", participants: [] }
```

When layer 2 supplies an override the loader will be re-pointed at
`~/.everos/prompt_slots/`; the runtime resolution logic (currently TBD)
sandbox-runs the merged slot before returning it.

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
