# Custom instructions

Custom instructions let a deployment add requirements to an everalgo
operator without touching the operator's prompt. The text is passed as the
operator's `custom_instructions=` argument; everalgo appends it after its
bundled rules, so it **adds** to them and can never replace a rule or change
the output shape the operator validates against.

They replace the former prompt-slot files, which swapped the whole prompt
template. everalgo's 2026-10 contract line removed
whole-prompt overrides: an operator's rules and its strict reply schema are
written together, so a replaced prompt could no longer be checked against
the schema it has to produce.

## Operators

| File | Operator | Where it is appended |
|---|---|---|
| `boundary_detection.yaml` | `everalgo.boundary.adetect_boundaries` | last block of the system message |
| `episode_extract.yaml` | `EpisodeExtractor.aextract` | last block of the system message |
| `profile_extract.yaml` | `ProfileExtractor.aextract` (every INIT / UPDATE / COMPACT / REGROUP call) | additional-requirements section before the data |
| `reflect.yaml` | `EpisodeReflector.areflect` | last block of the system message |

Atomic-fact and foresight extraction take no custom instructions on the
paths EverOS uses.

## File format

`src/everos/config/custom_instructions/<operator>.yaml`:

```yaml
enabled: false
instructions: ""
```

When `enabled` is `true` and `instructions` is non-blank,
[`CustomInstructions.load()`](../src/everos/memory/custom_instructions/loader.py)
returns the text verbatim; otherwise it returns `None` and the operator runs
on its own rules. The text is never rendered as a template, so braces stay
literal. Every file ships disabled.

```python
from pathlib import Path

from everos.memory.custom_instructions import CustomInstructions

instructions = CustomInstructions(Path("src/everos/config"))
instructions.load("reflect")  # str | None
```

`service/bootstrap.py` builds one `CustomInstructions` and injects it into
boundary detection, the user-memory pipeline and the strategies
(`MemoryDeps.instructions`).

## Migrating from prompt slots

| Before | After |
|---|---|
| the prompt-slot file `<name>.yaml` | `config/custom_instructions/<name>.yaml` |
| `template:` (a whole prompt) | `instructions:` (requirements only; the bundled prompt stays) |
| the prompt-slot loader's `load(name)` | `CustomInstructions.load(operator)` |

A slot that held a whole prompt cannot be carried over as is: keep only the
requirements it added on top of the default prompt.

A per-deployment overlay (`~/.everos/custom_instructions/<operator>.yaml`)
is planned; today only the bundled files are read.

## Why YAML

Instructions are prose, often multi-line; YAML's literal block scalar (`|`)
keeps them as written. `config/default.toml` stays TOML: it is one
validated settings tree, while these are independent texts looked up by
operator name.

## See also

- [`src/everos/memory/custom_instructions/`](../src/everos/memory/custom_instructions/) — `CustomInstructions`
- [`src/everos/component/yaml_loader/loader.py`](../src/everos/component/yaml_loader/loader.py) — generic `YamlConfigLoader`
- [`docs/architecture.md`](architecture.md) — layer placement
