# `everos.algo` — the algorithms

Every memory algorithm EverOS runs lives here, one family per subpackage:

| Family | What it holds |
|---|---|
| `types` | The data contracts every family and its hosts share: `MemCell`, `Episode`, `AtomicFact`, `Foresight`, `Profile`, `AgentCase`, `AgentSkill`, `KnowledgeMemory`, `CategorySpec`, `RawFile`, `ParsedContent`, the rank candidates |
| `llm` | The `LLMClient` protocol, `LLMMessage` and its content parts, `StrictOutputModel` / `parse_reply`, `LLMOutputError`, and the `to_*_response_format` schema builders |
| `utils` | Prompt rendering (`render_prompt`, `tagged`, `paragraphs`, `append_custom_instructions`), token budgets under `o200k_base`, UTC timestamp rendering, chat-line rendering |
| `boundary` | Conversation -> MemCell boundary detection (`adetect_boundaries`, `BoundaryDetector`) |
| `user_memory` | `EpisodeExtractor`, `EpisodeSummarizer`, `AtomicFactExtractor`, `ForesightExtractor`, `ProfileExtractor`, `EpisodeReflector`, `OutputLanguage` |
| `agent_memory` | `AgentCaseExtractor` with its gate (`AgentCaseGate`, `CaseJudge`), `AgentSkillExtractor`, `AgentProfileExtractor`, the skip reasons |
| `clustering` | `Cluster` and the online operators `cluster_by_geometry` / `acluster_by_llm` |
| `knowledge` | `KnowledgeExtractor`: parsed document -> topic-tree memories, with category classification |
| `rank` | Fusion (`rrf` / `lr` / `vector_anchored`), the `arank` dispatcher and its facades, LLM rerank, the retrieval compositions (maxsim / hybrid / cluster / agentic), heap-expand episode/fact competition and multi-round evidence selection |

Prompts live beside their operator as module-level string constants under
`<family>/prompts/`; hosts add requirements through an operator's
`custom_instructions=` argument, never by editing a prompt at runtime.

The multimodal parser is not here: it fetches URLs, writes temporary files and
shells out to LibreOffice, so it is a provider in
`everos/component/parser/backend/` that consumes `algo.types` and `algo.llm`.

## Contract

- Pass all input data, configuration values, and prior workflow state explicitly.
  Return results or updated state as values; the caller decides what to persist.
- Keep working data local to one call. Do not retain requests, conversation
  history, caches, locks, or mutable configuration between calls. An extractor
  object may hold the capabilities it was constructed with (an LLM client); the
  caller still owns their lifecycle, and the object accumulates nothing per call.
- Do not mutate caller-owned inputs. Produce new result objects.
- Receive external capabilities such as an LLM, embedding function, retrieval
  callback, clock, or random source as explicit arguments when needed. The caller
  constructs them, owns their state and lifecycle, and supplies credentials.
- Do not read environment variables, application settings, global context, files,
  or databases. Do not discover or construct providers, stores, or schedulers.
  The one lazy load is tiktoken's `o200k_base` encoding (`utils.tokens`), which
  `service/_warmup.py` fetches at startup so no request pays for it.
- Import no other EverOS layer, including its models or configuration types.
  Storage-neutral contracts live in `types`; hosts map their own models in
  `memory/mappers`. An algorithm may call an injected capability; stateless does
  not require that every algorithm be a pure mathematical function.
- Expose each family's public API through its `__init__.py`. Everything is
  native async; there are no synchronous bridges.

`make lint` checks the dependency boundary (`algo-independent`) and rejects
direct imports of runtime implementations (`algo-no-runtime`). These checks
supplement review; they do not prove that every function is stateless. Unit
tests should also verify input preservation and independence between
invocations.

## Example

```python
from everos.algo.rank import heap_expand, select_injection

# The caller has already fetched and grouped the candidates.
ranked = heap_expand(
    sparse=sparse_candidates,
    dense=dense_candidates,
    episode_to_facts=facts_by_episode,
    top_k=10,
)

# Multi-round search owns these collections and passes a snapshot to selection.
selection = select_injection(
    top_k=10,
    core_ids=core_ids,
    core_order=core_order,
    candidate_ids=candidates_by_id,
    subq_hits=scores_by_subquery,
    guarantee=guaranteed_candidates,
    core_overflow=False,
)
```

For a live policy, `core_overflow` also accepts a caller-owned callback returning
a boolean. The algorithm invokes it at each core-candidate decision, preserving
the host's configuration read boundaries without importing application settings.

## History

Until EverOS 1.6 these families were consumed as the `everalgo-*` PyPI packages.
They were moved in as-is, with their 120-column docstrings (`E501` is exempt
under `src/everos/algo/`; reflow a docstring when you edit it, never a prompt).
A family can still become a separate package later without taking EverOS state
with it.
