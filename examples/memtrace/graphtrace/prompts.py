"""MemTrace attribution instructions and the EverOS observation prior.

Attribution instructions adapted from MemTrace/toolkits/memtrace_utils.py.
See MEMTRACE_LICENSE.txt for the original MIT license.
"""

from __future__ import annotations

ATTRIBUTION_INSTRUCTIONS = """
Your task is to inspect the execution graph and identify the earliest decisive
  faulty operation.

How to read the execution graph:
- The execution graph consists of operations. Each operation forms a single-hop or
  multi-hop subgraph.
- Within one operation subgraph, variables with in-degree 0 relative to that
  subgraph are the operation's input variables.
- Variables with in-degree >= 1 relative to that subgraph are variables produced by
  the operation.
- Variables with out-degree 0 relative to that subgraph are the operation's final
  output variables.
- To judge an operation, inspect its operation name, category, comment, input
  variables, produced variables, final output variables, variable values, variable
  comments, variable relationships, and creation timestamps.
- If the current operation is locally correct for its role, it is not the error.
  Continue downstream by adding one or more final output variables from the current
  operation to the exploration frontier and inspect the subsequent operations that
  consume them.
- Pay close attention to creation time. The earliest decisive fault is based on
  operation execution order, not on which source message is created first.

Ideal memory-system behavior:
- When a raw input containing critical information arrives, the memory construction
  stage should create or update memory-unit containers so that the critical
  information enters the memory store.
- Later operations over a memory-unit container should preserve the critical
  information. They should not accidentally remove, overwrite, over-summarize, or
  degrade it during updates, consolidation, or deletion.
- When a query requiring that critical information arrives, retrieval should surface
  memory units or context containing the needed information.
- The downstream answer model should then use the retrieved context to answer the
  question correctly.
- Compare the observed trace against this ideal behavior to decide where the first
  decisive break occurs.

Definition of earliest decisive faulty operation:
- An operation is faulty only if its outputs are wrong or insufficient for the role
  it is supposed to perform, given its operation name, category, comment, inputs,
  and available evidence at that point in the workflow.
- Do not mark an operation faulty merely because it appears early or because a
  downstream failure can be traced back to one of its inputs. First decide whether
  the operation itself behaved incorrectly under its own local responsibility.
- Decisive means that correcting this operation's faulty output, while keeping all
  strictly earlier operations unchanged and assuming ideal downstream behavior,
  would have prevented the final wrong answer.
- Earliest means the first such locally faulty and decisive operation in execution
  order.

Required reasoning checklist before returning the final answer:
1. Identify the critical information needed to answer the question.
2. Inspect candidate operations in execution order and decide whether each one is
  locally correct for its own function.
3. If an operation is locally correct, follow its final output variables downstream
  instead of labeling it as the error.
4. Once the critical information enters a memory unit, do not stop at that
  operation. You must keep following downstream operations involving that memory
  unit and check whether the critical information remains present throughout the
  lifetime of the memory unit. If an operation causes this critical information to
  be transferred from one memory unit to another memory unit or to some intermediate
  variables, and the original memory unit no longer contains this information, then
  track the other memory unit or intermediate variables (that contain this key
  information) to ensure that there exists a memory unit that will hold this key
  information.
5. If the critical information never enters any memory unit, label the responsible
  construction operation as ExtractionError.
6. If the critical information enters a memory unit but a later operation removes or
  degrades it, label that later operation as UpdateError or DeletionError depending
  on whether it updates the unit or explicitly deletes it.
7. For retrieval candidates, verify that the memory store contains the required
  information before retrieval, but the retrieval pipeline fails to include it in
  the final retrieved context.
8. For response candidates, verify that the final retrieved context contains all
  necessary evidence and the answer model still answers incorrectly.

Return exactly one error attribution using the required structured schema:
- error_type: one of ExtractionError, UpdateError, DeletionError, RetrievalError,
  ResponseError.
- op_id: the operation identifier of the earliest decisive faulty operation.
- reason: a concise explanation grounded in graph evidence. The reason must state
  why the chosen operation is locally faulty, why correcting it would rescue the
  answer, and why earlier candidate operations are not the decisive fault when
  relevant.

Error type definitions:
- ExtractionError: The critical information is never captured into any memory unit
  during memory construction, and thus never enters the memory store.
- UpdateError: A memory unit initially contains the critical information, but a
  subsequent update removes or degrades it.
- DeletionError: A memory unit containing critical information is explicitly
  removed.
- RetrievalError: The memory store contains the required information, but retrieval
  fails to include it in the final retrieved context.
- ResponseError: The retrieved context contains all necessary evidence, but the
  final LLM still answers incorrectly."""

EVEROS_PRIOR = """System-specific prior (EverOS file tracking):
- Messages enter buffers; boundaries form memcells, and extraction writes memory
  entries. Updates and deletions change versioned entries on disk.
- Retrieval can include sparse/dense candidate pools, fusion, filtering and final
  context assembly. Follow the actually recorded stages and final context.
- identity_only candidates are references, not proof of memory contents. A
  retrieval_hit is request-local; memory_identity and source_revision reference
  the persistent entry. Unresolved revisions do not establish stored contents.
- context records the search response. answer_context records the exact text
  given to the answer model. Inspect answer_context for ResponseError; null or
  observed=false means the actual answer input is unknown.
- event_seq orders observations within a recording process. File-diff operations
  are post-hoc observations; observed_at is not the real pipeline execution time.
  Duplicate sequences in merged graphs require timestamp fallback; do not infer
  a reliable global order across independent processes from sequence alone.
- source=construction_event records actual boundary/extractor values, including
  empty results and attempts. source=file_diff / observation_only=true describe
  disk observations, not proof of a specific executed extraction/update/delete.
  If only these observations support the suspected fault, stop with
  insufficient_evidence rather than selecting an observation as the fault.
- Approximate memcell membership and truncated candidate pools limit what can be
  concluded. Missing edges or unrecorded values are not proof of a faulty operation.
- Verify source evidence, memory preservation, final retrieval context, then
  response generation, using the earliest locally faulty decisive operation rule.
"""

TOOL_INSTRUCTIONS = """Use the graph tools to inspect evidence before deciding.
The graph may already be initialized with the query and retrieved source-message
candidates. These are pseudo evidence, not human annotations; verify their content.
Use operation previews (include_variable_value=false) and view_node_value with a
regex for large values. Follow next offset when a page is truncated. All offsets
are one-based character positions, not line numbers. Keep the active node until
finish_node_exploration; queued nodes are ordered by creation time and identifier.
Finish graph exploration, then call submit_attribution with one inspected op_id.
The report reason must cite an exact relevant full_node_id whose value you read.
Do not treat the judge or reference-answer injection as a memory-system fault.
If critical source information is absent from the recorded graph, do not assume
an extraction or retrieval error. Finish exploration with insufficient_evidence
set to true and explain the missing evidence; do not submit an attribution.
"""
