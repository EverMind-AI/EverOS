"""Bounded, serial tool-calling attribution with a complete audit transcript."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, Protocol

from .cases import decode_snapshot
from .models import AttributionConfig, AttributionResult
from .notebook import GraphTraceNotebook, graph_tools
from .prompts import ATTRIBUTION_INSTRUCTIONS, EVEROS_PRIOR, TOOL_INSTRUCTIONS
from .retrieval import retrieve_starting_nodes

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from smartcomment.runtime import ExecNetwork

    from .models import FailedQueryCase


class ChatProvider(Protocol):
    """Small injectable model boundary; tests use scripted responses."""

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return a message and optional integer token usage."""
        ...


class OpenAIChatProvider:
    """Use an already-configured OpenAI-compatible async client."""

    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self.client = client
        self.model = model

    async def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools,  # type: ignore[arg-type]
            parallel_tool_calls=False,
            temperature=0,
        )
        message = response.choices[0].message
        payload: dict[str, Any] = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            payload["tool_calls"] = [call.model_dump() for call in message.tool_calls]
        return {
            "message": payload,
            "usage": response.usage.model_dump() if response.usage else {},
        }


class ContextBudgetError(ValueError):
    """The protected task, tool schema and newest turn exceed the budget."""


def trim_history(
    history: list[dict[str, Any]], hint: str, tools: list[dict[str, Any]], budget: int
) -> tuple[list[dict[str, Any]], int]:
    """Drop whole oldest turns; never split one assistant's tool-call batch."""
    protected = history[:2]
    groups: list[list[dict[str, Any]]] = []
    for message in history[2:]:
        if message["role"] != "tool":
            groups.append([message])
        elif not groups:
            raise ValueError("Orphaned tool response in history.")
        else:
            groups[-1].append(message)
    removed = 0
    while True:
        messages = protected + [message for group in groups for message in group]
        request = [messages[0], {"role": "system", "content": hint}, *messages[1:]]
        size = len(
            json.dumps({"messages": request, "tools": tools}, ensure_ascii=False)
        )
        if size <= budget:
            return messages, removed
        if len(groups) <= 1:
            raise ContextBudgetError(
                "Protected messages and latest turn exceed context_chars. "
                "Increase the budget or use smaller tool pages."
            )
        groups.pop(0)
        removed += 1


def _validate_arguments(schema: dict[str, Any], arguments: Any) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object.")
    properties = schema["properties"]
    if arguments.keys() - properties.keys():
        raise ValueError("Unknown tool argument.")
    if set(schema.get("required", [])) - arguments.keys():
        raise ValueError("Required tool argument is missing.")
    types = {"string": str, "integer": int, "boolean": bool, "array": list}
    for name, value in arguments.items():
        spec = properties[name]
        if type(value) is not types[spec["type"]]:
            raise ValueError(f"Invalid type for {name}.")
        if "minimum" in spec and value < spec["minimum"]:
            raise ValueError(f"{name} is below its minimum.")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"Invalid value for {name}.")
        if spec["type"] == "array" and any(not isinstance(v, str) for v in value):
            raise ValueError(f"{name} must contain strings.")
    return arguments


async def attribute(
    graph: ExecNetwork,
    case: FailedQueryCase,
    provider: ChatProvider,
    config: AttributionConfig | None = None,
) -> dict[str, Any]:
    """Run one case and return a report; service/limit failures have no attribution."""
    config = config or AttributionConfig()
    notebook = GraphTraceNotebook(
        graph, config.max_trace_nodes, case.query_full_node_id
    )
    notebook.page_limit = min(16_000, max(128, config.context_chars // 4))
    starting = await asyncio.to_thread(
        retrieve_starting_nodes,
        graph,
        case,
        min(config.num_starting_points, max(1, config.max_trace_nodes - 1)),
    )
    initial = [case.query_full_node_id, *starting][: config.max_trace_nodes]
    initialized = notebook.initialize_execution_graph(initial if starting else None)
    tools = [
        *graph_tools(),
        {
            "type": "function",
            "function": {
                "name": "submit_attribution",
                "description": "Submit a fault after graph exploration.",
                "parameters": AttributionResult.model_json_schema(),
            },
        },
    ]
    schemas = {
        tool["function"]["name"]: tool["function"]["parameters"] for tool in tools
    }
    history: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": ATTRIBUTION_INSTRUCTIONS
            + "\n"
            + EVEROS_PRIOR
            + "\n"
            + TOOL_INSTRUCTIONS,
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": case.query,
                    "golden_answers": case.golden_answers,
                    "wrong_answer": case.prediction,
                    "query_node": case.query_full_node_id,
                    "pseudo_evidence_nodes": starting,
                    "initialization": initialized,
                }
            ),
        },
    ]
    transcript: list[dict[str, Any]] = [
        {
            "event": "initialization",
            "initial_nodes": initial if starting else [],
            "content": initialized,
        },
        *[{"event": "message", "message": message} for message in history],
    ]
    report: dict[str, Any] = {
        "status": "failed",
        "graph_id": graph.graph_id,
        "case": case.model_dump(),
        "config": config.model_dump(),
        "pseudo_evidence_nodes": starting,
        "attribution": None,
        "failure": None,
        "iterations": 0,
        "transcript": transcript,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    operation_ids = {op.op_id for op in graph.search_operations()}
    try:
        for iteration in range(config.max_iters):
            hint = notebook.hint()
            history, removed = trim_history(history, hint, tools, config.context_chars)
            if removed:
                transcript.append({"event": "context_trim", "removed_turns": removed})
            transcript.append(
                {"event": "hint", "iteration": iteration + 1, "content": hint}
            )
            request = [history[0], {"role": "system", "content": hint}, *history[1:]]
            report["iterations"] = iteration + 1
            response = await provider.complete(request, tools)
            for key in report["usage"]:
                report["usage"][key] += int(response.get("usage", {}).get(key, 0) or 0)
            message = response["message"]
            history.append(message)
            transcript.append({"event": "message", "message": message})
            calls = message.get("tool_calls") or []
            if not calls:
                correction = {
                    "role": "user",
                    "content": (
                        "Inspect evidence, finish graph exploration, then call "
                        "submit_attribution. Plain text is not a result."
                    ),
                }
                history.append(correction)
                transcript.append({"event": "message", "message": correction})
                continue
            attribution = None
            notebook.page_limit = min(
                16_000, max(128, config.context_chars // (4 * len(calls)))
            )
            for call in calls:
                name = call["function"]["name"]
                try:
                    if name not in schemas:
                        raise ValueError("Unknown tool.")
                    arguments = _validate_arguments(
                        schemas[name], json.loads(call["function"]["arguments"])
                    )
                    if name == "submit_attribution":
                        if len(calls) != 1:
                            raise ValueError(
                                "Submit attribution in a separate tool call."
                            )
                        result = AttributionResult.model_validate(arguments)
                        if result.op_id not in operation_ids:
                            raise ValueError("The submitted operation does not exist.")
                        if notebook.state is not None or notebook.final_state is None:
                            raise ValueError(
                                "Finish graph exploration before submitting."
                            )
                        if notebook.insufficient_evidence is not None:
                            raise ValueError(
                                "Exploration ended with insufficient evidence."
                            )
                        if result.op_id not in notebook.inspected_operations:
                            raise ValueError("Inspect the submitted operation first.")
                        op = graph.get_operation(result.op_id)
                        if op.op_name == "llm-judge" or op.op_name.startswith(
                            ("judge_answer_", "inject_golden_answers_")
                        ):
                            raise ValueError(
                                "Judging and reference injection are "
                                "not attributable faults."
                            )
                        if not notebook.operation_allowed(op):
                            raise ValueError(
                                "The submitted operation belongs to "
                                "another request or a future event."
                            )
                        if (
                            op.metadata.get("observation_only")
                            or op.metadata.get("source") == "file_diff"
                        ):
                            raise ValueError(
                                "File observations do not establish the faulty "
                                "operation; stop with insufficient_evidence."
                            )
                        evidence = {
                            node.full_node_id
                            for node in graph.filter_by_operation(op.op_id).nodes
                        }
                        if result.error_type == "ResponseError":
                            for node in graph.filter_by_operation(op.op_id).nodes:
                                snapshot = decode_snapshot(node.value)
                                if (
                                    node.class_name == "answer_context"
                                    and isinstance(snapshot, dict)
                                    and snapshot.get("observed") is False
                                ):
                                    raise ValueError(
                                        "The actual answer context is unrecorded; "
                                        "stop with insufficient_evidence."
                                    )
                        cited = {
                            node_id for node_id in evidence if node_id in result.reason
                        }
                        if not cited & notebook.read_nodes:
                            raise ValueError(
                                "Cite a relevant node whose value was read "
                                "in the attribution reason."
                            )
                        attribution = result.model_dump()
                        content = "Attribution accepted."
                    else:
                        content = await asyncio.to_thread(
                            getattr(notebook, name), **arguments
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    content = f"Tool error: {exc}"
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": content,
                }
                history.append(tool_message)
                transcript.append(
                    {"event": "message", "tool_name": name, "message": tool_message}
                )
            if attribution is not None:
                report.update(status="success", attribution=attribution)
                break
            if notebook.insufficient_evidence is not None:
                report["failure"] = {
                    "type": "InsufficientEvidence",
                    "message": notebook.insufficient_evidence,
                }
                break
        else:
            report["failure"] = {
                "type": "IterationLimit",
                "message": "No validated attribution before max_iters.",
            }
    except ContextBudgetError as exc:
        report["failure"] = {"type": "ContextBudgetError", "message": str(exc)}
    except Exception as exc:
        # Do not serialize provider exceptions: request URLs/headers may hold secrets.
        report["failure"] = {
            "type": type(exc).__name__,
            "status_code": getattr(exc, "status_code", None),
        }
    report["final_state"] = notebook.snapshot()
    report["inspected_operations"] = sorted(notebook.inspected_operations)
    return report
