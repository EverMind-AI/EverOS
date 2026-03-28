"""
AP Automation Agent - LangGraph + EverMemOS + Claude
=====================================================
Multi-agent accounts payable pipeline with shared episodic memory.

Architecture:
    UI (Streamlit) → LangGraph Orchestrator → [Invoice Agent → Risk Agent → Approval Agent → Memory Updater]
                                                      ↕              ↕             ↕                ↕
                                                         EverMemOS (Shared Memory Bus)
"""

import os
import json
import re
import uuid
from datetime import datetime
from typing import Any, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from evermem_client import EverMemClient

load_dotenv()


def _extract_json(text: str) -> dict[str, Any] | None:
    """
    Extract JSON from Claude response. Handles markdown code blocks and extra text.
    """
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    # Strip markdown code blocks (```json ... ``` or ``` ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        text = match.group(1).strip()
    # Find first { and last } to extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    # Fallback: try parsing the whole string
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ─────────────────────────────────────────────
# GRAPH STATE
# ─────────────────────────────────────────────

class APState(TypedDict):
    invoice: dict
    memory_context: str
    risk_flags: list[str]
    risk_score: int
    decision: str
    reasoning: str
    invoice_id: str
    processed_at: str


# ─────────────────────────────────────────────
# INITIALISE SHARED SERVICES
# ─────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-opus-4-6",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0,
)

evermem = EverMemClient()


# ─────────────────────────────────────────────
# NODE 1: INVOICE AGENT
# ─────────────────────────────────────────────

def invoice_agent(state: APState) -> APState:
    invoice = state["invoice"]
    invoice_id = str(uuid.uuid4())[:8]
    vendor = invoice.get("vendor_name", "Unknown Vendor")

    print(f"\n[Invoice Agent] Processing invoice from {vendor}...")

    memory_content = (
        f"Invoice received from vendor '{vendor}'. "
        f"Invoice number: {invoice.get('invoice_number')}. "
        f"Amount: ${invoice.get('amount')}. "
        f"Payment terms: {invoice.get('payment_terms', 'net-30')}. "
        f"Due date: {invoice.get('due_date', 'not specified')}. "
        f"Line items: {invoice.get('line_items', 'not provided')}."
    )
    evermem.write_memory(
        content=memory_content,
        role="assistant",
        memory_type="episodic_memory",
    )

    vendor_memories = evermem.search_memories(
        query=f"{vendor} invoice payment dispute late rejected duplicate",
        retrieve_method="keyword",
        memory_types=["episodic_memory", "event_log"],
        top_k=20,
    )

    # Filter search results to this vendor (keyword search returns all groups)
    vendor_lower = vendor.lower()
    vendor_memories = [
        m for m in vendor_memories
        if vendor_lower in str(m.get("content", m.get("episode", m.get("summary", m.get("subject", m.get("title", "")))))).lower()
    ]

    vendor_profile = evermem.fetch_memories_by_type(
        memory_types=["profile"],
        top_k=3,
    )

    all_memories = vendor_profile + vendor_memories
    memory_context = evermem.format_memory_context(all_memories)
    print(
        f"[Invoice Agent] Retrieved {len(vendor_profile)} profile + "
        f"{len(vendor_memories)} episodic records for {vendor}"
    )

    return {
        **state,
        "invoice_id": invoice_id,
        "memory_context": memory_context,
        "processed_at": datetime.now().isoformat(),
    }


# ─────────────────────────────────────────────
# NODE 2: RISK AGENT
# ─────────────────────────────────────────────

def risk_agent(state: APState) -> APState:
    invoice = state["invoice"]
    memory_context = state["memory_context"]

    print(f"\n[Risk Agent] Analysing risk for invoice {invoice.get('invoice_number')}...")

    system_prompt = """You are an AP Risk Agent for a finance team.
Your job is to analyse incoming invoices against vendor payment history and flag anomalies.
Be precise, concise, and always ground your findings in the memory context provided.

CRITICAL: Return ONLY a raw JSON object. No markdown, no code blocks, no preamble, no explanation.
Start your response with { and end with }."""

    user_prompt = f"""
CURRENT INVOICE:
{json.dumps(invoice, indent=2)}

VENDOR MEMORY (past interactions retrieved from shared memory system):
{memory_context}

Analyse this invoice and return a JSON object with exactly this structure:
{{
  "risk_flags": ["list of specific risk flags, or empty list if clean"],
  "risk_score": <integer 0-100, where 0=no risk and 100=definite fraud>,
  "summary": "one sentence summary of your finding"
}}

Check for:
- Duplicate invoice numbers vs history
- Amount significantly higher than vendor average
- Unusual or changed payment terms
- Prior disputes or late payments with this vendor
- Any other anomalies in the memory context
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)
    result = _extract_json(content)
    if result is not None:
        risk_flags = result.get("risk_flags", [])
        risk_score = int(result.get("risk_score", 0))
        print(f"[Risk Agent] Score: {risk_score}/100 | Flags: {risk_flags or 'None'}")
    else:
        risk_flags = ["Unable to parse risk analysis — manual review required"]
        risk_score = 50
        print("[Risk Agent] Warning: Could not parse Claude response as JSON")

    return {
        **state,
        "risk_flags": risk_flags,
        "risk_score": risk_score,
    }


# ─────────────────────────────────────────────
# NODE 3: APPROVAL AGENT
# ─────────────────────────────────────────────

def approval_agent(state: APState) -> APState:
    invoice = state["invoice"]
    risk_flags = state["risk_flags"]
    risk_score = state["risk_score"]
    memory_context = state["memory_context"]

    print(f"\n[Approval Agent] Making decision (risk score: {risk_score})...")

    system_prompt = """You are a senior AP approver. You make final payment decisions
based on invoice data, risk analysis, and vendor history.
Be decisive and explain your reasoning in plain English for the finance team.

CRITICAL: Return ONLY a raw JSON object. No markdown, no code blocks, no preamble.
Start your response with { and end with }."""

    user_prompt = f"""
INVOICE: {json.dumps(invoice, indent=2)}

RISK ANALYSIS:
- Risk Score: {risk_score}/100
- Flags Raised: {risk_flags if risk_flags else "None"}

VENDOR HISTORY CONTEXT:
{memory_context}

Decision rules:
- risk_score 0-30  → "approve" (unless a critical flag overrides)
- risk_score 31-60 → "hold" (needs human review)
- risk_score 61+   → "reject" (do not process)
- Any duplicate invoice number → always "reject"
- Any unresolved prior dispute → always "hold"

Return JSON:
{{
  "decision": "approve" | "hold" | "reject",
  "reasoning": "2-3 sentence plain English explanation citing specific memory evidence"
}}
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    content = response.content if isinstance(response.content, str) else str(response.content)
    result = _extract_json(content)
    if result is not None:
        decision = result.get("decision", "hold")
        reasoning = result.get("reasoning", "No reasoning provided.")
        if decision not in ("approve", "hold", "reject"):
            decision = "hold"
    else:
        decision = "hold"
        reasoning = "Could not parse approval decision — defaulting to hold for manual review."

    print(f"[Approval Agent] Decision: {decision.upper()}")
    print(f"[Approval Agent] Reasoning: {reasoning}")

    return {
        **state,
        "decision": decision,
        "reasoning": reasoning,
    }


# ─────────────────────────────────────────────
# NODE 4: MEMORY UPDATER
# ─────────────────────────────────────────────

def memory_updater(state: APState) -> APState:
    invoice = state["invoice"]
    vendor = invoice.get("vendor_name", "Unknown Vendor")
    decision = state["decision"]
    reasoning = state["reasoning"]
    risk_flags = state["risk_flags"]

    print(f"\n[Memory Updater] Writing outcome to EverMemOS...")

    outcome_memory = (
        f"Invoice {invoice.get('invoice_number')} from vendor '{vendor}' "
        f"for ${invoice.get('amount')} was {decision.upper()}. "
        f"Risk score: {state['risk_score']}/100. "
        f"Flags: {', '.join(risk_flags) if risk_flags else 'none'}. "
        f"Reasoning: {reasoning}"
    )
    evermem.write_memory(
        content=outcome_memory,
        role="assistant",
        memory_type="episodic_memory",
    )

    if state["risk_score"] >= 40 or decision in ["hold", "reject"]:
        profile_update = (
            f"Vendor '{vendor}' has a flagged invoice history. "
            f"Most recent outcome: {decision.upper()} on invoice "
            f"{invoice.get('invoice_number')} (${invoice.get('amount')}). "
            f"Risk flags observed: {', '.join(risk_flags) if risk_flags else 'none'}."
        )
        evermem.write_vendor_profile(vendor=vendor, profile_content=profile_update)
        print(f"[Memory Updater] Vendor profile updated due to risk score {state['risk_score']}")

    print("[Memory Updater] Resolution written to shared memory ✓")
    return state


# ─────────────────────────────────────────────
# ROUTING & GRAPH
# ─────────────────────────────────────────────

def route_after_approval(state: APState) -> str:
    return "memory_updater"


def build_ap_graph() -> StateGraph:
    graph = StateGraph(APState)

    graph.add_node("invoice_agent", invoice_agent)
    graph.add_node("risk_agent", risk_agent)
    graph.add_node("approval_agent", approval_agent)
    graph.add_node("memory_updater", memory_updater)

    graph.set_entry_point("invoice_agent")
    graph.add_edge("invoice_agent", "risk_agent")
    graph.add_edge("risk_agent", "approval_agent")
    graph.add_conditional_edges("approval_agent", route_after_approval, {
        "memory_updater": "memory_updater",
    })
    graph.add_edge("memory_updater", END)

    return graph.compile()


# ─────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Running CLI test...")
    graph = build_ap_graph()
    test_invoice = {
        "vendor_name": "Acme Corp",
        "invoice_number": "INV-2025-0042",
        "amount": 12500.00,
        "payment_terms": "net-30",
        "due_date": "2025-03-30",
        "line_items": "Software licenses x5",
    }
    initial_state: APState = {
        "invoice": test_invoice,
        "memory_context": "",
        "risk_flags": [],
        "risk_score": 0,
        "decision": "",
        "reasoning": "",
        "invoice_id": "",
        "processed_at": "",
    }
    result = graph.invoke(initial_state)
    print(f"\n{'='*50}")
    print(f"FINAL DECISION: {result['decision'].upper()}")
    print(f"REASONING: {result['reasoning']}")
    print(f"RISK SCORE: {result['risk_score']}/100")
