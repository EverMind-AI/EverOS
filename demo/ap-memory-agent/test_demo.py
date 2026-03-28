"""
Test script for the AP Memory Agent demo.

Run from EverMemOS root:
    uv run python demo/ap-memory-agent/test_demo.py

Or from demo/ap-memory-agent/:
    uv run python test_demo.py

Prerequisites:
    - EverMemOS running at http://localhost:1995
    - seed_memory.py already run (vendor history populated)
    - ANTHROPIC_API_KEY in .env
"""

import sys
from pathlib import Path

# Ensure demo/ap-memory-agent is on path when run from project root
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from ap_agent_graph import build_ap_graph, APState


# Example invoices to test (matches the demo script from README)
TEST_INVOICES = [
    {
        "name": "Acme Corp (clean) — expect APPROVE",
        "invoice": {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-2025-0042",
            "amount": 500.0,
            "payment_terms": "net-30",
            "due_date": "2025-03-30",
            "line_items": "Office supplies",
        },
    },
    {
        "name": "Globex Supplies (prior dispute) — expect HOLD",
        "invoice": {
            "vendor_name": "Globex Supplies",
            "invoice_number": "INV-2025-0105",
            "amount": 1200.0,
            "payment_terms": "net-30",
            "due_date": "2025-04-15",
            "line_items": "Industrial equipment",
        },
    },
    {
        "name": "Shadow LLC (duplicate INV-777) — expect REJECT",
        "invoice": {
            "vendor_name": "Shadow LLC",
            "invoice_number": "INV-777",
            "amount": 3000.0,
            "payment_terms": "net-30",
            "due_date": "2025-04-01",
            "line_items": "Consulting services",
        },
    },
    {
        "name": "Acme Corp (new invoice) — expect APPROVE",
        "invoice": {
            "vendor_name": "Acme Corp",
            "invoice_number": "INV-2025-0099",
            "amount": 550.0,
            "payment_terms": "net-30",
            "due_date": "2025-05-01",
            "line_items": "Software licenses",
        },
    },
]


def run_test(invoice: dict, name: str) -> dict:
    """Run the graph with a single invoice and return the result."""
    graph = build_ap_graph()
    initial_state: APState = {
        "invoice": invoice,
        "memory_context": "",
        "risk_flags": [],
        "risk_score": 0,
        "decision": "",
        "reasoning": "",
        "invoice_id": "",
        "processed_at": "",
    }
    return graph.invoke(initial_state)


def main() -> None:
    print("=" * 60)
    print("AP Memory Agent — Demo Test Script")
    print("=" * 60)
    print("\nEnsure EverMemOS is running and seed_memory.py has been run.\n")

    for i, test in enumerate(TEST_INVOICES, 1):
        print(f"\n--- Test {i}: {test['name']} ---")
        result = run_test(test["invoice"], test["name"])
        print(f"  Decision: {result['decision'].upper()}")
        print(f"  Risk Score: {result['risk_score']}/100")
        if result["risk_flags"]:
            print(f"  Flags: {result['risk_flags']}")
        print(f"  Reasoning: {result['reasoning'][:80]}...")
        print()

    print("=" * 60)
    print("Tests complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
