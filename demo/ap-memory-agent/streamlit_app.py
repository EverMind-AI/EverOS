"""
AP Memory Agent - Streamlit UI

Run with: streamlit run streamlit_app.py
(From demo/ap-memory-agent/ directory)
"""

import streamlit as st

from ap_agent_graph import build_ap_graph, APState


def run_streamlit_ui() -> None:
    st.set_page_config(page_title="AP Memory Agent", page_icon="🧾", layout="wide")
    st.title("🧾 AP Automation Agent")
    st.caption(
        "Powered by LangGraph + EverMemOS + Claude — "
        "Multi-agent invoice processing with shared memory"
    )

    st.subheader("Submit Invoice")
    with st.form("invoice_form"):
        col1, col2 = st.columns(2)
        with col1:
            vendor_name = st.text_input("Vendor Name", placeholder="e.g. Acme Corp")
            invoice_number = st.text_input(
                "Invoice Number", placeholder="e.g. INV-2025-0042"
            )
            amount = st.number_input("Amount ($)", min_value=0.0, step=100.0)
        with col2:
            payment_terms = st.selectbox(
                "Payment Terms",
                ["net-30", "net-15", "net-60", "due-on-receipt"],
            )
            due_date = st.date_input("Due Date")
            line_items = st.text_area(
                "Line Items (optional)",
                placeholder="e.g. Software licenses x5, Support hours x10",
            )

        submitted = st.form_submit_button("🚀 Process Invoice", use_container_width=True)

    if submitted and vendor_name and invoice_number:
        invoice = {
            "vendor_name": vendor_name,
            "invoice_number": invoice_number,
            "amount": amount,
            "payment_terms": payment_terms,
            "due_date": str(due_date),
            "line_items": line_items,
        }

        with st.spinner("Agents processing invoice..."):
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
            final_state = graph.invoke(initial_state)

        st.divider()
        st.subheader("Agent Decision")

        decision = final_state["decision"]
        decision_colors = {"approve": "🟢", "hold": "🟡", "reject": "🔴"}
        emoji = decision_colors.get(decision, "⚪")
        st.markdown(f"## {emoji} {decision.upper()}")
        st.info(final_state["reasoning"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Score", f"{final_state['risk_score']}/100")
        with col2:
            flags = final_state["risk_flags"]
            st.metric("Risk Flags", len(flags))

        if flags:
            st.subheader("⚠️ Risk Flags")
            for flag in flags:
                st.warning(flag)

        with st.expander("🧠 Vendor Memory Used in Decision", expanded=True):
            st.caption(
                "This is what EverMemOS retrieved from shared agent memory "
                "before making the decision:"
            )
            st.text(final_state["memory_context"])

    elif submitted:
        st.error("Please fill in at least Vendor Name and Invoice Number.")


if __name__ == "__main__":
    run_streamlit_ui()
