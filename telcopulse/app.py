import json
import uuid

import streamlit as st
import streamlit.components.v1 as components

from common.config import load_config
from telcopulse.database import init_db
from telcopulse.supervisor import ask, build_app


AGENT_LABELS = {
    "sql": "Subscriber Analytics",
    "research": "Telecom Research Team",
    "chart": "Campaign Visualization",
    "rag": "Internal Policy Knowledge",
    "conversation": "Multilingual Conversation",
}

CHART_HTML = """
<div style="background:#fff;padding:12px;border-radius:8px;">
  <canvas id="telco-chart" height="260"></canvas>
</div>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
  new Chart(document.getElementById('telco-chart').getContext('2d'), SPEC);
</script>
"""


def render_chart(spec):
    safe_spec = json.dumps(spec).replace("<", "\\u003c")
    components.html(CHART_HTML.replace("SPEC", safe_spec), height=320)


@st.cache_resource
def get_app():
    load_config()
    init_db()
    return build_app()


def main():
    st.set_page_config(page_title="TelcoPulse AI", page_icon="📡", layout="wide")
    st.title("TelcoPulse AI")
    st.caption("Multilingual Telecom Retention & Operations Copilot")
    st.info("Portfolio prototype using fictional subscriber records. All customer-facing actions require human approval.")
    app = get_app()
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []
    if "stats" not in st.session_state:
        st.session_state.stats = {"total": 0, "completed": 0, "retries": 0, "duration": 0.0}

    with st.sidebar:
        st.subheader("Specialist team")
        for route, label in AGENT_LABELS.items():
            st.write(f"**{label}** — `{route}`")
        st.divider()
        stats = st.session_state.stats
        st.subheader("Session metrics")
        st.metric("Requests", stats["total"])
        completion_rate = stats["completed"] / stats["total"] * 100 if stats["total"] else 0
        st.metric("Completion rate", f"{completion_rate:.0f}%")
        st.metric("Retries", stats["retries"])
        average = stats["duration"] / stats["total"] if stats["total"] else 0
        st.metric("Average response", f"{average:.0f} ms")
        st.divider()
        st.caption("Example requests")
        st.code(
            "Which subscribers have the highest churn risk?\n"
            "What offer is allowed after two failed renewals?\n"
            "Chart risk: Lebanon 2, Jordan 2, Egypt 1\n"
            "Research mobile VAS retention in MENA\n"
            "اكتب مسودة عرض مجانية لمدة ثلاثة أيام",
            language=None,
        )

    for entry in st.session_state.history:
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            label = AGENT_LABELS.get(entry["route"], entry["route"])
            st.caption(f"{label} · {entry['elapsed_ms']} ms")
            st.write(entry["answer"])
            if entry.get("chart"):
                render_chart(entry["chart"])
            if entry.get("sources"):
                with st.expander("Sources"):
                    for source in entry["sources"]:
                        st.write(f"- {source}")
            with st.expander("Execution details"):
                st.write(" → ".join(entry["agent_trail"]))
                st.write(f"Retries: {entry['retries']}")
                st.write(f"Completed: {entry['completed']}")

    question = st.chat_input("Ask TelcoPulse about fictional operations")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Routing to the right specialist..."):
            result = ask(app, question, thread_id=st.session_state.thread_id)
        route = result["route"]
        completed = not result.get("failed", False)
        st.caption(f"{AGENT_LABELS.get(route, route)} · {result['elapsed_ms']} ms")
        st.write(result["answer"])
        if result.get("chart"):
            render_chart(result["chart"])
        if result.get("sources"):
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.write(f"- {source}")
        with st.expander("Execution details"):
            st.write(" → ".join(result.get("agent_trail", [])))
            st.write(f"Retries: {result.get('retries', 0)}")
            st.write(f"Completed: {completed}")

    stats = st.session_state.stats
    stats["total"] += 1
    stats["completed"] += int(completed)
    stats["retries"] += result.get("retries", 0)
    stats["duration"] += result["elapsed_ms"]
    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
        "route": route,
        "chart": result.get("chart"),
        "sources": result.get("sources", []),
        "agent_trail": result.get("agent_trail", []),
        "retries": result.get("retries", 0),
        "completed": completed,
        "elapsed_ms": result["elapsed_ms"],
    })
    st.rerun()


if __name__ == "__main__":
    main()
