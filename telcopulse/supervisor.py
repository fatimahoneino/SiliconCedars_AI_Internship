import re
import sys
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from common.config import load_config
from telcopulse import agents
from telcopulse.database import get_backend, init_db
from telcopulse.state import TelcoPulseState


MAX_RETRIES = 2
ROUTES = {
    "sql": "questions requiring fictional subscriber, subscription, service, renewal, engagement, complaint, country, or churn-risk database records",
    "research": "requests for current public information or reports about telecom, mobile VAS, retention, mobile gaming, or multilingual engagement",
    "chart": "requests to create a bar, line, or pie chart when labels and numeric values are supplied in the message",
    "rag": "questions about internal fictional retention policy, VAS catalog, SMS compliance, customer support rules, or approved offers",
    "conversation": "greetings, capability questions, follow-up conversation, or drafting English or Arabic customer messages",
}


def has_word(text, word):
    # \b = word boundary. Long words only need a boundary at the start so that
    # "subscriber" also matches "subscribers". Short words like "hi" need a
    # boundary at both ends, otherwise they match inside "sushi" or "highest".
    pattern = rf"\b{re.escape(word)}"
    if len(word) <= 3:
        pattern += r"\b"
    return re.search(pattern, text) is not None


CHART_WORDS = ("chart", "graph", "plot", "visualize", "visualise")


def wants_database_chart(lowered):
    """A chart request with no numbers in it, about database data.

    Example: "Chart the number of subscribers per country".
    The supervisor sends it to sql first, then passes the rows to chart.
    """
    asks_chart = any(has_word(lowered, word) for word in CHART_WORDS)
    has_numbers = re.search(r"\d", lowered) is not None
    about_data = any(has_word(lowered, word) for word in ("subscriber", "subscription", "service", "churn", "country"))
    return asks_chart and about_data and not has_numbers


def route_node(state: TelcoPulseState):
    question = state["messages"][-1].content
    lowered = question.lower()

    # Multi-step request: the data must come from the database before charting.
    if wants_database_chart(lowered):
        return {
            "route": "sql",
            "then_chart": True,
            "agent_trail": ["supervisor"],
            "chart": None,
            "data_rows": [],
        }

    # First try cheap keyword rules. The order matters: the first match wins.
    # Only if nothing matches do we spend a model call on routing.
    keyword_routes = [
        ("chart", CHART_WORDS),
        ("research", ("research", "latest", "web search", "market trend")),
        ("rag", ("policy", "allowed", "permitted", "compliance", "rule", "approved offer",
                 "approval", "price", "catalog", "playbook")),
        ("sql", ("subscriber", "subscription", "churn risk", "failed renewal", "engagement score")),
        ("conversation", ("draft", "remember", "recall", "hello", "hi", "thank")),
    ]
    choice = next(
        (route for route, words in keyword_routes if any(has_word(lowered, word) for word in words)),
        "",
    )
    if not choice:
        options = "\n".join(f"{name}: {description}" for name, description in ROUTES.items())
        choice = agents.get_llm().invoke([
            SystemMessage(
                "Choose the single TelcoPulse specialist that should handle the request. "
                "Return exactly one route name and nothing else. Personal statements, "
                "remember/recall requests, pronoun-based follow-ups, and message drafting "
                "always use conversation. Use sql only when the user explicitly asks about "
                "fictional subscriber or service records.\n\n"
                f"{options}"
            ),
            HumanMessage(question),
        ]).content.strip().lower()
    choice = "".join(character for character in choice if character.isalpha())
    if choice not in ROUTES:
        choice = "conversation"
    # Reset per-request fields, because memory keeps the state between turns.
    return {
        "route": choice,
        "then_chart": False,
        "agent_trail": ["supervisor"],
        "chart": None,
        "data_rows": [],
    }


def run_with_retries(function, *args):
    # try once, then retry up to MAX_RETRIES times before giving up
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return function(*args), attempt
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error))


def make_agent_node(name, function, needs_history=False, uses_rows=False):
    def node(state: TelcoPulseState):
        question = state["messages"][-1].content
        if needs_history:
            arguments = (question, state["messages"])
        elif uses_rows:
            arguments = (question, state.get("data_rows") or None)
        else:
            arguments = (question,)

        # When this agent is the second step, keep the work of the first step.
        earlier_retries = state.get("retries", 0) if uses_rows and state.get("data_rows") else 0
        earlier_sources = state.get("sources", []) if uses_rows and state.get("data_rows") else []

        try:
            result, retries = run_with_retries(function, *arguments)
        except RuntimeError as exc:
            message = f"The {name} agent failed after {MAX_RETRIES} retries: {exc}"
            return {
                "answer": message,
                "chart": None,
                "sources": earlier_sources,
                "failed": True,
                "retries": earlier_retries + MAX_RETRIES,
                "then_chart": False,
                "agent_trail": state.get("agent_trail", []) + [name],
                "messages": [AIMessage(message)],
            }
        answer = result.get("answer", "")
        trail = result.get("trail", [name])
        update = {
            "answer": answer,
            "chart": result.get("chart"),
            "sources": earlier_sources + result.get("sources", []),
            "retries": earlier_retries + retries,
            "failed": False,
            "agent_trail": state.get("agent_trail", []) + trail,
        }
        if "rows" in result:
            update["data_rows"] = result["rows"]

        going_to_chart = name == "sql" and state.get("then_chart") and result.get("rows")
        if not going_to_chart:
            # Only the final step adds its answer to the conversation memory.
            update["messages"] = [AIMessage(answer)]
            update["then_chart"] = False
        return update
    return node


def pick_route(state: TelcoPulseState):
    return state["route"]


def after_sql(state: TelcoPulseState):
    """After the SQL agent: finish, or pass the rows on to the chart agent."""
    if state.get("then_chart") and state.get("data_rows") and not state.get("failed"):
        return "chart"
    return "end"


def build_app():
    builder = StateGraph(TelcoPulseState)
    builder.add_node("supervisor", route_node)
    builder.add_node("sql", make_agent_node("sql", agents.sql_agent))
    builder.add_node("research", make_agent_node("research", agents.research_team))
    builder.add_node("chart", make_agent_node("chart", agents.chart_agent, uses_rows=True))
    builder.add_node("rag", make_agent_node("rag", agents.rag_agent))
    builder.add_node(
        "conversation",
        make_agent_node("conversation", agents.conversation_agent, needs_history=True),
    )
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", pick_route, list(ROUTES))
    # After sql the supervisor may send the rows on to chart (multi-step request).
    builder.add_conditional_edges("sql", after_sql, {"chart": "chart", "end": END})
    for route in ROUTES:
        if route != "sql":
            builder.add_edge(route, END)
    return builder.compile(checkpointer=MemorySaver())


@traceable(name="TelcoPulse request", run_type="chain")
def ask(app, question, thread_id="default"):
    config = {"configurable": {"thread_id": thread_id}}
    started = time.perf_counter()
    result = app.invoke({"messages": [HumanMessage(question)]}, config)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)

    # Record the request metrics on the LangSmith trace, so they can be
    # filtered and charted in the LangSmith dashboard.
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata({
            "route": result.get("route"),
            "agent_trail": " -> ".join(result.get("agent_trail", [])),
            "response_time_ms": result["elapsed_ms"],
            "fallback_count": result.get("retries", 0),
            "completed": not result.get("failed", False),
        })
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_config()
    init_db()
    print(f"Database: {get_backend()} | Web search: {agents.search_provider()}")
    app = build_app()
    questions = [
        "Which three fictional subscribers have the highest churn risk scores?",
        "What retention offer is permitted after two failed renewals?",
        "Create a bar chart of at-risk subscribers: Lebanon 2, Jordan 2, Egypt 1, UAE 1",
        "Research current mobile VAS retention practices in MENA.",
        "Draft a friendly Arabic message offering a three-day GameBox pass.",
    ]
    for index, question in enumerate(questions):
        result = ask(app, question, thread_id=f"demo-{index}")
        print(f"\n{'=' * 72}")
        print(f"Q: {question}")
        print(f"route: {result['route']}")
        print(f"trail: {' -> '.join(result.get('agent_trail', []))}")
        print(f"A: {result['answer']}")
        if result.get("sources"):
            print(f"sources: {', '.join(result['sources'][:3])}")
        if result.get("chart"):
            print(f"chart: {result['chart']['type']}")
        print(
            f"retries: {result.get('retries', 0)} | "
            f"completed: {not result.get('failed', False)} | "
            f"{result['elapsed_ms']} ms"
        )


if __name__ == "__main__":
    main()
