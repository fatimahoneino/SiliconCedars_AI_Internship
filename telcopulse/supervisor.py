import sys
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from common.config import load_config
from telcopulse import agents
from telcopulse.database import init_db
from telcopulse.state import TelcoPulseState


MAX_RETRIES = 2
ROUTES = {
    "sql": "questions requiring fictional subscriber, subscription, service, renewal, engagement, complaint, country, or churn-risk database records",
    "research": "requests for current public information or reports about telecom, mobile VAS, retention, mobile gaming, or multilingual engagement",
    "chart": "requests to create a bar, line, or pie chart when labels and numeric values are supplied in the message",
    "rag": "questions about internal fictional retention policy, VAS catalog, SMS compliance, customer support rules, or approved offers",
    "conversation": "greetings, capability questions, follow-up conversation, or drafting English or Arabic customer messages",
}


def route_node(state: TelcoPulseState):
    question = state["messages"][-1].content
    lowered = question.lower()
    keyword_routes = [
        ("chart", ("chart", "graph", "plot", "visualize", "visualise")),
        ("research", ("research", "latest", "current", "web search", "market trend")),
        ("rag", ("policy", "allowed", "permitted", "compliance", "rule", "approved offer")),
        ("sql", ("subscriber", "subscription", "churn risk", "failed renewal", "engagement score")),
        ("conversation", ("draft", "remember", "recall", "hello", "hi ", "thank")),
    ]
    choice = next(
        (route for route, words in keyword_routes if any(word in lowered for word in words)),
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
    return {"route": choice, "agent_trail": ["supervisor"]}


def run_with_retries(function, *args):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            return function(*args), attempt
        except Exception as exc:
            last_error = exc
    raise RuntimeError(str(last_error))


def make_agent_node(name, function, needs_history=False):
    def node(state: TelcoPulseState):
        question = state["messages"][-1].content
        arguments = (question, state["messages"]) if needs_history else (question,)
        try:
            result, retries = run_with_retries(function, *arguments)
        except RuntimeError as exc:
            message = f"The {name} agent failed after {MAX_RETRIES} retries: {exc}"
            return {
                "answer": message,
                "chart": None,
                "sources": [],
                "failed": True,
                "retries": MAX_RETRIES,
                "agent_trail": state.get("agent_trail", []) + [name],
                "messages": [AIMessage(message)],
            }
        answer = result.get("answer", "")
        trail = result.get("trail", [name])
        return {
            "answer": answer,
            "chart": result.get("chart"),
            "sources": result.get("sources", []),
            "retries": retries,
            "failed": False,
            "agent_trail": state.get("agent_trail", []) + trail,
            "messages": [AIMessage(answer)],
        }
    return node


def pick_route(state: TelcoPulseState):
    return state["route"]


def build_app():
    builder = StateGraph(TelcoPulseState)
    builder.add_node("supervisor", route_node)
    builder.add_node("sql", make_agent_node("sql", agents.sql_agent))
    builder.add_node("research", make_agent_node("research", agents.research_team))
    builder.add_node("chart", make_agent_node("chart", agents.chart_agent))
    builder.add_node("rag", make_agent_node("rag", agents.rag_agent))
    builder.add_node(
        "conversation",
        make_agent_node("conversation", agents.conversation_agent, needs_history=True),
    )
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", pick_route, list(ROUTES))
    for route in ROUTES:
        builder.add_edge(route, END)
    return builder.compile(checkpointer=MemorySaver())


def ask(app, question, thread_id="default"):
    config = {"configurable": {"thread_id": thread_id}}
    started = time.perf_counter()
    result = app.invoke({"messages": [HumanMessage(question)]}, config)
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 1)
    return result


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_config()
    init_db()
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
