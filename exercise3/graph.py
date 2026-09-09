import ast
import operator
import re
import sys
from pathlib import Path
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from common.config import load_config

MODEL = "openai/gpt-oss-20b"
GRAPH_IMAGE = Path(__file__).parent / "graph.png"

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    kind: str #math,fact, chat
    path: list[str] #names of nodes visited
    error: str #calc error if exists

load_config()
llm = ChatGroq(model=MODEL)

def classify(state: State):
    user_message = state["messages"][-1]
    system_message = SystemMessage(
        content=(
            "Classify the user's message into exactly one category: "
            "math, factual, or chat. "
            "Return only the category name."
        )
    )

    response = llm.invoke(
        [system_message, user_message]
    )

    kind = response.content.strip().lower()
    if kind not in {"math","factual","chat"}:
        kind = "factual"
    return {
        "kind": kind,
        "path": state.get("path",[]) + ["classify"]
    }

BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def evaluate_expression(expression: str):
    tree = ast.parse(expression, mode="eval")

    def walk(node):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in {int, float}
        ):
            return node.value

        if (
            isinstance(node, ast.BinOp)
            and type(node.op) in BINARY_OPERATORS
        ):
            left = walk(node.left)
            right = walk(node.right)
            operation = BINARY_OPERATORS[type(node.op)]
            return operation(left, right)

        if (
            isinstance(node, ast.UnaryOp)
            and type(node.op) in UNARY_OPERATORS
        ):
            value = walk(node.operand)
            operation = UNARY_OPERATORS[type(node.op)]
            return operation(value)

        raise ValueError("Unsupported expression")

    return walk(tree.body)

def extract_expression(text: str):
    matches = re.findall(
        r"[0-9+\-*/().%\s]+",
        text,
    )

    candidates = [
        match.strip()
        for match in matches
        if any(character.isdigit() for character in match)
    ]

    if not candidates:
        raise ValueError("No arithmetic expression found")

    return max(candidates, key=len)

def calculate(state: State):
    question = state["messages"][-1].content
    path = state.get("path", []) + ["calculate"]

    try:
        expression = extract_expression(question)
        result = evaluate_expression(expression)

        reply = AIMessage(
            content=f"{expression} = {result}"
        )

        return {
            "messages": [reply],
            "path": path,
            "error": "",
        }

    except Exception as error:
        message = f"I could not calculate that: {error}"

        return {
            "messages": [AIMessage(content=message)],
            "path": path,
            "error": str(error),
        }

def answer(state: State):
    system_message = SystemMessage(
        content=(
            "Answer the user's factual question clearly. "
            "Use at most two sentences."
        )
    )

    response = llm.invoke(
        [system_message, *state["messages"]]
    )

    return {
        "messages": [response],
        "path": state.get("path", []) + ["answer"],
        "error": "",
    }

def chat(state: State):
    system_message = SystemMessage(
        content=(
            "Have a warm and brief conversation with the user. "
            "Use at most two sentences."
        )
    )

    response = llm.invoke(
        [system_message, *state["messages"]]
    )

    return {
        "messages": [response],
        "path": state.get("path", []) + ["chat"],
        "error": "",
    }

def route(
    state: State,
) -> Literal["calculate", "answer", "chat"]:
    if state["kind"] == "math":
        return "calculate"

    if state["kind"] == "chat":
        return "chat"

    return "answer"


def build_graph():
    builder = StateGraph(State)

    builder.add_node("classify", classify)
    builder.add_node("calculate", calculate)
    builder.add_node("answer", answer)
    builder.add_node("chat", chat)

    builder.add_edge(START, "classify")

    builder.add_conditional_edges(
        "classify",
        route,
        {
            "calculate": "calculate",
            "answer": "answer",
            "chat": "chat",
        },
    )

    builder.add_edge("calculate", END)
    builder.add_edge("answer", END)
    builder.add_edge("chat", END)

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


def run_message(graph, question, thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=question)
            ],
            "path": [],
            "error": "",
        },
        config=config,
    )

    print(f"\nQuestion: {question}")
    print(f"Kind: {result['kind']}")
    print(f"Path: {' -> '.join(result['path'])}")
    print(f"Answer: {result['messages'][-1].content}")

    if result["error"]:
        print(f"Error: {result['error']}")

    return result
def test_branching(graph):
    print("\n--- branching ---")

    tests = [
        (
            "What is 47 * 13 + 8?",
            "math-test",
        ),
        (
            "What causes seasons on Earth?",
            "factual-test",
        ),
        (
            "Hey, how is your day going?",
            "chat-test",
        ),
    ]

    for question, thread_id in tests:
        run_message(graph, question, thread_id)
def test_memory(graph):
    print("\n--- memory on one thread ---")

    config = {
        "configurable": {
            "thread_id": "memory-demo"
        }
    }

    first = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "My name is Sam and I work on "
                        "the payments team."
                    )
                )
            ],
            "path": [],
            "error": "",
        },
        config=config,
    )

    print(first["messages"][-1].content)

    second = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Which team did I say I work on?"
                )
            ],
            "path": [],
            "error": "",
        },
        config=config,
    )

    print(second["messages"][-1].content)
    print(f"Messages remembered: {len(second['messages'])}")

def save_graph_image(graph):
    try:
        image = graph.get_graph().draw_mermaid_png()
        GRAPH_IMAGE.write_bytes(image)
        print(f"Graph image saved to {GRAPH_IMAGE}")

    except Exception as error:
        print(f"Graph image could not be created: {error}")

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(
            encoding="utf-8",
            errors="replace",
        )

    graph = build_graph()

    save_graph_image(graph)
    test_branching(graph)
    test_memory(graph)


if __name__ == "__main__":
    main()