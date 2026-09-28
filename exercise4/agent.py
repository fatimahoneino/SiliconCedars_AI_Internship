from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from common.config import load_config
from exercise4.tools import ALL_TOOLS

MODEL = "openai/gpt-oss-20b"


def main():
    load_config()
    llm = ChatGroq(model=MODEL).bind_tools(ALL_TOOLS)

    reply = llm.invoke(
        [HumanMessage("Translate 'good morning' to Japanese.")]
    )

    print("tool_calls:", reply.tool_calls)
    print("content:", reply.content)

import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq

from common.config import load_config
from exercise4.tools import ALL_TOOLS, init_db

MODEL = "openai/gpt-oss-20b"
MAX_STEPS = 6

SYSTEM = SystemMessage(
    "You are a helpful assistant about Japan, with access to tools. "
    "Use the translator for any Japanese translation request. "
    "Use the search tool for current facts about Japan you are not sure of. "
    "Use the prefecture lookup for questions about a specific prefecture "
    "or Japanese city, such as its capital or population. "
    "When a question needs several steps, call the tools one after another."
)

TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


def run_agent(question, llm):
    """Loop: ask the model, run any tools it requests, feed results back."""
    messages = [SYSTEM, HumanMessage(question)]
    used = []

    for _ in range(MAX_STEPS):
        reply = llm.invoke(messages)
        messages.append(reply)

        if not reply.tool_calls:
            return reply.content, used

        for call in reply.tool_calls:
            name = call["name"]
            used.append(name)
            tool = TOOLS_BY_NAME.get(name)

            if tool is None:
                result = f"unknown tool {name}"
            else:
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:
                    result = f"{name} failed: {exc}"

            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    return "Stopped after too many steps without a final answer.", used

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_config()
    init_db()
    llm = ChatGroq(model=MODEL).bind_tools(ALL_TOOLS)

    questions = [
        "Translate 'Where is the nearest train station?' to Japanese.",
        "What is the capital of Osaka prefecture and its population?",
        "What is the best time of year to see cherry blossoms in Japan?",
        "Translate 'Thank you very much' to Japanese, then tell me the "
        "capital of Hokkaido.",
        "Tell me a one-line joke about sushi.",
    ]

    for question in questions:
        start = time.time()
        answer, used = run_agent(question, llm)
        elapsed = round((time.time() - start) * 1000)

        print(f"\nQ: {question}")
        print(f"A: {answer}")
        print(f"tools used: {', '.join(used) if used else 'none'}")
        print(f"took {elapsed} ms")


if __name__ == "__main__":
    main()