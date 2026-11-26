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
    "Use the search tool when the user asks to search, or asks about news, "
    "forecasts, or anything current about Japan. "
    "Use the prefecture lookup for questions about a specific prefecture "
    "or Japanese city, such as its capital or population. "
    "When a question needs several steps, call the tools one after another. "
    "Call each tool at most once per question, then answer with what it returned. "
    "When you use search results, answer briefly, use only facts from the results, "
    "and list the source links. Do not add dates or numbers that are not in the results. "
    "If a tool reports an error, tell the user which tool failed."
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
            tool = TOOLS_BY_NAME.get(name)

            if name in used:
                # Each tool runs at most once per question.
                result = f"You already called {name}. Use its earlier result and answer now."
            elif tool is None:
                result = f"unknown tool {name}"
            else:
                used.append(name)
                try:
                    result = tool.invoke(call["args"])
                except Exception as exc:
                    # Report the failing tool by name and let the model continue.
                    result = f"Tool '{name}' failed: {exc}"

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
        "Search the web for the latest cherry blossom forecast in Japan.",
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