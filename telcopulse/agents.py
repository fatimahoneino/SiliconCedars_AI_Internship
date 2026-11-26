import json
import os
import re
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from telcopulse.database import describe_schema, execute_select, get_backend


MODEL = "openai/gpt-oss-20b"
DOCS_DIR = Path(__file__).parent / "documents"
INDEX_DIR = Path(__file__).parent / "faiss_index"
CHART_TYPES = {"bar", "line", "pie"}
OTHER_CHART_TYPES = {"radar", "scatter", "bubble", "doughnut", "donut", "area", "histogram", "heatmap", "polar"}
PALETTE = [
    "rgba(34,211,238,0.75)",
    "rgba(59,130,246,0.75)",
    "rgba(139,92,246,0.75)",
    "rgba(16,185,129,0.75)",
    "rgba(249,115,22,0.75)",
]


def get_llm(temperature=0.0):
    return ChatGroq(model=MODEL, temperature=temperature)


def extract_json(text: str):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The model did not return a JSON object.")
    return json.loads(match.group(0))


def extract_sql(text: str):
    fenced = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else text.strip()

CHANGE_WORDS = re.compile(
    r"\b(delete|remove|drop|erase|wipe|truncate|update|insert|add|change|modify|edit|alter|reset)\b",
    re.IGNORECASE,
)


def sql_agent(question: str):
    # Guard on the request itself: this agent only reads data.
    if CHANGE_WORDS.search(question):
        return {
            "answer": "This request was refused. The SQL agent is read-only: it can only "
                      "read data with one SELECT query. Nothing was changed.",
            "sources": [],
            "rows": [],
        }

    llm = get_llm()
    dialect = "PostgreSQL" if get_backend() == "postgresql" else "SQLite"
    drafted = llm.invoke([
        SystemMessage(
            f"Write one {dialect} read-only SELECT query that answers the question. "
            "Return only SQL with no markdown or explanation. Subscriber records "
            "are fictional. Use joins when service details are needed. "
            "Each row in subscriptions is one service of one subscriber, so compare "
            "failed_renewals per row unless the user asks for a total.\n\n"
            f"Schema:\n{describe_schema()}"
        ),
        HumanMessage(question),
    ]).content
    sql = extract_sql(drafted)

    if not sql.lower().lstrip("(").startswith(("select", "with")):
        # The model refused to write a query (for example for a delete request).
        return {
            "answer": "This request was refused. I can only read data with one SELECT query. Nothing was changed.",
            "sources": [],
            "rows": [],
        }

    try:
        rows = execute_select(sql)
    except ValueError as exc:
        # Blocked by the read-only check.
        return {"answer": f"This request was refused: {exc} Nothing was changed.", "sources": [sql], "rows": []}
    except Exception as exc:
        # Report the database error message (for example a wrong column name).
        return {"answer": f"The database request could not be completed: {exc}", "sources": [sql], "rows": []}
    if not rows:
        return {"answer": "The database query returned no matching subscribers.", "sources": [sql], "rows": []}

    explained = llm.invoke([
        SystemMessage(
            "Explain the fictional telecom query result in two or three clear sentences. "
            "Quote relevant IDs and numbers exactly as they appear in the rows. "
            "Do not recommend contacting customers automatically."
        ),
        HumanMessage(f"Question: {question}\nSQL: {sql}\nRows: {json.dumps(rows, default=str)}"),
    ]).content
    return {"answer": explained, "sources": [sql], "rows": rows}


def search_provider():
    return "tavily" if os.getenv("TAVILY_API_KEY") else "duckduckgo"


def web_search(query: str, max_results=4):
    """Tavily when TAVILY_API_KEY is set, otherwise DuckDuckGo (no key needed).

    Both return a list of {"title", "body", "href"} dictionaries.
    """
    if search_provider() == "tavily":
        import httpx

        response = httpx.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {os.getenv('TAVILY_API_KEY')}"},
            json={"query": query, "max_results": max_results},
            timeout=15,
        )
        response.raise_for_status()
        return [
            {"title": r.get("title", ""), "body": r.get("content", ""), "href": r.get("url", "")}
            for r in response.json().get("results", [])
        ]

    import time

    from ddgs import DDGS

    # DuckDuckGo sometimes refuses a request for a moment, so try twice.
    last_error = None
    for attempt in range(2):
        try:
            with DDGS() as ddgs:
                hits = list(ddgs.text(query, max_results=max_results, timeout=8))
            if hits:
                return hits
        except Exception as exc:
            last_error = exc
        time.sleep(1.5)
    if last_error:
        raise last_error
    return []


class ResearchState(TypedDict, total=False):
    question: str
    query: str
    findings: str
    sources: list[str]
    answer: str
    attempts: int
    trail: list[str]


def research_plan(state: ResearchState):
    raw = get_llm().invoke([
        SystemMessage(
            "Convert the request into one concise public web search about telecom, "
            "mobile VAS, subscriber retention, or multilingual customer engagement. "
            "Return only the search query. Never include private subscriber data."
        ),
        HumanMessage(state["question"]),
    ]).content
    return {"query": raw.strip(), "attempts": 0, "trail": ["research:sub-supervisor"]}


def research_search(state: ResearchState):
    try:
        hits = web_search(state["query"])
    except Exception:
        hits = []
    findings = []
    sources = []
    for number, hit in enumerate(hits, start=1):
        findings.append(
            f"[{number}] {hit.get('title', 'Untitled')}: {hit.get('body', '')} "
            f"(source: {hit.get('href', 'unknown')})"
        )
        if hit.get("href"):
            sources.append(hit["href"])
    return {
        "findings": "\n\n".join(findings),
        "sources": sources,
        "attempts": state.get("attempts", 0) + 1,
        "trail": state.get("trail", []) + ["research:researcher"],
    }

def research_retry(state: ResearchState):
    return {
        "query": "telecom subscriber retention mobile VAS customer engagement",
        "trail": state.get("trail", []) + ["research:sub-supervisor(retry)"],
    }


def research_writer(state: ResearchState):
    if not state.get("findings"):
        return {
            "answer": (
                "No public web results were found for this request, even after a retry "
                f"with a simpler query ('{state.get('query', '')}')."
            ),
            "trail": state.get("trail", []) + ["research:writer"],
        }
    answer = get_llm(temperature=0.2).invoke([
        SystemMessage(
            "Write a concise report using only the supplied public findings. "
            "Cite each claim with its finding number, like [1] or [2]. "
            "Do not invent claims. If a finding is not related to the question, ignore it. "
            "End with a 'Sources' list of the URLs you used, and say that sources "
            "should be verified before business use."
        ),
        HumanMessage(f"Question: {state['question']}\n\nFindings:\n{state['findings']}"),
    ]).content
    return {"answer": answer, "trail": state.get("trail", []) + ["research:writer"]}


def after_research(state: ResearchState):
    if state.get("findings") or state.get("attempts", 0) >= 2:
        return "writer"
    return "retry"


def build_research_team():
    builder = StateGraph(ResearchState)
    builder.add_node("sub_supervisor", research_plan)
    builder.add_node("researcher", research_search)
    builder.add_node("retry", research_retry)
    builder.add_node("writer", research_writer)
    builder.add_edge(START, "sub_supervisor")
    builder.add_edge("sub_supervisor", "researcher")
    builder.add_conditional_edges("researcher", after_research, {"retry": "retry", "writer": "writer"})
    builder.add_edge("retry", "researcher")
    builder.add_edge("writer", END)
    return builder.compile()


RESEARCH_TEAM = build_research_team()


def research_team(question: str):
    result = RESEARCH_TEAM.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
        "trail": result.get("trail", []),
    }


def chart_agent(question: str, rows=None):
    """Build a Chart.js spec.

    The data comes from the user's message, or from SQL rows when the
    supervisor sends the SQL result on to this agent.
    Bad input gets a clear answer instead of an error, because retrying would not help.
    """
    request = question
    if rows:
        request += f"\n\nUse these database rows as the data:\n{json.dumps(rows, default=str)}"
    parsed = extract_json(get_llm().invoke([
        SystemMessage(
            "Extract chart data as JSON with chart_type, title, labels, and values. "
            "chart_type is exactly the chart type the user asked for, even if it is unusual "
            "(use bar only if the user does not name a type). "
            "labels must be strings and values numbers. Return JSON only."
        ),
        HumanMessage(request),
    ]).content)
    chart_type = str(parsed.get("chart_type", "bar")).lower()
    # Double check in code: if the user named an unsupported type, do not replace it silently.
    asked = re.search(r"\b(\w+)\s+(?:chart|graph|plot)\b", question.lower())
    if asked and asked.group(1) in OTHER_CHART_TYPES:
        chart_type = asked.group(1)
    title = str(parsed.get("title", "Telecom chart"))
    labels = parsed.get("labels", [])
    values = parsed.get("values", [])

    # These are problems with the user's request, so retrying would not help.
    # Return a clear message instead of raising an error.
    if chart_type not in CHART_TYPES:
        return {"answer": f"'{chart_type}' charts are not supported. Supported types: bar, line, pie."}
    if not labels or not values:
        return {"answer": "No data is available to chart. Please give labels and numbers, e.g. 'Lebanon 2, Jordan 3'."}
    if len(labels) != len(values):
        return {"answer": "Each label needs exactly one number. Please check the data."}
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        return {"answer": "Chart values must be numbers."}
    labels = [str(label) for label in labels]

    colours = [PALETTE[index % len(PALETTE)] for index in range(len(values))]
    chart = {
        "type": chart_type,
        "data": {
            "labels": labels,
            "datasets": [{
                "label": title,
                "data": values,
                "backgroundColor": colours if chart_type == "pie" else colours[0],
                "borderColor": colours[0],
                "borderWidth": 1,
            }],
        },
        "options": {"responsive": True, "plugins": {"title": {"display": True, "text": title}}},
    }
    return {"answer": f"Created a {chart_type} chart titled {title}.", "chart": chart}


_store = None


def get_store():
    global _store
    if _store is not None:
        return _store

    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = FastEmbedEmbeddings()
    index_file = INDEX_DIR / "index.faiss"
    metadata_file = INDEX_DIR / "index.pkl"
    if index_file.exists() and metadata_file.exists():
        _store = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        return _store

    documents = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(DOCS_DIR.glob("*.txt"))
    ]
    if not documents:
        raise ValueError("No internal policy documents were found.")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    ).split_documents(documents)
    if not chunks:
        raise ValueError("The internal documents produced no searchable chunks.")
    _store = FAISS.from_documents(chunks, embeddings)
    _store.save_local(str(INDEX_DIR))
    return _store


def rag_agent(question: str):
    hits = get_store().similarity_search(question, k=3)
    if not hits:
        return {"answer": "No relevant internal policy information was found.", "sources": []}
    context = "\n\n".join(
        f"[{hit.metadata['source']}]\n{hit.page_content}" for hit in hits
    )
    answer = get_llm().invoke([
        SystemMessage(
            "Answer only from the fictional TelcoPulse policy context. "
            "If the answer is absent, say you do not know. Always remind the user "
            "that customer-facing actions need human approval.\n\n"
            f"Context:\n{context}"
        ),
        HumanMessage(question),
    ]).content
    sources = sorted({hit.metadata["source"] for hit in hits})
    return {"answer": answer, "sources": sources}


def conversation_agent(question: str, history: list):
    recent = [message for message in history if isinstance(message, (HumanMessage, AIMessage))][-10:]
    answer = get_llm(temperature=0.3).invoke([
        SystemMessage(
            "You are TelcoPulse AI, a portfolio prototype for fictional telecom operations. "
            "You are talking to an operations employee, not a customer. Reply briefly in the "
            "user's language and use the earlier conversation when it is relevant. "
            "Only when the user asks you to write a message for a customer, draft it in English, "
            "Modern Standard Arabic, or simple Lebanese Arabic, and start it with "
            "'Draft for human review:'. Do not add that label to normal replies. "
            "Never claim to contact subscribers or change accounts. If asked about "
            "capabilities, mention subscriber analytics, policy lookup, public research, and charts."
        ),
        *recent,
    ]).content
    return {"answer": answer, "sources": []}
