import json
import re
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from telcopulse.database import describe_schema, execute_select


MODEL = "openai/gpt-oss-20b"
DOCS_DIR = Path(__file__).parent / "documents"
INDEX_DIR = Path(__file__).parent / "faiss_index"
CHART_TYPES = {"bar", "line", "pie"}
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

def sql_agent(question: str):
    llm = get_llm()
    drafted = llm.invoke([
        SystemMessage(
            "Write one SQLite read-only SELECT query that answers the question. "
            "Return only SQL with no markdown or explanation. Subscriber records "
            "are fictional. Use joins when service details are needed.\n\n"
            f"Schema:\n{describe_schema()}"
        ),
        HumanMessage(question),
    ]).content
    sql = extract_sql(drafted)

    try:
        rows = execute_select(sql)
    except Exception as exc:
        return {"answer": f"The database request could not be completed: {exc}", "sources": [sql]}
    if not rows:
        return {"answer": "The database query returned no matching subscribers.", "sources": [sql]}

    explained = llm.invoke([
        SystemMessage(
            "Explain the fictional telecom query result in two or three clear sentences. "
            "Quote relevant IDs and numbers. Do not recommend contacting customers automatically."
        ),
        HumanMessage(f"Question: {question}\nSQL: {sql}\nRows: {json.dumps(rows)}"),
    ]).content
    return {"answer": explained, "sources": [sql]}


def web_search(query: str, max_results=4):
    from ddgs import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results, timeout=8))


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
    for hit in hits:
        findings.append(f"{hit.get('title', 'Untitled')}: {hit.get('body', '')}")
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
            "answer": "No public web results were available for this research request.",
            "trail": state.get("trail", []) + ["research:writer"],
        }
    answer = get_llm(temperature=0.2).invoke([
        SystemMessage(
            "Write a concise report using only the supplied public findings. "
            "Do not invent claims. State that sources should be verified before business use."
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


def chart_agent(question: str):
    parsed = extract_json(get_llm().invoke([
        SystemMessage(
            "Extract chart data as JSON with chart_type, title, labels, and values. "
            "chart_type must be bar, line, or pie. labels must be strings and values numbers. "
            "Return JSON only."
        ),
        HumanMessage(question),
    ]).content)
    chart_type = str(parsed.get("chart_type", "bar")).lower()
    title = str(parsed.get("title", "Telecom chart"))
    labels = parsed.get("labels", [])
    values = parsed.get("values", [])

    if chart_type not in CHART_TYPES:
        raise ValueError("Only bar, line, and pie charts are supported.")
    if not labels or len(labels) != len(values):
        raise ValueError("Chart labels and values must have equal non-zero lengths.")
    if not all(isinstance(label, str) for label in labels):
        raise ValueError("Chart labels must be text.")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        raise ValueError("Chart values must be numbers.")

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
            "Reply briefly in the user's language. You may draft English, Modern Standard Arabic, "
            "or simple Lebanese Arabic messages, but clearly mark customer messages as drafts for "
            "human review. Never claim to contact subscribers or change accounts. If asked about "
            "capabilities, mention subscriber analytics, policy lookup, public research, and charts."
        ),
        *recent,
    ]).content
    return {"answer": answer, "sources": []}
