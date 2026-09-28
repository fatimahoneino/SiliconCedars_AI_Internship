import sqlite3
from pathlib import Path

from langchain_core.tools import tool

DB_PATH = Path(__file__).parent / "japan.db"

PREFECTURES = [
    ("Tokyo", "Tokyo", "Kanto", 14000000),
    ("Osaka", "Osaka", "Kansai", 8800000),
    ("Kyoto", "Kyoto", "Kansai", 2550000),
    ("Hokkaido", "Sapporo", "Hokkaido", 5140000),
    ("Okinawa", "Naha", "Kyushu", 1460000),
    ("Aichi", "Nagoya", "Chubu", 7500000),
    ("Fukuoka", "Fukuoka", "Kyushu", 5100000),
]

def init_db():
    """Create the Japan facts database if it does not exist yet."""
    if DB_PATH.exists():
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE prefectures (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                capital TEXT NOT NULL,
                region TEXT NOT NULL,
                population INTEGER NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO prefectures (name, capital, region, population) "
            "VALUES (?, ?, ?, ?)",
            PREFECTURES,
        )

@tool
def lookup_prefecture(name: str) -> str:
    """Look up a Japanese prefecture's capital city, region and population.

    Use this for any question about a specific Japanese prefecture or city.
    Accepts a full or partial name, such as "Tokyo" or "Osaka".
    """
    init_db()
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT name, capital, region, population FROM prefectures "
                "WHERE name LIKE ?",
                (f"%{name}%",),
            ).fetchall()
    except Exception as exc:
        return f"database error: {exc}"

    if not rows:
        return f"No prefecture found matching {name!r}."

    return "\n".join(
        f"{r['name']} - capital {r['capital']}, region {r['region']}, "
        f"population {r['population']:,}"
        for r in rows
    )
if __name__ == "__main__":
    init_db()
    print(lookup_prefecture.invoke({"name": "Osaka"}))

@tool
def search_japan(query: str) -> str:
    """Search the public web for current information about Japan.

    Use this for news, travel facts, culture, or anything about Japan
    that you do not already know for certain.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return "search error: no search package installed"

    japan_query = f"{query} Japan"

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(japan_query, max_results=3))
    except Exception as exc:
        return f"search error: {exc}"

    if not hits:
        return f"No results found for {japan_query!r}."

    lines = []
    for hit in hits:
        title = hit.get("title", "untitled")
        body = hit.get("body", "")
        lines.append(f"{title}: {body}")

    return "\n".join(lines)
if __name__ == "__main__":
    print(search_japan.invoke({"query": "cherry blossom season"}))


# translator starts
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from common.config import load_config

TRANSLATE_MODEL = "openai/gpt-oss-20b"

TRANSLATE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Translate the user's English text into natural Japanese. "
            "Reply with only the Japanese translation, nothing else.",
        ),
        ("human", "{text}"),
    ]
)


@tool
def translate_to_japanese(text: str) -> str:
    """Translate English text into Japanese.

    Use this whenever the user asks for a Japanese translation of a word,
    phrase, or sentence.
    """
    try:
        load_config()
        llm = ChatGroq(model=TRANSLATE_MODEL)
        filled_prompt = TRANSLATE_PROMPT.invoke({"text": text})
        reply = llm.invoke(filled_prompt)
        return reply.content
    except Exception as exc:
        return f"translation error: {exc}"
ALL_TOOLS = [translate_to_japanese, search_japan, lookup_prefecture]