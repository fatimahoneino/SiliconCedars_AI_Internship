import sqlite3
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from common.config import load_config

DB_PATH = Path(__file__).parent / "japan.db"
TRANSLATE_MODEL = "openai/gpt-oss-20b"

PREFECTURES = [
    ("Tokyo", "Tokyo", "Kanto", 14000000),
    ("Osaka", "Osaka", "Kansai", 8800000),
    ("Kyoto", "Kyoto", "Kansai", 2550000),
    ("Hokkaido", "Sapporo", "Hokkaido", 5140000),
    ("Okinawa", "Naha", "Kyushu", 1460000),
    ("Aichi", "Nagoya", "Chubu", 7500000),
    ("Fukuoka", "Fukuoka", "Kyushu", 5100000),
]

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
def translate_to_japanese(text: str) -> str:
    """Translate English text into Japanese.

    Use this whenever the user asks for a Japanese translation of a word,
    phrase, or sentence.
    """
    try:
        load_config()
        llm = ChatGroq(model=TRANSLATE_MODEL)
        reply = llm.invoke(TRANSLATE_PROMPT.invoke({"text": text}))
        return reply.content
    except Exception as exc:
        return f"translate_to_japanese error: {exc}"


@tool
def search_japan(query: str) -> str:
    """Search the public web for current information about Japan.

    Use this for news, forecasts, events, or anything current about Japan.
    """
    from ddgs import DDGS

    japan_query = f"{query} Japan"
    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(japan_query, max_results=3))
    except Exception as exc:
        return f"search_japan error: {exc}"

    if not hits:
        return f"No results found for {japan_query!r}."

    return "\n".join(
        f"{hit.get('title', 'untitled')}: {hit.get('body', '')} (source: {hit.get('href', 'unknown')})"
        for hit in hits
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
            # The value is passed as a parameter (?), never glued into the SQL text.
            rows = conn.execute(
                "SELECT name, capital, region, population FROM prefectures "
                "WHERE name LIKE ?",
                (f"%{name}%",),
            ).fetchall()
    except Exception as exc:
        return f"lookup_prefecture error: {exc}"

    if not rows:
        return f"No prefecture found matching {name!r}."

    return "\n".join(
        f"{r['name']} - capital {r['capital']}, region {r['region']}, "
        f"population {r['population']:,}"
        for r in rows
    )


ALL_TOOLS = [translate_to_japanese, search_japan, lookup_prefecture]


if __name__ == "__main__":
    init_db()
    print(lookup_prefecture.invoke({"name": "Osaka"}))
