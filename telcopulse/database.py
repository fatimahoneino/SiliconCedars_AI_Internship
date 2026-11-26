"""Fictional telecom database.

PostgreSQL is the main database (DATABASE_URL in .env). If PostgreSQL is not
reachable, the project falls back to a local SQLite file so it still runs.
Both databases get the same tables and the same fictional rows.
"""

import os
import re
import sqlite3
from pathlib import Path


SQLITE_PATH = Path(__file__).parent / "telcopulse.db"

SCHEMA = """
subscribers(
    subscriber_id TEXT PRIMARY KEY,
    country TEXT,
    preferred_language TEXT,
    days_inactive INTEGER,
    unresolved_complaints INTEGER,
    engagement_score INTEGER,
    churn_risk_score INTEGER
)
services(
    service_id TEXT PRIMARY KEY,
    name TEXT,
    category TEXT,
    monthly_price_usd REAL
)
subscriptions(
    subscriber_id TEXT REFERENCES subscribers,
    service_id TEXT REFERENCES services,
    status TEXT,
    failed_renewals INTEGER,
    start_date TEXT
)
"""

CREATE_TABLES = [
    """
    CREATE TABLE IF NOT EXISTS subscribers (
        subscriber_id TEXT PRIMARY KEY,
        country TEXT NOT NULL,
        preferred_language TEXT NOT NULL,
        days_inactive INTEGER NOT NULL,
        unresolved_complaints INTEGER NOT NULL,
        engagement_score INTEGER NOT NULL,
        churn_risk_score INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS services (
        service_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        monthly_price_usd REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS subscriptions (
        subscriber_id TEXT NOT NULL REFERENCES subscribers(subscriber_id),
        service_id TEXT NOT NULL REFERENCES services(service_id),
        status TEXT NOT NULL,
        failed_renewals INTEGER NOT NULL,
        start_date TEXT NOT NULL,
        PRIMARY KEY (subscriber_id, service_id)
    )
    """,
]

SUBSCRIBERS = [
    ("SUB-1001", "Lebanon", "Arabic", 3, 0, 88, 12),
    ("SUB-1002", "Jordan", "Arabic", 21, 1, 52, 58),
    ("SUB-1003", "Lebanon", "English", 42, 1, 28, 82),
    ("SUB-1004", "Egypt", "Arabic", 7, 0, 76, 25),
    ("SUB-1005", "UAE", "English", 31, 2, 35, 77),
    ("SUB-1006", "Lebanon", "Arabic", 12, 0, 65, 38),
    ("SUB-1007", "Jordan", "English", 55, 2, 18, 91),
    ("SUB-1008", "Egypt", "Arabic", 2, 0, 92, 8),
]

SERVICES = [
    ("VAS-01", "GameBox", "Gaming", 2.00),
    ("VAS-02", "GoalAlert", "Sports", 1.00),
    ("VAS-03", "MusicDaily", "Music", 2.50),
    ("VAS-04", "LearnArabic", "Education", 1.50),
]

SUBSCRIPTIONS = [
    ("SUB-1001", "VAS-01", "active", 0, "2025-08-10"),
    ("SUB-1002", "VAS-02", "active", 2, "2025-05-14"),
    ("SUB-1003", "VAS-01", "active", 3, "2025-09-02"),
    ("SUB-1004", "VAS-04", "active", 0, "2026-01-20"),
    ("SUB-1005", "VAS-03", "active", 2, "2025-11-08"),
    ("SUB-1006", "VAS-02", "paused", 1, "2025-04-15"),
    ("SUB-1007", "VAS-01", "active", 4, "2025-02-18"),
    ("SUB-1008", "VAS-04", "active", 0, "2026-03-05"),
]

WRITE_WORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|grant|revoke"
    r"|attach|pragma|copy|vacuum|call|do)\b",
    re.IGNORECASE,
)

_backend = None


def _ensure_postgres_database(url):
    """Create the database named in DATABASE_URL if it does not exist yet."""
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    name = conninfo_to_dict(url).get("dbname")
    if not name or name == "postgres":
        return
    admin = make_conninfo(url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True, connect_timeout=5) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (name,)
        ).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def get_backend():
    """Return 'postgresql' when DATABASE_URL works, otherwise 'sqlite'."""
    global _backend
    if _backend:
        return _backend

    url = os.getenv("DATABASE_URL")
    if url:
        try:
            import psycopg

            _ensure_postgres_database(url)
            with psycopg.connect(url, connect_timeout=5):
                pass
            _backend = "postgresql"
            return _backend
        except Exception as exc:
            print(f"PostgreSQL is not available ({type(exc).__name__}). Using SQLite instead.")

    _backend = "sqlite"
    return _backend


def init_db():
    """Create the tables and insert the fictional rows (safe to run many times)."""
    if get_backend() == "postgresql":
        import psycopg

        with psycopg.connect(os.getenv("DATABASE_URL")) as conn:
            for statement in CREATE_TABLES:
                conn.execute(statement)
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO subscribers VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    SUBSCRIBERS,
                )
                cur.executemany(
                    "INSERT INTO services VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    SERVICES,
                )
                cur.executemany(
                    "INSERT INTO subscriptions VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT DO NOTHING",
                    SUBSCRIPTIONS,
                )
        return

    with sqlite3.connect(SQLITE_PATH) as conn:
        for statement in CREATE_TABLES:
            conn.execute(statement)
        conn.executemany("INSERT OR IGNORE INTO subscribers VALUES (?, ?, ?, ?, ?, ?, ?)", SUBSCRIBERS)
        conn.executemany("INSERT OR IGNORE INTO services VALUES (?, ?, ?, ?)", SERVICES)
        conn.executemany("INSERT OR IGNORE INTO subscriptions VALUES (?, ?, ?, ?, ?)", SUBSCRIPTIONS)


def describe_schema():
    return SCHEMA


def is_read_only(sql: str):
    """First safety layer: accept one SELECT (or WITH ... SELECT) and nothing else."""
    stripped = sql.strip().rstrip(";")
    return (
        len(stripped) <= 2000
        and ";" not in stripped
        and stripped.lower().startswith(("select", "with"))
        and WRITE_WORDS.search(stripped) is None
    )


def execute_select(sql: str):
    if not is_read_only(sql):
        raise ValueError("Only one read-only SELECT query is allowed.")

    # Second safety layer: the connection itself is read-only,
    # so the database refuses any write even if one got past the check above.
    if get_backend() == "postgresql":
        import psycopg
        from psycopg.rows import dict_row

        with psycopg.connect(os.getenv("DATABASE_URL"), row_factory=dict_row) as conn:
            conn.read_only = True
            conn.execute("SET statement_timeout = 5000")
            return conn.execute(sql).fetchall()

    conn = sqlite3.connect(f"file:{SQLITE_PATH.as_posix()}?mode=ro", uri=True)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


if __name__ == "__main__":
    from common.config import load_config

    load_config()
    init_db()
    print("Database:", get_backend())
    for row in execute_select(
        "SELECT subscriber_id, country, churn_risk_score FROM subscribers "
        "ORDER BY churn_risk_score DESC"
    ):
        print(row)
