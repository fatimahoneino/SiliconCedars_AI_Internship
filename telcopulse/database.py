import re
import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).parent / "telcopulse.db"

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
    subscriber_id TEXT,
    service_id TEXT,
    status TEXT,
    failed_renewals INTEGER,
    start_date TEXT
)
"""

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
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|pragma)\b",
    re.IGNORECASE,
)


def init_db():
    if DB_PATH.exists():
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(
            """
            CREATE TABLE subscribers (
                subscriber_id TEXT PRIMARY KEY,
                country TEXT NOT NULL,
                preferred_language TEXT NOT NULL,
                days_inactive INTEGER NOT NULL,
                unresolved_complaints INTEGER NOT NULL,
                engagement_score INTEGER NOT NULL,
                churn_risk_score INTEGER NOT NULL
            );
            CREATE TABLE services (
                service_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                monthly_price_usd REAL NOT NULL
            );
            CREATE TABLE subscriptions (
                subscriber_id TEXT NOT NULL,
                service_id TEXT NOT NULL,
                status TEXT NOT NULL,
                failed_renewals INTEGER NOT NULL,
                start_date TEXT NOT NULL,
                FOREIGN KEY (subscriber_id) REFERENCES subscribers(subscriber_id),
                FOREIGN KEY (service_id) REFERENCES services(service_id)
            );
            """
        )
        conn.executemany("INSERT INTO subscribers VALUES (?, ?, ?, ?, ?, ?, ?)", SUBSCRIBERS)
        conn.executemany("INSERT INTO services VALUES (?, ?, ?, ?)", SERVICES)
        conn.executemany("INSERT INTO subscriptions VALUES (?, ?, ?, ?, ?)", SUBSCRIPTIONS)


def describe_schema():
    return SCHEMA


def is_read_only(sql: str):
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

    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(sql).fetchall()]
