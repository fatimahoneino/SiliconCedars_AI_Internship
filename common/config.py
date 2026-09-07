import os

from dotenv import load_dotenv


REQUIRED = [
    "GROQ_API_KEY",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_PROJECT",
]


def load_config():
    load_dotenv()
    missing = [name for name in REQUIRED if not os.getenv(name)]

    if missing:
        raise SystemExit("Missing values in .env: " + ", ".join(missing))

    return {
        "groq_api_key": os.getenv("GROQ_API_KEY"),
        "langchain_api_key": os.getenv("LANGCHAIN_API_KEY"),
        "langchain_tracing_v2": os.getenv("LANGCHAIN_TRACING_V2"),
        "langchain_project": os.getenv("LANGCHAIN_PROJECT"),
    }


if __name__ == "__main__":
    load_config()
    print("All keys loaded successfully.")