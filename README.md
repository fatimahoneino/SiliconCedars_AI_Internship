# Silicon Cedars AI Internship

Agentic AI work built step by step with LangChain, LangGraph, Groq and LangSmith.

| Folder | What it covers |
|---|---|
| `common/` | Loads and checks the API keys from `.env` |
| `exercise1/` | First model call, prompt templates, timed model comparison |
| `exercise2/` | RAG over local documents (FAISS + FastEmbed) with a LangSmith evaluation |
| `exercise3/` | LangGraph with four nodes, conditional branching and memory |
| `exercise4/` | Agent with three tools (translator, web search, database) with a LangSmith evaluation |
| `telcopulse/` | Final project: hierarchical multi-agent system with a supervisor and a Streamlit UI |

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then fill in the keys
python -m common.config     # checks the keys
```

## Run

```powershell
python -m exercise1.hello_groq
python -m exercise1.prompt_template
python -m exercise1.compare_models
python -m exercise2.rag
python -m exercise3.graph
python -m exercise4.agent
python -m telcopulse.supervisor
streamlit run telcopulse/app.py
```

## Evaluation (LangSmith)

```powershell
python -m exercise2.create_dataset;   python -m exercise2.evaluate_rag
python -m exercise4.create_dataset;   python -m exercise4.evaluate_agent
python -m telcopulse.create_dataset;  python -m telcopulse.evaluate
```

| Dataset | Examples | Checks |
|---|---|---|
| `exercise-2-rag-evaluation` | 5 | required phrase, correct source, correct refusal |
| `exercise-4-tool-selection` | 5 | correct tool sequence, no repeated calls |
| `telcopulse-supervisor-evaluation` | 11 | correct route, completed, facts, chart created, no fallback |
