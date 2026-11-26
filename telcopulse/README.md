# TelcoPulse AI

TelcoPulse AI is a prototype for fictional telecom operations. A LangGraph supervisor routes each request to one of five specialists: subscriber analytics (SQL), public telecom research, chart creation, internal policy retrieval (RAG), and multilingual conversation.

## Architecture

```text
Main Supervisor
├── SQL Agent            (can hand its rows to the Chart Agent)
├── Research Team        (subgraph)
│   ├── Sub-Supervisor
│   ├── Researcher
│   └── Report Writer
├── Chart Agent
├── Policy RAG Agent
└── Conversation Agent
```

The research team is a LangGraph subgraph, which makes the system hierarchical. The main graph uses per-thread memory. For every request it records the route, execution trail, response time, fallback (retry) count and completion, both in the app and as metadata on the LangSmith trace.

## Safety and scope

- All subscriber records, services and policies are fictional.
- SQL is read-only in three layers: the request check, the single-SELECT check, and a read-only database connection.
- Customer messages are drafts that need human approval.
- The prototype never contacts customers, charges accounts or changes subscriptions.
- FAISS indexes are only loaded from files generated locally.

## Configuration

| Setting | Effect |
|---|---|
| `DATABASE_URL` | PostgreSQL connection. If it is missing or unreachable, a local SQLite file is used. |
| `TAVILY_API_KEY` | Optional. Uses Tavily for web search. Without it, DuckDuckGo is used. |

## Run

```powershell
python -m telcopulse.supervisor        # command-line demo
streamlit run telcopulse/app.py        # web interface
python -m telcopulse.database          # create and show the database
python -m telcopulse.create_dataset    # create the LangSmith dataset
python -m telcopulse.evaluate          # run the evaluation
```

## Example requests

- `Which three fictional subscribers have the highest churn risk scores?`
- `What retention offer is permitted after two failed renewals?`
- `Create a bar chart of at-risk subscribers: Lebanon 2, Jordan 2, Egypt 1, UAE 1`
- `Chart the number of subscribers per country` (SQL, then chart)
- `Research current mobile VAS retention practices in MENA.`
- `Draft a friendly Arabic message offering a three-day GameBox pass.`

## Evaluation

`telcopulse-supervisor-evaluation` has 11 requests covering all five routes. Each is scored for correct route, completion, required facts in the answer, chart creation and no fallback.

## Main files

- `database.py`: fictional telecom data (PostgreSQL or SQLite) and read-only SQL enforcement
- `agents.py`: the five specialists and the research subgraph
- `state.py`: shared LangGraph state
- `supervisor.py`: routing, SQL-to-chart handoff, retries, memory and metrics
- `app.py`: Streamlit chat and Chart.js rendering
- `create_dataset.py`, `evaluate.py`: LangSmith evaluation
- `documents/`: fictional policies indexed by the RAG agent
