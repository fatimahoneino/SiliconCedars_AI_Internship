# TelcoPulse AI

TelcoPulse AI is a portfolio prototype for fictional telecom operations. A LangGraph supervisor routes requests to five specialists: subscriber analytics, public telecom research, campaign visualization, internal policy retrieval, and multilingual conversation.

## Architecture

```text
Main Supervisor
├── SQL Agent
├── Research Team
│   ├── Sub-Supervisor
│   ├── Researcher
│   └── Report Writer
├── Chart Agent
├── Policy RAG Agent
└── Conversation Agent
```

The research department is a LangGraph subgraph, making the system hierarchical. The main graph uses per-thread memory and records route, execution trail, duration, retries, and completion.

## Safety and scope

- All subscriber records, services, and policies are fictional.
- SQL is limited to one read-only query.
- Customer messages are drafts requiring human approval.
- The prototype does not contact customers, charge accounts, or change subscriptions.
- The local demo uses SQLite instead of PostgreSQL.
- DuckDuckGo search is used instead of Tavily.
- FAISS indexes are loaded only from files generated locally.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m telcopulse.supervisor
streamlit run telcopulse/app.py
```

## Example requests

- `Which three fictional subscribers have the highest churn risk scores?`
- `What retention offer is permitted after two failed renewals?`
- `Create a bar chart of at-risk subscribers: Lebanon 2, Jordan 2, Egypt 1, UAE 1`
- `Research current mobile VAS retention practices in MENA.`
- `Draft a friendly Arabic message offering a three-day GameBox pass.`

## Main files

- `database.py`: synthetic telecom data and read-only SQL enforcement
- `agents.py`: five specialists and the research subgraph
- `state.py`: shared LangGraph state
- `supervisor.py`: routing, retries, memory, and metrics
- `app.py`: Streamlit chat and Chart.js rendering
- `documents/`: fictional policies indexed by the RAG agent
