from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class TelcoPulseState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    answer: str
    chart: dict[str, Any] | None
    sources: list[str]
    agent_trail: list[str]
    retries: int
    failed: bool
    elapsed_ms: float
    then_chart: bool  # True when the SQL result should be passed on to the chart agent
    data_rows: list[dict[str, Any]]  # rows returned by the SQL agent