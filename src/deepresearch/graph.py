"""LangGraph runtime assembly with intake, supervisor, and researcher subgraphs."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, MessagesState, StateGraph

from .intake import scope_intake
from .report import final_report_generation
from .researcher_subgraph import build_researcher_subgraph
from .state import ResearchState
from .supervisor_subgraph import build_supervisor_subgraph


def build_app(*, checkpointer: Any | None = None):
    """Construct and compile the LangGraph runtime."""
    builder = StateGraph(ResearchState, input_schema=MessagesState)
    builder.add_node("scope_intake", scope_intake)
    builder.add_node("research_supervisor", build_supervisor_subgraph())
    builder.add_node("final_report_generation", final_report_generation)

    builder.add_edge(START, "scope_intake")
    builder.add_edge("research_supervisor", "final_report_generation")
    builder.add_edge("final_report_generation", END)
    return builder.compile(checkpointer=checkpointer)


# Keep the single canonical export expected by langgraph.json.
app = build_app()


__all__ = [
    "app",
    "build_app",
    "scope_intake",
    "build_researcher_subgraph",
    "build_supervisor_subgraph",
    "final_report_generation",
]
