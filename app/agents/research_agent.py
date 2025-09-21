"""Deep research agent built with LangGraph."""
from __future__ import annotations

from typing import List, TypedDict

import json

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from ..config import get_settings


class ResearchState(TypedDict, total=False):
    topic: str
    context: str
    plan: str
    notes: List[str]
    outline: object


class DeepResearchAgent:
    """Generates structured outlines for CIM topics."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.has_openai:
            raise RuntimeError("OPENAI_API_KEY is required for research agent")
        self.llm = ChatOpenAI(model=settings.research_model, temperature=0.2)
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("plan", self._plan_sections)
        workflow.add_node("research", self._gather_insights)
        workflow.add_node("synthesize", self._compose_outline)

        workflow.set_entry_point("plan")
        workflow.add_edge("plan", "research")
        workflow.add_edge("research", "synthesize")
        workflow.add_edge("synthesize", END)
        return workflow.compile()

    def _plan_sections(self, state: ResearchState) -> ResearchState:
        prompt = f"""
You are designing an outline for the section "{state['topic']}" in a Confidential Information Memorandum.
Given the document context, propose 4-6 high-level sections that should be covered.
Return them as a numbered list with short titles.
Document context:
{state['context']}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return {"plan": response.content}

    def _gather_insights(self, state: ResearchState) -> ResearchState:
        prompt = f"""
You are preparing research notes for the topic "{state['topic']}".
Follow this plan:
{state['plan']}
Summarise supporting points from the document context. Capture concrete data points such as growth, revenue, investment terms.
Return the answer as bullet points where each bullet covers one insight with an optional supporting quote.
Document context:
{state['context']}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        notes = [line.strip("- ") for line in response.content.splitlines() if line.strip()]
        return {"notes": notes}

    def _compose_outline(self, state: ResearchState) -> ResearchState:
        notes = "\n".join(f"- {note}" for note in state.get("notes", []))
        prompt = f"""
You are finalising the outline for "{state['topic']}" based on these notes:
{notes}
Write a detailed outline with sections, subsections, and notes on the evidence to reference.
Return the outline as JSON with keys `section`, `subsections` (list) and `evidence`.
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        try:
            outline = json.loads(response.content)
        except json.JSONDecodeError:
            outline = response.content
        return {"outline": outline}

    def run(self, *, topic: str, context: str) -> dict:
        result = self.graph.invoke({"topic": topic, "context": context})
        return {
            "topic": topic,
            "plan": result.get("plan"),
            "notes": result.get("notes"),
            "outline": result.get("outline"),
        }


__all__ = ["DeepResearchAgent"]
