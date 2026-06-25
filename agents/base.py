"""
agents/base.py
Shared types for the planner -> executor -> critic loop.
No agent file imports another at module level — the orchestrator below
uses lazy imports to avoid a circular import.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from api.document_store import document_store
from tools.doc_extractor import ExtractionResult

logger = logging.getLogger(__name__)

MAX_LOOP_ITERATIONS = 2  # plan -> execute -> critique, one revision pass max


class AgentError(Exception):
    """Raised when the loop cannot proceed at all (e.g. unknown job_id)."""


@dataclass
class AgentContext:
    """Per-request state shared across planner, executor, and critic."""
    job_id: str
    question: str
    document: ExtractionResult
    history: list[str] = field(default_factory=list)  # critic feedback, newest last


    @classmethod
    def from_job(cls, job_id: str, question: str) -> "AgentContext":

        document = document_store.get(job_id)
        if document is None:
                    raise AgentError(f"No document found for job_id '{job_id}'.")
        return cls(job_id=job_id, question=question, document=document)


@dataclass
class ToolResult:
    """One tool call's outcome. `output` shape is tool-specific."""
    tool_name: str
    success: bool
    output: Any
    error: str | None = None


class Tool(ABC):
    """Contract every tool implements. Register instances in a ToolRegistry."""
    name: str
    description: str

    @abstractmethod
    async def run(self, tool_input: dict[str, Any], context: AgentContext) -> ToolResult:
        ...


class ToolRegistry:
    """Name -> Tool lookup. Planner picks names; Executor resolves them."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def describe_all(self) -> str:
        """Tool list fed to the planner's prompt — keeps it from naming fake tools."""
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())


@dataclass
class PlanStep:
    tool_name: str
    tool_input: dict[str, Any]
    rationale: str


@dataclass
class Plan:
    steps: list[PlanStep]
    raw_reasoning: str = ""


@dataclass
class ExecutionResult:
    step_results: list[ToolResult]
    final_answer: str
    pages_used: list[int] = field(default_factory=list)
    hallucination_risk: float = 0.0
    hallucination_signals: list[str] = field(default_factory=list)
    secrets_found: list[str] = field(default_factory=list)


class Verdict(str, Enum):
    ACCEPT = "accept"
    REVISE = "revise"


@dataclass
class Critique:
    verdict: Verdict
    feedback: str


@dataclass
class AgentResult:
    """Final, caller-facing output of the loop."""
    answer: str
    pages_used: list[int]
    hallucination_risk: float
    hallucination_signals: list[str]
    iterations: int
    accepted: bool  # False if the loop exhausted MAX_LOOP_ITERATIONS without an ACCEPT


async def run_agent_loop(job_id: str, question: str, registry: ToolRegistry | None = None) -> AgentResult:
    """Runs Planner -> Executor -> Critic, looping once on REVISE, then returns best effort."""
    from agents.planner import Planner
    from agents.executor import Executor, build_default_registry
    from agents.critic import Critic

    registry = registry or build_default_registry()
    context = AgentContext.from_job(job_id, question)
    planner, executor, critic = Planner(registry), Executor(registry), Critic()

    result: ExecutionResult | None = None
    for iteration in range(1, MAX_LOOP_ITERATIONS + 1):
        plan = await planner.create_plan(context)
        result = await executor.execute(plan, context)
        critique = await critic.review(context, result)

        if critique.verdict == Verdict.ACCEPT:
            return AgentResult(
                answer=result.final_answer,
                pages_used=result.pages_used,
                hallucination_risk=result.hallucination_risk,
                hallucination_signals=result.hallucination_signals,
                iterations=iteration,
                accepted=True,
            )

        context.history.append(critique.feedback)
        logger.info("job=%s revision requested (iter %d): %s", job_id, iteration, critique.feedback)

    # Loop exhausted — return the last attempt; a flagged best-effort beats nothing.
    return AgentResult(
        answer=result.final_answer,
        pages_used=result.pages_used,
        hallucination_risk=result.hallucination_risk,
        hallucination_signals=result.hallucination_signals,
        iterations=MAX_LOOP_ITERATIONS,
        accepted=False,
    )