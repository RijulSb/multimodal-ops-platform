"""
agents/planner.py
Turns a question (+ prior critic feedback) into a Plan of tool calls.
Falls back to a single safe document_qa step if the LLM's plan is
missing, malformed, or names an unregistered tool.
"""

from __future__ import annotations

import json
import logging
import re

from agents.base import AgentContext, Plan, PlanStep, ToolRegistry
from llm.client import LLMClient, LLMClientError

logger = logging.getLogger(__name__)

_llm_client = LLMClient()

_PLANNER_SYSTEM = """You are a planning agent for a document Q&A system. \
Output a short plan as JSON only — no prose, no markdown fences.

Format: {"steps": [{"tool_name": "...", "tool_input": {"question": "..."}, \
"rationale": "..."}], "reasoning": "one sentence"}

Rules:
- Only use tool names from AVAILABLE TOOLS. Never invent one.
- Most questions need exactly one document_qa step.
- If feedback from a prior attempt is given, sharpen tool_input.question to address it."""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class Planner:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def create_plan(self, context: AgentContext) -> Plan:
        try:
            completion = _llm_client.complete([
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": self._build_prompt(context)},
            ])
            plan = self._parse_plan(completion.text)
            if plan:
                return plan

        except LLMClientError as exc:
            logger.warning("Planner LLM call failed, using fallback: %s", exc)


        return self._fallback_plan(context)


    def _build_prompt(self, context: AgentContext) -> str:
        feedback = context.history[-1] if context.history else "None — first attempt."
        return (
            f"AVAILABLE TOOLS:\n{self.registry.describe_all()}\n\n"
            f"QUESTION: {context.question}\nPRIOR FEEDBACK: {feedback}"
        )


    def _parse_plan(self, raw_text: str) -> Plan | None:
        cleaned = _FENCE_RE.sub("", raw_text.strip())
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Planner returned non-JSON output, falling back.")
            return None

        steps = []
        for s in data.get("steps", []):
            tool_name = s.get("tool_name")
            if not tool_name or self.registry.get(tool_name) is None:
                logger.warning("Planner named unknown tool '%s', dropping step.", tool_name)
                continue
            steps.append(PlanStep(tool_name, s.get("tool_input", {}), s.get("rationale", "")))

        return Plan(steps, data.get("reasoning", "")) if steps else None


    def _fallback_plan(self, context: AgentContext) -> Plan:
        """Deterministic, always-valid — keeps the loop alive if planning fails."""
        return Plan(
            steps=[PlanStep("document_qa", {"question": context.question}, "Fallback: direct lookup.")],
            raw_reasoning="Fallback plan — planner LLM unavailable or invalid output.",
        )