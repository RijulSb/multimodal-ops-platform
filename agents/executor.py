"""
agents/executor.py
Runs a Plan's steps against registered tools and merges results into
one ExecutionResult. Today's only tool: document_qa (RAG via llm/).
"""

from __future__ import annotations

import logging
from typing import Any

from agents.base import AgentContext, ExecutionResult, Plan, Tool, ToolRegistry, ToolResult
from llm.client import LLMClient, LLMClientError
from llm.output_guard import scrub_secrets, score_hallucination
from llm.prompt_builder import build_rag_prompt

logger = logging.getLogger(__name__)

_llm_client = LLMClient()  # shared singleton, same pattern as api/routes/query.py


class DocumentQATool(Tool):
    """Answers a question via RAG over the document already in context."""
    name = "document_qa"
    description = "Answer a question using the content of the ingested document."

    async def run(self, tool_input: dict[str, Any], context: AgentContext) -> ToolResult:
        question = tool_input.get("question", context.question)
        bundle = build_rag_prompt(context.document, question, client_token_counter=_llm_client.count_tokens)

        try:
            completion = await _llm_client.complete([
                {"role": "system", "content": bundle.system},
                {"role": "user", "content": bundle.user},
            ])
        except LLMClientError as exc:
            logger.error("document_qa failed job=%s: %s", context.job_id, exc)
            return ToolResult(self.name, success=False, output=None, error=str(exc))

        scrub = scrub_secrets(completion.text)
        risk = score_hallucination(completion.text, bundle.user, bundle.trust, bundle.pages_included)

        return ToolResult(self.name, success=True, output={
            "answer": scrub.scrubbed_text,
            "pages_used": bundle.pages_included,
            "hallucination_risk": risk.risk_score,
            "hallucination_signals": risk.signals,
            "secrets_found": scrub.secrets_found,
        })


def build_default_registry() -> ToolRegistry:
    """Today: one tool. tools/action_dispatcher.py is the seam for more later."""
    registry = ToolRegistry()
    registry.register(DocumentQATool())
    return registry


class Executor:
    """Resolves each PlanStep's tool, runs it, merges outputs into one answer."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def execute(self, plan: Plan, context: AgentContext) -> ExecutionResult:
        step_results: list[ToolResult] = []

        for step in plan.steps:
            tool = self.registry.get(step.tool_name)
            if tool is None:
                step_results.append(ToolResult(step.tool_name, False, None, f"Unknown tool '{step.tool_name}'."))
                continue
            try:
                step_results.append(await tool.run(step.tool_input, context))
            except Exception as exc:  # one bad step shouldn't kill the whole plan
                logger.error("Tool '%s' crashed: %s", step.tool_name, exc, exc_info=True)
                step_results.append(ToolResult(step.tool_name, False, None, str(exc)))

        return self._merge(step_results)

    def _merge(self, step_results: list[ToolResult]) -> ExecutionResult:
        """Combines one or more tool outputs. Pass-through today; extends to multi-step plans later."""
        ok = [r for r in step_results if r.success]
        if not ok:
            errors = "; ".join(r.error or "unknown error" for r in step_results)
            return ExecutionResult(step_results, f"Couldn't answer — every step failed ({errors}).")

        answers = [r.output["answer"] for r in ok if isinstance(r.output, dict) and r.output.get("answer")]
        pages = sorted({p for r in ok for p in r.output.get("pages_used", [])})
        risk = max((r.output.get("hallucination_risk", 0.0) for r in ok), default=0.0)
        signals = [s for r in ok for s in r.output.get("hallucination_signals", [])]
        secrets = [s for r in ok for s in r.output.get("secrets_found", [])]

        return ExecutionResult(
            step_results=step_results,
            final_answer="\n\n".join(answers) if answers else "No answer produced.",
            pages_used=pages,
            hallucination_risk=risk,
            hallucination_signals=signals,
            secrets_found=secrets,
        )