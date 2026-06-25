"""
agents/critic.py
Reviews an ExecutionResult before it's accepted. Reuses the
hallucination score llm/output_guard.py already computed during
execution — no second LLM call judging the first one.
"""

from __future__ import annotations

import logging

from agents.base import AgentContext, Critique, ExecutionResult, Verdict

logger = logging.getLogger(__name__)

REVISE_RISK_THRESHOLD = 0.6  # at/above this, or with no source pages, force a revision



class Critic:
    async def review(self, context: AgentContext, result: ExecutionResult) -> Critique:
        if not result.final_answer or "couldn't answer" in result.final_answer.lower():
            return Critique(Verdict.REVISE, "No usable answer — try a broader or rephrased lookup.")

        if not result.pages_used:
            return Critique(Verdict.REVISE, "Answer cites no source pages — re-ground it in the document text.")

        if result.hallucination_risk >= REVISE_RISK_THRESHOLD:
            reasons = "; ".join(result.hallucination_signals) or "high hallucination risk"
            return Critique(Verdict.REVISE,
                            f"Low-confidence answer ({reasons}). Re-answer more cautiously and cite pages.")

        logger.debug("job=%s accepted on review", context.job_id)
        return Critique(Verdict.ACCEPT, "Answer is grounded and within risk thresholds.")