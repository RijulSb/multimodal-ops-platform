"""
llm/client.py — OpenAI abstraction layer.

Why a wrapper instead of calling the OpenAI SDK directly from routes:
  1. Retry logic belongs in exactly one place. Without this, every caller
     (query.py today, agents/executor.py later) would reinvent backoff —
     and inevitably get it slightly wrong in one of them.
  2. Token counting needs to happen BEFORE the request, not after, so we
     can budget context (prompt_builder.py) instead of discovering we blew
     the context window via a 400 error from the API.
  3. A single chokepoint for swapping providers later (Anthropic, local
     model, etc.) without touching every call site.

Design mirrors the async pattern already established in api/routes/ingest.py:
  ingest.py wraps a BLOCKING call (pdfplumber/Tesseract) in run_in_executor
  because those libraries have no async API.
  Here, AsyncOpenAI is natively async — no run_in_executor needed. We are
  not fighting a blocking C-call here, we're awaiting a real coroutine.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import tiktoken
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Errors worth retrying: transient/server-side. NOT retried: auth errors,
# bad-request errors (malformed payload won't fix itself on retry — retrying
# those just burns latency and quota for a guaranteed-repeat failure).
_RETRYABLE_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)

# Fallback encoding for models tiktoken doesn't recognise by name yet.
# cl100k_base is the encoding used by gpt-4/gpt-4o family — close enough
# for budgeting purposes even if the exact model is newer than this list.
_FALLBACK_ENCODING = "cl100k_base"

# Per-message token overhead from OpenAI's documented counting recipe:
# every message costs a few tokens for role/name/separator wrapping on top
# of its content. Without this, count_messages_tokens() under-counts and
# a "budgeted" prompt can still overflow the real context window.
_TOKENS_PER_MESSAGE = 3
_TOKENS_PER_REPLY_PRIMER = 3


class LLMClientError(Exception):
    """Raised when a completion fails after exhausting retries, or fails
    on a non-retryable error. Callers (query.py) catch this one type
    instead of needing to know about every OpenAI exception subclass."""

    def __init__(self, message: str, *, retries_used: int = 0, cause: Exception | None = None):
        super().__init__(message)
        self.retries_used = retries_used
        self.cause = cause


@dataclass
class LLMConfig:
    """Every tunable in one place — same pattern as ExtractionConfig in
    tools/doc_extractor.py. Lets tests override behaviour without touching
    LLMClient's logic, and gives one spot to tune for prod vs dev."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_output_tokens: int = 1024
    request_timeout_seconds: float = 30.0

    # ── Retry settings ──────────────────────────────────────────────────
    max_retries: int = 3
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 20.0
    # Jitter prevents a thundering-herd retry pattern if many requests
    # hit a rate limit at the same moment.
    jitter_seconds: float = 0.5


@dataclass
class LLMResponse:
    """What every caller gets back. Token counts travel with the response
    so callers (query.py, mlops/cost_logger.py later) never need a second
    round-trip or a separate tiktoken call just to log cost."""

    text: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_seconds: float
    retries_used: int = 0
    finish_reason: str | None = None


class LLMClient:
    """
    Thin async wrapper around the OpenAI Chat Completions API.

    Usage:
        client = LLMClient(api_key=settings.OPENAI_API_KEY)
        response = await client.complete([
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
        ])
        print(response.text, response.total_tokens)
    """

    def __init__(self, api_key: str | None = None, config: LLMConfig | None = None) -> None:
        self.cfg = config or LLMConfig()
        # api_key=None lets the SDK fall back to the OPENAI_API_KEY env var —
        # same "don't force callers to thread a secret through every layer"
        # principle as everywhere else in this stack.
        self._client = AsyncOpenAI(api_key=api_key, timeout=self.cfg.request_timeout_seconds)
        self._encoding = self._resolve_encoding(self.cfg.model)

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Run a chat completion with retry on transient failures.

        Args:
            messages: OpenAI-format message list, e.g.
                [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
            model/temperature/max_output_tokens: per-call overrides of LLMConfig.
                None means "use the configured default" — same override pattern
                ExtractionConfig uses (pass a custom config, or override per call).

        Returns:
            LLMResponse

        Raises:
            LLMClientError: on auth/bad-request failures (no retry), or after
                all retries are exhausted on transient failures.
        """
        model = model or self.cfg.model
        temperature = self.cfg.temperature if temperature is None else temperature
        max_output_tokens = max_output_tokens or self.cfg.max_output_tokens

        input_tokens = self.count_messages_tokens(messages, model=model)

        start = time.perf_counter()
        last_exc: Exception | None = None

        for attempt in range(self.cfg.max_retries + 1):
            try:
                completion = await self._client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_output_tokens,
                    messages=messages,
                )
                choice = completion.choices[0]
                usage = completion.usage

                return LLMResponse(
                    text=choice.message.content or "",
                    model=completion.model,
                    input_tokens=usage.prompt_tokens if usage else input_tokens,
                    output_tokens=usage.completion_tokens if usage else self.count_tokens(
                        choice.message.content or "", model=model
                    ),
                    total_tokens=usage.total_tokens if usage else 0,
                    latency_seconds=time.perf_counter() - start,
                    retries_used=attempt,
                    finish_reason=choice.finish_reason,
                )

            except (AuthenticationError, BadRequestError) as exc:
                # Non-retryable by definition — a bad key or malformed
                # request will fail identically on attempt 2 as attempt 1.
                logger.error("LLM call failed (non-retryable): %s", exc)
                raise LLMClientError(
                    f"Non-retryable LLM error: {exc}", retries_used=attempt, cause=exc
                ) from exc

            except _RETRYABLE_EXCEPTIONS as exc:
                last_exc = exc
                if attempt >= self.cfg.max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM call failed (attempt %d/%d), retrying in %.2fs: %s",
                    attempt + 1, self.cfg.max_retries + 1, delay, exc,
                )
                await asyncio.sleep(delay)

        logger.error("LLM call failed after %d retries: %s", self.cfg.max_retries, last_exc)
        raise LLMClientError(
            f"LLM call failed after {self.cfg.max_retries} retries: {last_exc}",
            retries_used=self.cfg.max_retries,
            cause=last_exc,
        )

    def count_tokens(self, text: str, *, model: str | None = None) -> int:
        """Token count for a raw string. Used by prompt_builder.py to budget
        document context against the model's window BEFORE sending anything."""
        encoding = self._encoding if model is None else self._resolve_encoding(model)
        return len(encoding.encode(text))

    def count_messages_tokens(self, messages: list[dict[str, str]], *, model: str | None = None) -> int:
        """
        Token count for a full chat message list, including the per-message
        and per-reply-primer overhead OpenAI's API actually charges for.

        Why not just sum len(encode(content)) per message:
            That under-counts by a few tokens per message (role/name/
            separator wrapping), which compounds across a multi-page RAG
            prompt. Under-counting here means a prompt that looks "under
            budget" in prompt_builder.py can still get rejected by the API
            for exceeding the real context window.
        """
        encoding = self._encoding if model is None else self._resolve_encoding(model)
        total = 0
        for message in messages:
            total += _TOKENS_PER_MESSAGE
            for value in message.values():
                total += len(encoding.encode(value))
        total += _TOKENS_PER_REPLY_PRIMER
        return total

    # =========================================================================
    # PRIVATE
    # =========================================================================

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped at max_backoff_seconds."""
        exp_delay = self.cfg.base_backoff_seconds * (2 ** attempt)
        capped = min(exp_delay, self.cfg.max_backoff_seconds)
        return capped + random.uniform(0, self.cfg.jitter_seconds)

    @staticmethod
    def _resolve_encoding(model: str) -> Any:
        """tiktoken doesn't always recognise brand-new model names yet.
        Fall back to cl100k_base rather than raising — an approximate
        token count is far more useful than a crashed request.

        Broad except (not just KeyError) is deliberate: encoding_for_model
        can also fail on a vocab-file fetch (network blip, sandboxed/offline
        environment, first-run cache miss) — a failure mode that has nothing
        to do with whether the model name is recognised. Either way, token
        counting must never be the reason a request never goes out."""
        try:
            return tiktoken.encoding_for_model(model)
        except Exception as exc:
            logger.debug("Falling back to %s encoding for model '%s': %s", _FALLBACK_ENCODING, model, exc)
        try:
            return tiktoken.get_encoding(_FALLBACK_ENCODING)
        except Exception as exc:
            logger.warning(
                "tiktoken encoding unavailable (%s) — using approximate "
                "whitespace-based token counter instead", exc,
            )
            return _ApproxEncoding()


class _ApproxEncoding:
    """Last-resort token counter when tiktoken's vocab files are
    unreachable (offline/sandboxed environments, registry outages).

    ~4 chars/token is OpenAI's own published rule of thumb for English
    text. It will under/over-count vs. the real BPE tokenizer, so this
    path should only ever be exercised when the real encoding is
    genuinely unavailable — callers budgeting against it should keep
    extra headroom to compensate for the imprecision."""

    def encode(self, text: str) -> list[int]:
        approx_count = max(1, len(text) // 4) if text else 0
        return [0] * approx_count