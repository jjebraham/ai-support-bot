"""LLM backends: Claude (Anthropic SDK) and a deterministic mock.

Both take the question plus the retrieved chunks and return an ``LLMReply``.
A reply either answers the question or says it should go to a human.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol

import anthropic

from app.retrieval import Hit, tokenize

log = logging.getLogger(__name__)

ESCALATE_TOKEN = "ESCALATE"

SYSTEM_PROMPT = f"""You are the customer support assistant for Parcelo, a parcel delivery service.

Answer the customer's question using only the support articles in <context>.
Rules:
- If the articles do not contain the answer, or you would have to guess, reply with exactly {ESCALATE_TOKEN} and nothing else. A human agent will take over.
- Also reply {ESCALATE_TOKEN} if the customer asks for something only a human can do (an exception, a complaint about a specific order, legal threats).
- Keep answers short: 1 to 3 sentences, plain text, no Markdown headings.
- Do not invent prices, dates or policies that are not in the articles."""


@dataclass(frozen=True)
class LLMReply:
    text: str
    escalate: bool
    reason: str = ""  # why it escalated: "model_unsure", "llm_error", "refusal"


class LLMClient(Protocol):
    async def answer(self, question: str, hits: list[Hit]) -> LLMReply: ...


def format_context(hits: list[Hit]) -> str:
    parts = []
    for h in hits:
        parts.append(f'<article source="{h.chunk.source}">\n{h.chunk.title}\n{h.chunk.text}\n</article>')
    return "<context>\n" + "\n".join(parts) + "\n</context>"


class ClaudeLLM:
    """Calls the Claude Messages API with the retrieved articles as context."""

    def __init__(self, api_key: str, model: str = "claude-opus-5-5", timeout: float = 30.0):
        # The SDK retries 408/409/429/5xx and connection errors (max_retries) itself.
        self.client = anthropic.AsyncAnthropic(api_key=api_key or None, timeout=timeout, max_retries=2)
        self.model = model

    async def answer(self, question: str, hits: list[Hit]) -> LLMReply:
        started = time.monotonic()
        try:
            response = await self.client.beta.messages.create(
                model=self.model,
                max_tokens=1024,
                # Short factual answers: low effort is enough and keeps latency down.
                output_config={"effort": "low"},
                # If a safety classifier declines, the API retries on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": f"{format_context(hits)}\n\nCustomer question: {question}"}],
            )
        except anthropic.AuthenticationError:
            log.error("llm_auth_error", extra={"fields": {"model": self.model}})
            return LLMReply("", escalate=True, reason="llm_error")
        except anthropic.RateLimitError:
            log.warning("llm_rate_limited", extra={"fields": {"model": self.model}})
            return LLMReply("", escalate=True, reason="llm_error")
        except anthropic.APIStatusError as e:
            log.error(
                "llm_api_error",
                extra={"fields": {"status": e.status_code, "request_id": e.request_id, "model": self.model}},
            )
            return LLMReply("", escalate=True, reason="llm_error")
        except anthropic.APIConnectionError as e:
            log.error("llm_connection_error", extra={"fields": {"error": type(e).__name__}})
            return LLMReply("", escalate=True, reason="llm_error")

        latency_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "llm_response",
            extra={
                "fields": {
                    "model": response.model,
                    "request_id": response._request_id,
                    "stop_reason": response.stop_reason,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "latency_ms": latency_ms,
                }
            },
        )
        if response.stop_reason == "refusal":
            return LLMReply("", escalate=True, reason="refusal")

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text or text.startswith(ESCALATE_TOKEN):
            return LLMReply("", escalate=True, reason="model_unsure")
        return LLMReply(text, escalate=False)


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class MockLLM:
    """Offline stand-in for Claude, used when MOCK_LLM=true and in tests.

    It answers extractively: it picks the sentences of the best chunk that share
    the most words with the question. If the chunks cover less than
    ``min_coverage`` of the question's keywords, it escalates, like the real
    model is told to do when the articles do not contain the answer.
    """

    def __init__(self, min_coverage: float = 0.5, max_sentences: int = 2):
        self.min_coverage = min_coverage
        self.max_sentences = max_sentences

    async def answer(self, question: str, hits: list[Hit]) -> LLMReply:
        q = set(tokenize(question))
        if not hits or not q:
            return LLMReply("", escalate=True, reason="model_unsure")
        context_tokens = set()
        for h in hits:
            context_tokens.update(tokenize(f"{h.chunk.title} {h.chunk.text}"))
        coverage = len(q & context_tokens) / len(q)
        if coverage < self.min_coverage:
            return LLMReply("", escalate=True, reason="model_unsure")

        best = hits[0].chunk
        sentences = [s for s in _SENTENCE_RE.split(best.text.replace("\n", " ")) if s.strip()]
        ranked = sorted(
            range(len(sentences)),
            key=lambda i: (-len(q & set(tokenize(sentences[i]))), i),
        )
        keep = sorted(ranked[: self.max_sentences])
        return LLMReply(" ".join(sentences[i].strip() for i in keep), escalate=False)
