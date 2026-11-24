"""The support pipeline: retrieve, decide, answer or escalate.

Two gates decide when a human takes over:

1. Retrieval gate: if the best BM25 score is below ``min_retrieval_score``,
   the knowledge base has nothing on the topic, so the LLM is not called.
2. Model gate: the LLM is told to reply ``ESCALATE`` when the articles do not
   answer the question. API errors and refusals also escalate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from app.llm import LLMClient
from app.retrieval import BM25Index

log = logging.getLogger(__name__)

ESCALATION_MESSAGE = (
    "I'm not sure about that one, so I've passed your question to our support team. "
    "A person will reply here (Monday to Friday, 09:00 to 18:00)."
)


@dataclass(frozen=True)
class Answer:
    text: str
    escalated: bool
    reason: str = ""  # "low_retrieval_score", "model_unsure", "llm_error", "refusal"
    sources: list[str] = field(default_factory=list)
    top_score: float = 0.0


class SupportAgent:
    def __init__(self, index: BM25Index, llm: LLMClient, min_retrieval_score: float = 2.0, top_k: int = 3):
        self.index = index
        self.llm = llm
        self.min_retrieval_score = min_retrieval_score
        self.top_k = top_k

    async def answer(self, question: str) -> Answer:
        started = time.monotonic()
        hits = self.index.search(question, top_k=self.top_k)
        top_score = hits[0].score if hits else 0.0
        sources = [h.chunk.source for h in hits]

        if top_score < self.min_retrieval_score:
            result = Answer(ESCALATION_MESSAGE, True, "low_retrieval_score", sources, top_score)
        else:
            reply = await self.llm.answer(question, hits)
            if reply.escalate:
                result = Answer(ESCALATION_MESSAGE, True, reply.reason, sources, top_score)
            else:
                result = Answer(reply.text, False, "", sources, top_score)

        log.info(
            "question_handled",
            extra={
                "fields": {
                    "escalated": result.escalated,
                    "reason": result.reason,
                    "top_score": round(top_score, 3),
                    "sources": sources,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                }
            },
        )
        return result
