"""Ask the support agent a question from the terminal, without Telegram.

    MOCK_LLM=true python -m app.cli "How long does a refund take?"
"""

from __future__ import annotations

import asyncio
import sys

from app.bot import build_agent
from app.config import Settings
from app.logging_setup import setup_logging


async def run(question: str) -> None:
    settings = Settings.from_env()
    setup_logging("WARNING")
    agent = build_agent(settings)
    result = await agent.answer(question)
    status = f"ESCALATED ({result.reason})" if result.escalated else "ANSWERED"
    print(f"[{status}] top_score={result.top_score:.2f} sources={result.sources}")
    print(result.text)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('usage: python -m app.cli "your question"')
    asyncio.run(run(" ".join(sys.argv[1:])))
