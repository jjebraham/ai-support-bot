"""Reproducible eval for the support agent.

    python -m eval.run_eval              # mock LLM, no API key, fully deterministic
    python -m eval.run_eval --live       # real Claude API (needs ANTHROPIC_API_KEY)
    python -m eval.run_eval --markdown   # print the results table for the README

Each line of eval/questions.jsonl says whether the bot should answer or
escalate. For "answer" items it also gives the doc section that must be the top
retrieval hit and keywords that must appear in the answer.

Metrics:
* accuracy: share of questions handled correctly (right decision, and for
  answers, right source and keywords present)
* escalation rate: share of all questions sent to a human
* retrieval hit@1: for answerable questions, top hit is the expected section
* false escalations: answerable questions sent to a human
* missed escalations: questions that should go to a human but were answered
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from app.agent import SupportAgent
from app.llm import ClaudeLLM, MockLLM
from app.retrieval import BM25Index

ROOT = Path(__file__).resolve().parent.parent


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def grade(item: dict, result) -> tuple[bool, str]:
    if item["expect"] == "escalate":
        return (result.escalated, "" if result.escalated else "missed_escalation")
    if result.escalated:
        return False, f"false_escalation({result.reason})"
    if not result.sources or result.sources[0] != item["source"]:
        return False, "wrong_source"
    missing = [k for k in item["keywords"] if k.lower() not in result.text.lower()]
    if missing:
        return False, f"missing_keywords{missing}"
    return True, ""


async def run(live: bool, questions_path: Path) -> dict:
    index = BM25Index.from_dir(ROOT / "docs")
    if live:
        llm = ClaudeLLM(os.environ["ANTHROPIC_API_KEY"], os.getenv("CLAUDE_MODEL", "claude-opus-5-5"))
    else:
        llm = MockLLM()
    agent = SupportAgent(index, llm, float(os.getenv("MIN_RETRIEVAL_SCORE", "2.0")), int(os.getenv("TOP_K", "3")))

    items = load_questions(questions_path)
    rows = []
    for item in items:
        result = await agent.answer(item["question"])
        ok, why = grade(item, result)
        rows.append({"id": item["id"], "expect": item["expect"], "ok": ok, "why": why, "escalated": result.escalated,
                     "top_source": result.sources[0] if result.sources else None})

    answerable = [r for r, i in zip(rows, items) if i["expect"] == "answer"]
    should_escalate = [r for r in rows if r["expect"] == "escalate"]
    n = len(rows)
    return {
        "mode": "live" if live else "mock",
        "questions": n,
        "accuracy": sum(r["ok"] for r in rows) / n,
        "escalation_rate": sum(r["escalated"] for r in rows) / n,
        "retrieval_hit_at_1": sum(r["top_source"] == i["source"] for r, i in zip(rows, items) if i["expect"] == "answer")
        / len(answerable),
        "false_escalations": sum(r["escalated"] for r in answerable),
        "missed_escalations": sum(not r["escalated"] for r in should_escalate),
        "answerable": len(answerable),
        "should_escalate": len(should_escalate),
        "failures": [r for r in rows if not r["ok"]],
    }


def to_markdown(report: dict) -> str:
    return "\n".join(
        [
            "| Metric | Value |",
            "| --- | --- |",
            f"| Questions | {report['questions']} ({report['answerable']} answerable, {report['should_escalate']} should escalate) |",
            f"| Accuracy | {report['accuracy']:.0%} |",
            f"| Escalation rate | {report['escalation_rate']:.0%} |",
            f"| Retrieval hit@1 | {report['retrieval_hit_at_1']:.0%} |",
            f"| False escalations | {report['false_escalations']} |",
            f"| Missed escalations | {report['missed_escalations']} |",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="call the real Claude API")
    parser.add_argument("--markdown", action="store_true", help="print a Markdown table")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.jsonl")
    parser.add_argument("--min-accuracy", type=float, default=0.0, help="exit 1 if accuracy is below this")
    args = parser.parse_args()

    report = asyncio.run(run(args.live, args.questions))
    if args.markdown:
        print(to_markdown(report))
    else:
        print(json.dumps(report, indent=2))
    return 1 if report["accuracy"] < args.min_accuracy else 0


if __name__ == "__main__":
    sys.exit(main())
