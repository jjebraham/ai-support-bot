"""Settings loaded from environment variables (optionally a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    bot_token: str
    admin_chat_id: int
    anthropic_api_key: str
    claude_model: str
    mock_llm: bool
    docs_dir: Path
    min_retrieval_score: float
    top_k: int
    log_level: str

    @classmethod
    def from_env(cls, dotenv: bool = True) -> "Settings":
        if dotenv:
            load_dotenv(os.getenv("DOTENV_PATH", ".env"))
        return cls(
            bot_token=os.getenv("BOT_TOKEN", ""),
            admin_chat_id=int(os.getenv("ADMIN_CHAT_ID", "0") or 0),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            claude_model=os.getenv("CLAUDE_MODEL", "claude-opus-5-5"),
            mock_llm=_bool(os.getenv("MOCK_LLM"), default=False),
            docs_dir=Path(os.getenv("DOCS_DIR", "docs")),
            min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "2.0")),
            top_k=int(os.getenv("TOP_K", "3")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )
