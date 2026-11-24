"""Telegram layer (aiogram 3).

* Users send a question, the bot answers from the docs or escalates.
* Escalations are posted to ADMIN_CHAT_ID. When an admin replies to that post,
  the reply is sent back to the user.
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from app.agent import SupportAgent
from app.config import Settings
from app.llm import ClaudeLLM, LLMClient, MockLLM
from app.logging_setup import setup_logging
from app.retrieval import BM25Index

log = logging.getLogger(__name__)

WELCOME = (
    "Hi! I'm the Parcelo support bot. Ask me about shipping, tracking, returns, "
    "billing or your account. If I don't know, I'll pass you to a person."
)


class EscalationStore:
    """Maps the admin-chat message id of an escalation to the user's chat id.

    In memory on purpose: this is a demo. A production bot would keep this in
    Redis or a database so it survives restarts.
    """

    def __init__(self) -> None:
        self._by_admin_msg: dict[int, int] = {}

    def add(self, admin_message_id: int, user_chat_id: int) -> None:
        self._by_admin_msg[admin_message_id] = user_chat_id

    def user_for(self, admin_message_id: int) -> int | None:
        return self._by_admin_msg.get(admin_message_id)


def build_router(agent: SupportAgent, admin_chat_id: int, store: EscalationStore | None = None) -> Router:
    router = Router()
    store = store or EscalationStore()

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(WELCOME)

    @router.message(Command("help"))
    async def help_(message: Message) -> None:
        await message.answer(WELCOME)

    # Admin replies to an escalation -> relay to the user.
    @router.message(F.chat.id == admin_chat_id, F.reply_to_message, F.text)
    async def admin_reply(message: Message, bot: Bot) -> None:
        user_chat_id = store.user_for(message.reply_to_message.message_id)
        if user_chat_id is None:
            return
        await bot.send_message(user_chat_id, f"Support team: {message.text}")
        log.info("admin_reply_relayed", extra={"fields": {"user_chat_id": user_chat_id}})

    @router.message(F.text, F.chat.id != admin_chat_id)
    async def question(message: Message, bot: Bot) -> None:
        await bot.send_chat_action(message.chat.id, "typing")
        result = await agent.answer(message.text)
        await message.answer(result.text)
        if result.escalated:
            if not admin_chat_id:
                log.warning("escalation_dropped_no_admin_chat")
                return
            user = message.from_user
            who = f"@{user.username}" if user and user.username else f"user {message.chat.id}"
            posted = await bot.send_message(
                admin_chat_id,
                f"Escalated question from {who} (reason: {result.reason}):\n\n{message.text}\n\n"
                "Reply to this message to answer the user.",
            )
            store.add(posted.message_id, message.chat.id)

    return router


def build_llm(settings: Settings) -> LLMClient:
    if settings.mock_llm:
        return MockLLM()
    if not settings.anthropic_api_key:
        raise SystemExit("ANTHROPIC_API_KEY is not set. Set it, or run with MOCK_LLM=true.")
    return ClaudeLLM(api_key=settings.anthropic_api_key, model=settings.claude_model)


def build_agent(settings: Settings) -> SupportAgent:
    index = BM25Index.from_dir(settings.docs_dir)
    return SupportAgent(index, build_llm(settings), settings.min_retrieval_score, settings.top_k)


async def main() -> None:
    settings = Settings.from_env()
    setup_logging(settings.log_level)
    if not settings.bot_token:
        raise SystemExit("BOT_TOKEN is not set. Copy .env.example to .env and fill it in.")
    agent = build_agent(settings)
    bot = Bot(settings.bot_token)
    dp = Dispatcher()
    dp.include_router(build_router(agent, settings.admin_chat_id))
    log.info(
        "bot_starting",
        extra={"fields": {"mock_llm": settings.mock_llm, "model": settings.claude_model, "chunks": len(agent.index.chunks)}},
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
