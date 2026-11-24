"""Telegram handler tests with aiogram's Bot replaced by a mock."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.agent import Answer
from app.bot import EscalationStore, build_router

ADMIN = -100123


class StubAgent:
    def __init__(self, answer: Answer):
        self._answer = answer

    async def answer(self, question):
        return self._answer


def handler(router, name):
    for h in router.message.handlers:
        if h.callback.__name__ == name:
            return h.callback
    raise LookupError(name)


def user_message(text="hi", chat_id=42):
    return SimpleNamespace(
        text=text,
        chat=SimpleNamespace(id=chat_id),
        from_user=SimpleNamespace(username="alice"),
        answer=AsyncMock(),
    )


async def test_answer_is_sent_to_user_and_admin_not_bothered():
    router = build_router(StubAgent(Answer("Within 7 days.", False)), ADMIN)
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message=AsyncMock())
    msg = user_message("When is my refund?")
    await handler(router, "question")(msg, bot)
    msg.answer.assert_awaited_once_with("Within 7 days.")
    bot.send_message.assert_not_awaited()


async def test_escalation_goes_to_admin_and_reply_is_relayed():
    store = EscalationStore()
    router = build_router(StubAgent(Answer("passed on", True, "model_unsure")), ADMIN, store)
    bot = SimpleNamespace(send_chat_action=AsyncMock(), send_message=AsyncMock(return_value=SimpleNamespace(message_id=777)))
    await handler(router, "question")(user_message("Bitcoin?", chat_id=42), bot)

    admin_chat, admin_text = bot.send_message.await_args.args
    assert admin_chat == ADMIN
    assert "Bitcoin?" in admin_text and "model_unsure" in admin_text
    assert store.user_for(777) == 42

    bot.send_message.reset_mock()
    reply = SimpleNamespace(text="No, card or PayPal only.", reply_to_message=SimpleNamespace(message_id=777))
    await handler(router, "admin_reply")(reply, bot)
    bot.send_message.assert_awaited_once_with(42, "Support team: No, card or PayPal only.")


async def test_admin_reply_to_unknown_message_is_ignored():
    router = build_router(StubAgent(Answer("x", False)), ADMIN)
    bot = SimpleNamespace(send_message=AsyncMock())
    reply = SimpleNamespace(text="hello", reply_to_message=SimpleNamespace(message_id=1))
    await handler(router, "admin_reply")(reply, bot)
    bot.send_message.assert_not_awaited()
