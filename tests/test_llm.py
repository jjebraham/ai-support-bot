"""ClaudeLLM tests. The Anthropic client is replaced, so no network is used."""

from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest

from app.llm import ESCALATE_TOKEN, SYSTEM_PROMPT, ClaudeLLM, MockLLM, format_context


def make_response(text: str, stop_reason: str = "end_turn"):
    resp = SimpleNamespace(
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20),
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
    )
    resp._request_id = "req_test"
    return resp


class FakeMessages:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


def claude_with(messages: FakeMessages) -> ClaudeLLM:
    llm = ClaudeLLM(api_key="test-key")
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return llm


@pytest.fixture
def hits(index):
    return index.search("When will I get my refund?")


async def test_claude_answer_and_request_shape(hits):
    messages = FakeMessages(make_response("Within 7 business days."))
    reply = await claude_with(messages).answer("When will I get my refund?", hits)
    assert reply.text == "Within 7 business days." and not reply.escalate
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5"
    assert kw["system"] == SYSTEM_PROMPT
    assert "returns.md#Refund timing" in kw["messages"][0]["content"]
    assert kw["fallbacks"] == "default"


async def test_claude_escalate_token(hits):
    reply = await claude_with(FakeMessages(make_response(ESCALATE_TOKEN))).answer("q", hits)
    assert reply.escalate and reply.reason == "model_unsure"


async def test_claude_refusal_escalates(hits):
    reply = await claude_with(FakeMessages(make_response("", stop_reason="refusal"))).answer("q", hits)
    assert reply.escalate and reply.reason == "refusal"


def _status_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(status, request=request, headers={"request-id": "req_err"})
    return cls("boom", response=response, body=None)


@pytest.mark.parametrize(
    "error",
    [
        _status_error(anthropic.RateLimitError, 429),
        _status_error(anthropic.InternalServerError, 500),
        _status_error(anthropic.AuthenticationError, 401),
        anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")),
    ],
)
async def test_claude_api_errors_escalate(hits, error):
    reply = await claude_with(FakeMessages(error=error)).answer("q", hits)
    assert reply.escalate and reply.reason == "llm_error"


def test_format_context_tags_sources(hits):
    ctx = format_context(hits)
    assert ctx.startswith("<context>") and 'source="returns.md#Refund timing"' in ctx


async def test_mock_escalates_when_docs_do_not_cover_question(index):
    hits = index.search("Can I send a refund to Mars by rocket?")
    reply = await MockLLM().answer("Can I send a refund to Mars by rocket?", hits)
    assert reply.escalate
