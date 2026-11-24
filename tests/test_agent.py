from app.agent import ESCALATION_MESSAGE, SupportAgent
from app.llm import LLMReply, MockLLM


class FakeLLM:
    """Records calls and returns a fixed reply."""

    def __init__(self, reply: LLMReply):
        self.reply = reply
        self.calls = []

    async def answer(self, question, hits):
        self.calls.append((question, hits))
        return self.reply


async def test_answers_from_docs_with_mock_llm(index):
    agent = SupportAgent(index, MockLLM())
    result = await agent.answer("How many days do I have to return an item?")
    assert not result.escalated
    assert "30 days" in result.text
    assert result.sources[0] == "returns.md#Return window"


async def test_low_retrieval_score_escalates_without_calling_llm(index):
    llm = FakeLLM(LLMReply("should not be used", escalate=False))
    agent = SupportAgent(index, llm, min_retrieval_score=2.0)
    result = await agent.answer("What is the weather in Istanbul?")
    assert result.escalated
    assert result.reason == "low_retrieval_score"
    assert result.text == ESCALATION_MESSAGE
    assert llm.calls == []


async def test_model_unsure_escalates(index):
    agent = SupportAgent(index, FakeLLM(LLMReply("", escalate=True, reason="model_unsure")))
    result = await agent.answer("Can I pay my invoice with Bitcoin?")
    assert result.escalated and result.reason == "model_unsure"


async def test_llm_error_escalates_with_friendly_message(index):
    agent = SupportAgent(index, FakeLLM(LLMReply("", escalate=True, reason="llm_error")))
    result = await agent.answer("When will I get my refund?")
    assert result.escalated
    assert result.reason == "llm_error"
    assert result.text == ESCALATION_MESSAGE


async def test_llm_gets_top_k_hits(index):
    llm = FakeLLM(LLMReply("ok", escalate=False))
    agent = SupportAgent(index, llm, top_k=2)
    await agent.answer("refund shipping fees damaged")
    _, hits = llm.calls[0]
    assert len(hits) == 2
