import pytest

from app.retrieval import BM25Index, split_markdown, tokenize


def test_tokenize_drops_stopwords_and_plurals():
    assert tokenize("How do I return my Parcels?") == ["return", "parcel"]


def test_split_markdown_uses_h2_sections():
    md = "# Title\n\nintro ignored\n\n## First\nOne.\n\n## Second\nTwo.\n"
    chunks = split_markdown("x.md", md)
    assert [(c.title, c.text) for c in chunks] == [("First", "One."), ("Second", "Two.")]
    assert chunks[0].source == "x.md#First"


@pytest.mark.parametrize(
    "question, source",
    [
        ("How do I reset my password?", "account.md#Reset your password"),
        ("refund timing", "returns.md#Refund timing"),
        ("what does a tracking number look like", "tracking.md#Tracking number"),
        ("Which payment methods do you accept?", "billing.md#Payment methods"),
    ],
)
def test_search_ranks_expected_section_first(index, question, source):
    hits = index.search(question)
    assert hits[0].chunk.source == source
    assert hits == sorted(hits, key=lambda h: h.score, reverse=True)


def test_unrelated_question_has_no_hits(index):
    assert index.search("weather Istanbul") == []


def test_empty_index_is_rejected():
    with pytest.raises(ValueError):
        BM25Index([])
