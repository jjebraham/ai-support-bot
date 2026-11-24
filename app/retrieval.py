"""Small BM25 retriever over Markdown files.

Each ``docs/*.md`` file is split into chunks at ``##`` headings, so one chunk is
one topic ("Refund timing", "Reset your password", ...). Chunks are scored with
Okapi BM25. No external search service or embedding model is needed, and the
ranking is deterministic, which keeps tests and the eval reproducible.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    """a an and are as at be by can do does for from get got have how i if in into
    is it its me my of on or our so than that the their then there this to was
    we what when where which who why will with you your""".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        # very light stemming: "parcels" -> "parcel", "refunds" -> "refund"
        if len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


@dataclass(frozen=True)
class Chunk:
    doc: str  # file name, e.g. "returns.md"
    title: str  # section heading
    text: str  # section body

    @property
    def source(self) -> str:
        return f"{self.doc}#{self.title}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


def split_markdown(doc_name: str, markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    title = None
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if title and body:
            chunks.append(Chunk(doc=doc_name, title=title, text=body))

    for line in markdown.splitlines():
        if line.startswith("## "):
            flush()
            title = line[3:].strip()
            lines = []
        elif line.startswith("# "):
            continue
        elif title is not None:
            lines.append(line)
    flush()
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("cannot build an index with no chunks")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        # the heading is part of the indexed text: it is a strong topic signal
        self._docs = [tokenize(f"{c.title} {c.title} {c.text}") for c in chunks]
        self._tf = [Counter(d) for d in self._docs]
        self._avgdl = sum(len(d) for d in self._docs) / len(self._docs)
        df: Counter[str] = Counter()
        for d in self._docs:
            df.update(set(d))
        n = len(self._docs)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    @classmethod
    def from_dir(cls, path: Path) -> "BM25Index":
        chunks: list[Chunk] = []
        for file in sorted(Path(path).glob("*.md")):
            chunks.extend(split_markdown(file.name, file.read_text(encoding="utf-8")))
        return cls(chunks)

    def score(self, query_tokens: list[str], i: int) -> float:
        tf = self._tf[i]
        dl = len(self._docs[i])
        total = 0.0
        for t in query_tokens:
            f = tf.get(t)
            if not f:
                continue
            idf = self._idf[t]
            total += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
        return total

    def search(self, query: str, top_k: int = 3) -> list[Hit]:
        q = tokenize(query)
        if not q:
            return []
        hits = [Hit(c, self.score(q, i)) for i, c in enumerate(self.chunks)]
        hits = [h for h in hits if h.score > 0]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]
