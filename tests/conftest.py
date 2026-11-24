from pathlib import Path

import pytest

from app.retrieval import BM25Index

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def index() -> BM25Index:
    return BM25Index.from_dir(ROOT / "docs")
