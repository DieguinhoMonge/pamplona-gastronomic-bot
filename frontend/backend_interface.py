"""The single contract between the Streamlit UI and the RAG backend.

See INTERFACE.md for the full description. The UI only ever calls `ask()`
from this module, which forwards to the mock or the real implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator

from frontend.config import DEFAULT_BACKEND


@dataclass
class Source:
    restaurant: str
    file: str
    page: int
    chunk: str
    score: float | None = None


@dataclass
class RAGResponse:
    answer: str | Iterator[str]
    sources: list[Source]
    grounded: bool
    debug: dict = field(default_factory=dict)


class BackendError(Exception):
    """Expected backend failure; the message is shown to the user."""


class BackendConnectionError(BackendError):
    """The LLM / embeddings service could not be reached."""


def backend_name() -> str:
    return os.getenv("PAMPLONA_BACKEND", DEFAULT_BACKEND).lower()


def ask(question: str, history: list[dict], settings: dict) -> RAGResponse:
    if backend_name() == "real":
        try:
            from rag.pipeline import ask as impl  # implemented by the backend team
        except ImportError as exc:
            raise BackendError(
                "The real RAG backend (rag/pipeline.py) is not available yet. "
                "Unset PAMPLONA_BACKEND to use the mock."
            ) from exc
    else:
        from frontend.mock_backend import ask as impl
    return impl(question, history, settings)
