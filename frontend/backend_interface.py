"""The single contract between the Streamlit UI and the RAG backend.

See INTERFACE.md for the full description. The UI only ever calls `ask()`
from this module, which forwards to the mock or the real implementation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterator

from frontend.config import DEFAULT_BACKEND, ROOT

try:  # load .env (GOOGLE_API_KEY, PAMPLONA_BACKEND) before anything reads the environment
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

_PLACEHOLDER_KEYS = {"", "tu_api_key_aqui"}


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


def has_api_key() -> bool:
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or ""
    return key.strip() not in _PLACEHOLDER_KEYS


def backend_name() -> str:
    """'real' or 'mock'. 'auto' picks the real backend when an API key is configured."""
    name = os.getenv("PAMPLONA_BACKEND", DEFAULT_BACKEND).strip().lower()
    if name == "auto":
        return "real" if has_api_key() else "mock"
    return name


def ask(question: str, history: list[dict], settings: dict) -> RAGResponse:
    if backend_name() == "real":
        try:
            from rag.pipeline import ask as impl  # implemented by the backend team
        except ImportError as exc:
            raise BackendError(
                "The real RAG backend (rag/pipeline.py) is not available yet. "
                f"Unset PAMPLONA_BACKEND to use the mock. ({exc})"
            ) from exc
    else:
        from frontend.mock_backend import ask as impl
    return impl(question, history, settings)
