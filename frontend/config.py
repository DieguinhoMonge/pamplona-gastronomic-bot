"""Central configuration for the Streamlit frontend."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data"          # the 20 restaurant PDFs (shared with the RAG backend)
CHROMA_DIR = ROOT / "chroma_db"   # persisted vector store, built on first use (git-ignored)

APP_TITLE = "Pamplona Gastronomic Assistant"
BOT_NAME = "Txoko"  # a 'txoko' is a traditional Basque-Navarrese gastronomic society
APP_TAGLINE = "Your guide to the restaurants of Pamplona and Navarra, grounded in real documents."

# Gemini model IDs shown in Settings. The first one is the default and must match
# the model the backend was tested with (src/rag_engine.py).
MODEL_OPTIONS = ["gemini-3.5-flash-lite", "gemini-2.5-flash"]

DEFAULT_SETTINGS = {
    "model": MODEL_OPTIONS[0],
    "top_k": 4,
    "temperature": 0.2,
}

# Which backend implementation frontend/backend_interface.py loads:
#   "auto" (default): the real RAG backend when GOOGLE_API_KEY is set in .env, else the mock
#   "real" / "mock":  force one. Can be overridden with the env var PAMPLONA_BACKEND.
DEFAULT_BACKEND = "auto"

BOT_AVATAR = "🍷"
USER_AVATAR = "🧑"
