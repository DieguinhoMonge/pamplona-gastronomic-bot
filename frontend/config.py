"""Central configuration for the Streamlit frontend."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "Restaurant docs-20261108T214532Z-1-001" / "Restaurant docs"

APP_TITLE = "Pamplona Gastronomic Assistant"
BOT_NAME = "Txoko"  # a 'txoko' is a traditional Basque-Navarrese gastronomic society
APP_TAGLINE = "Your guide to the restaurants of Pamplona and Navarra, grounded in real documents."

# Gemini model IDs shown in the sidebar. The backend team should replace these
# with the exact model IDs they use from Google AI Studio / the Gemini API.
MODEL_OPTIONS = ["gemini-2.5-flash", "gemini-2.5-pro"]

DEFAULT_SETTINGS = {
    "model": MODEL_OPTIONS[0],
    "top_k": 4,
    "temperature": 0.2,
}

# Which backend implementation frontend/backend_interface.py loads:
# "mock" (default) or "real". Can be overridden with the env var PAMPLONA_BACKEND.
DEFAULT_BACKEND = "mock"

BOT_AVATAR = "🍷"
USER_AVATAR = "🧑"
