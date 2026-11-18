"""Shared UI helpers: conversations state, settings, styling, formatting and exports."""

from __future__ import annotations

import html
import json
import re
import unicodedata
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st

from frontend.backend_interface import backend_name
from frontend.config import DEFAULT_SETTINGS

# Empty-state suggestions (label == question)
SUGGESTIONS = [
    "Where can I eat good pintxos?",
    "What's in the Hamburguesa Babylon?",
    "Tell me about Café Iruña",
    "A vegetarian burger in Pamplona?",
]

# Demo scenarios for the video / technical defense (label -> question)
DEMO_QUESTIONS: dict[str, str] = {
    "Grounded answer": "What does the Hamburguesa Babylon contain?",
    "Follow-up (memory)": "And how much is it?",
    "Restaurant info": "Tell me about Café Iruña",
    "Out of scope (Madrid)": "Which restaurant has the best sushi in Madrid?",
    "Off-topic": "What is the capital of France?",
    "Nonsense": "asdkjh qwe?",
}

SPEEDS = {"Slow": 0.06, "Normal": 0.025, "Fast": 0.008, "Instant": 0.0}

ICONS = {
    "search": '<svg viewBox="0 0 24 24"><path d="M15.5 14h-.79l-.28-.27A6.47 6.47 0 0 0 16 9.5 6.5 6.5 0 1 0 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/></svg>',
    "check": '<svg viewBox="0 0 24 24"><path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>',
    "block": '<svg viewBox="0 0 24 24"><path d="M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20zM4 12a8 8 0 0 1 12.9-6.3L5.7 16.9A7.9 7.9 0 0 1 4 12zm8 8a7.9 7.9 0 0 1-4.9-1.7L18.3 7.1A8 8 0 0 1 12 20z"/></svg>',
    "cloud_off": '<svg viewBox="0 0 24 24"><path d="M19.35 10.04A7.49 7.49 0 0 0 12 4c-1.48 0-2.85.43-4.01 1.17l1.46 1.46A5.47 5.47 0 0 1 17.5 11.5v.5H19a3 3 0 0 1 3 3c0 1.13-.64 2.11-1.56 2.62l1.45 1.45A4.98 4.98 0 0 0 24 15c0-2.64-2.05-4.78-4.65-4.96zM3 5.27l2.75 2.74C2.56 8.15 0 10.77 0 14a6 6 0 0 0 6 6h11.73l2 2L21 20.73 4.27 4 3 5.27zM7.73 10l8 8H6a4 4 0 0 1 0-8h1.73z"/></svg>',
    "copy": '<svg class="cp" viewBox="0 0 24 24"><path d="M16 1H4a2 2 0 0 0-2 2v14h2V3h12V1zm3 4H8a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h11a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2zm0 16H8V7h11v14z"/></svg>',
    "ok": '<svg class="ok" viewBox="0 0 24 24"><path d="M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/></svg>',
}


# --------------------------------------------------------------------------- state

def _new_conv() -> dict:
    return {"id": uuid.uuid4().hex[:8], "title": None, "messages": [],
            "created": datetime.now().isoformat(timespec="seconds")}


def init_state() -> None:
    ss = st.session_state
    if "convs" not in ss:
        first = _new_conv()
        ss.convs = {first["id"]: first}
        ss.active = first["id"]
    ss.setdefault("pending_question", None)
    ss.setdefault("pending_regen", False)
    for key, value in DEFAULT_SETTINGS.items():
        ss.setdefault(f"set_{key}", value)
    ss.setdefault("set_simulate_failure", False)
    ss.setdefault("set_speed", "Normal")


def active_conv() -> dict:
    return st.session_state.convs[st.session_state.active]


def messages() -> list[dict]:
    return active_conv()["messages"]


def new_chat() -> None:
    if not messages():  # current chat is already empty
        return
    conv = _new_conv()
    st.session_state.convs[conv["id"]] = conv
    st.session_state.active = conv["id"]


def select_chat(cid: str) -> None:
    st.session_state.active = cid


def delete_active_chat() -> None:
    ss = st.session_state
    for m in messages():
        ss.pop(f"fb_{m['id']}", None)
    del ss.convs[ss.active]
    if not ss.convs:
        conv = _new_conv()
        ss.convs[conv["id"]] = conv
    ss.active = next(reversed(ss.convs))


def set_title_from(question: str) -> None:
    conv = active_conv()
    if not conv["title"]:
        q = question.strip().rstrip("?")
        conv["title"] = q if len(q) <= 38 else q[:36].rstrip() + "…"


def new_message(role: str, content: str, **extra) -> dict:
    return {"id": uuid.uuid4().hex[:10], "role": role, "content": content,
            "ts": datetime.now().isoformat(timespec="seconds"), **extra}


def chat_history(msgs: list[dict]) -> list[dict]:
    """History in the format the backend expects (errors excluded)."""
    return [{"role": m["role"], "content": m["content"]} for m in msgs if not m.get("error")]


def queue_question(question: str) -> None:
    st.session_state.pending_question = question


def queue_from_pill(key: str) -> None:
    """on_change callback for suggestion pills."""
    value = st.session_state.get(key)
    if value:
        st.session_state.pending_question = value
        st.session_state[key] = None


def queue_regenerate() -> None:
    st.session_state.pending_regen = True


# --------------------------------------------------------------------------- settings

def sync_setting(name: str) -> None:
    """Copy a dialog widget value (w_*) into its persistent key (set_*).

    Widget keys are dropped by Streamlit when the dialog closes, so the real
    values live in set_* keys.
    """
    st.session_state[f"set_{name}"] = st.session_state[f"w_{name}"]


def prime_widget(name: str) -> None:
    st.session_state[f"w_{name}"] = st.session_state[f"set_{name}"]


def current_settings() -> dict:
    ss = st.session_state
    settings = {"model": ss.set_model, "top_k": ss.set_top_k, "temperature": ss.set_temperature}
    if backend_name() == "mock":
        settings |= {"simulate_failure": ss.set_simulate_failure,
                     "stream_delay": SPEEDS[ss.set_speed]}
    return settings


# --------------------------------------------------------------------------- styling / formatting

def load_css() -> None:
    css = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")
    st.html(f"<style>{css}</style>")


def format_answer(text: str) -> str:
    """Escape raw HTML and turn [n] citations into superscript chips."""
    safe = text.replace("<", "&lt;")
    return re.sub(r"\[(\d{1,2})\]", r'<sup class="cite">\1</sup>', safe)


_MD_SPECIAL = re.compile(r"([\\`*_\[\]<>#|$~])")


def _fold(ch: str) -> str:
    base = unicodedata.normalize("NFKD", ch)
    return (base[0] if base else ch).lower()


def _md_escape(text: str) -> str:
    return _MD_SPECIAL.sub(r"\\\1", text)


def highlight(text: str, terms: list[str]) -> str:
    """Escape markdown and highlight query terms (accent-insensitive, prefix match)."""
    terms = [t for t in terms if len(t) >= 4]
    folded = "".join(_fold(c) for c in text)  # same length as text
    spans: list[tuple[int, int]] = []
    if terms:
        pattern = re.compile(r"\b(?:" + "|".join(re.escape(t[:max(4, len(t) - 1)]) for t in terms)
                             + r")\w*")
        spans = [m.span() for m in pattern.finditer(folded)]
    out, pos = [], 0
    for a, b in spans:
        out.append(_md_escape(text[pos:a]))
        out.append(f":red-background[{_md_escape(text[a:b])}]")
        pos = b
    out.append(_md_escape(text[pos:]))
    return "".join(out)


def copy_button(text: str) -> None:
    data = html.escape(text, quote=True)
    st.html(
        f'<button class="txk-copy" title="Copy answer" data-t="{data}" '
        "onclick=\"navigator.clipboard.writeText(this.dataset.t);this.classList.add('done');"
        "setTimeout(()=>this.classList.remove('done'),1400)\">"
        f'{ICONS["copy"]}{ICONS["ok"]}</button>',
        width="content", unsafe_allow_javascript=True,
    )


def feedback_counts() -> tuple[int, int]:
    up = down = 0
    for conv in st.session_state.convs.values():
        for m in conv["messages"]:
            v = st.session_state.get(f"fb_{m['id']}")
            up += v == 1
            down += v == 0
    return up, down


# --------------------------------------------------------------------------- export

def export_markdown() -> str:
    conv = active_conv()
    lines = [f"# {conv['title'] or 'Conversation'}",
             f"_Txoko · Pamplona Gastronomic Assistant · exported {datetime.now():%Y-%m-%d %H:%M}_", ""]
    for m in conv["messages"]:
        lines += [f"**{'You' if m['role'] == 'user' else 'Txoko'}:** {m['content']}", ""]
        for i, s in enumerate(m.get("sources") or [], 1):
            lines.append(f"> [{i}] {s['restaurant']} ({s['file']}, p.{s['page']})")
        lines.append("")
    return "\n".join(lines)


def export_json() -> str:
    data = []
    for m in messages():
        item = {k: v for k, v in m.items() if k != "debug"}
        item["feedback"] = {1: "up", 0: "down"}.get(st.session_state.get(f"fb_{m['id']}"))
        data.append(item)
    return json.dumps({"title": active_conv()["title"], "messages": data},
                      ensure_ascii=False, indent=2)
