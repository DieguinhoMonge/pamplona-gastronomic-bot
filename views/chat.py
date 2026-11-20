"""Chat page: an answer-engine style conversation over the restaurant documents."""

import re
import time
from dataclasses import asdict

import streamlit as st

from frontend.backend_interface import BackendConnectionError, BackendError, ask
from frontend.restaurant_data import load_restaurants
from frontend.ui import (ICONS, SUGGESTIONS, chat_history, copy_button, current_settings,
                         format_answer, highlight, messages, new_message, queue_from_pill,
                         queue_regenerate, set_title_from)

INPUT_PLACEHOLDER = "Ask about restaurants, dishes, prices…"


@st.cache_data(show_spinner=False)
def n_restaurants() -> int:
    return len(load_restaurants())


# ----------------------------------------------------------------------------- pieces
def progress_line(text: str, running: bool = False, icon: str = "search") -> str:
    cls = "txk-progress running" if running else "txk-progress"
    return f'<div class="{cls}">{ICONS[icon]}<span>{text}</span></div>'


def render_sources(msg: dict) -> None:
    sources = msg.get("sources") or []
    if not sources:
        return
    terms = msg.get("debug", {}).get("expanded_terms") or re.findall(r"\w+", msg.get("question", ""))
    st.html('<div class="txk-label">Sources</div>')
    rel_css = []
    with st.container(horizontal=True, gap="small", key=f"srcs_{msg['id']}"):
        for i, s in enumerate(sources, 1):
            rel = min(max(float(s.get("score") or 0), 0.0), 1.0)
            rel_css.append(f".st-key-src_{msg['id']}_{i}{{--rel:{rel:.3f}}}")
            with st.container(width=172, key=f"src_{msg['id']}_{i}"):
                with st.popover(f"**{i}** · {s['restaurant']}", key=f"pop_{msg['id']}_{i}",
                                width="stretch"):
                    st.html(f'<div class="txk-src-head">{s["file"]} · page {s["page"]}</div>')
                    if s.get("score") is not None:
                        st.progress(rel, text=f"Relevance {rel:.0%}")
                    st.markdown(highlight(s["chunk"], terms))
    st.html(f"<style>{''.join(rel_css)}</style>")


def render_summary(msg: dict) -> None:
    n = len(msg.get("sources") or [])
    if not msg.get("grounded", True):
        st.html(f'<div class="txk-notfound">{ICONS["block"]}<span>Not found in the knowledge base'
                "</span></div>")
    elif n:
        bits = [f"Reviewed {n} source{'s' * (n != 1)}"]
        if msg.get("elapsed") is not None:
            bits.append(f"{msg['elapsed']:.1f}s")
        if msg.get("debug", {}).get("rewritten_query"):
            bits.append("follow-up resolved")
        st.html(progress_line(" · ".join(bits), icon="check"))


def render_actions(msg: dict, is_last: bool) -> None:
    with st.container(horizontal=True, gap="small", vertical_alignment="center",
                      key=f"act_{msg['id']}"):
        copy_button(msg["content"])
        st.feedback("thumbs", key=f"fb_{msg['id']}")
        if is_last:
            st.button("", icon=":material/refresh:", type="tertiary", help="Regenerate",
                      key=f"regen_{msg['id']}", on_click=queue_regenerate)


def render_user(msg: dict) -> None:
    with st.container(key=f"u_{msg['id']}"):
        st.markdown(msg["content"])


def render_assistant(msg: dict, is_last: bool) -> None:
    if msg.get("error"):
        with st.container(key=f"err_{msg['id']}"):
            st.html(f'<div class="txk-err">{ICONS["cloud_off"]}<span>{msg["content"]}</span></div>')
            if is_last:
                st.button("Retry", icon=":material/refresh:", key=f"retry_{msg['id']}",
                          on_click=queue_regenerate)
        return
    with st.container(key=f"a_{msg['id']}"):
        render_summary(msg)
        render_sources(msg)
        if msg.get("sources"):
            st.html('<div class="txk-label">Answer</div>')
        st.markdown(format_answer(msg["content"]), unsafe_allow_html=True)
        render_actions(msg, is_last)


def render_empty() -> str | None:
    st.html(f"""
<div class="txk-hero">
  <div class="txk-wordmark">Txoko</div>
  <div class="txk-sub">Ask about {n_restaurants()} restaurants in Pamplona and Navarra.
  Every answer is grounded in their documents.</div>
</div>""")
    with st.container(key="hero_input"):
        typed = st.chat_input(INPUT_PLACEHOLDER, key="chat_input")
    with st.container(horizontal=True, horizontal_alignment="center"):
        st.pills("Suggestions", SUGGESTIONS, key="suggest_pill", label_visibility="collapsed",
                 on_change=queue_from_pill, args=("suggest_pill",))
    return typed


# ----------------------------------------------------------------------------- generation
def generate(question: str, history: list[dict]) -> dict:
    msg = new_message("assistant", "", question=question)
    t0 = time.perf_counter()
    with st.container(key=f"a_{msg['id']}"):
        status = st.empty()
        status.html(progress_line(f"Searching {n_restaurants()} restaurant menus…", running=True))
        try:
            response = ask(question, history, current_settings())
            msg |= {"grounded": response.grounded, "debug": response.debug or {},
                    "sources": [asdict(s) for s in response.sources]}
            n = len(response.sources)
            if n:
                status.html(progress_line(f"Reading {n} source{'s' * (n != 1)}…", running=True))
            else:
                status.empty()
            render_sources(msg)
            if n:
                st.html('<div class="txk-label">Answer</div>')
            body = st.empty()
            text = ""
            chunks = [response.answer] if isinstance(response.answer, str) else response.answer
            for piece in chunks:
                text += str(piece)
                body.markdown(format_answer(text) + '<span class="txk-caret"></span>',
                              unsafe_allow_html=True)
            body.markdown(format_answer(text), unsafe_allow_html=True)
            msg["content"] = text
            msg["elapsed"] = time.perf_counter() - t0
            status.empty()
        except BackendConnectionError as exc:
            msg = _error(question, "Couldn't reach Gemini. Check the connection and try again.", exc)
        except BackendError as exc:
            msg = _error(question, str(exc), exc)
        except Exception as exc:  # noqa: BLE001 - last line of defence for the UI
            msg = _error(question, "Something unexpected went wrong. Please try again.", exc)
    return msg


def _error(question: str, text: str, exc: Exception) -> dict:
    return new_message("assistant", text, question=question, error=True,
                       debug={"exception": f"{type(exc).__name__}: {exc}"})


def ask_new(question: str) -> None:
    msgs = messages()
    history = chat_history(msgs)
    set_title_from(question)
    msgs.append(new_message("user", question))
    render_user(msgs[-1])
    msgs.append(generate(question, history))
    st.rerun()


def regenerate() -> None:
    msgs = messages()
    if not msgs or msgs[-1]["role"] != "assistant":
        return
    last = msgs.pop()
    st.session_state.pop(f"fb_{last['id']}", None)
    history = chat_history(msgs[:-1])  # everything before the user's question
    msgs.append(generate(last["question"], history))
    st.rerun()


# ----------------------------------------------------------------------------- page
msgs = messages()
pending = st.session_state.pending_question
regen = st.session_state.pending_regen
st.session_state.pending_question = None
st.session_state.pending_regen = False

if not msgs and not pending:
    typed = render_empty()
else:
    for idx, m in enumerate(msgs):
        if m["role"] == "user":
            render_user(m)
        else:
            render_assistant(m, is_last=(idx == len(msgs) - 1) and not pending)
    typed = st.chat_input(INPUT_PLACEHOLDER, key="chat_input")

question = (typed or pending or "").strip()
if regen and not question:
    regenerate()
elif question:
    ask_new(question)
