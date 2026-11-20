"""Entry point: `streamlit run app.py`."""

import streamlit as st

from frontend.backend_interface import backend_name
from frontend.config import APP_TITLE, MODEL_OPTIONS
from frontend.ui import (DEMO_QUESTIONS, SPEEDS, delete_active_chat, export_json,
                         export_markdown, feedback_counts, init_state, load_css, messages,
                         new_chat, prime_widget, queue_question, select_chat, sync_setting)

st.set_page_config(page_title=f"Txoko · {APP_TITLE}", page_icon=":material/restaurant:",
                   layout="centered", initial_sidebar_state="expanded")
init_state()
load_css()

pages = [
    st.Page("views/chat.py", title="Chat", icon=":material/forum:", default=True),
]
nav = st.navigation(pages, position="top")


# ----------------------------------------------------------------------------- settings dialog
@st.dialog("Settings", width="medium", icon=":material/tune:")
def settings_dialog() -> None:
    model_tab, demo_tab, export_tab = st.tabs(["Model", "Demo", "Export"])

    with model_tab:
        for name in ("model", "top_k", "temperature"):
            prime_widget(name)
        st.selectbox("Gemini model", MODEL_OPTIONS, key="w_model",
                     on_change=sync_setting, args=("model",))
        st.slider("Fragments to retrieve (top-k)", 1, 8, key="w_top_k",
                  on_change=sync_setting, args=("top_k",),
                  help="How many document chunks are given to the model as context.")
        st.slider("Temperature", 0.0, 1.0, step=0.05, key="w_temperature",
                  on_change=sync_setting, args=("temperature",),
                  help="Lower is more factual. Keep it low for strict grounding.")
        up, down = feedback_counts()
        st.caption(f"Session feedback: {up} helpful · {down} not helpful")

    with demo_tab:
        st.caption("One-click scenarios for the demo video. Each one is asked in the current chat.")
        for label, question in DEMO_QUESTIONS.items():
            if st.button(label, key=f"demo_{label}", width="stretch", help=question):
                queue_question(question)
                st.rerun()
        if backend_name() == "mock":
            st.divider()
            for name in ("simulate_failure", "speed"):
                prime_widget(name)
            st.toggle("Simulate connection failure", key="w_simulate_failure",
                      on_change=sync_setting, args=("simulate_failure",),
                      help="The backend raises a connection error, to show error handling.")
            st.select_slider("Streaming speed", list(SPEEDS), key="w_speed",
                             on_change=sync_setting, args=("speed",))

    with export_tab:
        has_msgs = bool(messages())
        st.caption("Download or delete the current conversation.")
        c1, c2 = st.columns(2)
        c1.download_button("Markdown", export_markdown(), "conversation.md", "text/markdown",
                           icon=":material/description:", disabled=not has_msgs, width="stretch")
        c2.download_button("JSON", export_json(), "conversation.json", "application/json",
                           icon=":material/data_object:", disabled=not has_msgs, width="stretch")
        if st.button("Delete this conversation", icon=":material/delete:", disabled=not has_msgs,
                     width="stretch"):
            delete_active_chat()
            st.rerun()


# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.html('<div class="txk-brand">Txoko</div>')
    st.button("New chat", icon=":material/edit_square:", on_click=new_chat, width="stretch",
              type="primary")

    started = [c for c in st.session_state.convs.values() if c["messages"]]
    if started:
        st.html('<div class="txk-group">Today</div>')
        for conv in reversed(started):
            active = conv["id"] == st.session_state.active
            st.button(conv["title"] or "New conversation", key=f"conv_{conv['id']}",
                      on_click=select_chat, args=(conv["id"],), width="stretch",
                      type="secondary" if active else "tertiary")

    st.space("large")
    st.divider()
    left, right = st.columns([3, 2], vertical_alignment="center")
    with left:
        live = backend_name() != "mock"
        st.html(f'<div class="txk-status"><span class="txk-dot {"live" if live else "mock"}"></span>'
                f'{"Live · Gemini" if live else "Mock backend"}</div>')
    with right:
        if st.button("Settings", icon=":material/settings:", type="tertiary"):
            settings_dialog()

nav.run()
