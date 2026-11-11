# Txoko · Pamplona Gastronomic Assistant

A RAG (Retrieval-Augmented Generation) chatbot that works as a gastronomic guide to
Pamplona and Navarra. It answers questions about 20 local restaurants (their history,
dishes, prices and ingredients) **using only the provided PDF documents**. When the
answer is not in the documents, it says so instead of making something up.

> *A **txoko** is a traditional Basque-Navarrese gastronomic society where friends cook
> and eat together.*

Final project for the AI Technology course. **Due: Sunday, November 22, 23:59.**

---

## Project requirements (from the brief)

| Requirement | How it is covered |
|---|---|
| Python + LangChain RAG pipeline (PDF loading, chunking, embeddings, vector store, retrieval) | Backend (in progress, see [INTERFACE.md](INTERFACE.md)) |
| LLM & embeddings provider | **Google Gemini API** |
| Frontend in Streamlit or Chainlit | **Streamlit** (this repo, `app.py`) |
| Strict grounding, no hallucinations | Backend prompt + `grounded` flag; the UI shows a "Not found in the knowledge base" state |
| Context and memory for follow-up questions | Chat history is sent to the backend on every question |
| Error handling | Typed backend errors and a friendly error card with **Retry** |
| Deliverables | Source code, a 2–3 minute demo video, and a 7-minute presentation |

---

## Quick start

```bash
# 1. Create and activate a virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at <http://localhost:8501>. By default it runs on the **mock backend**,
so you don't need an API key to try the interface.

To use the real RAG backend once it exists (`rag/pipeline.py`):

```bash
set PAMPLONA_BACKEND=real        # Windows cmd (PowerShell: $env:PAMPLONA_BACKEND="real")
streamlit run app.py
```

The Gemini API key must go in `.streamlit/secrets.toml`, which is git-ignored. **Never
commit it.**

---

## Project structure

```
pamplona-gastronomic-bot/
├─ app.py                     # Entry point: page config, navigation, sidebar, settings dialog
├─ views/
│  └─ chat.py                 # Chat page (answer-engine style)
├─ frontend/
│  ├─ backend_interface.py    # THE contract: ask() -> RAGResponse, plus error types
│  ├─ mock_backend.py         # Stand-in backend that searches the real PDFs (no LLM)
│  ├─ restaurant_data.py      # Parses the 20 PDFs into restaurants and dishes
│  ├─ ui.py                   # Session state, conversations, settings, formatting, exports
│  ├─ styles.css              # All custom styling (pinned to streamlit==1.67.0)
│  └─ config.py               # App name, Gemini model list, default settings
├─ .streamlit/config.toml     # Theme (warm Navarra palette)
├─ data/                      # Project brief + 20 restaurant PDFs
├─ INTERFACE.md               # Frontend ↔ backend contract for the team
└─ requirements.txt
```

---

## Frontend (Streamlit)

### Design

The interface is designed to feel like a real chat product, not a default Streamlit
demo. It follows an **"answer engine"** layout, where grounding is the visual identity:

- **Empty state:** the "Txoko" wordmark, the input box in the center and four
  suggestion chips.
- **User messages:** right-aligned bubbles.
- **Assistant answers:**
  1. A progress line ("Searching 20 restaurant menus…", then "Reviewed 4 sources · 0.8s").
  2. **Sources:** compact cards with a relevance bar. Clicking one opens the PDF name,
     page, relevance score and the excerpt with the search terms highlighted.
  3. **Answer:** streamed text with inline citation chips (¹ ²).
  4. Actions: copy, 👍/👎 feedback and regenerate.
- **Not grounded:** an amber "Not found in the knowledge base" line and no sources.
- **Errors:** a muted card ("Couldn't reach Gemini…") with a **Retry** button.
- **Sidebar:** New chat, a list of conversations titled from their first question, the
  backend status (mock or live) and **Settings**.
- **Settings dialog:**
  - *Model*: Gemini model, top-k, temperature.
  - *Demo*: one-click demo scenarios, a simulated connection failure, streaming speed.
  - *Export*: Markdown and JSON downloads, delete the conversation.
- **Visual style:** a warm Navarra palette (cream background, Navarra red accent), the
  Inter font, Material icons and no emojis in the interface.

### Streamlit features used

`st.navigation` / `st.Page`, `st.chat_input`, streaming through `st.empty` placeholders,
`st.session_state` (multi-conversation memory), `st.dialog`, `st.tabs`, `st.popover`,
`st.pills`, `st.feedback`, `st.progress`, `st.download_button`, `st.html` (including a
JavaScript copy button), horizontal `st.container`s with keys for CSS targeting,
`st.cache_data`, and a theme defined in `config.toml`.

### Backend contract

The UI only calls a single function:

```python
ask(question: str, history: list[dict], settings: dict) -> RAGResponse
# RAGResponse(answer: str | Iterator[str], sources: list[Source], grounded: bool, debug: dict)
```

[INTERFACE.md](INTERFACE.md) covers the full contract, the error types, the expected
behaviours and how to plug in the real backend.

### Mock backend

While the LangChain + Gemini pipeline is being built, `frontend/mock_backend.py`
implements the same `ask()` contract:

- It searches the **real PDFs** with BM25 keyword search and a small English→Spanish
  synonym map, so every source shown is real text from the documents.
- It writes answers from templates using only the retrieved chunks, so it never
  invents information.
- It simulates the demo edge cases: follow-ups (memory), out-of-scope places,
  off-topic questions, nonsense input, greetings, and connection failures.

Its answers are lists, not natural prose; Gemini will produce the real wording.

### Demo scenarios (Settings → Demo)

| Scenario | Question | Expected behaviour |
|---|---|---|
| Grounded answer | What does the Hamburguesa Babylon contain? | Ingredients with sources from `1. Babylon.pdf` |
| Follow-up (memory) | And how much is it? | Resolved with history: **14,50 €** |
| Restaurant info | Tell me about Café Iruña | History and description, with sources |
| Out of scope | Which restaurant has the best sushi in Madrid? | Not grounded: only Pamplona/Navarra is covered |
| Off-topic | What is the capital of France? | "I don't have that information in my documents" |
| Nonsense | asdkjh qwe? | Asks the user to rephrase |
| Connection error | (toggle "Simulate connection failure") | Error card with Retry |

---

## Data

There are 20 Spanish PDFs (one per restaurant, about 400 words each) in
`data/Restaurant docs-20261108T214532Z-1-001/Restaurant docs/`. Each one has a history
paragraph, a cuisine paragraph and about 10 dishes in the format
`Name (price €): ingredients`. That makes **200 dishes** across **8 towns** (Pamplona plus 7 other Navarra towns).

- With `pymupdf` the € sign is extracted correctly. Some other extractors (e.g.
  `pdftotext`) lose it.
- Dish lists are bulleted (`•`, `*`) in some files and numbered (`1.`) in others.
- File 3 (*More Than Burgers*) has no title line.
- Restaurant categories (Burgers, Haute cuisine, …) are **manual labels** added in
  `frontend/restaurant_data.py`. They are not part of the documents.

---

## Progress

| Dates | Work | Status |
|---|---|---|
| Nov 11 | Planning, interface contract (`INTERFACE.md`), environment, PDF parser, mock backend | ✅ Done |
| Nov 11–12 | Chat page v1 (streaming, sources, memory, feedback, export, error handling) | ✅ Done |
| Nov 12 | UI redesign: answer-engine layout, conversation history, settings dialog, warm Navarra theme | ✅ Done (pending visual review) |
| Nov 16 | Restaurant Explorer page (cards and filters) | ⏳ Next |
| Nov 17 | Insights page (price and category charts) and "Under the hood" page (chunks, rewritten query, prompt) | ⏳ Planned |
| Nov 18–19 | Connect the real LangChain + Gemini backend; test edge cases | ⏳ Planned |
| Nov 20 | Polish and final README | ⏳ Planned |
| Nov 21 | Record the demo video, prepare slides | ⏳ Planned |
| Nov 22 | Buffer and submission | ⏳ Planned |

### Changelog

- **Frontend v2: redesign.** Replaced the default-Streamlit look (emojis, gradient
  banner, control-panel sidebar) with an answer-engine layout. Added multiple
  conversations, the settings dialog, inline citations, source cards, the copy and
  regenerate actions, the Retry flow and `frontend/styles.css`. Pinned
  `streamlit==1.67.0`.
- **Frontend v1.** Chat page with streaming, a status panel, a sources section,
  feedback, export, the simulated failure toggle and the first theme.
- **Foundations.** `INTERFACE.md`, `frontend/backend_interface.py`, the PDF parser and
  the mock backend.

---

## Team

Group project (up to 3 people).

| Area | Owner |
|---|---|
| Frontend (Streamlit UI, mock backend, interface contract) | *your name* |
| RAG backend (LangChain, Gemini, vector store) | *teammate* |
| *(to define)* | *teammate* |
