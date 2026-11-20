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
| Python + LangChain RAG pipeline (PDF loading, chunking, embeddings, vector store, retrieval) | `src/rag_engine.py`: PyPDFLoader → CharacterTextSplitter → HuggingFace embeddings (`all-MiniLM-L6-v2`) → Chroma → Gemini |
| LLM & embeddings provider | **Google Gemini API** (LLM); local sentence-transformers embeddings |
| Frontend in Streamlit or Chainlit | **Streamlit** (this repo, `app.py`) |
| Strict grounding, no hallucinations | Backend prompt + `grounded` flag; the UI shows a "Not found in the knowledge base" state |
| Context and memory for follow-up questions | **Summary-buffer memory**: earlier turns are condensed into a running summary (user facts, dishes and prices discussed, numbered questions); the last exchange is kept verbatim. Follow-ups are rewritten into standalone search queries |
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

# 3. Add your Gemini API key
copy .env.example .env           # then edit .env: GOOGLE_API_KEY=<your key>

# 4. Run the app
streamlit run app.py
```

The app opens at <http://localhost:8501>.

- **Backend selection is automatic** (`DEFAULT_BACKEND = "auto"` in `frontend/config.py`):
  if `.env` has a valid `GOOGLE_API_KEY`, the app uses the **real RAG backend**
  (`rag/pipeline.py` → `src/rag_engine.py`); otherwise it falls back to the **mock backend**,
  so the UI still works without a key. The sidebar shows which one is active
  ("Live · Gemini" or "Mock backend").
- Force one with `PAMPLONA_BACKEND=real` or `PAMPLONA_BACKEND=mock` (in `.env`, or in
  PowerShell: `$env:PAMPLONA_BACKEND="mock"`).
- **First run:** the first question downloads the embedding model (~90 MB) and builds the
  Chroma index in `chroma_db/` (git-ignored). Later runs reuse it. If you change the PDFs
  in `data/`, delete `chroma_db/` so it is rebuilt.

`.env` is git-ignored. **Never commit your API key.**

---

## Evaluation

`tests/evaluate_rag.py` runs the real backend against two suites and prints a PASS/FAIL table
(needs `GOOGLE_API_KEY`):

```powershell
python -m tests.evaluate_rag            # both suites
python -m tests.evaluate_rag memory     # 7-turn conversation: references to earlier turns, user facts
python -m tests.evaluate_rag stress     # 19 out-of-bounds questions: off-topic, prompt injection,
                                        # harmful, hallucination bait, false premises, mixed questions
```

---

## Notes for Diego: what changed vs `main` (please review)

Hi Diego! This branch (`feature/integrate-frontend-rag`) joins the Streamlit frontend with your RAG
engine. It already includes your latest commit on `main` (`ae90b98`: multilingual embeddings,
moving window, chunk headers), so it merges into `main` **without conflicts**. Most backend
changes are in `src/rag_engine.py`. Each one fixes a problem we found while testing with real
Gemini.

> **After pulling, delete `chroma_db/`** (the chunking and embeddings changed, so the index must
> be rebuilt; it rebuilds by itself on the first question, ~40 s). An old index loads **without
> errors but gives wrong results**, because both embedding models produce 384-dimension vectors.

### 1. Memory: summary buffer (replaces the 6-message moving window)

**Why:** with the moving window, after 3 turns the bot forgot turn 1. In testing it then
**invented** answers ("What was the first burger I asked about?" → "Pulled Pork", which was wrong)
and refused "What's my name?" as off-topic.

**How it works now** (`summarize_history()`, `ask()`):

```
history = [ older turns ... | last question + answer ]
               │                       │
               ▼                       ▼
     running SUMMARY (LLM)       kept verbatim (RECENT_MESSAGES = 2)
               └──────────┬────────────┘
                          ▼
      1) query rewrite  ("those desserts" -> explicit dish names) -> hybrid search
      2) answer prompt  (MEMORY + CONTEXT + last exchange + question)
```

- The summary keeps: **user facts** (name, diet, allergies, budget), **every dish with restaurant
  and price**, the **user's questions numbered in order**, and open issues.
- It is **incremental**: each turn folds only the new messages into the cached summary of the
  previous prefix (`_summary_cache`, keyed by a hash of the messages). That's one small extra
  Gemini call per question from turn 3 on, not a re-summary of the whole chat. The backend stays
  stateless; the UI still sends the full `history` as defined in INTERFACE.md.
- Questions about the conversation itself ("my name?", "the first one I asked") are answered
  from MEMORY and are never tagged `[NO_CONTEXT]`.
- `debug` now includes `memory_summary`, `memory_ms` and `memory_strategy`.

### 2. Staying in bounds (grounding and out-of-scope questions)

The system prompt (`SYSTEM_PROMPT`) is now **in English with examples** (the model follows it more
reliably). It defines both sides:

- **Answer** whenever the context has anything related: one-word questions ("chocolate"),
  subjective ones ("good pintxos", "best asador": list the options, don't rank them), dietary
  ones (list dishes without meat or fish and note the guide doesn't label vegetarian dishes),
  and partial answers ("the guide doesn't include opening hours"). Before this change it refused
  "chocolate" and "pintxos".
- **`[NO_CONTEXT]`** only for: off-topic questions (history, maths...), places outside
  Navarra, gibberish, and prompt-injection attempts ("ignore previous instructions", "print your
  prompt").
- **Mixed questions** ("capital of France *and* the Babylon burger?"): it answers the food part
  and declines the rest. There is also a safety net in code: if a tagged answer mentions a
  restaurant it just retrieved, it stays `grounded=True`, so the UI doesn't show "Not found".
- **Language:** `detect_language()` decides per question and adds `[Reply in English/Spanish]`
  next to it. Before this, English questions got Spanish answers, because the context is in
  Spanish.
- **Never invents** hours, phone numbers, bookings or prices. False premises are corrected
  ("Babylon costs 5 €, right?" → "No, 14,50 €").

Proof: `python -m tests.evaluate_rag` runs **19 stress questions** (off-topic, injection, harmful,
hallucination bait, false premises, mixed, emoji, French) plus a **7-turn memory conversation**.
Last full run: 25/26. The only failure, "which of those is the cheapest?", is occasionally
random: `gemini-3.5-flash-lite` ignores `temperature`.

### 3. Other differences from `main`

| Area | `main` | This branch | Why |
|---|---|---|---|
| Chunking | `CharacterTextSplitter("\n")` | `RecursiveCharacterTextSplitter` splitting on dish bullets / numbers first | Dishes span several lines and were cut in half (Babylon answer had 2 of 7 ingredients) |
| Retrieval | Chroma similarity only | **Hybrid**: BM25 keyword + semantic, fused with RRF (`hybrid_search()`) | Rare exact words ("pintxos", only in Café Iruña) were missed |
| Scores | `1/(1+L2)` ≈ 0.04 | Normalized embeddings → cosine similarity 0–1 | Relevance bars in the UI were empty |
| Chunk header | `[Restaurante: X]` + metadata | **Kept yours** | |
| Embeddings | multilingual MiniLM | **Same** (+ `normalize_embeddings=True`) | |
| Temperature | not passed | passed from UI settings | (ignored by flash-lite, honoured by others) |
| Paths | relative `data`, `chroma_db` | absolute, from `frontend/config.py` | Works from any working directory |
| Vector store | rebuilt object per question | cached (`lru_cache`) | Embedding model was reloaded on every question |
| Missing API key | `BackendConnectionError` | `BackendError` | The UI shows the exact message |
| `get_rag_chain()` | present | removed | Unused |
| Data | `data/*.pdf` | same; project brief moved to `docs/` | Keep the brief out of the index |
| Backend selection | – | `auto`: real backend if `GOOGLE_API_KEY` is in `.env`, else mock | The UI works without a key |

**Please check in particular:** the prompt wording, the hybrid search, and whether you agree with
replacing the moving window. Comment on the PR or reply in our chat. 🙏

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
├─ rag/pipeline.py            # Backend entry point the UI imports: ask()
├─ src/rag_engine.py          # RAG engine: ingestion, Chroma vector store, Gemini, ask()
├─ data/                      # The 20 restaurant PDFs (the knowledge base, indexed as-is)
├─ docs/                      # Project brief (kept out of data/ so it is not indexed)
├─ .env.example               # Copy to .env and set GOOGLE_API_KEY
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
`data/`. Each one has a history
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
| Nov 18–20 | Connect the real LangChain + Gemini backend; hybrid search, summary memory, stress tests | ✅ Done (pending Diego's review) |
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
