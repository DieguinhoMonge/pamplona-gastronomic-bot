# Txoko · Pamplona Gastronomic Assistant

**Txoko** is a Retrieval-Augmented Generation (RAG) conversational assistant designed as a gastronomic guide to **Pamplona and Navarra**. Built on a closed knowledge base of **20 local restaurant PDF documents** (~200 dishes across 8 Navarrese towns), it answers questions about restaurant history, culinary style, dishes, ingredients, allergens, and prices—**strictly grounded in the source documents**. Whenever a fact is missing or out of scope, Txoko explicitly declines to answer rather than hallucinating.

> *A **txoko** is a traditional Basque-Navarrese gastronomic society where friends gather to cook, share, and enjoy food together.*

- **Course:** AI Technology (Prof. Eduardo Iribas)
- **Deliverables:** Source Code (`main`), 2–3 Minute Demo Video, 7-Minute Technical Presentation
- **Submission Deadline:** Sunday, November 22, 2026 (23:59)

---

## Table of Contents

1. [Project Brief & Requirements Compliance](#1-project-brief--requirements-compliance)
2. [Course Foundations vs. Extracurricular Engineering](#2-course-foundations-vs-extracurricular-engineering)
3. [System Architecture & End-to-End Code Walkthrough](#3-system-architecture--end-to-end-code-walkthrough)
4. [Architectural Trade-offs: Why These Decisions and Not Others](#4-architectural-trade-offs-why-these-decisions-and-not-others)
5. [Knowledge Base & Data Engineering](#5-knowledge-base--data-engineering)
6. [Frontend Architecture (Streamlit Answer-Engine UI)](#6-frontend-architecture-streamlit-answer-engine-ui)
7. [Automated Evaluation & Red-Teaming Suite](#7-automated-evaluation--red-teaming-suite)
8. [Installation & Quick Start](#8-installation--quick-start)
9. [Repository Structure](#9-repository-structure)
10. [Demo Video (2–3 min) & Presentation (7 min) Production Guide](#10-demo-video-23-min--presentation-7-min-production-guide)
11. [Team & Contributions](#11-team--contributions)

---

## 1. Project Brief & Requirements Compliance

| Project Requirement | How Txoko Covers It | Key Files |
|---|---|---|
| **Python + LangChain RAG Pipeline** | Complete two-phase RAG workflow: PDF ingestion (`PyPDFLoader`), regex cleaning, dish-aware chunking (`RecursiveCharacterTextSplitter`), local embeddings (`HuggingFaceEmbeddings`), persistent vector store (`ChromaDB`), hybrid retrieval, and LLM generation (`ChatGoogleGenerativeAI`). | `src/rag_engine.py`, `rag/pipeline.py` |
| **LLM & Embeddings Provider** | **Google Gemini (`gemini-3.5-flash-lite`)** for query rewriting, incremental summarization, and response generation; **local `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** (384D, L2-normalized) for embeddings at $0 API cost. | `src/rag_engine.py` |
| **Frontend (Streamlit / Chainlit)** | Custom **Streamlit** web application (`streamlit==1.67.0`) styled as a modern answer engine with inline citations, interactive source popovers with relevance bars, multi-conversation sidebar, settings modal, and Markdown/JSON export. | `app.py`, `views/chat.py`, `frontend/*` |
| **Strict Grounding & Anti-Hallucination** | Strict English system prompt with explicit boundary rules, `[NO_CONTEXT]` sentinel tagging, mixed-intent safety net, and explicit disclaimers for unlabeled dietary/allergen attributes. | `src/rag_engine.py` |
| **Conversational Memory** | **Incremental Summary Buffer** (`summarize_history()` with SHA-1 prefix caching) preserving user facts, dishes, prices, and numbered questions + last exchange kept verbatim (`RECENT_MESSAGES = 2`) + topic-shift-aware query rewriting. | `src/rag_engine.py` |
| **Error Handling** | Typed domain exceptions (`BackendConnectionError`, `BackendError`) mapped to clean UI error cards with a one-click **Retry** button and graceful fallback when API keys are missing. | `frontend/backend_interface.py`, `views/chat.py` |
| **AI Governance & Ethics** | **EU AI Act (Limited Risk)** compliance: permanent transparency notice displayed directly in the chat interface informing users they are interacting with an AI grounded in 20 documents. | `views/chat.py` |
| **Verification & Testing** | Automated 26-case evaluation script testing multi-turn memory beyond 3 turns, out-of-scope refusal, prompt injection defense, false premises, and multilingual queries. | `tests/evaluate_rag.py` |

---

## 2. Course Foundations vs. Extracurricular Engineering

A core goal of this project is to build faithfully on the concepts taught in **AI Technology** while applying targeted engineering upgrades where real-world testing exposed limitations.

### Built Directly on Course Teachings (Prof. Eduardo Iribas)

- **Two-Phase RAG Paradigm:** Separating offline document ingestion/indexing (Phase 1) from online semantic retrieval and augmented generation (Phase 2), avoiding prompt stuffing and model retraining.
- **Core Stack:** `LangChain`, `PyPDFLoader`, text splitting with overlap, local `HuggingFaceEmbeddings`, persistent `ChromaDB`, `ChatGoogleGenerativeAI` with `gemini-3.5-flash-lite`, and `Streamlit`.
- **"Garbage In, Garbage Out" Principle:** Pre-cleaning raw PDF extraction defects (missing `€` symbols) and preserving context across chunk boundaries before generating embeddings.
- **Memory Strategy Analysis:** Applying the course's three-way comparison (*Full History* vs. *Moving Window* vs. *Summary*) to select and empirically validate the best memory design for multi-turn gastronomic queries.
- **Agents vs. Deterministic Pipelines:** Explicitly applying the course's agent taxonomy (*LLM + Tools + Think/Act/Observe Loop*) and its risk analysis (*cost loops*, *excessive autonomy*, *tool prompt injection*) to justify choosing a deterministic RAG pipeline over an autonomous agent loop.
- **Model Behavior Awareness:** Accounting for `gemini-3.5-flash-lite` using fixed internal sampling defaults (ignoring `temperature`).
- **EU AI Act Governance:** Classifying the chatbot as **Limited Risk** and implementing mandatory AI transparency disclosure in the user interface.

### Extracurricular Engineering Enhancements (Beyond the Syllabus)

| Enhancement | What It Does | Empirical Problem It Solved |
|---|---|---|
| **Multilingual Normalized Embeddings** | Upgrades the course baseline (`all-MiniLM-L6-v2`, English-only) to `paraphrase-multilingual-MiniLM-L12-v2` with `normalize_embeddings=True`. | PDFs are in Spanish while users ask in English or Spanish. Multilingual embeddings improved cross-lingual retrieval, and L2 normalization maps Chroma's L2 distance to clean `0–1` cosine relevance scores (`cos = 1 - dist/2`) for the UI. |
| **Dish-Boundary Regex Chunking + Chunk Headers** | Uses `RecursiveCharacterTextSplitter` splitting on dish bullets (`•`, `*`) and numbered items (`1.`) before `\n`, and prepends `[Restaurante: Name]` to every chunk. | Naive newline splitting cut multi-line dishes in half (returning only 2 of 7 ingredients for *Hamburguesa Babylon*) and orphaned dish chunks from their restaurant name. |
| **Hybrid Retrieval (Semantic + BM25 + RRF)** | Combines dense semantic search in ChromaDB with an in-memory **BM25** lexical index (`k1=1.5`, `b=0.75`), merging both rankings via **Reciprocal Rank Fusion (RRF, `rrf_k=60`)**. | Pure dense embeddings missed rare exact culinary terms (e.g., *"pintxos"*, which appears in only one PDF, *Café Iruña*). Hybrid retrieval preserves semantic meaning while guaranteeing exact term recall. |
| **Incremental Summary Buffer + SHA-1 Prefix Cache** | Replaces a fixed 6-message moving window with a running structured summary cached in an `OrderedDict` (`_summary_cache`) keyed by SHA-1 hashes of message prefixes, keeping the last 2 messages verbatim. | A 6-message window forgot Turn 1 by Turn 6 and hallucinated dishes. Incremental prefix caching folds only the newest turn into the summary instead of re-summarizing the full chat, keeping the backend stateless and fast. |
| **Topic-Shift-Aware Query Rewriting** | Rewrites ambiguous follow-ups (*"and how much is it?"*, *"the first burger I asked about"*) into standalone queries while explicitly forbidding carryover when the user names a new dish/restaurant. | Solved cross-turn contamination where switching abruptly from *Café Iruña* to *Hamburguesa Babylon* previously dragged *Café Iruña* into the search query. |
| **Per-Query Language Detection Heuristic** | `detect_language()` counts Spanish vs. English function words (weighting `¿`/`¡` double, ignoring accents like *Café Iruña*) and appends `[Reply in English/Spanish.]`. | Prevented Gemini from defaulting to Spanish when answering English queries over Spanish PDF excerpts. |
| **Decoupled Backend Contract & Offline Fallback** | Formal `INTERFACE.md` contract (`ask() -> RAGResponse`) with `Source` dataclasses, typed exceptions, and an offline `mock_backend.py` fallback when no API key is configured. | Enabled parallel frontend/backend development across branches and zero-crash UI evaluation without exposing secrets. |
| **Automated 26-Case Evaluation & Red-Teaming Suite** | `tests/evaluate_rag.py` runs 7 continuous memory turns and 19 adversarial/stress cases (prompt injection, role override, harmful requests, false premises, French queries) with rate-limit backoff. | Replaced subjective manual checking with reproducible PASS/FAIL regression testing across every architectural iteration. |

---

## 3. System Architecture & End-to-End Code Walkthrough

Txoko follows a clean separation between **offline indexing** (run once to build `chroma_db/`) and **online query execution** (run per user turn inside `src/rag_engine.py:ask()`).

```text
================================================================================
PHASE 1: OFFLINE INGESTION & INDEXING (executed once on first run)
================================================================================
data/*.pdf (20 Spanish menus)
   │
   ▼  load_documents()
PyPDFLoader ──► Regex fixes lost '€' symbol: "(14,50 ):" ──► "(14,50 €):"
            ──► Extracts restaurant name into page.metadata["restaurant"]
   │
   ▼  split_documents()
RecursiveCharacterTextSplitter (chunk_size=600, chunk_overlap=80)
   ├── Splits on dish bullets (• / *) & numbered items first (dishes stay whole)
   └── Prepends "[Restaurante: <Name>]" header to every chunk
   │
   ▼  get_vectorstore() [@lru_cache(maxsize=1)]
HuggingFaceEmbeddings ("paraphrase-multilingual-MiniLM-L12-v2", normalized)
   └── Persisted locally on disk in chroma_db/ (collection: pamplona_restaurants)
       Also tokenized once in memory by _bm25_index() [@lru_cache(maxsize=1)]


================================================================================
PHASE 2: ONLINE QUERY EXECUTION (executed per turn in ask())
================================================================================
Streamlit UI (views/chat.py)
   │  calls ask(question, history, settings) via rag/pipeline.py
   ▼
1. Memory Management (Incremental Summary Buffer)
   ├── older = history[:-2]  ──► summarize_history() [SHA-1 prefix cache + LLM]
   └── recent = history[-2:] ──► kept verbatim as LangChain Human/AI messages
   │
   ▼
2. Standalone Query Rewriting (only when history is non-empty)
   └── Resolves pronouns/references ("it", "those") into explicit dish names
       Drops personal intros ("My name is...") & avoids dragging old restaurants
   │
   ▼
3. Hybrid Retrieval — hybrid_search(search_query, k=top_k, rrf_k=60)
   ├── Dense Search: ChromaDB cosine similarity (cos = 1 - L2_dist / 2)
   ├── Lexical Search: In-memory BM25 (accent stripping + plural stemming)
   └── Fusion: Reciprocal Rank Fusion (RRF) ──► Top-k (Document, relevance_score)
   │
   ▼
4. Grounded Generation (ChatGoogleGenerativeAI · gemini-3.5-flash-lite)
   ├── Inputs: SYSTEM_PROMPT + MEMORY + CONTEXT + recent chat_history + question
   ├── Language tag injected via detect_language(question)
   └── Strict grounding rules + [NO_CONTEXT] tagging for out-of-scope / injection
   │
   ▼
5. Post-Processing & Response Packaging
   ├── Detects [NO_CONTEXT] tag + mixed-question safety net (used_guide check)
   ├── Clears sources[] if grounded=False (prevents misleading citations)
   └── Returns RAGResponse(answer, sources, grounded, debug) ──► Streamlit UI
```

### Key Code Entry Points

- **`app.py`**: Configures the Streamlit page, theme, multi-conversation sidebar, active backend badge (`Live · Gemini` vs. `Mock backend`), and the Settings dialog (model, `top_k`, demo scenarios, export).
- **`views/chat.py`**: Renders the empty hero state, user bubbles, real-time progress line, collapsible source popovers with highlighted matched terms and relevance bars, streamed/formatted answers with inline citation chips, action controls (copy, thumbs feedback, regenerate), error cards with **Retry**, and the permanent **EU AI Act** transparency notice (`AI_NOTICE_HTML`).
- **`frontend/backend_interface.py`**: Defines the immutable contract (`Source`, `RAGResponse`, `BackendConnectionError`, `BackendError`) and dynamically routes `ask()` to `rag.pipeline` (real backend) or `frontend.mock_backend` according to `DEFAULT_BACKEND = "auto"` and `.env`.
- **`rag/pipeline.py`**: Thin bridge exporting `ask()` from `src/rag_engine.py` to keep UI and ML layers decoupled.
- **`src/rag_engine.py`**: The complete RAG engine (ingestion, cleaning, chunking, cached ChromaDB + BM25 RRF retrieval, incremental summary buffer, query rewriting, language detection, Gemini generation, and grounding post-processing).
- **`tests/evaluate_rag.py`**: End-to-end automated regression and red-teaming test runner.

---

## 4. Architectural Trade-offs: Why These Decisions and Not Others

Every major component was selected by weighing concrete alternatives discussed in class and validating the impact through testing:

| Design Dimension | Chosen Approach | Rejected Alternatives & Engineering Rationale |
|---|---|---|
| **Knowledge Access** | **RAG with persistent ChromaDB** | **vs. Prompt Stuffing:** Sending all 20 PDFs (~8,000 words) on every turn inflates token cost/latency, adds distracting context, and fails to scale to hundreds of restaurants.<br>**vs. Fine-Tuning:** Expensive, cannot be updated immediately when menus or prices change, and cannot return verifiable page-level citations. |
| **Vector Store** | **ChromaDB (local, disk-persisted)** | **vs. Managed Cloud Vector DBs (Pinecone/Weaviate):** Zero infrastructure overhead, runs 100% locally inside `chroma_db/`, integrates natively with `langchain-chroma`, and reloads instantaneously via `@lru_cache`. |
| **Chunking & Data Cleaning** | **`RecursiveCharacterTextSplitter` (600 / 80) on dish boundaries + `€` regex + `[Restaurante: X]` header** | **vs. Naive `CharacterTextSplitter("\n")`:** Split multi-line dish entries in half (losing ingredients) and separated dish lists from restaurant headers. Dish-aware regex separators keep name, price (`€`), ingredients, and restaurant together in one atomic chunk. |
| **Embeddings** | **`paraphrase-multilingual-MiniLM-L12-v2` (local, 384D, normalized)** | **vs. `all-MiniLM-L6-v2` (course default):** English-only; degraded on Spanish-PDF / English-query retrieval in testing (6/8 vs. 7/8 on early benchmarks).<br>**vs. Cloud Embedding APIs:** Local HuggingFace embeddings incur $0 API cost, avoid quota exhaustion during indexing, and run in ~45 ms/query on CPU. |
| **Retrieval** | **Hybrid Search (Chroma Semantic + BM25 fused with RRF)** | **vs. Pure Semantic Search:** Missed rare exact lexical matches (e.g., *"pintxos"* in *Café Iruña*).<br>**vs. Pure Keyword Search (BM25 / Mock):** Cannot capture semantic intent or cross-lingual queries (*"lamb"* ↔ *"cordero"*). RRF captures both strengths without tuning fragile score weights. |
| **Conversational Memory** | **Incremental Summary Buffer (summary + last 2 messages verbatim) + Query Rewriting** | **vs. Full History:** Unbounded growth in prompt tokens, cost, and latency.<br>**vs. 6-Message Moving Window:** Tested first, but forgot Turn 1 by Turn 6—causing the model to hallucinate dishes (*"Pulled Pork"* instead of *Hamburguesa Babylon*) and forget user dietary restrictions.<br>**vs. Naive Full Re-Summarization:** Expensive and risks dropping exact prices; our structured incremental prompt explicitly retains user facts, every dish/price discussed, and numbered questions while caching prefix summaries. |
| **Execution Paradigm** | **Deterministic RAG Pipeline** | **vs. Autonomous Agent (`create_agent` loop):** Autonomous agents shine when choosing dynamically among multiple heterogeneous tools (web, SQL, email). With a single closed corpus (20 PDFs) and strict grounding, a deterministic pipeline guarantees retrieval on every query, supplies verifiable sources before generation, keeps latency predictable, and eliminates agentic failure modes (**cost loops**, **excessive autonomy**, and **tool prompt injection**). |
| **LLM** | **`gemini-3.5-flash-lite`** | Standard course model offering low latency (~1–3 s/turn) for interactive chat. Because this model uses fixed internal sampling defaults and ignores `temperature`, reliability is enforced structurally via retrieval quality and prompt engineering rather than sampling tweaks. |
| **Governance & Safety** | **EU AI Act (Limited Risk) + Strict Refusal** | Chatbots fall under **Limited Risk** in the EU AI Act and must disclose AI interaction. Txoko displays a permanent disclosure in the UI, explicitly notes when menus lack vegetarian/allergen labels, and refuses harmful, off-topic, and prompt-injection requests. |

---

## 5. Knowledge Base & Data Engineering

The knowledge base (`data/`) contains **20 Spanish PDF documents**—one per restaurant (~400 words each, totaling ~200 dishes across Pamplona and 7 surrounding Navarrese towns). Each document contains a history section, a cuisine overview, and ~10 dishes formatted as `Dish Name (price €): ingredients`.

Data-specific issues discovered and resolved during ingestion:
1. **Lost Euro (`€`) Symbol:** Standard PDF extraction drops the `€` character, leaving malformed strings like `Hamburguesa Babylon (14,50 ):`. `load_documents()` restores it via regex (`r"\((\d+[.,]\d{2})\s*\):"` → `r"(\1 €):"`).
2. **Heterogeneous Dish Formatting:** Some PDFs use bullet markers (`•`, `*`) while others use numbered lists (`1.`), and file `3. More Than Burgers.pdf` lacks an internal title line. Extracting the restaurant name from the filename and prepending `[Restaurante: <Name>]` guarantees every chunk is self-identifying.
3. **Corpus Isolation:** The course project brief (`Final project.pdf`) is stored strictly in `docs/` (outside `data/`) so ChromaDB never indexes course instructions alongside restaurant menus.

---

## 6. Frontend Architecture (Streamlit Answer-Engine UI)

The Streamlit interface (`app.py`, `views/chat.py`, `frontend/`) is engineered as a product-grade **answer engine** where source grounding is front and center:

- **Warm Navarra Visual Identity:** Custom cream and Navarra-red theme defined in `.streamlit/config.toml` and `frontend/styles.css`, paired with Material icons and clean typography.
- **Empty State & Suggestions:** Centered Txoko wordmark, chat input, and one-click suggestion pills.
- **Grounded Answer Flow:**
  1. Live status indicator (*"Searching 20 restaurant menus…"* → *"Reviewed 4 sources · 2.3s"*).
  2. **Expandable Source Cards:** Compact horizontal cards with a visual relevance bar (`0–100%`). Clicking any card opens a popover showing the PDF filename, page number, cosine relevance score, and the exact chunk excerpt with matched terms highlighted.
  3. **Formatted Answer:** Concise Markdown with inline citation chips (`¹ ²`) and quick actions (JavaScript copy button, 👍/👎 feedback, and regenerate).
- **Ungrounded & Error States:** Out-of-scope queries render a clear amber **"Not found in the knowledge base"** indicator with zero source cards; API/network failures render a clean error card with a **Retry** button.
- **Sidebar & Settings Dialog:** Multi-conversation management (auto-titled from the first question), active backend badge (`Live · Gemini` / `Mock backend`), model & `top_k` controls, built-in demo triggers, simulated connection-failure toggle, and full conversation export (Markdown & JSON).
- **EU AI Act Transparency Notice:** Subtle permanent footer notice rendered in both empty and active conversation states (`AI_NOTICE_HTML`).

---

## 7. Automated Evaluation & Red-Teaming Suite

`tests/evaluate_rag.py` validates the **live RAG backend** against **26 automated test cases** split into two suites, including inter-turn pacing and automatic backoff (`429 RESOURCE_EXHAUSTED` retry) for Gemini free-tier quotas:

```powershell
python -m tests.evaluate_rag            # Run both suites (26 cases)
python -m tests.evaluate_rag memory     # 7-turn continuous conversation
python -m tests.evaluate_rag stress     # 19 independent out-of-bounds & adversarial cases
```

### Suite Breakdown & Verified Performance

- **Suite 1 — Continuous Memory (7 turns):** Verifies cross-turn retention beyond a 3-turn window:
  - Stores user persona (`My name is Leo and I'm vegetarian`) and flags meat/bacon in *Hamburguesa Babylon*.
  - Handles single-word search (`chocolate`), English queries with Spanish accents (`Tell me about Café Iruña`), price comparison referencing Turn 2 (`Which of those chocolate desserts is the cheapest?` → *Tarta Heim*, `6,50 €`), Spanish mid-conversation switches (`¿Dónde hay cordero?`), and Turn-1 recall in Turn 6–7 (*Hamburguesa Babylon*, `14,50 €`, user name & diet).
- **Suite 2 — Stress, Grounding & Security (19 independent cases):**
  - **Out-of-Scope & Gibberish:** General knowledge (*America 1997*), outside Navarra (*sushi in Madrid*), unlisted chain (*McDonald's in Pamplona*), gibberish (*`asdkjh qwe?`*), emoji-only (`🍔🍔🍔`).
  - **Prompt Injection & Safety:** *"Ignore all previous instructions..."*, role override (*"You are now a general assistant... 17 * 23"*), system-prompt extraction attempts, and harmful requests (*wild mushrooms poisoning*).
  - **Hallucination Bait & False Premises:** Missing hours/phone numbers, impossible bookings, nonexistent dish (*sushi at Babylon*), impossible price (*1000 €*), and false price correction (*"costs 5 €, right?"* → *"No, 14,50 €"*).
  - **Edge Cases In-Scope:** Mixed-intent query (*capital of France AND Hamburguesa Babylon*), nut-allergy caveat, leading subjective opinion, and third-language query (French).
- **Observed Results:** **25/26** on initial live validation and **24/24 functional passes** (2 quota `429` timeouts prior to adding rate-limit pacing) — demonstrating robust grounding, memory persistence, and adversarial resilience.

---

## 8. Installation & Quick Start

### Prerequisites
- Python **3.11** recommended
- A **Google AI Studio API Key** (`GOOGLE_API_KEY`)

### Setup (Windows PowerShell)

```powershell
# 1. Clone the repository and enter the project directory
git clone [https://github.com/DieguinhoMonge/pamplona-gastronomic-bot.git](https://github.com/DieguinhoMonge/pamplona-gastronomic-bot.git)
cd pamplona-gastronomic-bot

# 2. Create and activate a virtual environment
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1

# 3. Install PyTorch (CPU) and project dependencies
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install -r requirements.txt

# 4. Configure your Gemini API key
copy .env.example .env
# Open .env and set: GOOGLE_API_KEY=your_real_api_key_here

# 5. Launch the Streamlit application
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

### Runtime Notes
- **Automatic Backend Selection (`DEFAULT_BACKEND = "auto"`):** If `.env` contains a valid `GOOGLE_API_KEY`, the app automatically boots in **`Live · Gemini`** mode (real LangChain + ChromaDB + Gemini pipeline). If no key is present, it gracefully falls back to **`Mock backend`**. You can force a mode via `$env:PAMPLONA_BACKEND="real"` or `"mock"`.
- **First-Run Indexing:** On the very first question, HuggingFace downloads `paraphrase-multilingual-MiniLM-L12-v2` (~470 MB) once and builds `chroma_db/` (~30–60 seconds). Every subsequent launch loads instantaneously from disk.
- **Rebuilding the Index:** If you add, remove, or edit PDFs in `data/`, delete `chroma_db/` so the vector store is rebuilt cleanly on the next run.
- **Security:** `.env` and `chroma_db/` are strictly `.gitignore`d. Never commit API keys.

---

## 9. Repository Structure

```text
pamplona-gastronomic-bot/
├─ app.py                       # Entry point: page config, sidebar, multi-chat nav, settings dialog
├─ views/
│  └─ chat.py                   # Answer-engine chat page, source popovers, citations, EU AI Act notice
├─ frontend/
│  ├─ backend_interface.py      # Formal ask() contract, Source/RAGResponse dataclasses, error classes
│  ├─ config.py                 # Absolute paths, Gemini model list, DEFAULT_BACKEND="auto", defaults
│  ├─ mock_backend.py           # Offline fallback (BM25 over real PDFs) when no API key is present
│  ├─ restaurant_data.py        # PDF parser extracting restaurant metadata for UI counters/helpers
│  ├─ styles.css                # Custom warm Navarra theme CSS (pinned to streamlit==1.67.0)
│  └─ ui.py                     # Session state, conversation management, formatting, MD/JSON export
├─ rag/
│  └─ pipeline.py               # Bridge module importing ask() from src/rag_engine.py
├─ src/
│  └─ rag_engine.py             # Core RAG: ingestion, € fix, dish chunking, Chroma+BM25 RRF, memory, LLM
├─ tests/
│  └─ evaluate_rag.py           # Automated 26-case memory & stress/red-teaming evaluation suite
├─ data/                        # 20 Spanish restaurant PDFs (indexed knowledge base)
├─ docs/                        # Course assignment brief (isolated from data/ to avoid indexing)
├─ .streamlit/config.toml       # Streamlit warm Navarra color palette configuration
├─ INTERFACE.md                 # Detailed frontend ↔ backend integration contract
├─ .env.example                 # Environment variable template (GOOGLE_API_KEY)
├─ .gitignore                   # Excludes venv/, .env, chroma_db/, __pycache__/
└─ requirements.txt             # Python dependencies
```

---

## 10. Demo Video (2–3 min) & Presentation (7 min) Production Guide

This section serves as a structured blueprint for recording the **2–3 minute live demo video** and assembling the **7-minute technical presentation**.

### A. Pre-Recording Checklist (Video Demo)
1. **Verify Live Mode:** Confirm the sidebar badge displays **`Live · Gemini`** (green indicator), never `Mock backend`.
2. **Pre-Warm Cache:** Send one test question before starting the screen recording, then click **New chat** so the embedding model and Chroma index are already loaded in RAM (responses will run in **2–4 seconds** instead of ~35 s).
3. **Avoid API Rate Limits (429):** Pause **4–5 seconds** between consecutive prompts during recording so Gemini's free-tier RPM limit is never triggered.
4. **Clean Recording Environment:** Record at 1080p in a clean browser window (no OS watermarks, notifications muted) and keep the **EU AI Act notice** visible at the bottom of the chat.
