# Frontend ↔ RAG Backend Interface

The Streamlit frontend talks to the RAG pipeline through **one function**: `ask()`.
Until the real pipeline is ready, the frontend uses a mock implementation with the same
signature. To connect the real backend, implement `ask()` as described below. The UI
needs no other changes.

---

## 1. The contract

```python
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Source:
    restaurant: str           # e.g. "Restaurante Túbal"
    file: str                 # PDF file name, e.g. "11. Restaurante_Tubal.pdf"
    page: int                 # 1-based page number inside the PDF
    chunk: str                # the exact text fragment that was retrieved
    score: float | None = None  # similarity / relevance score (higher = more relevant)


@dataclass
class RAGResponse:
    answer: str | Iterator[str]   # full text, OR a generator of text pieces (streaming)
    sources: list[Source]         # chunks used to build the answer (empty if none)
    grounded: bool                # False when the answer is NOT in the documents
    debug: dict = field(default_factory=dict)  # optional, shown in "Under the hood"


def ask(question: str, history: list[dict], settings: dict) -> RAGResponse:
    ...
```

### Inputs

| Argument   | Type         | Description |
|------------|--------------|-------------|
| `question` | `str`        | The user's latest message. |
| `history`  | `list[dict]` | Previous turns, oldest first, **not** including `question`. Each item is `{"role": "user" \| "assistant", "content": str}`. Use it for follow-ups ("and how much does it cost?"). |
| `settings` | `dict`       | Values from the sidebar: `{"top_k": int, "temperature": float, "model": str \| None}`. Ignore any key you don't support. |

### Output: `RAGResponse`

- **`answer`**: either a plain string or a **generator that yields text pieces**.
  If you use LangChain, `chain.stream(...)` works directly; wrap it so it yields
  `str`. The UI streams it token by token with `st.write_stream`.
- **`sources`**: the chunks returned by the retriever. Return them even when streaming,
  so they need to be known **before** generation starts (retrieve first, then stream).
- **`grounded`**: set to `False` when the documents do not contain the answer
  (strict grounding). The UI then shows the answer with a distinct
  "Not in my knowledge base" style. In that case `sources` may be empty.
- **`debug`** (optional, but recommended for the technical defense). Any of these keys
  are displayed on the *Under the hood* page if present:

  | Key                 | Meaning |
  |---------------------|---------|
  | `rewritten_query`   | Standalone question after resolving the follow-up with history |
  | `prompt`            | Final prompt sent to the LLM |
  | `model`             | Model actually used |
  | `retrieval_ms`      | Retrieval time in milliseconds |
  | `generation_ms`     | Generation time in milliseconds (non-streaming only) |

---

## 2. Error handling

Raise these exceptions (defined in `frontend/backend_interface.py`) and the UI will show
a friendly message instead of a stack trace:

| Exception                 | When to raise | What the user sees |
|---------------------------|---------------|--------------------|
| `BackendConnectionError`  | LLM / embeddings API unreachable, timeout, invalid API key | "Can't reach the AI service right now, try again." |
| `BackendError`            | Any other expected failure (e.g. vector store not built) | The exception message |

Any other unexpected exception is also caught by the UI and shown as a generic error,
but please prefer the two above.

---

## 3. Expected behaviours (these will appear in the demo video)

| User asks | Expected `RAGResponse` |
|-----------|------------------------|
| "What does the Hamburguesa Babylon contain?" | `grounded=True`, answer from the docs, sources from `1. Babylon.pdf` |
| Follow-up: "And how much is it?" | Resolved with `history`; `grounded=True` |
| "Which restaurant has the best sushi in Madrid?" | `grounded=False`, says it only knows the provided Pamplona/Navarra restaurants |
| "asdkjh qwe?" (nonsense) | `grounded=False`, asks the user to rephrase |
| "What's the capital of France?" (off-topic) | `grounded=False`, politely says it only answers gastronomy questions about the documents |

---

## 4. Notes on the data (for whoever builds the ingestion)

- 20 PDFs in Spanish, ~400 words each, in
  `data/`.
- Each one has a history paragraph, a cuisine paragraph and a list of about 10 dishes as
  `Name (price): ingredients`.
- **The € sign is lost on text extraction** (`Hamburguesa Babylon (14,50 ):`).
  Consider restoring it during ingestion so the LLM reports prices correctly.
- Dish lists are bulleted in some files and numbered in others. Chunking per dish or per
  section keeps dish and price together in one chunk.

---

## 5. How to plug in the real backend

1. Implement `ask()` with the signature above in **`rag/pipeline.py`**. Import the
   dataclasses and exceptions from `frontend.backend_interface`.
2. Select the real backend, either with an environment variable
   (`set PAMPLONA_BACKEND=real` on Windows) or by changing `DEFAULT_BACKEND` in
   (default `"auto"`: real when `GOOGLE_API_KEY` is set in `.env`, else mock)
   `frontend/config.py` to `"real"`.
3. Run `streamlit run app.py` and test the five behaviours in section 3.

The sidebar may also send extra mock-only keys in `settings` (e.g. `simulate_failure`,
`stream_delay`). The real backend can ignore them.
