import math
import os
import re
import time
import unicodedata
from collections import Counter
from functools import lru_cache
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

from frontend.backend_interface import (
    Source,
    RAGResponse,
    BackendConnectionError,
    BackendError,
)

from frontend.config import CHROMA_DIR as _CHROMA_PATH, DOCS_DIR, ROOT

load_dotenv(ROOT / ".env")

# Absolute paths, so the app works no matter which folder it is launched from.
DATA_DIR = str(DOCS_DIR)
CHROMA_DIR = str(_CHROMA_PATH)
COLLECTION_NAME = "pamplona_restaurants"
# Multilingual: the PDFs are in Spanish and users ask in English or Spanish.
# (all-MiniLM-L6-v2 is English-only and missed Spanish queries in our tests: 6/8 vs 7/8.)
# Changing this requires deleting chroma_db/ so the index is rebuilt.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def load_documents(data_path=DATA_DIR):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"La carpeta '{data_path}' no existe.")

    docs = []
    for root, _, files in os.walk(data_path):
        for file_name in sorted(files):
            if file_name.endswith(".pdf"):
                full_path = os.path.join(root, file_name)
                loader = PyPDFLoader(full_path)
                loaded_pages = loader.load()
                for page in loaded_pages:
                    page.page_content = re.sub(
                        r"\((\d+[.,]\d{2})\s*\):",
                        r"(\1 €):",
                        page.page_content
                    )
                docs.extend(loaded_pages)

    if not docs:
        raise ValueError(f"No se encontraron archivos PDF en '{data_path}'.")
    return docs


def split_documents(docs, chunk_size=600, chunk_overlap=80):
    # Dishes wrap over several lines in the PDFs, so splitting on every "\n" cut
    # dishes in half (name + price in one chunk, ingredients in the next). Split on
    # dish bullets ("• ", "* ") or numbered items ("3. ") first, so each dish stays whole.
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[r"\n(?=[•*] )", r"\n(?=\d+\.\s)", r"\n", r" "],
        is_separator_regex=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    # Prefix every chunk with its restaurant, so a dish chunk still "knows" which
    # restaurant it belongs to (improves retrieval for "Hamburguesa Babylon", etc.).
    for chunk in chunks:
        chunk.page_content = f"Restaurante: {restaurant_from_path(chunk.metadata.get('source', ''))}\n{chunk.page_content}"
    return chunks


def restaurant_from_path(path: str) -> str:
    """'data/11. Restaurante_Tubal.pdf' -> 'Restaurante Tubal'."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"^\d+\.\s*", "", stem).replace("_", " ")


@lru_cache(maxsize=1)
def get_vectorstore():
    """Load (or build on first run) the Chroma store. Cached: the embedding model
    is loaded once per process instead of on every question."""
    # Normalized vectors -> Chroma's L2 distance maps to cosine similarity (cos = 1 - d/2),
    # which gives the UI a meaningful 0-1 relevance score.
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL,
                                       encode_kwargs={"normalize_embeddings": True})

    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings
        )

    docs = load_documents()
    chunks = split_documents(docs)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR
    )


# --------------------------------------------------------------------------- prompt
# Written in English: the model follows English instructions more reliably, and the
# examples pin down the borderline cases we saw in testing ("chocolate", "good pintxos").
SYSTEM_PROMPT = """You are Txoko, a food guide for the restaurants of Pamplona and Navarra.
You answer ONLY from the CONTEXT below: excerpts (in Spanish) from the guide's restaurant documents.

HOW TO ANSWER
- If the CONTEXT contains anything related to the question, answer with it. Partial answers are fine:
  give what the context has and briefly say what is missing.
- Short or one-word questions are valid searches. "chocolate" means "which dishes in the guide have
  chocolate?": list every matching dish in the context with its restaurant and price.
- Subjective questions ("good", "best", "recommend") are valid: don't rank by opinion, present the
  matching options from the context with their details.
- Dietary questions (e.g. "vegetarian"): list dishes whose ingredients in the context contain no meat
  or fish, and say that the guide does not label dishes as vegetarian.
- Never invent dishes, prices, places or facts that are not in the context.
- Be concise and use bullet lists for dishes. Do not greet or introduce yourself.
- Language: reply in the language requested at the end of the user's message. Translate the
  information, but keep dish and restaurant names as written.

WHEN NOT TO ANSWER (start the reply with the tag [NO_CONTEXT])
- The question is not about food/restaurants (history, geography, politics, general knowledge...)
  or it is about places outside Pamplona/Navarra. After the tag, politely say you only know the
  restaurants in the guide.
- The message is gibberish (e.g. "asdkjh qwe"). After the tag, ask the user to rephrase.
  A real food word is never gibberish.
- Never use the tag if you are listing any dish or restaurant from the context.

Examples:
- "chocolate" -> list of chocolate desserts with restaurant and price (no tag)
- "Where can I eat good pintxos?" -> the restaurants in the context that serve pintxos (no tag)
- "What happened in America in 1997?" -> [NO_CONTEXT] + "I only know about the restaurants in the guide..."

CONTEXT
{context}"""

_SPANISH_HINTS = re.compile(
    r"[¿¡ñáéíóú]|\b(qué|que|dónde|donde|cuál|cual|cuánto|cuanto|cómo|como|el|la|los|las|de|del|"
    r"hay|tiene|tienen|quiero|busco|restaurantes?|platos?|mejor|para|con|y|es|un|una)\b", re.I)


def detect_language(text: str) -> str:
    """Tiny heuristic: Spanish if the question shows Spanish markers, otherwise English.
    (Single food words like 'chocolate' default to English, matching the UI language.)"""
    return "Spanish" if len(_SPANISH_HINTS.findall(text)) >= 1 and not re.search(
        r"\b(the|what|where|which|how|is|are|can|do|does|i|you)\b", text, re.I) else "English"


# --------------------------------------------------------------------------- hybrid retrieval
# Semantic search alone misses rare exact words (e.g. "pintxos" appears in only one PDF),
# so we also run a BM25 keyword search and fuse both rankings (Reciprocal Rank Fusion).

def _tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))  # strip accents
    words = re.findall(r"[a-z0-9]+", text)
    # very light stemming (drop plural "s") so "pintxos" matches "pintxo", "chocolates" matches "chocolate"
    return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if w not in _STOPWORDS]


_STOPWORDS = set("""a al and are de del el en es esta este for good how i in is it la las los me my
of on que the to un una what where which with y can eat do does tiene hay donde cual cuales
restaurante restaurant restaurantes restaurants mejor best""".split())


@lru_cache(maxsize=1)
def _bm25_index():
    """Tokenized chunks + document frequencies, built once from the Chroma collection."""
    data = get_vectorstore().get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
    toks = [_tokens(d.page_content) for d in docs]
    df = Counter(w for t in toks for w in set(t))
    avgdl = sum(len(t) for t in toks) / max(len(toks), 1)
    return docs, toks, df, avgdl


def _bm25_search(query: str, k: int, k1: float = 1.5, b: float = 0.75) -> list[Document]:
    docs, toks, df, avgdl = _bm25_index()
    q = set(_tokens(query))
    n = len(docs)
    scored = []
    for doc, t in zip(docs, toks):
        tf = Counter(t)
        s = sum(math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5)) * tf[w] * (k1 + 1)
                / (tf[w] + k1 * (1 - b + b * len(t) / avgdl)) for w in q if tf[w])
        if s > 0:
            scored.append((s, doc))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]


def hybrid_search(query: str, k: int, rrf_k: int = 60) -> list[tuple[Document, float]]:
    """Return the top-k chunks as (doc, cosine_similarity), fusing semantic + keyword ranks."""
    vectorstore = get_vectorstore()
    n_total = vectorstore._collection.count()
    semantic = vectorstore.similarity_search_with_score(query, k=n_total)  # all chunks; corpus is tiny
    cosine = {d.page_content: round(max(0.0, 1 - float(dist) / 2), 3) for d, dist in semantic}
    keyword = _bm25_search(query, k=max(k * 3, 10))

    fused: dict[str, float] = {}
    by_text: dict[str, Document] = {}
    for ranking in ([d for d, _ in semantic[: max(k * 3, 10)]], keyword):
        for rank, d in enumerate(ranking):
            fused[d.page_content] = fused.get(d.page_content, 0.0) + 1.0 / (rrf_k + rank + 1)
            by_text[d.page_content] = d
    best = sorted(fused, key=lambda t: -fused[t])[:k]
    return [(by_text[t], cosine.get(t, 0.0)) for t in best]


def ask(question: str, history: list[dict] | None = None, settings: dict | None = None) -> RAGResponse:
    history = history or []
    settings = settings or {}

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "tu_api_key_aqui":
        # BackendError (not ConnectionError) so the UI shows this exact message.
        raise BackendError("Falta configurar GOOGLE_API_KEY en el archivo .env.")

    top_k = int(settings.get("top_k", 4))
    model_name = settings.get("model") or DEFAULT_MODEL
    temperature = float(settings.get("temperature", 0.2))

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key.strip()
        )

        chat_history = []
        for msg in history:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                chat_history.append(AIMessage(content=msg.get("content", "")))

        search_query = question
        if chat_history:
            rewrite_prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "Reescribe la última pregunta del usuario como una pregunta completa e independiente "
                    "usando el historial, sin responderla. Usa el historial SOLO si la pregunta depende de él "
                    "(p. ej. 'and how much is it?', '¿y el postre?'). Si cambia de tema o ya es independiente, "
                    "devuélvela EXACTAMENTE igual. Devuelve solo la pregunta."
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            rewrite_chain = rewrite_prompt | llm | StrOutputParser()
            search_query = rewrite_chain.invoke({
                "chat_history": chat_history,
                "input": question
            }).strip()

        t0 = time.perf_counter()
        results = hybrid_search(search_query, k=top_k)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

        sources = []
        context_blocks = []
        for doc, relevance in results:
            file_name = os.path.basename(doc.metadata.get("source", "documento.pdf"))
            restaurant_name = restaurant_from_path(file_name)
            page_num = int(doc.metadata.get("page", 0)) + 1

            sources.append(Source(
                restaurant=restaurant_name,
                file=file_name,
                page=page_num,
                chunk=doc.page_content,
                score=relevance
            ))
            context_blocks.append(f"[{restaurant_name} - Pág. {page_num}]\n{doc.page_content}")

        context_text = "\n\n".join(context_blocks)

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}\n\n[Reply in {language}.]")
        ])

        t1 = time.perf_counter()
        chain = prompt | llm | StrOutputParser()
        raw_answer = chain.invoke({
            "context": context_text,
            "chat_history": chat_history,
            "input": question,
            "language": detect_language(question),
        }).strip()
        generation_ms = round((time.perf_counter() - t1) * 1000, 1)

        if raw_answer.startswith("[NO_CONTEXT]"):
            grounded = False
            answer = raw_answer.replace("[NO_CONTEXT]", "", 1).strip()
            sources = []
        else:
            grounded = True
            answer = raw_answer

        return RAGResponse(
            answer=answer,
            sources=sources,
            grounded=grounded,
            debug={
                "rewritten_query": search_query,
                "model": model_name,
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
            }
        )

    except (BackendConnectionError, BackendError):
        raise
    except (FileNotFoundError, ValueError) as e:
        raise BackendError(str(e))
    except Exception as e:
        raise BackendConnectionError(f"No se pudo conectar con el servicio de IA: {e}")