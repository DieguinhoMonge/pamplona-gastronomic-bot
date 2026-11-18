import os
import re
import time
from functools import lru_cache
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
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
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

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
                    "usando el historial, sin responderla. Si ya es independiente, devuélvela igual."
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
        vectorstore = get_vectorstore()
        results = vectorstore.similarity_search_with_score(search_query, k=top_k)
        retrieval_ms = round((time.perf_counter() - t0) * 1000, 1)

        sources = []
        context_blocks = []
        for doc, dist in results:
            file_name = os.path.basename(doc.metadata.get("source", "documento.pdf"))
            restaurant_name = re.sub(r"^\d+\.\s*", "", os.path.splitext(file_name)[0]).replace("_", " ")
            page_num = int(doc.metadata.get("page", 0)) + 1
            relevance = round(1.0 / (1.0 + float(dist)), 3)

            sources.append(Source(
                restaurant=restaurant_name,
                file=file_name,
                page=page_num,
                chunk=doc.page_content,
                score=relevance
            ))
            context_blocks.append(f"[{restaurant_name} - Pág. {page_num}]\n{doc.page_content}")

        context_text = "\n\n".join(context_blocks)

        system_prompt = (
            "Eres Txoko, un asistente gastronómico experto en restaurantes de Pamplona y Navarra. "
            "Responde basándote únicamente en el contexto proporcionado.\n"
            "IDIOMA: responde SIEMPRE en el idioma de la última pregunta del usuario. "
            "Si pregunta en inglés, responde en inglés aunque el contexto esté en español "
            "(traduce la información, pero mantén los nombres de platos y restaurantes tal cual). "
            "Si pregunta en español, responde en español.\n\n"
            "Reglas estrictas:\n"
            "1. Si la pregunta es ininteligible o no tiene sentido, empieza tu respuesta con '[NO_CONTEXT]' "
            "y pide amablemente al usuario que reformule su consulta.\n"
            "2. Si la respuesta no está en el contexto (restaurantes fuera de Pamplona/Navarra o temas ajenos "
            "a la gastronomía de la guía), empieza tu respuesta con '[NO_CONTEXT]' y explica amablemente "
            "que solo tienes información sobre los restaurantes de Pamplona y Navarra de la guía.\n"
            "3. No inventes platos, precios, ubicaciones ni datos externos.\n"
            "4. Sé claro, cordial y directo. No te presentes ni saludes en cada respuesta.\n"
            "5. Las reglas 1 y 2 también siguen la regla de IDIOMA: si la pregunta está en inglés, "
            "la respuesta después de '[NO_CONTEXT]' debe estar en inglés.\n\n"
            "Contexto:\n{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        t1 = time.perf_counter()
        chain = prompt | llm | StrOutputParser()
        raw_answer = chain.invoke({
            "context": context_text,
            "chat_history": chat_history,
            "input": question
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