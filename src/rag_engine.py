import os
import re
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from frontend.backend_interface import (
    Source,
    RAGResponse,
    BackendConnectionError,
    BackendError,
)

load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pamplona_restaurants"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"
MAX_HISTORY_MESSAGES = 6  # Moving window: últimos 3 turnos (usuario + asistente)


def _extract_restaurant_name(file_name: str) -> str:
    return re.sub(r"^\d+\.\s*", "", os.path.splitext(file_name)[0]).replace("_", " ")


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
                restaurant_name = _extract_restaurant_name(file_name)
                for page in loaded_pages:
                    page.page_content = re.sub(
                        r"\((\d+[.,]\d{2})\s*\):",
                        r"(\1 €):",
                        page.page_content
                    )
                    page.metadata["restaurant"] = restaurant_name
                docs.extend(loaded_pages)

    if not docs:
        raise ValueError(f"No se encontraron archivos PDF en '{data_path}'.")
    return docs


def split_documents(docs, chunk_size=600, chunk_overlap=80):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        restaurant = chunk.metadata.get("restaurant", "")
        if restaurant and not chunk.page_content.startswith(f"[Restaurante: {restaurant}]"):
            chunk.page_content = f"[Restaurante: {restaurant}]\n{chunk.page_content}"
    return chunks


def get_vectorstore():
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


def get_rag_chain():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise ValueError("Configura una GOOGLE_API_KEY válida en el archivo .env")

    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model=DEFAULT_LLM_MODEL,
        google_api_key=api_key.strip()
    )

    system_prompt = (
        "Eres un asistente gastronómico experto en restaurantes de Pamplona y Navarra. "
        "Responde a las preguntas del usuario utilizando únicamente la información del contexto proporcionado.\n\n"
        "Reglas estrictas:\n"
        "1. Si la respuesta no está en el contexto, responde exactamente: "
        "'No dispongo de esa información en la guía gastronómica.'\n"
        "2. No inventes platos, precios, ubicaciones ni datos externos.\n"
        "3. Mantén un tono cordial, claro y conciso.\n\n"
        "Contexto:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    return (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )


def ask(question: str, history: list[dict] | None = None, settings: dict | None = None) -> RAGResponse:
    history = history or []
    settings = settings or {}

    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "tu_api_key_aqui":
        raise BackendConnectionError("Falta configurar GOOGLE_API_KEY en el archivo .env.")

    top_k = int(settings.get("top_k", 4))
    model_name = settings.get("model") or DEFAULT_LLM_MODEL

    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key.strip()
        )

        # Estrategia Moving Window: conservar solo los últimos N mensajes
        recent_history = history[-MAX_HISTORY_MESSAGES:]
        chat_history = []
        for msg in recent_history:
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
            restaurant_name = doc.metadata.get("restaurant") or _extract_restaurant_name(file_name)
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
            "Responde en el mismo idioma en el que pregunte el usuario (español o inglés) "
            "basándote únicamente en el contexto proporcionado.\n\n"
            "Reglas estrictas:\n"
            "1. Si la pregunta es ininteligible o no tiene sentido, empieza tu respuesta con '[NO_CONTEXT]' "
            "y pide amablemente al usuario que reformule su consulta.\n"
            "2. Si la respuesta no está en el contexto (restaurantes fuera de Pamplona/Navarra o temas ajenos "
            "a la gastronomía de la guía), empieza tu respuesta con '[NO_CONTEXT]' y explica amablemente "
            "que solo dispones de información sobre los restaurantes de Pamplona y Navarra de los documentos.\n"
            "3. No inventes platos, precios, ubicaciones ni datos externos.\n"
            "4. Sé claro, cordial y directo.\n\n"
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
                "embedding_model": EMBEDDING_MODEL,
                "memory_strategy": f"moving_window ({MAX_HISTORY_MESSAGES} msgs)",
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