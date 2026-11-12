import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "pamplona_restaurants"


def load_documents(data_path=DATA_DIR):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"La carpeta '{data_path}' no existe.")

    docs = []
    for file_name in os.listdir(data_path):
        if file_name.endswith(".pdf"):
            full_path = os.path.join(data_path, file_name)
            loader = PyPDFLoader(full_path)
            docs.extend(loader.load())

    if not docs:
        raise ValueError(f"No se encontraron archivos PDF en '{data_path}'.")
    return docs


def split_documents(docs, chunk_size=600, chunk_overlap=80):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)


def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

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
        model="gemini-3.5-flash-lite",
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

    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["input"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain