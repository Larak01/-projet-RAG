from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Configuration manuelle (remplace les valeurs par tes propres clés si besoin)
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = ""  # Laisse vide si tu n'en as pas

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialiser les composants avec vérification minimale
embedder = AzureOpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=OPENAI_API_KEY
)

llm = AzureChatOpenAI(
    model=CHAT_MODEL,
    api_key=OPENAI_API_KEY
)

# Pas de vecteur tant qu'on n'a pas de documents
vector_store = None

def store_pdf_file(file_path: str, doc_name: str):
    global vector_store

    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata = {
            "document_name": doc_name,
            "insert_date": datetime.now()
        }

    # Création du vecteur uniquement si on a des chunks
    if chunks:
        vector_store = FAISS.from_documents(chunks, embedder)


def retrieve(question: str):
    if vector_store is None:
        return []
    return vector_store.similarity_search(question)


def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        (
            "system",
            f"""Use the following context to answer in 3 sentences max. Say 'I don't know' if unsure.\n{context}"""
        ),
        ("user", question)
    ]


def answer_question(question: str) -> str:
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, context)
    response = llm.invoke(messages)
    return response.content
