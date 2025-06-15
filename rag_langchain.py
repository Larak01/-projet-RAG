from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.fake import FakeEmbeddings  # Simule un embedder
from langchain.llms.fake import FakeLLM  # Simule un LLM
from langchain_core.documents import Document

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedder factice pour usage hors-ligne
embedder = FakeEmbeddings()
llm = FakeLLM()

vector_store = None  # Créé après ajout de documents


def store_pdf_file(file_path: str, doc_name: str):
    global vector_store
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)

    for doc in splits:
        doc.metadata = {
            'document_name': doc_name,
            'insert_date': datetime.now()
        }

    # Indexation avec vecteur factice
    vector_store = FAISS.from_documents(splits, embedder)


def retrieve(question: str):
    if not vector_store:
        return []
    return vector_store.similarity_search(question)


def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are a helpful assistant."),
        ("system", f"Here is the context: {context}"),
        ("user", question)
    ]


def answer_question(question: str, k: int = 5) -> str:
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, context)
    return llm.invoke(messages)
