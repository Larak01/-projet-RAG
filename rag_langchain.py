from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import FakeEmbeddings
from langchain_core.documents import Document

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding mock
embedder = FakeEmbeddings()
vector_store = FAISS.from_documents([Document(page_content="Document vide")], embedder)

def store_pdf_file(file_path: str, doc_name: str):
    # Simule l’ajout d’un document (sans lecture réelle)
    doc = Document(page_content=f"Contenu simulé de {doc_name}", metadata={"document_name": doc_name, "insert_date": datetime.now()})
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents([doc])
    vector_store.add_documents(splits)

def retrieve(question: str):
    return vector_store.similarity_search(question)

def answer_question(question: str, k: int = 5) -> str:
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    return f"[Réponse simulée]\n\nContexte utilisé:\n{context}"
