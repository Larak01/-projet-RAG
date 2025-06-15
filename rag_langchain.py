from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Simule un embedder (aucune API requise)
class FakeEmbedder:
    def embed_documents(self, texts):
        return [[0.0] * 1536 for _ in texts]  # fake 1536-dim vectors
    def embed_query(self, text):
        return [0.0] * 1536

# Simule un LLM (aucune API requise)
class FakeLLM:
    def invoke(self, messages):
        return type("Response", (), {"content": "Réponse simulée. (Pas de clé API utilisée)"})

embedder = FakeEmbedder()
llm = FakeLLM()
vector_store = FAISS.from_documents([], embedder)

def get_meta_doc(extract: str) -> str:
    return "title: unknown\nauthor: unknown\nsource: unknown\nlanguage: unknown\ntype: unknown\nthemes: []"

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool = True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    for c in chunks:
        c.metadata = {
            "document_name": doc_name,
            "insert_date": datetime.now()
        }

    if use_meta_doc:
        extract = "\n".join(c.page_content for c in chunks[:10])
        meta_doc = Document(
            page_content=get_meta_doc(extract),
            metadata={"document_name": doc_name, "insert_date": datetime.now()}
        )
        chunks.append(meta_doc)

    vector_store.add_documents(chunks)

def retrieve(question: str):
    return vector_store.similarity_search(question)

def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are a simple assistant."),
        ("system", f"Context:\n{context}"),
        ("user", question)
    ]

def answer_question(question: str, k: int = 5) -> str:
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, context)
    response = llm.invoke(messages)
    return response.content
