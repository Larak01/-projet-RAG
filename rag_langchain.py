from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# ✅ Paramètres Azure DIRECTEMENT dans le code (⚠️ à ne pas publier en ligne)
embedder = AzureOpenAIEmbeddings(
    azure_endpoint="https://mon-instance.openai.azure.com",
    azure_deployment="text-embedding-ada-002",
    api_key="sk-...votre-clé-embedding...",
    api_version="2024-02-15"
)

llm = AzureChatOpenAI(
    azure_endpoint="https://mon-instance.openai.azure.com",
    azure_deployment="gpt-35-turbo",
    api_key="sk-...votre-clé-chat...",
    api_version="2024-02-15"
)

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ✅ FAISS vide
vector_store = FAISS.from_documents([], embedder)


def get_meta_doc(extract: str) -> str:
    messages = [
        ("system", "You are a librarian extracting metadata from documents."),
        ("user", f"""Extract this metadata. Answer 'unknown' if missing.
- title
- author
- source
- type (e.g. scientific, news, literature)
- language
- themes (keywords)

<content>
{extract}
</content>""")
    ]
    return llm.invoke(messages).content


def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool = True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    now = datetime.now()
    for c in chunks:
        c.metadata = {"document_name": doc_name, "insert_date": now}

    if use_meta_doc:
        extract = "\n\n".join(c.page_content for c in chunks[:min(10, len(chunks))])
        meta = Document(
            page_content=get_meta_doc(extract),
            metadata={"document_name": doc_name, "insert_date": now}
        )
        chunks.append(meta)

    vector_store.add_documents(chunks)


def retrieve(question: str):
    return vector_store.similarity_search(question)


def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"""Use the following context to answer in 3 sentences max. Say 'I don't know' if unsure.

{context}"""),
        ("user", question)
    ]


def answer_question(question: str) -> str:
    docs = retrieve(question)
    context = "\n\n".join(d.page_content for d in docs)
    messages = build_qa_messages(question, context)
    return llm.invoke(messages).content
