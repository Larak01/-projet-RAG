import yaml
from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

vector_store = None

# Config directe sans secrets
config = {
    "embedding": {
        "azure_endpoint": "https://mon-instance.openai.azure.com",
        "azure_deployment": "text-embedding-ada-002",
        "azure_api_version": "2024-02-15",
        "api_key": "sk-xxxxx"
    },
    "chat": {
        "azure_endpoint": "https://mon-instance.openai.azure.com",
        "azure_deployment": "gpt-35-turbo",
        "azure_api_version": "2024-02-15",
        "api_key": "sk-yyyyy"
    }
}

embedder = AzureOpenAIEmbeddings(**config["embedding"])
llm = AzureChatOpenAI(**config["chat"])

def get_meta_doc(extract: str) -> str:
    messages = [
        ("system", "You are a librarian extracting metadata from documents."),
        ("user", f"""Extract from the content the following metadata.\nAnswer 'unknown' if you cannot find or generate the information.\nMetadata list:\n- title\n- author\n- source\n- type of content (e.g. scientific paper, literature, news, etc.)\n- language\n- themes as a list of keywords\n\n<content>\n{extract}\n</content>""")
    ]
    return llm.invoke(messages).content

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool = True):
    global vector_store
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = splitter.split_documents(docs)

    for split in all_splits:
        split.metadata = {
            "document_name": doc_name,
            "insert_date": datetime.now()
        }

    if use_meta_doc:
        extract = "\n\n".join(split.page_content for split in all_splits[:min(10, len(all_splits))])
        meta_doc = Document(
            page_content=get_meta_doc(extract),
            metadata={"document_name": doc_name, "insert_date": datetime.now()}
        )
        all_splits.append(meta_doc)

    vector_store = FAISS.from_documents(all_splits, embedder)

def retrieve(question: str):
    if vector_store is None:
        raise ValueError("Vector store not initialized.")
    return vector_store.similarity_search(question)

def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"""Use the following context to answer in 3 sentences max. Say 'I don't know' if unsure.\n{context}"""),
        ("user", question)
    ]

def answer_question(question: str) -> str:
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, context)
    return llm.invoke(messages).content
