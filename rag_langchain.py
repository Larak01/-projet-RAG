import yaml
from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Erreur YAML: {e}")
            return None

# ✅ Charger config locale
config = read_config("config.yaml")

# ✅ Embeddings
embedder = AzureOpenAIEmbeddings(
    azure_endpoint=config["embedding"]["azure_endpoint"],
    azure_deployment=config["embedding"]["azure_deployment"],
    api_key=config["embedding"]["azure_api_key"],
    api_version=config["embedding"]["azure_api_version"]
)

# ✅ LLM
llm = AzureChatOpenAI(
    azure_endpoint=config["chat"]["azure_endpoint"],
    azure_deployment=config["chat"]["azure_deployment"],
    api_key=config["chat"]["azure_api_key"],
    api_version=config["chat"]["azure_api_version"]
)

# ✅ Index vide initial
vector_store = FAISS.from_documents([], embedder)


def get_meta_doc(extract: str) -> str:
    messages = [
        ("system", "You are a librarian extracting metadata from documents."),
        ("user", f"""Extract the following metadata. Answer 'unknown' if missing.
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
    response = llm.invoke(messages)
    return response.content


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
    response = llm.invoke(messages)
    return response.content
