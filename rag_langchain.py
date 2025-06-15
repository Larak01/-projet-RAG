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

# ✅ Charger depuis config.yaml
config = read_config("secrets/config.yaml")

embedder = AzureOpenAIEmbeddings(**config["embedding"])
llm = AzureChatOpenAI(**config["chat"])

# ✅ Initialiser la base FAISS vide
vector_store = FAISS.from_documents([], embedder)


def get_meta_doc(extract: str) -> str:
    messages = [
        ("system", "You are a librarian extracting metadata from documents."),
        ("user", f"""Extract from the content the following metadata.\nAnswer 'unknown' if you cannot find or generate the information.\nMetadata list:\n- title\n- author\n- source\n- type of content (e.g. scientific paper, literature, news, etc.)\n- language\n- themes as a list of keywords\n\n<content>\n{extract}\n</content>""")
    ]
    response = llm.invoke(messages)
    return response.content


def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool = True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(docs)

    for split in all_splits:
        split.metadata = {
            'document_name': doc_name,
            'insert_date': datetime.now()
        }

    if use_meta_doc:
        extract = '\n\n'.join(split.page_content for split in all_splits[:min(10, len(all_splits))])
        meta_doc = Document(
            page_content=get_meta_doc(extract),
            metadata={'document_name': doc_name, 'insert_date': datetime.now()}
        )
        all_splits.append(meta_doc)

    vector_store.add_documents(all_splits)


def retrieve(question: str):
    return vector_store.similarity_search(question)


def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"""Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
{context}"""),
        ("user", question)
    ]


def answer_question(question: str) -> str:
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, context)
    response = llm.invoke(messages)
    return response.content
