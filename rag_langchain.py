import os
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Paramètres
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 🔐 Lecture sécurisée de la clé API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialisation des composants
embedder = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
vector_store = FAISS.from_documents([], embedder)

# Génère un document de métadonnées à partir d’un extrait
def get_meta_doc(extract: str) -> str:
    messages = [
        ("system", "You are a librarian extracting metadata from documents."),
        ("user", f"""Extract metadata:
- title
- author
- source
- type (e.g. scientific paper, news, literature, etc.)
- language
- themes (list of keywords)

<content>
{extract}
</content>""")
    ]
    response = llm.invoke(messages)
    return response.content

# Indexe un fichier PDF
def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool = True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = splitter.split_documents(docs)

    for split in all_splits:
        split.metadata = {"document_name": doc_name, "insert_date": datetime.now()}

    if use_meta_doc:
        extract = "\n\n".join(split.page_content for split in all_splits[:10])
        meta_doc = Document(
            page_content=get_meta_doc(extract),
            metadata={"document_name": doc_name, "insert_date": datetime.now()}
        )
        all_splits.append(meta_doc)

    vector_store.add_documents(all_splits)

# Recherche vectorielle
def retrieve(question: str):
    return vector_store.similarity_search(question)

# Construction du prompt QA
def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"""Use this context to answer the question below in 3 short sentences. Say "I don't know" if not sure.\n\n{context}"""),
        ("user", question)
    ]

# Répondre à une question
def answer_question(question: str, k: int = 5) -> str:
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, context)
    response = llm.invoke(messages)
    return response.content
