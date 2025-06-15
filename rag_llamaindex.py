import streamlit as st
from datetime import datetime

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ✅ Lecture des credentials sécurisés depuis Streamlit Cloud
llm = AzureOpenAI(
    model=st.secrets["chat"]["azure_deployment"],
    deployment_name=st.secrets["chat"]["azure_deployment"],
    api_key=st.secrets["chat"]["azure_api_key"],
    azure_endpoint=st.secrets["chat"]["azure_endpoint"],
    api_version=st.secrets["chat"]["azure_api_version"],
)

embedder = AzureOpenAIEmbedding(
    model=st.secrets["embedding"]["azure_deployment"],
    deployment_name=st.secrets["embedding"]["azure_deployment"],
    api_key=st.secrets["embedding"]["azure_api_key"],
    azure_endpoint=st.secrets["embedding"]["azure_endpoint"],
    api_version=st.secrets["embedding"]["azure_api_version"],
)

Settings.llm = llm
Settings.embed_model = embedder

vector_store = SimpleVectorStore()

def store_pdf_file(file_path: str, doc_name: str):
    loader = PyMuPDFReader()
    documents = loader.load(file_path)

    text_parser = SentenceSplitter(chunk_size=CHUNK_SIZE)
    text_chunks = []
    doc_idxs = []
    for doc_idx, doc in enumerate(documents):
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_idxs.extend([doc_idx] * len(cur_text_chunks))

    nodes = []
    for idx, text_chunk in enumerate(text_chunks):
        node = TextNode(text=text_chunk)
        src_doc = documents[doc_idxs[idx]]
        node.metadata = {
            'document_name': doc_name,
            'insert_date': datetime.now(),
            **src_doc.metadata
        }
        node.embedding = embedder.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        nodes.append(node)

    vector_store.add(nodes)

def retrieve(question: str):
    query_embedding = embedder.get_query_embedding(question)
    query_mode = "default"

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding,
        similarity_top_k=5,
        mode=query_mode
    )

    result = vector_store.query(vector_store_query)
    return result.nodes

def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"""Use the following context to answer in 3 sentences max. Say 'I don't know' if unsure.\n{context}"""),
        ("user", question)
    ]

def answer_question(question: str, k: int = 5) -> str:
    docs = retrieve(question)
    context = "\n\n".join(doc.get_content() for doc in docs)
    messages = build_qa_messages(question, context)
    response = llm.invoke(messages)
    return response.content
