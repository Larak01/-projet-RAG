import streamlit as st
from datetime import datetime

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

llm = AzureOpenAI(
    model=st.secrets["chat_azure_deployment"],
    deployment_name=st.secrets["chat_azure_deployment"],
    api_key=st.secrets["chat_azure_api_key"],
    azure_endpoint=st.secrets["chat_azure_endpoint"],
    api_version=st.secrets["chat_azure_api_version"]
)

embedder = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name=st.secrets["embedding_azure_deployment"],
    azure_endpoint=st.secrets["embedding_azure_endpoint"],
    api_key=st.secrets["embedding_azure_api_key"],
    api_version=st.secrets["embedding_azure_api_version"]
)

Settings.llm = llm
Settings.embed_model = embedder

st.write("🟢 Clé détectée:", st.secrets["chat_azure_deployment"])


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
        node.metadata = src_doc.metadata
        nodes.append(node)

    for node in nodes:
        node_embedding = embedder.get_text_embedding(node.get_content(metadata_mode="all"))
        node.embedding = node_embedding

    vector_store.add(nodes)
    return

def delete_file_from_store(name: str) -> int:
    raise NotImplementedError('function not implemented for Llamaindex')

def inspect_vector_store(top_n: int=10) -> list:
    raise NotImplementedError('function not implemented for Llamaindex')

def get_vector_store_info():
    raise NotImplementedError('function not implemented for Llamaindex')

def retrieve(question: str):
    query_embedding = embedder.get_query_embedding(question)

    vector_store_query = VectorStoreQuery(
        query_embedding=query_embedding, similarity_top_k=5, mode="default"
    )

    query_result = vector_store.query(vector_store_query)
    return query_result.nodes

def build_qa_messages(question: str, context: str) -> list[str]:
    messages = [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"""Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise.
        {context}"""),
        ("user", question)
    ]
    return messages

def answer_question(question: str) -> str:
    docs = retrieve(question)
    docs_content = "\n\n".join(doc.get_content() for doc in docs)
    messages = build_qa_messages(question, docs_content)
    response = llm.invoke(messages)
    return response.content
