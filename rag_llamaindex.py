from datetime import datetime
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
import streamlit as st

CHUNK_SIZE = 1000
embedder = AzureOpenAIEmbedding(**st.secrets["embedding"])
llm = AzureOpenAI(**st.secrets["chat"])
Settings.llm = llm
Settings.embed_model = embedder
vector_store = SimpleVectorStore()

def store_pdf_file(file_path: str, doc_name: str):
    reader = PyMuPDFReader()
    docs = reader.load(file_path)
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE)
    chunks, ids = [], []
    for i, doc in enumerate(docs):
        texts = splitter.split_text(doc.text)
        chunks.extend(texts)
        ids.extend([i] * len(texts))
    nodes = []
    for idx, chunk in enumerate(chunks):
        node = TextNode(text=chunk)
        node.metadata = {'document_name': doc_name, 'insert_date': datetime.now()}
        node.embedding = embedder.get_text_embedding(chunk)
        nodes.append(node)
    vector_store.add(nodes)

def retrieve(question: str, k: int = 5):
    embedding = embedder.get_query_embedding(question)
    result = vector_store.query(VectorStoreQuery(query_embedding=embedding, similarity_top_k=k))
    return result.nodes

def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"Use the context below to answer. Be brief.\n{context}"),
        ("user", question)
    ]

def answer_question(question: str, k: int = 5) -> str:
    nodes = retrieve(question, k)
    context = "\n\n".join(n.get_content() for n in nodes)
    messages = build_qa_messages(question, context)
    return llm.invoke(messages).content
