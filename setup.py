
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

def load_documents(file_paths):
    documents = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())
    return documents

def split_text(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

def create_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vectorstore(texts, embeddings):
    return FAISS.from_documents(texts, embeddings)

def create_response_chain(retriever, llm):
    system_prompt = (
        "You are a doctor speaking to a patient. "
        "Use the following context to answer the question. "
        "If you don't know the answer, say you don't know.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, document_chain)

def initialize_llm():
    GROQ_API_KEY = os.getenv("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")
    return ChatGroq(
        model="gemma2-9b-it",
        temperature=0.7,
        max_tokens=512
    )
