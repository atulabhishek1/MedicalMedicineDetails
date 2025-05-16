# app.py

import streamlit as st
import os
from setup import load_documents, split_text, create_embeddings, create_vectorstore, create_response_chain, initialize_llm

# Set page configuration
st.set_page_config(page_title="Medical Assistant", layout="wide")

st.title("🩺 Medical Assistant Chatbot")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'llm' not in st.session_state:
    st.session_state.llm = initialize_llm()

# Load default PDF
default_pdf_path = "MedicineBook.pdf"
if os.path.exists(default_pdf_path):
    st.success("Medicine PDF is already uploaded and loaded.")
    documents = load_documents([default_pdf_path])
    texts = split_text(documents)
    embeddings = create_embeddings()
    st.session_state.vectorstore = create_vectorstore(texts, embeddings)
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
    st.session_state.chain = create_response_chain(st.session_state.retriever, st.session_state.llm)
else:
    st.error(f"Default PDF '{default_pdf_path}' not found.")

# File uploader for additional PDFs
uploaded_files = st.file_uploader("Upload additional PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    uploaded_paths = []
    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        uploaded_paths.append(uploaded_file.name)

    # Load and process uploaded PDFs
    documents = load_documents(uploaded_paths)
    texts = split_text(documents)
    embeddings = create_embeddings()
    st.session_state.vectorstore = create_vectorstore(texts, embeddings)
    st.session_state.retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2})
    st.session_state.chain = create_response_chain(st.session_state.retriever, st.session_state.llm)
    st.success("Uploaded PDFs have been processed and added to the knowledge base.")

# Chat interface
if st.session_state.chain:
    query = st.text_input("Ask a medical question:")
    if query:
        response = st.session_state.chain.invoke({"input": query})
        st.write("**Answer:**")
        st.write(response['answer'])
