import os
import pickle
import time
import requests
import streamlit as st
from bs4 import BeautifulSoup

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Streamlit UI setup
st.set_page_config(page_title="Deepak: News Research Tool", layout="wide")
st.title("🧠 Deepak: News Research Tool")
st.sidebar.title("🔗 Input Settings")

# User Configs
st.sidebar.markdown("### Example:")
st.sidebar.markdown("- https://news.un.org/en/story/2025/07/1165146")
urls = [st.sidebar.text_input(f"URL {i+1}", "") for i in range(3)]
chunk_size = st.sidebar.slider("🧩 Chunk Size", 300, 1500,0, step=100)
process_btn = st.sidebar.button("🚀 Process")

VECTORSTORE_PATH = "faiss_index.pkl"

# Utilities
def fetch_article_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, "html.parser")
        text = ' '.join(p.get_text() for p in soup.find_all("p"))
        return text if len(text.strip()) > 100 else None
    except Exception as e:
        st.warning(f"❌ {url}: {e}")
        return None

def create_vectorstore(docs, chunk_sz):
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", ".", ","],
                                               chunk_size=chunk_sz,
                                               chunk_overlap=chunk_sz // 5)
    split_docs = splitter.split_documents(docs)
    if not split_docs:
        return None
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(split_docs, embedder)

def save_vectorstore(store, path):
    with open(path, "wb") as f:
        pickle.dump(store, f)

def load_vectorstore(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def load_qa_chain():
    try:
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Article Processor
if process_btn:
    valid_urls = [url.strip() for url in urls if url.strip()]
    if not valid_urls:
        st.sidebar.error("⚠️ Please enter at least one URL.")
        st.stop()

    with st.status("Fetching articles...", expanded=False):
        docs = []
        for url in valid_urls:
            text = fetch_article_text(url)
            if text:
                docs.append(Document(page_content=text, metadata={"source": url}))
            else:
                st.warning(f"⚠️ Skipped or empty: {url}")

    if not docs:
        st.error("❌ No valid content found.")
        st.stop()

    with st.status("Indexing..."):
        vectorstore = create_vectorstore(docs, chunk_size)
        if not vectorstore:
            st.error("⚠️ Failed to build vectorstore.")
            st.stop()
        save_vectorstore(vectorstore, VECTORSTORE_PATH)
        st.success("✅ Index created and saved.")

# Question-answer UI
query = st.text_input("💬 Ask something from the articles:")
if query:
    vectorstore = load_vectorstore(VECTORSTORE_PATH)
    if not vectorstore:
        st.warning("📦 Please process URLs first.")
        st.stop()

    llm = load_qa_chain()
    if not llm:
        st.stop()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    with st.spinner("🧠 Thinking..."):
        response = chain({"query": query})
        st.subheader("🔎 Answer")
        st.write(response["result"])

        sources = response.get("source_documents", [])
        if sources:
            st.markdown("#### 🔗 Sources")
            for s in sources:
                st.markdown(f"- [{s.metadata['source']}]({s.metadata['source']})")
