import streamlit as st
import requests
import os
import uuid

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="TechDocAI",
    page_icon="🧠",
    layout="centered"
)

API_URL = "http://localhost:8000/api"

# --- UI HEADER ---
st.title("🧠 TechDocAI")
st.markdown("*Your Production-Grade RAG Assistant*")
st.divider()

# --- SESSION STATE INITIALIZATION ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- SIDEBAR: DOCUMENT INGESTION ---
with st.sidebar:
    st.header("📂 Ingestion")
    st.info("Upload PDF, TXT, or MD files to build your knowledge base.")
    
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])
    
    if uploaded_file and st.button("🚀 Process Document", use_container_width=True):
        with st.spinner("Processing..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                res = requests.post(f"{API_URL}/ingest/", files=files)
                if res.status_code == 200:
                    st.success(f"Successfully indexed: {uploaded_file.name}")
                else:
                    st.error(f"Error: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

    st.divider()
    st.header("📊 Quality Audit (Ragas)")
    if st.button("⚖️ Run Accuracy Check", use_container_width=True):
        with st.spinner("LLaMA-3 is auditing the RAG pipeline..."):
            try:
                res = requests.post(f"{API_URL}/evaluate/")
                if res.status_code == 200:
                    data = res.json()["scores"]
                    st.success("Audit Complete!")
                    
                    # Display metrics in a grid
                    col1, col2 = st.columns(2)
                    col1.metric("Faithfulness", f"{data.get('faithfulness', 0):.2f}")
                    col2.metric("Relevancy", f"{data.get('answer_relevancy', 0):.2f}")
                    
                    col3, col4 = st.columns(2)
                    col3.metric("Precision", f"{data.get('context_precision', 0):.2f}")
                    col4.metric("Recall", f"{data.get('context_recall', 0):.2f}")
                else:
                    st.error("Audit Failed.")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.caption("Powered by Groq, ChromaDB & BGE")

# --- MAIN: CHAT INTERFACE ---
# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}** ({src['domain']})")
                    st.caption(src['content'][:200] + "...")
                    st.divider()

query = st.chat_input("Ask a technical question...")

if query:
    # 1. Add and display user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 2. Get Response from Backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "question": query, 
                    "k": 5,
                    "session_id": st.session_state.session_id
                }
                res = requests.post(f"{API_URL}/search/", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer")
                    sources = data.get("results", [])

                    # 3. Display Answer
                    st.markdown(answer)
                    
                    # 4. Save and Show Sources
                    if sources:
                        with st.expander("📚 View Sources"):
                            for i, src in enumerate(sources):
                                st.markdown(f"**Source {i+1}** ({src['domain']})")
                                st.caption(src['content'][:200] + "...")
                                st.divider()
                    
                    # 5. Save to session state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error("Could not connect to the backend server.")
                st.info("Please ensure `python src/main.py` is running.")
