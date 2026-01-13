import streamlit as st
import requests
import json
import pandas as pd

# Constants
API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="TechDocAI Workbench", layout="wide", page_icon="🤖")

st.title("🤖 TechDocAI Verification Workbench")
st.markdown("Use this interface to verify the **Ingestion Pipeline** and **Hybrid Search** logic.")

# Custom CSS
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; }
    .reportview-container { background: #f0f2f6; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 1.1rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR: INGESTION ---
with st.sidebar:
    st.header("📂 Document Ingestion")
    st.info("Upload PDF, TXT, or MD files to add them to the Vector Database.")
    
    uploaded_file = st.file_uploader("Select Document", type=["pdf", "txt", "md"])
    
    if uploaded_file:
        if st.button("🚀 Run Ingestion Pipeline"):
            with st.spinner("Uploading & Processing..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    res = requests.post(f"{API_URL}/ingest/", files=files)
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.success("✅ Ingestion Successful!")
                        st.json(data)
                    else:
                        st.error(f"❌ Error {res.status_code}: {res.text}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")

    st.markdown("---")
    st.header("⚙️ Database Controls")
    if st.button("🗑️ Reset Database (DANGER)"):
        st.warning("Feature not connected to API yet for safety.")

# --- MAIN CONTENT: SEARCH VERIFICATION ---
st.subheader("🔍 Hybrid Search Verification")

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("Enter a technical query:", placeholder="e.g., How does binary search work?")
with col2:
    k_val = st.number_input("Top K", min_value=1, max_value=20, value=5)

if query:
    if st.button("Search") or query:
        with st.spinner("Searching..."):
            try:
                payload = {"question": query, "k": k_val}
                res = requests.post(f"{API_URL}/search/", json=payload)
                
                if res.status_code == 200:
                    results = res.json()
                    answer = results.get("answer", "No answer generated.")
                    hits = results.get("results", [])
                    
                    st.success("✅ Generated Answer:")
                    st.markdown(f"### {answer}")
                    st.markdown("---")
                    
                    st.subheader(f"📚 Sources ({len(hits)} chunks used)")
                    
                    for i, doc in enumerate(hits):
                        score_display = "" # Score is fused, might not display directly easily without change
                        with st.expander(f"Result #{i+1} | {doc['domain'].upper()} | {doc['metadata'].get('source', 'Unknown')}"):
                            st.markdown(f"**Content Snippet:**")
                            st.code(doc['content'], language="text")
                            st.markdown("**Metadata:**")
                            st.json(doc['metadata'])
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Failed to connect to backend: {e}")
                st.info("Make sure `src/main.py` is running!")

st.markdown("---")
st.caption("TechDocAI Verification UI")
