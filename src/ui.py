import streamlit as st
import requests

st.set_page_config(page_title="TechDocAI", layout="wide")
st.title("TechDocAI Assistant")

# Sidebar for Ingestion
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "txt", "md"])
    if uploaded_file is not None:
        if st.button("Ingest"):
            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
            res = requests.post("http://localhost:8000/api/ingest/", files=files)
            if res.status_code == 200:
                st.success("Ingestion Complete!")
            else:
                st.error(f"Error: {res.text}")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a technical question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = requests.post("http://localhost:8000/api/ask/", json={"question": prompt})
                if res.status_code == 200:
                    answer = res.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Show sources (Optional)
                    with st.expander("Sources"):
                         st.write(res.json().get("sources", []))
                else:
                    st.error(f"API Error: {res.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")
