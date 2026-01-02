# TechDocAI

A production-grade multi-domain RAG assistant using:
- FastAPI
- Streamlit
- Groq LLaMA-3
- ChromaDB
- Hybrid Retrieval (Vector + BM25)

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables in `.env`:
   ```
   GROQ_API_KEY=your_api_key_here
   ```

3. Run Backend:
   ```bash
   uvicorn src.api:app --reload
   ```

4. Run Frontend:
   ```bash
   streamlit run src/app.py
   ```
