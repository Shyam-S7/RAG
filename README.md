# 🧠 TechDocAI: RAG Assistant

TechDocAI is a modular, high-performance Retrieval-Augmented Generation (RAG) assistant designed for technical documentation. It combines semantic search with keyword-based retrieval to provide highly accurate, document-backed answers.

## 🚀 Key Features

- **Hybrid Retrieval**: Combines **ChromaDB** (Vector Search) and **BM25** (Keyword Search) using Reciprocal Rank Fusion (RRF).
- **Deep Reranking**: Uses a Cross-Encoder model to refine search results for maximum relevance.
- **Context Optimization**: Implements redundancy filtering, context compression, and attention-based reordering (Lost-in-the-Middle fix).
- **Session Memory**: Intelligent conversation history management with automatic message trimming.
- **Query Rewriting**: Automatically transforms shorthand follow-up questions into standalone search queries.
- **Production-Grade API**: Built with **FastAPI** for high performance and scalability.
- **Real-time UI**: Clean **Streamlit** interface for easy document ingestion and chat.

## 🛠️ Tech Stack

- **LLM**: Groq LLaMA-3.3-70B (Ultra-fast inference)
- **Embeddings**: BGE-Small-EN-v1.5 (High-efficiency vectors)
- **Vector DB**: ChromaDB
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Logic**: LangChain

## 📦 Installation

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

## 🚦 How to Run

### 1. Start the Backend Server
```bash
python src/main.py
```
*The API will be available at `http://localhost:8000`*

### 2. Start the Frontend UI
```bash
streamlit run src/ui.py
```

## 🧪 Testing the Pipelines

Each component can be tested individually using the built-in test blocks:

1. **Ingestion**: `python -m src.pipeline.ingestion_pipeline`
2. **Retrieval**: `python -m src.pipeline.retrieval_pipeline`
3. **Generation**: `python -m src.pipeline.generation_pipeline`

Full test results and intermediate states are saved in the `test/` directory for verification.

## 📂 Project Structure

- `src/ingestion`: PDF processing, embedding generation, and vector storage.
- `src/retrieval`: Hybrid search, reranking, and post-processing.
- `src/generation`: LLM interaction, prompt management, and memory.
- `src/pipeline`: End-to-end coordination of ingestion, retrieval, and generation.
- `src/api`: FastAPI route definitions.
- `src/ui.py`: Streamlit interface.
- `test/`: JSON logs of system performance and results.
