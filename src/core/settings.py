import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Centralized configuration management for the RAG system.
    Loads values from environment variables with sensible defaults.
    """
    
    # Project Paths
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = BASE_DIR / "data"
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", str(DATA_DIR / "chroma_db"))
    EVAL_DATASET_PATH = BASE_DIR / "src" / "evaluation" / "datasets" / "ground_truth.json"
    
    # Ingestion Settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    DOMAINS = ["programming", "system_design", "iot", "web_development", "ml_ai", "data_science"]
    
    # Model Names
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
    RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Retrieval Settings
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 5))
    
    # LLM Settings
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq") # e.g., groq, openai
    LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama-3.1-8b-instant")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    @classmethod
    def validate(cls):
        """Basic validation for required settings."""
        if cls.LLM_PROVIDER == "groq" and not cls.GROQ_API_KEY:
            # We don't want to crash on import if just building docs, 
            # but we should log a warning or handled it in pipelines.
            pass

# Initialize
Settings.validate()
