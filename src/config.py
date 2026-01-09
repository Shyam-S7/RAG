import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # Paths relative to the project root (assuming running from root)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
    
    EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    DOMAINS = ["programming", "system_design", "iot", "web_development", "ml_ai", "data_science"]
