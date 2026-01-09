from langchain_huggingface import HuggingFaceEmbeddings
import os
import sys

# Add root to sys.path to ensure absolute imports work if run as script
try:
    from src.config import Config
    from src.utils.logging import get_logger
    from src.utils.exception import EmbeddingError
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.config import Config
    from src.utils.logging import get_logger
    from src.utils.exception import EmbeddingError

logger = get_logger(__name__)

class Embedder:
    """Manages the Embedding Model (BGE-Small)."""
    
    def __init__(self):
        # Config.EMBEDDING_MODEL should be "BAAI/bge-small-en-v1.5"
        self.model_name = Config.EMBEDDING_MODEL
        # Auto-detect device (cuda/cpu) could be added here
        self.device = 'cpu' 
        self._embedding_function = None

    def get_function(self) -> HuggingFaceEmbeddings:
        """Returns the LangChain embedding function for Chroma integration."""
        if self._embedding_function is None:
            try:
                logger.info(f"Loading Embedding Model: {self.model_name}...")
                self._embedding_function = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={'device': self.device},
                    encode_kwargs={'normalize_embeddings': True} # BGE recommends normalization
                )
                logger.info("Embedding Model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise EmbeddingError(f"Model load failed: {self.model_name}", detail=str(e))
        return self._embedding_function
        
    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string."""
        return self.get_function().embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of documents/chunks."""
        return self.get_function().embed_documents(texts)

if __name__ == "__main__":
    # Test Block
    embedder = Embedder()
    vector = embedder.embed_query("What is RAG?")
    print(f"Embedding Success! Vector Dimension: {len(vector)}")
    print(vector)
