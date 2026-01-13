from langchain_huggingface import HuggingFaceEmbeddings
import torch
import os
import sys
import time

# Add root to sys.path to ensure absolute imports work if run as script
try:
    from src.config import Config
    from src.utils.logging import get_logger
    from src.utils.exception import EmbeddingError
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.config import Config
    from src.utils.logging import get_logger
    from src.utils.exception import EmbeddingError

logger = get_logger(__name__)


class Embedder:
    """Manages the Embedding Model (BGE-Small) with GPU support."""

    def __init__(self):
        # Config.EMBEDDING_MODEL should be "BAAI/bge-small-en-v1.5"
        self.model_name = Config.EMBEDDING_MODEL
        # Auto-detect device (cuda if available, else cpu)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for embeddings (slower)")

        self._embedding_function = None
        self.embedding_dim = None

    def get_function(self) -> HuggingFaceEmbeddings:
        """Returns the LangChain embedding function for Chroma integration."""
        if self._embedding_function is None:
            try:
                logger.info(
                    f"Loading Embedding Model: {self.model_name} on {self.device}..."
                )
                self._embedding_function = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": self.device},
                    encode_kwargs={
                        "normalize_embeddings": True
                    },  # BGE recommends normalization
                )
                # Get embedding dimension by embedding a test string
                test_embedding = self._embedding_function.embed_query("test")
                self.embedding_dim = len(test_embedding)
                logger.info(
                    f"Embedding Model loaded successfully. Dimension: {self.embedding_dim}"
                )
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise EmbeddingError(
                    f"Model load failed: {self.model_name}", detail=str(e)
                )
        return self._embedding_function

    def get_embedding_dimension(self) -> int:
        """Returns the embedding vector dimension."""
        if self.embedding_dim is None:
            self.get_function()  # Initialize if not done yet
        return self.embedding_dim

    def embed_query(self, text: str) -> list[float]:
        """Embeds a single query string."""
        start_time = time.time()
        embedding = self.get_function().embed_query(text)
        elapsed = time.time() - start_time
        logger.debug(f"Query embedding completed in {elapsed:.3f}s")
        return embedding

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embeds a list of documents/chunks with validation."""
        if not texts:
            logger.warning("No texts provided for embedding")
            return []

        try:
            start_time = time.time()
            embeddings = self.get_function().embed_documents(texts)
            elapsed = time.time() - start_time

            # Validate all embeddings have same dimension
            if embeddings:
                dims = set(len(e) for e in embeddings)
                if len(dims) > 1:
                    logger.warning(f"Inconsistent embedding dimensions: {dims}")
                    raise EmbeddingError(
                        "Inconsistent dimensions in embeddings", detail=f"Dims: {dims}"
                    )

                logger.debug(
                    f"Embedded {len(embeddings)} documents in {elapsed:.3f}s (avg: {elapsed/len(embeddings):.3f}s per doc)"
                )

            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise EmbeddingError("Document embedding failed", detail=str(e))


if __name__ == "__main__":
    # Test Block
    embedder = Embedder()
    vector = embedder.embed_query("What is RAG?")
    print(f"Embedding Success! Vector Dimension: {len(vector)}")
    print(vector)
