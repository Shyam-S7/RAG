from typing import List
import os
import sys
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

try:
    from src.config import Config
    from src.utils.logging import get_logger
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.config import Config
    from src.utils.logging import get_logger

logger = get_logger(__name__)

class Reranker:
    """
    Reranks document candidates using a Cross-Encoder model.
    Cross-encoders are significantly more accurate than Bi-Encoders (Embeddings)
    as they perform full-attention over the query and document together.
    """

    def __init__(self):
        try:
            import torch
            model_name = Config.CROSS_ENCODER_MODEL
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading Cross-Encoder model: {model_name} on {device}...")
            self.model = CrossEncoder(model_name, device=device)
            logger.info("Cross-Encoder model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Reranker model: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Document], k: int = 5) -> List[Document]:
        """
        Reranks a list of documents based on semantic relevance to the query.
        """
        if not self.model or not documents:
            logger.warning("Reranker not initialized or no documents provided. Returning original list.")
            return documents[:k]

        try:
            # Prepare pairs for cross-encoder: [[query, doc1], [query, doc2], ...]
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Sort documents by score in descending order
            scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            
            logger.info(f"Reranked {len(documents)} candidates.")
            return [doc for doc, score in scored_docs[:k]]
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            return documents[:k]
