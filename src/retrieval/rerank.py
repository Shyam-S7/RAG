import logging
from typing import List
from langchain_core.documents import Document
from src.core.models import ModelRegistry

logger = logging.getLogger(__name__)

class Reranker:
    """
    Reranks document candidates using a Cross-Encoder model.
    Uses the shared instance from ModelRegistry to prevent redundant memory usage.
    """

    def __init__(self):
        try:
            self.model = ModelRegistry.get_reranker_model()
            logger.info("✅ Reranker initialized with shared model.")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Reranker: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Document], k: int = 5) -> List[Document]:
        """
        Reranks a list of documents based on semantic relevance to the query.
        """
        if not self.model or not documents:
            logger.warning("⚠️ Reranker not ready or no documents provided.")
            return documents[:k]

        try:
            # Prepare pairs for cross-encoder
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Sort documents by score
            scored_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
            
            logger.info(f"✅ Reranked {len(documents)} candidates.")
            return [doc for doc, score in scored_docs[:k]]
            
        except Exception as e:
            logger.error(f"❌ Error during reranking: {e}")
            return documents[:k]
