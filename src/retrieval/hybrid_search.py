from typing import List, Tuple, Dict, Any
import hashlib
import re
import os
import sys

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

try:
    from src.ingestion.vector_store import ChromaStore
    from src.utils.logging import get_logger
    from src.utils.exception import RetrievalError
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.ingestion.vector_store import ChromaStore
    from src.utils.logging import get_logger
    from src.utils.exception import RetrievalError

logger = get_logger(__name__)

class HybridSearch:
    """
    Modular Hybrid Searcher:
    - Semantic Search: Via ChromaDB (Vector)
    - Keyword Search: Via BM25 (Rank-BM25)
    - Fusion: Reciprocal Rank Fusion (RRF)
    """

    def __init__(self):
        try:
            self.store = ChromaStore()
            self.vectorstore = self.store.get_vectorstore()
            self.bm25 = None
            self.documents = []
            self.build_bm25()
            logger.info("HybridSearch initialized.")
        except Exception as e:
            raise RetrievalError(f"Initialization Failed: {e}", sys)

    def refresh(self):
        """Rebuilds the BM25 index from current vector store state."""
        self.build_bm25()

    def build_bm25(self):
        """Pulls all documents from VectorStore and builds Keyword index."""
        try:
            data = self.vectorstore.get(include=["metadatas", "documents"])
            if not data["documents"]:
                logger.warning("No documents in DB for BM25.")
                return

            self.documents = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(data["documents"], data["metadatas"])
            ]
            
            tokenized_corpus = [self._tokenize(doc.page_content) for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 Index built with {len(self.documents)} docs.")
        except Exception as e:
            logger.error(f"BM25 Build Error: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase + alpha-numeric only."""
        return re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower()).split()

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, Dict[str, Any]]]:
        """Runs Vector + BM25 and fuses them."""
        logger.info(f"Searching: '{query}'")
        
        # 1. Vector Search (Semantic)
        vec_results = self.vectorstore.similarity_search_with_score(query, k=k*2)
        
        # 2. BM25 Search (Keyword)
        if not self.bm25:
            return [(doc, {"score": score, "source": "vector"}) for doc, score in vec_results[:k]]
        
        tokenized_query = self._tokenize(query)
        bm25_results = self.bm25.get_top_n(tokenized_query, self.documents, n=k*2)
        
        # 3. Fusion
        return self._rrf_fusion(vec_results, bm25_results, k=k)

    def _rrf_fusion(self, vec_results, bm25_results, k=5, c=60) -> List[Tuple[Document, Dict]]:
        """Reciprocal Rank Fusion - merges two ranked lists."""
        scores = {}
        doc_map = {}

        # Use content hash to unique-identify documents for merging
        def get_key(content): return hashlib.md5(content.encode()).hexdigest()

        # Score Vector Results
        for rank, (doc, _) in enumerate(vec_results):
            key = get_key(doc.page_content)
            scores[key] = scores.get(key, 0) + 1 / (c + rank)
            doc_map[key] = doc

        # Score BM25 Results
        for rank, doc in enumerate(bm25_results):
            key = get_key(doc.page_content)
            scores[key] = scores.get(key, 0) + 1 / (c + rank)
            doc_map[key] = doc

        # Sort by total fusion score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [(doc_map[key], {"rrf_score": scores[key]}) for key in sorted_keys[:k]]
