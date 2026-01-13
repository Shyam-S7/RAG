from typing import List, Tuple, Dict, Any
import time

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

import os
import sys

# Ensure imports work
try:
    from src.ingestion.vector_store import ChromaStore
    from src.utils.logging import get_logger
    from src.utils.exception import RetrievalError
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.ingestion.vector_store import ChromaStore
    from src.utils.logging import get_logger
    from src.utils.exception import RetrievalError

logger = get_logger(__name__)

class HybridSearch:
    """
    Implements Hybrid Search using BM25 (Keyword) + ChromaDB (Vector).
    Results are combined using Reciprocal Rank Fusion (RRF).
    """
    
    def __init__(self):
        try:
            self.store = ChromaStore()
            self.vectorstore = self.store.get_vectorstore()
            self.bm25_index = None
            self.docs_map = []
            
            # Initialize BM25 Index
            self.build_bm25()
        except Exception as e:
            logger.error(f"Failed to initialize HybridSearch: {e}")
            raise RetrievalError("Hybrid Search Init Failed", detail=str(e))

    def refresh(self):
        """Refreshes the BM25 index (e.g., after new ingestion)."""
        logger.info("Refreshing Hybrid Search Index...")
        self.build_bm25()

    def build_bm25(self):
        """
        Fetches all documents from ChromaDB to build the BM25 index in memory.
        Optimized to use 'documents' (text content) only.
        """
        logger.info("Building BM25 Index...")
        try:
            # Fetch all docs (text and metadata)
            # content=True is default for get()
            data = self.vectorstore.get(include=["metadatas", "documents"]) 
            
            if not data['documents']: 
                logger.warning("Vector store is empty. BM25 index will not be built.")
                return
            
            # Reconstruct Document objects for mapping back provided indices
            self.docs_map = [
                Document(page_content=t, metadata=m) 
                for t, m in zip(data['documents'], data['metadatas'])
            ]
            
            # Tokenize for BM25 (Simple whitespace split for speed)
            # For production, use specific tokenizers (e.g. NLTK, spaCy)
            tokenized_corpus = [doc.page_content.lower().split() for doc in self.docs_map]
            
            self.bm25_index = BM25Okapi(tokenized_corpus)
            logger.info(f"BM25 Index built with {len(self.docs_map)} documents.")
            
        except Exception as e:
            logger.error(f"Failed to build BM25: {e}")
            # Non-critical failure - we can fallback to vector only?
            # But let's log it.

    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Performs Hybrid Search:
        1. Vector Search (Top 2*k)
        2. BM25 Search (Top 2*k)
        3. RRF Fusion
        4. Return Top k
        """
        logger.info(f"Searching for: '{query}'")
        
        try:
            # 1. Vector Search
            # similarity_search_with_score returns (doc, score)
            # We fetch more candidates (2*k) for better fusion
            vec_res = self.vectorstore.similarity_search_with_score(query, k=k*2)
            logger.debug(f"Vector search returned {len(vec_res)} results.")

            # 2. BM25 Search
            bm25_res = []
            if self.bm25_index:
                tokenized_query = query.lower().split()
                # get_top_n returns just the documents
                bm25_res = self.bm25_index.get_top_n(tokenized_query, self.docs_map, n=k*2)
                logger.debug(f"BM25 search returned {len(bm25_res)} results.")
            else:
                logger.warning("BM25 index not available, using only Vector Search.")

            # 3 & 4. RRF Fusion & Return
            if not bm25_res:
                return [doc for doc, _ in vec_res[:k]]
            
            fused_docs = self._rrf_fusion(vec_res, bm25_res, k=k)
            logger.info(f"Hybrid search finished. Returning {len(fused_docs)} documents.")
            return fused_docs
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RetrievalError(f"Search failed for query '{query}'", detail=str(e))

    def _rrf_fusion(self, vec_res: List[Tuple[Document, float]], bm25_res: List[Document], k: int, c: int = 60) -> List[Document]:
        """
        Combines results using Reciprocal Rank Fusion.
        Score = 1 / (c + rank)
        """
        scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Document] = {}
        
        # Helper to generate unique key for docs (using content hash or ID if available)
        # Assuming content is unique enough or we trust it for now. 
        # Ideally, we should use the ID stored in Chroma.
        def get_doc_key(doc):
             # Fallback to content hash if ID not immediately accessible in object 
             # (though separate ID query adds latency)
             return hashlib.md5(doc.page_content.encode()).hexdigest()

        import hashlib # Ensure import inside method or at top

        # Process Vector Results
        for rank, (doc, _) in enumerate(vec_res):
            key = get_doc_key(doc)
            doc_lookup[key] = doc
            scores[key] = scores.get(key, 0) + (1 / (c + rank))

        # Process BM25 Results
        for rank, doc in enumerate(bm25_res):
            key = get_doc_key(doc)
            # BM25 docs come from self.docs_map, Vector docs come from Chroma search
            # They should be identical in content but checking key is safer
            if key not in doc_lookup:
                doc_lookup[key] = doc
            scores[key] = scores.get(key, 0) + (1 / (c + rank))

        # Sort by RRF Score
        sorted_keys = sorted(scores.keys(), key=scores.get, reverse=True)
        
        # Return Top K
        return [doc_lookup[key] for key in sorted_keys[:k]]

if __name__ == "__main__":
    # Test
    try:
        searcher = HybridSearch()
        results = searcher.search("algorithm complexity", k=2)
        print("\n--- Top Results ---")
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.page_content[:100]}... (Source: {doc.metadata.get('source')})")
    except Exception as e:
        logger.error(f"Test failed: {e}")
