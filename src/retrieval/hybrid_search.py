from typing import List, Tuple, Dict, Any
import hashlib
import re

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
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.ingestion.vector_store import ChromaStore
    from src.utils.logging import get_logger
    from src.utils.exception import RetrievalError

logger = get_logger(__name__)


class HybridSearch:
    """
    Implements Hybrid Search using BM25 (Keyword) + Vector Search.
    Flow: BM25 Search + Vector Search → RRF Fusion → Top-K Results
    """

    def __init__(self):
        try:
            self.store = ChromaStore()
            self.vectorstore = self.store.get_vectorstore()
            self.bm25_index = None
            self.docs_map = []

            # Initialize BM25 Index from ChromaDB
            self.build_bm25()
            logger.info("HybridSearch initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize HybridSearch: {e}")
            raise RetrievalError("Hybrid Search Init Failed", detail=str(e))

    def refresh(self):
        """Refreshes the BM25 index after new documents are ingested."""
        logger.info("Refreshing Hybrid Search Index...")
        self.build_bm25()

    def build_bm25(self):
        """
        Builds BM25 index from all documents in ChromaDB.
        BM25 = Keyword-based ranking algorithm.
        """
        logger.info("Building BM25 Index from ChromaDB...")
        try:
            # Fetch all documents from ChromaDB
            data = self.vectorstore.get(include=["metadatas", "documents"])

            if not data["documents"]:
                logger.warning("ChromaDB is empty. BM25 index will be empty.")
                self.bm25_index = None
                self.docs_map = []
                return

            # Reconstruct Document objects for BM25
            self.docs_map = [
                Document(page_content=text, metadata=meta)
                for text, meta in zip(data["documents"], data["metadatas"])
            ]

            # Tokenize documents for BM25
            tokenized_corpus = [
                self._tokenize(doc.page_content) for doc in self.docs_map
            ]

            # Build BM25 index
            self.bm25_index = BM25Okapi(tokenized_corpus)
            logger.info(
                f"BM25 Index built successfully with {len(self.docs_map)} documents"
            )

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            self.bm25_index = None

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text for BM25.
        Removes punctuation, converts to lowercase, filters short tokens.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and special characters
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Split by whitespace
        tokens = text.split()
        # Filter out very short tokens (< 2 chars)
        tokens = [t for t in tokens if len(t) > 2]
        return tokens

    def search(self, query: str, k: int = 5) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Hybrid Search Pipeline:
        1. Vector Search (Semantic) - Top 2*k
        2. BM25 Search (Keyword) - Top 2*k
        3. RRF Fusion (Combine rankings)
        4. Return Top k results

        Args:
            query: Search query string
            k: Number of top results to return

        Returns:
            List of (Document, metadata) tuples with scores
        """
        logger.info(f"Hybrid search query: '{query}'")

        try:
            # 1. Vector Search (Semantic similarity via embeddings)
            logger.debug("Running vector search...")
            vec_results = self.vectorstore.similarity_search_with_score(query, k=k * 2)
            logger.debug(f"Vector search: {len(vec_results)} results")

            # 2. BM25 Search (Keyword matching)
            logger.debug("Running BM25 search...")
            bm25_results = []
            if self.bm25_index and self.docs_map:
                tokenized_query = self._tokenize(query)
                bm25_results = self.bm25_index.get_top_n(
                    tokenized_query, self.docs_map, n=k * 2
                )
                logger.debug(f"BM25 search: {len(bm25_results)} results")
            else:
                logger.warning("BM25 index unavailable, using vector search only")

            # 3 & 4. RRF Fusion & Return Top-K
            if not bm25_results:
                # Fallback: vector search only
                fused_results = [
                    (doc, {"source": "vector_only", "score": score})
                    for doc, score in vec_results[:k]
                ]
                logger.info(
                    f"Returned {len(fused_results)} results (vector search only)"
                )
            else:
                # Full hybrid fusion
                fused_results = self._rrf_fusion(vec_results, bm25_results, k=k)
                logger.info(
                    f"Hybrid fusion complete. Returning {len(fused_results)} results"
                )

            return fused_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RetrievalError(f"Search failed for query '{query}'", detail=str(e))

    def _rrf_fusion(
        self,
        vec_results: List[Tuple[Document, float]],
        bm25_results: List[Document],
        k: int,
        c: int = 60,
    ) -> List[Tuple[Document, Dict[str, Any]]]:
        """
        Reciprocal Rank Fusion (RRF) combines vector & BM25 results.

        RRF Formula: Score = 1 / (c + rank)
        - c: constant (default 60) to avoid zero-division
        - Documents appear in both searches get higher score

        Args:
            vec_results: List of (Document, similarity_score) from vector search
            bm25_results: List of Document from BM25 search
            k: Top k documents to return
            c: RRF constant

        Returns:
            Top k documents with fusion scores
        """
        scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Tuple[Document, Dict]] = {}

        def get_doc_key(doc: Document) -> str:
            """Generate unique key for document using content hash."""
            return hashlib.md5(doc.page_content.encode()).hexdigest()

        # Process Vector Search Results
        logger.debug("Processing vector search results...")
        for rank, (doc, sim_score) in enumerate(vec_results):
            key = get_doc_key(doc)
            doc_lookup[key] = (
                doc,
                {
                    "vector_score": float(sim_score),
                    "vector_rank": rank + 1,
                    "sources": ["vector"],
                },
            )
            scores[key] = scores.get(key, 0) + (1 / (c + rank))

        # Process BM25 Search Results
        logger.debug("Processing BM25 search results...")
        for rank, doc in enumerate(bm25_results):
            key = get_doc_key(doc)

            if key in doc_lookup:
                # Document appears in both searches - boost score
                doc_lookup[key][1]["bm25_rank"] = rank + 1
                doc_lookup[key][1]["sources"].append("bm25")
                logger.debug(
                    f"Found in both searches (vector_rank={doc_lookup[key][1]['vector_rank']}, bm25_rank={rank+1})"
                )
            else:
                # Document only in BM25
                doc_lookup[key] = (doc, {"bm25_rank": rank + 1, "sources": ["bm25"]})

            scores[key] = scores.get(key, 0) + (1 / (c + rank))

        # Sort by RRF Score (descending)
        sorted_keys = sorted(scores.keys(), key=scores.get, reverse=True)

        # Return Top K with metadata
        fused_results = []
        for key in sorted_keys[:k]:
            doc, metadata = doc_lookup[key]
            metadata["rrf_score"] = scores[key]
            fused_results.append((doc, metadata))

        return fused_results


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING HYBRID SEARCH")
    print("=" * 60)

    try:
        # Initialize hybrid searcher
        searcher = HybridSearch()
        print(f"\nHybridSearch initialized")

        # Test search query
        query = "python programming algorithms"
        print(f"\nQuery: '{query}'")

        # Run hybrid search
        results = searcher.search(query, k=3)

        print(f"\n{'='*60}")
        print("SEARCH RESULTS")
        print(f"{'='*60}")

        if results:
            for i, (doc, metadata) in enumerate(results, 1):
                print(f"\n#{i} (RRF Score: {metadata.get('rrf_score', 0):.4f})")
                print(f"   Source: {metadata.get('sources')}")
                print(f"   Domain: {doc.metadata.get('domain')}")
                print(f"   Content: {doc.page_content[:80]}...")
        else:
            print("No results found")

        print(f"\n{'='*60}")
        print("HYBRID SEARCH TEST COMPLETE")
        print(f"{'='*60}")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"Error: {e}")
