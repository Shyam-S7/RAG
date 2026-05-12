import logging
from typing import List, Dict, Any, Optional
from src.core.settings import Settings
from src.core.services import VectorStoreService, RetrievalService, GenerationService
from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.pipeline.generation_pipeline import GenerationPipeline

from src.utils.logging import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    """
    Main Orchestrator for the RAG Application.
    
    Why Centralized?
    - Prevents multiple loads of heavy models (BGE, LLaMA, Reranker).
    - Ensures consistent configuration across ingestion, search, and generation.
    - Simplifies dependency management through the core/ module.
    """

    def __init__(self, vs_service: VectorStoreService = None):
        """
        Initializes the RAG pipeline using shared core services.
        Dependencies are injected to allow for flexible reuse and testing.
        """
        # Initialize Core Services (shared across pipelines)
        self.vs_service = vs_service or VectorStoreService()
        self.ret_service = RetrievalService(self.vs_service)
        self.gen_service = GenerationService()

        # Initialize Modular Pipelines
        self.retrieval_pipeline = RetrievalPipeline(
            vs_service=self.vs_service, 
            ret_service=self.ret_service
        )
        self.generation_pipeline = GenerationPipeline(
            gen_service=self.gen_service
        )
        
        logger.info("✅ RAG Pipeline initialized with shared core infrastructure.")

    def run(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the full RAG cycle:
        1. Query Rewriting (if history exists)
        2. Hybrid Retrieval (Vector + BM25)
        3. Reranking (Cross-Encoder)
        4. Response Generation (LLM with Context)
        """
        logger.info(f"🚀 Processing Query: '{query}'")

        # 1. Retrieval Phase (includes Query Rewriting)
        # Note: RetrievalPipeline.run handles rewriting internally if history is needed,
        # but here we pass an empty list if no session_id to simplify.
        retrieved_docs = self.retrieval_pipeline.run(query)

        if not retrieved_docs:
            logger.warning("⚠️ No relevant documents found.")
            return {
                "query": query,
                "answer": "I'm sorry, I couldn't find any relevant information to answer your question.",
                "source_count": 0,
                "sources": []
            }

        # 2. Generation Phase
        answer = self.generation_pipeline.run(
            query=query, 
            context_docs=retrieved_docs, 
            session_id=session_id
        )

        logger.info("✅ Generation complete.")

        return {
            "query": query,
            "answer": answer,
            "source_count": len(retrieved_docs),
            "sources": [
                {
                    "source": doc.metadata.get("source", "Unknown"),
                    "content": doc.page_content[:200] + "...",
                    "domain": doc.metadata.get("domain", "unknown")
                } for doc in retrieved_docs
            ]
        }

if __name__ == "__main__":
    # Setup Logging
    logging.basicConfig(level=logging.INFO)
    
    # Example Execution
    pipeline = RAGPipeline()
    test_query = "What is a RAG pipeline and why is centralized architecture better?"
    
    result = pipeline.run(test_query)
    
    print("\n" + "="*60)
    print(f"🤖 RAG RESPONSE")
    print("="*60)
    print(f"Query: {result['query']}")
    print("-" * 60)
    print(f"Answer:\n{result['answer']}")
    print("-" * 60)
    print(f"Sources Used: {result['source_count']}")
    for i, src in enumerate(result['sources']):
        print(f" [{i+1}] {src['source']} ({src['domain']})")
    print("="*60)
