import logging
from typing import List
from langchain_core.documents import Document
from src.core.settings import Settings
from src.core.services import VectorStoreService, RetrievalService
from src.core.models import ModelRegistry

logger = logging.getLogger(__name__)

class RetrievalPipeline:
    """
    Facade for the complete Retrieval Logic:
    1. Query Rewriting (Context-aware)
    2. Hybrid Retrieval & Reranking (via core.services)
    """

    def __init__(self, vs_service: VectorStoreService = None, ret_service: RetrievalService = None):
        self.vs_service = vs_service or VectorStoreService()
        self.ret_service = ret_service or RetrievalService(self.vs_service)
        self.llm = ModelRegistry.get_llm()
        logger.info("✅ Retrieval Pipeline initialized with shared services.")

    def _rewrite_query(self, query: str, history: List[dict]) -> str:
        """Transforms shorthand queries into standalone search queries based on history."""
        if not history:
            return query
            
        history_str = ""
        for msg in history[-3:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"

        prompt = f"""
        You are a search query optimizer. Your goal is to rewrite the "Follow-up Question" into a single, standalone search query that includes all necessary context from the "Conversation History".
        
        RULES:
        1. Keep it as a search-friendly phrase (e.g., "types of RAG paradigms" instead of "Tell me about types").
        2. If the question is already clear, do not change it.
        3. Only return the rewritten query text.
        
        CONVERSATION HISTORY:
        {history_str}
        
        FOLLOW-UP QUESTION: {query}
        
        STANDALONE SEARCH QUERY:"""
        
        try:
            rewritten = self.llm.invoke(prompt).content
            clean_query = rewritten.strip().strip('"').strip("'")
            logger.info(f"🔄 Query Rewritten: '{query}' -> '{clean_query}'")
            return clean_query
        except Exception as e:
            logger.error(f"❌ Query rewrite failed: {e}")
            return query

    def run(self, query: str, k: int = Settings.TOP_K_RETRIEVAL, history: List[dict] = None) -> List[Document]:
        """
        Executes the end-to-end retrieval flow with optional query rewriting.
        """
        # 1. Rewrite Query if history exists
        search_query = self._rewrite_query(query, history) if history else query
        
        logger.info(f"🚀 Pipeline running retrieval for: '{search_query}'")
        
        try:
            # Stage: Retrieve and Rerank (using centralized service)
            final_docs = self.ret_service.retrieve(search_query, top_k=k)
            
            logger.info(f"✅ Pipeline complete. Returning {len(final_docs)} optimized documents.")
            return final_docs

        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {e}")
            return []

if __name__ == "__main__":
    # Test script for retrieval
    pipeline = RetrievalPipeline()
    test_query = "what is RAG paradigms?"
    results = pipeline.run(test_query, k=3)
    for i, doc in enumerate(results):
        print(f"[{i+1}] {doc.page_content[:100]}...")
