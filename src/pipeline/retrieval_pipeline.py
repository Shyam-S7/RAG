import os
import sys
from typing import List
from langchain_core.documents import Document

try:
    from src.retrieval.hybrid_search import HybridSearch
    from src.retrieval.rerank import Reranker
    from src.retrieval.post_processing import PostProcessor
    from src.utils.logging import get_logger
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.retrieval.hybrid_search import HybridSearch
    from src.retrieval.rerank import Reranker
    from src.retrieval.post_processing import PostProcessor
    from src.utils.logging import get_logger

logger = get_logger(__name__)

class RetrievalPipeline:
    """
    Facade for the complete Retrieval Logic:
    1. Hybrid Search (Vector + Keyword)
    2. Reranking (Cross-Encoder)
    3. Post-Processing (Filter, Compress, Reorder)
    """

    def __init__(self):
        try:
            from src.generation.llm import LLMClient
            self.search_engine = HybridSearch()
            self.reranker = Reranker()
            self.llm = LLMClient()  # For query rewriting
            logger.info("Retrieval Pipeline fully initialized with Rewriter.")
        except Exception as e:
            logger.error(f"Retrieval Pipeline Init Failed: {e}")
            raise e

    def _rewrite_query(self, query: str, history: List[dict]) -> str:
        """Transforms shorthand queries into standalone search queries based on history."""
        if not history:
            return query
            
        history_str = ""
        for msg in history[-3:]: # Use last 3 messages for context
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
            rewritten = self.llm.generate(system_prompt="You are a query optimizer.", user_query=prompt)
            # Remove quotes if the LLM adds them
            clean_query = rewritten.strip().strip('"').strip("'")
            logger.info(f"Query Rewritten: '{query}' -> '{clean_query}'")
            return clean_query
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return query

    def run(self, query: str, k: int = 5, history: List[dict] = None) -> List[Document]:
        """
        Executes the end-to-end retrieval flow with optional query rewriting.
        """
        # 0. Rewrite Query if history exists
        search_query = self._rewrite_query(query, history) if history else query
        
        logger.info(f"Pipeline running for query: '{search_query}'")
        
        try:
            # Stage 1: Hybrid Search (Use the REWRITTEN query for search)
            candidates_with_meta = self.search_engine.search(search_query, k=k*2)
            candidates = [doc for doc, meta in candidates_with_meta]
            
            if not candidates:
                logger.warning("No candidates found in Hybrid Search.")
                return []

            # Stage 2: Rerank (Deep semantic analysis of top candidates)
            reranked_docs = self.reranker.rerank(query, candidates, k=k)
            
            # Stage 3: Optimize (Redundancy filtering -> Compression -> Attention reordering)
            final_docs = PostProcessor.optimize(reranked_docs, k=k)
            
            logger.info(f"Pipeline complete. Returning {len(final_docs)} optimized documents.")
            return final_docs

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            # Fallback: Return empty list rather than crashing the API
            return []

    def refresh(self):
        """Refreshes underlying search indices (e.g. BM25)."""
        self.search_engine.refresh()

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING COMPLETE RETRIEVAL PIPELINE")
    print("=" * 60)
    
    try:
        import json
        
        pipeline = RetrievalPipeline()
        # Ensure BM25 index is built from stored data
        print("\n🔄 Refreshing Search Indices...")
        pipeline.refresh()
        
        test_query = "what is RAG paradigms?"
        print(f"\n🔍 Running End-to-End Retrieval for: '{test_query}'")
        
        # Run the full pipeline
        final_docs = pipeline.run(test_query, k=5)
        
        if not final_docs:
            print("❌ No results found. Ensure you have ingested documents first.")
        else:
            print(f"✅ Pipeline returned {len(final_docs)} optimized documents.")
            
            # Save results to test folder
            output_data = {
                "query": test_query,
                "total_results": len(final_docs),
                "results": []
            }
            
            for i, doc in enumerate(final_docs):
                output_data["results"].append({
                    "position": i + 1,
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            test_dir = os.path.join(os.getcwd(), "test")
            os.makedirs(test_dir, exist_ok=True)
            output_file = os.path.join(test_dir, "retrieval_pipeline_results.json")
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4)
                
            print(f"💾 Retrieval summary saved to: {output_file}")
            
            print("\nPreview of top optimized result:")
            print(f"📄 Content: {final_docs[0].page_content[:200]}...")
            print(f"📂 Source: {final_docs[0].metadata.get('source')}")

        print(f"\n{'='*60}")
        print("✅ RETRIEVAL PIPELINE TEST COMPLETE")
        print(f"{'='*60}")

    except Exception as e:
        print(f"❌ Retrieval Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
