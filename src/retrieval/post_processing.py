from typing import List
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder

class PostProcessor:
    """
    Modular Post-Processing for Retrieval:
    - Reorder: Handles 'Lost in the Middle' bias.
    - Diversity (MMR-style): Reduces redundancy.
    - Compression: Trims context to save tokens.
    """

    @staticmethod
    def reorder(documents: List[Document]) -> List[Document]:
        """
        Reorders documents so the most relevant are at the start and end.
        LLMs tend to pay more attention to the beginning and end of context.
        """
        if not documents:
            return []
        reorder = LongContextReorder()
        return reorder.transform_documents(documents)

    @staticmethod
    def filter_redundant(documents: List[Document], threshold: float = 0.95) -> List[Document]:
        """
        Simple text-based redundancy filter.
        Ensures we don't send identical or extremely similar chunks to the LLM.
        """
        unique_docs = []
        seen_contents = set()

        for doc in documents:
            # Simple content cleaning for comparison
            content_key = " ".join(doc.page_content.lower().split())
            if content_key not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(content_key)
        
        return unique_docs

    @staticmethod
    def compress_context(documents: List[Document], max_chunk_chars: int = 800) -> List[Document]:
        """
        Basic context compression.
        Trims individual documents to a maximum length to ensure space for more sources.
        """
        for doc in documents:
            if len(doc.page_content) > max_chunk_chars:
                doc.page_content = doc.page_content[:max_chunk_chars] + "... [trimmed]"
        return documents

    @staticmethod
    def optimize(documents: List[Document], k: int = 5) -> List[Document]:
        """
        The Final Pipeline: Redundancy Filter -> Trimming -> Reordering.
        """
        # 1. Remove duplicates/redundant chunks
        docs = PostProcessor.filter_redundant(documents)
        
        # 2. Limit to top K
        docs = docs[:k]
        
        # 3. Compress if chunks are too bulky
        docs = PostProcessor.compress_context(docs)
        
        # 4. Reorder for LLM attention
        return PostProcessor.reorder(docs)

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING POST-PROCESSOR (OPTIMIZATION)")
    print("=" * 60)
    
    try:
        from src.retrieval.hybrid_search import HybridSearch
        from src.retrieval.rerank import Reranker
        import os
        import sys
        
        # Ensure pathing for absolute imports
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
        
        # 1. Get real data from Search -> Rerank
        print("\n🔍 Stage 1: Retrieving & Reranking real chunks...")
        searcher = HybridSearch()
        reranker = Reranker()
        
        test_query = "what is rag"
        
        # Get 10 hybrid results
        candidates_tuples = searcher.search(test_query, k=10)
        candidates = [doc for doc, meta in candidates_tuples]
        
        # Rerank to top 6
        reranked_docs = reranker.rerank(test_query, candidates, k=6)
        print(f"✅ Reranked to {len(reranked_docs)} documents.")

        # 2. Optimize with PostProcessor
        print("\n⚙️ Stage 2: Optimizing with PostProcessor...")
        # Optimize to final K=3
        final_docs = PostProcessor.optimize(reranked_docs, k=3)
        
        print(f"✅ Optimization complete. Final context order (for LLM attention):")
        import json
        
        output_data = {
            "query": test_query,
            "final_context": []
        }
        
        for i, doc in enumerate(final_docs):
            output_data["final_context"].append({
                "position": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata
            })
            
        output_file = os.path.join(os.getcwd(), "test", "final_context.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
            
        print(f"💾 Final optimized context saved to: {output_file}")
        
        for i, doc in enumerate(final_docs):
            print(f"\n[{i+1}] Source: {doc.metadata.get('source', 'unknown')}")
            print(f"📄 Content Preview: {doc.page_content[:100]}...")
            print("-" * 30)
            
        print("\n💡 NOTE: Post-processing reorders docs so the most relevant are at the start and end.")
        print("   This prevents the LLM from 'forgetting' information in the middle.")

    except Exception as e:
        print(f"❌ Post-Processing Error: {e}")
        import traceback
        traceback.print_exc()
