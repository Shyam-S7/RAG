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

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RERANKER (CHAINED FROM HYBRID RESULTS)")
    print("=" * 60)
    
    try:
        import json
        
        # 1. Load candidates from Hybrid Search result file
        input_file = os.path.join(os.getcwd(), "test", "hybrid_search_results.json")
        if not os.path.exists(input_file):
            print(f"❌ Error: {input_file} not found. Run hybrid_search.py first.")
            sys.exit(1)
            
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            
        test_query = input_data["query"]
        candidates_data = input_data["results"]
        
        # Convert JSON data back to LangChain Documents
        candidates = [
            Document(page_content=item["content"], metadata=item["metadata"])
            for item in candidates_data
        ]
        
        print(f"✅ Loaded {len(candidates)} candidates for query: '{test_query}'")
            
        # 2. Rerank the candidates
        print("\n🧠 Stage 2: Reranking with Cross-Encoder...")
        reranker = Reranker()
        # Rerank to top 3
        final_docs = reranker.rerank(test_query, candidates, k=3)
        
        # 3. Save Reranked Results
        output_data = {
            "query": test_query,
            "reranked_results": []
        }
        
        for i, doc in enumerate(final_docs):
            output_data["reranked_results"].append({
                "rank": i + 1,
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        output_file = os.path.join(os.getcwd(), "test", "reranked_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
            
        print(f"💾 Reranked results saved to: {output_file}")
        
        print(f"\n✅ Top 3 Refined Results:")
        for i, doc in enumerate(final_docs):
            print(f"   [{i+1}] {doc.page_content[:100]}...")

    except Exception as e:
        print(f"❌ Rerank Error: {e}")
        import traceback
        traceback.print_exc()
