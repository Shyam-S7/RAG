from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from src.config import Config

class Reranker:
    def __init__(self):
        # Load model once
        self.model = CrossEncoder(Config.CROSS_ENCODER_MODEL)

    def rerank(self, query: str, documents: List[Document], top_n: int = 5) -> List[Document]:
        if not documents: return []
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_n]]
