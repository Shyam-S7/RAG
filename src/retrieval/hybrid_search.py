from typing import List
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from src.ingestion.vector_store import ChromaStore

class HybridSearch:
    def __init__(self):
        self.store = ChromaStore()
        self.vectorstore = self.store.get_vectorstore()
        self.bm25_index = None
        self.docs_map = []
        self._build_bm25()

    def _build_bm25(self):
        print("Building BM25 Index...")
        data = self.vectorstore.get()
        if not data['documents']: 
            print("Vector store empty, BM25 skipped.")
            return
        
        self.docs_map = [Document(page_content=t, metadata=m) for t, m in zip(data['documents'], data['metadatas'])]
        # Simple whitespace tokenizer
        corpus = [d.page_content.lower().split() for d in self.docs_map]
        self.bm25_index = BM25Okapi(corpus)

    def search(self, query: str, k: int = 5) -> List[Document]:
        # Vector
        vec_res = self.vectorstore.similarity_search_with_score(query, k=k*2)
        # BM25
        if not self.bm25_index: 
            return [d for d, _ in vec_res[:k]]
        
        bm25_res = self.bm25_index.get_top_n(query.lower().split(), self.docs_map, n=k*2)
        
        return self._rrf_fusion(vec_res, bm25_res, k=k)

    def _rrf_fusion(self, vec_res, bm25_res, k, c=60):
        scores = {}
        doc_lookup = {}
        
        # Vector scores
        for rank, (doc, _) in enumerate(vec_res):
            # Using content hash/key would be safer but content works for now
            key = doc.page_content 
            doc_lookup[key] = doc
            scores[key] = scores.get(key, 0) + 1/(c+rank)
            
        # BM25 scores
        for rank, doc in enumerate(bm25_res):
            key = doc.page_content
            doc_lookup[key] = doc
            scores[key] = scores.get(key, 0) + 1/(c+rank)
            
        sorted_keys = sorted(scores.keys(), key=scores.get, reverse=True)
        return [doc_lookup[key] for key in sorted_keys[:k]]
