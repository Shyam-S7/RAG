import logging
import hashlib
from typing import List, Any, Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from src.core.settings import Settings
from src.core.models import ModelRegistry

from src.utils.logging import get_logger

logger = get_logger(__name__)

class VectorStoreService:
    """Service to handle interactions with ChromaDB (Storage and Search)."""
    
    def __init__(self, persist_directory: str = Settings.VECTOR_DB_PATH):
        self.persist_directory = persist_directory
        # Use the shared embedding model from the registry
        self.embeddings = ModelRegistry.get_embedding_model()
        self._vectorstore = None

    @property
    def vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            logger.info(f"📂 Connecting to Vector Store at: {self.persist_directory}")
            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="techdoc_collection"
            )
        return self._vectorstore

    def add_documents(self, documents: List[Document]):
        """Adds documents to the vector store with deduplication based on content hash."""
        if not documents:
            return
        
        # Generate deterministic IDs for deduplication
        ids = [self._generate_id(doc.page_content, doc.metadata.get("source", "")) for doc in documents]
        
        # Check existing
        existing = self.vectorstore.get(ids=ids)
        existing_ids = set(existing["ids"])
        
        new_docs = []
        new_ids = []
        for doc, doc_id in zip(documents, ids):
            if doc_id not in existing_ids:
                new_docs.append(doc)
                new_ids.append(doc_id)
        
        if new_docs:
            logger.info(f"📥 Adding {len(new_docs)} new documents to ChromaDB...")
            self.vectorstore.add_documents(new_docs, ids=new_ids)
            logger.info("✅ Documents added successfully.")
        else:
            logger.info("ℹ️ All documents already exist in the vector store.")

    def _generate_id(self, content: str, source: str) -> str:
        composite = f"{source}_{content}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]

    def count(self) -> int:
        return self.vectorstore._collection.count()

class RetrievalService:
    """Service to handle document retrieval (Hybrid) and reranking."""
    
    def __init__(self, vectorstore_service: VectorStoreService):
        from src.retrieval.hybrid_search import HybridSearch
        self.vs_service = vectorstore_service
        self.search_engine = HybridSearch(self.vs_service)
        self.reranker = ModelRegistry.get_reranker_model()

    def retrieve(self, query: str, top_k: int = Settings.TOP_K_RETRIEVAL) -> List[Document]:
        """Retrieves and reranks documents using Hybrid Search."""
        logger.info(f"🔍 Hybrid Retrieval started for: '{query}'")
        
        # 1. Hybrid Search (Vector + BM25)
        # Retrieve more candidates for reranking
        results = self.search_engine.search(query, k=top_k * 4)
        candidates = [doc for doc, meta in results]
        
        if not candidates:
            return []

        # 2. Reranking (Cross-Encoder)
        logger.info(f"⚖️ Reranking {len(candidates)} candidates...")
        pairs = [[query, doc.page_content] for doc in candidates]
        scores = self.reranker.predict(pairs)
        
        # Sort by score
        ranked_results = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in ranked_results[:top_k]]

    def refresh(self):
        """Refreshes the underlying search indices (BM25)."""
        self.search_engine.refresh()

class GenerationService:
    """Service to handle response generation using LLM."""
    
    def __init__(self):
        self.llm = ModelRegistry.get_llm()

    def generate(self, query: str, context_docs: List[Document]) -> str:
        """Generates an answer based on retrieved context."""
        logger.info("🧠 Generating response...")
        
        context_str = "\n\n".join([f"Source: {d.metadata.get('source', 'Unknown')}\nContent: {d.page_content}" for d in context_docs])
        
        system_prompt = f"""You are a helpful technical assistant. 
        Answer the question based strictly on the provided context. 
        If the answer is not in the context, say you don't know.
        
        CONTEXT:
        {context_str}
        """
        
        try:
            response = self.llm.invoke([
                ("system", system_prompt),
                ("human", query)
            ])
            return response.content
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            return "I encountered an error while generating the response."
