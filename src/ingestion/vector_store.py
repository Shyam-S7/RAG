import os
import shutil
import hashlib
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from src.config import Config
    from src.ingestion.embedding import Embedder
    from src.utils.logging import get_logger
    from src.utils.exception import VectorStoreError
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.config import Config
    from src.ingestion.embedding import Embedder
    from src.utils.logging import get_logger
    from src.utils.exception import VectorStoreError

logger = get_logger(__name__)


class ChromaStore:
    """
    Manages interactions with ChromaDB.
    Stores: Chunks (Text), Vectors (Embeddings), Metadata, and Unique IDs.
    """

    def __init__(self):
        self.persist_directory = Config.CHROMA_DB_DIR
        self.embedder = None
        self.embedding_fn = None

    def _init_embedder(self):
        if not self.embedder:
            try:
                self.embedder = Embedder()
                self.embedding_fn = self.embedder.get_function()
            except Exception as e:
                logger.error(f"Failed to initialize Embedder: {e}")
                raise VectorStoreError("Embedding initialization failed", detail=str(e))

    def get_vectorstore(self) -> Chroma:
        """Returns the Chroma vector store instance."""
        # Initialize embedder only when needed (adding docs or semantic search)
        # For simple retrieval/inspection, we might get away without it,
        # but LangChain's Chroma wrapper usually expects it.
        # To strictly answer "why load bge", it's because this Code initializes it.
        # We will initialize it lazily.
        self._init_embedder()
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_fn,
            collection_name="techdoc_collection",
        )

    def add_documents(self, documents: List[Document]):
        """
        Embeds and adds documents with unique IDs.
        Ensures strict storage of: Content, Embeddings, Metadata, ID.
        """
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return

        try:
            # Validate config directory exists
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory, exist_ok=True)
                logger.info(f"Created ChromaDB directory: {self.persist_directory}")

            # Test embedder with first chunk to verify it works
            self._init_embedder()
            test_vector = self.embedder.embed_query("test")
            if not test_vector or len(test_vector) == 0:
                raise VectorStoreError("Embedder produced empty vector", detail="")
            logger.debug(f"Embedding dimension verified: {len(test_vector)}")

            # Generate unique IDs based on content + source for deduplication
            ids = [
                self._generate_id(doc.page_content, doc.metadata.get("source", ""))
                for doc in documents
            ]

            store = self.get_vectorstore()

            # Check for duplicates before adding
            existing_ids = set(store.get()["ids"] or [])
            new_docs = []
            new_ids = []

            for doc, doc_id in zip(documents, ids):
                if doc_id not in existing_ids:
                    new_docs.append(doc)
                    new_ids.append(doc_id)
                else:
                    logger.warning(f"Skipping duplicate document: {doc_id}")

            if not new_docs:
                logger.warning("All documents already exist in ChromaDB.")
                return

            logger.info(
                f"Adding {len(new_docs)} documents to ChromaDB at {self.persist_directory}..."
            )

            # Add in batches to avoid overwhelming ChromaDB
            batch_size = 100
            for i in range(0, len(new_docs), batch_size):
                batch_docs = new_docs[i : i + batch_size]
                batch_ids = new_ids[i : i + batch_size]
                store.add_documents(documents=batch_docs, ids=batch_ids)
                logger.debug(f"Added batch {i//batch_size + 1}: {len(batch_docs)} docs")

            logger.info(
                f"Successfully stored {len(new_docs)} chunks, vectors, metadata, and IDs."
            )

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise VectorStoreError("Failed to add documents", detail=str(e))

    def _generate_id(self, content: str, source: str) -> str:
        """Generates a stable hash ID using SHA256 instead of MD5."""
        import time

        # Add timestamp to prevent exact duplicates of same content
        composite = f"{source}_{content}_{time.time()}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]

    def reset_db(self):
        """Clears the DB."""
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info("Vector database cleared.")
            except Exception as e:
                logger.error(f"Failed to clear DB: {e}")

    def inspect_db(self, limit: int = 3):
        """Debug method to verify what is actually stored."""
        logger.info(f"Inspecting top {limit} records in DB...")
        try:
            store = self.get_vectorstore()
            # Retrieve including embeddings to verify they exist
            data = store.get(
                limit=limit, include=["metadatas", "documents", "embeddings"]
            )

            if not data["ids"]:
                print("Database is empty.")
                return

            print(f"\n--- ChromaDB Inspection ({len(data['ids'])} records found) ---")
            for i in range(len(data["ids"])):
                print(f"ID: {data['ids'][i]}")
                print(f"Metadata: {data['metadatas'][i]}")
                print(f"Content: {data['documents'][i][:50]}...")
                if data["embeddings"]:
                    print(f"Vector: Present (Dim: {len(data['embeddings'][i])})")
                else:
                    print("Vector: MISSING!")
                print("-" * 20)
        except Exception as e:
            logger.error(f"Inspection failed: {e}")


if __name__ == "__main__":
    # Test
    store = ChromaStore()
    store.inspect_db()
