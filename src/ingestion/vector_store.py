import os
import sys
import shutil
import hashlib
from typing import List
from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from src.config import Config
    from src.ingestion.embedding import Embedder
    from src.utils.logging import get_logger
    from src.utils.exception import VectorStoreError
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.config import Config
    from src.ingestion.embedding import Embedder
    from src.utils.logging import get_logger
    from src.utils.exception import VectorStoreError

logger = get_logger(__name__)


class ChromaStore:
    """Manages interactions with ChromaDB."""

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
                raise VectorStoreError(
                    f"Embedding initialization failed: {str(e)}", sys
                )

    def get_vectorstore(self):
        self._init_embedder()
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_fn,
            collection_name="techdoc_collection",
        )

    def count(self) -> int:
        """Returns the number of documents in the collection."""
        try:
            store = self.get_vectorstore()
            return store._collection.count()
        except Exception:
            return 0

    def add_documents(self, documents: List[Document]):
        if not documents:
            logger.warning("No documents provided to add_documents.")
            return
        try:
            if not os.path.exists(self.persist_directory):
                os.makedirs(self.persist_directory, exist_ok=True)
                logger.info(f"Created ChromaDB directory: {self.persist_directory}")

            self._init_embedder()
            test_vector = self.embedder.embed_query("test")
            if not test_vector:
                raise VectorStoreError("Embedder produced empty vector", sys)

            ids = [
                self._generate_id(doc.page_content, doc.metadata.get("source", ""))
                for doc in documents
            ]
            store = self.get_vectorstore()
            existing_ids = set(store.get()["ids"] or [])

            # Filter against DB and track unique IDs in current batch to avoid internal duplicates
            new_docs_raw = []
            new_ids_raw = []
            for doc, doc_id in zip(documents, ids):
                if doc_id not in existing_ids:
                    new_docs_raw.append(doc)
                    new_ids_raw.append(doc_id)

            if not new_docs_raw:
                logger.warning("All documents already exist in ChromaDB.")
                return

            # Deduplicate within the current batch (in case chunks are identical)
            unique_new_docs = []
            unique_new_ids = []
            seen_ids = set()
            for doc, doc_id in zip(new_docs_raw, new_ids_raw):
                if doc_id not in seen_ids:
                    unique_new_docs.append(doc)
                    unique_new_ids.append(doc_id)
                    seen_ids.add(doc_id)

            batch_size = 100
            for i in range(0, len(unique_new_docs), batch_size):
                store.add_documents(
                    documents=unique_new_docs[i : i + batch_size],
                    ids=unique_new_ids[i : i + batch_size],
                )
                logger.info(
                    f"Added batch {i//batch_size + 1}: {len(unique_new_docs[i:i+batch_size])} docs"
                )

            logger.info(f"Successfully stored {len(unique_new_docs)} documents.")
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise VectorStoreError(
                f"Failed to add documents to ChromaDB: {str(e)}", sys
            )

    def _generate_id(self, content: str, source: str) -> str:
        """Generates a deterministic ID based on content and source for deduplication."""
        composite = f"{source}_{content}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()[:16]

    def reset_db(self):
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info("Vector database cleared.")
            except Exception as e:
                logger.error(f"Failed to clear DB: {e}")

    def inspect_db(self, limit: int = 3):
        logger.info(f"Inspecting top {limit} records in DB...")
        try:
            store = self.get_vectorstore()
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
                is_present = (
                    data["embeddings"] is not None and len(data["embeddings"]) > i
                )
                status = (
                    f"Present (len: {len(data['embeddings'][i])})"
                    if is_present
                    else "MISSING!"
                )
                print(f"Vector: {status}")
                print("-" * 20)
        except Exception as e:
            logger.error(f"Inspection failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING VECTOR STORE WITH SAVED EMBEDDINGS")
    print("=" * 60)

    try:
        import json
        from langchain_core.documents import Document

        # 1. Load Embedded Chunks from Test Folder
        embedded_file = os.path.join(os.getcwd(), "test", "embedded_chunks.json")
        if not os.path.exists(embedded_file):
            print(f"❌ Error: {embedded_file} not found. Run embedding.py first.")
            sys.exit(1)

        with open(embedded_file, "r", encoding="utf-8") as f:
            embedded_data = json.load(f)

        print(f"✅ Loaded {len(embedded_data)} embedded chunks from {embedded_file}")

        # 2. Convert to LangChain Documents
        # Note: In a real flow, we'd use use the 'embedding' field directly,
        # but ChromaStore.add_documents will re-verify them using the Embedder class.
        test_docs = []
        for item in embedded_data:
            test_docs.append(
                Document(
                    page_content=item["content"],
                    metadata={"source": item["file"], "domain": item["domain"]},
                )
            )

        # 3. Initialize and Store in Chroma
        print("\n🏟️ Stage 2: Storing in ChromaDB...")
        store = ChromaStore()

        # Optional: Reset DB for a clean test
        # store.reset_db()

        store.add_documents(test_docs)
        print(f"✅ Documents stored in ChromaDB.")

        # 4. Inspect and Save Records
        print("\n🔍 Stage 3: Inspecting & Saving Database Records...")
        store.inspect_db(limit=10)

        # Manually fetch for file storage
        db_store = store.get_vectorstore()
        db_data = db_store.get(limit=10, include=["metadatas", "documents"])

        records = []
        for i in range(len(db_data["ids"])):
            records.append(
                {
                    "id": db_data["ids"][i],
                    "metadata": db_data["metadatas"][i],
                    "content": db_data["documents"][i],
                }
            )

        output_file = os.path.join(os.getcwd(), "test", "vector_store_records.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=4)

        print(f"✅ Record inspection complete. Results saved to: {output_file}")

        print(f"\n{'='*60}")
        print("✅ VECTOR STORE TEST COMPLETE")
        print(f"{'='*60}")

    except Exception as e:
        print(f"❌ Vector Store Error: {e}")
        import traceback

        traceback.print_exc()
