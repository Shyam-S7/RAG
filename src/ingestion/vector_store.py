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
            new_docs = [
                doc for doc, doc_id in zip(documents, ids) if doc_id not in existing_ids
            ]
            new_ids = [
                doc_id
                for doc, doc_id in zip(documents, ids)
                if doc_id not in existing_ids
            ]

            if not new_docs:
                logger.warning("All documents already exist in ChromaDB.")
                return

            batch_size = 100
            for i in range(0, len(new_docs), batch_size):
                store.add_documents(
                    documents=new_docs[i : i + batch_size],
                    ids=new_ids[i : i + batch_size],
                )
                logger.info(
                    f"Added batch {i//batch_size + 1}: {len(new_docs[i:i+batch_size])} docs"
                )

            logger.info(f"Successfully stored {len(new_docs)} documents.")
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
                print(f"Vector: {'Present' if data['embeddings'] else 'MISSING!'}")
                print("-" * 20)
        except Exception as e:
            logger.error(f"Inspection failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING VECTOR STORE (CHROMADB)")
    print("=" * 60)
    try:
        store = ChromaStore()
        print(f"\n✅ ChromaStore initialized")
        print(f"📁 Database path: {store.persist_directory}")

        from langchain_core.documents import Document

        sample_docs = [
            Document(
                page_content="Python is a programming language used for web development and data science.",
                metadata={
                    "domain": "programming",
                    "source": "test_python.txt",
                    "char_count": 80,
                },
            ),
            Document(
                page_content="Binary search trees provide O(log n) time complexity for operations.",
                metadata={"domain": "dsa", "source": "test_dsa.txt", "char_count": 75},
            ),
            Document(
                page_content="REST API is the standard architecture for web services and microservices.",
                metadata={
                    "domain": "web_development",
                    "source": "test_web.txt",
                    "char_count": 78,
                },
            ),
        ]
        print(f"✅ Created {len(sample_docs)} sample documents")

        store.add_documents(sample_docs)
        print(f"✅ Documents added successfully")

        store.inspect_db()
        print(f"✅ Database inspection complete")

    except Exception as e:
        print(f"❌ Vector Store Error: {e}")
        import traceback

        traceback.print_exc()
