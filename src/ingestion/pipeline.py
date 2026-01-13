import os
import sys
from pathlib import Path
from typing import Dict, Any

# Ensure root import if run directly
try:
    from src.ingestion.preprocess import Preprocessor
    from src.ingestion.vector_store import ChromaStore
    from src.utils.logging import get_logger
    from src.utils.exception import IngestionError
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.ingestion.preprocess import Preprocessor
    from src.ingestion.vector_store import ChromaStore
    from src.utils.logging import get_logger
    from src.utils.exception import IngestionError

logger = get_logger(__name__)


class IngestionPipeline:
    # File extensions to process
    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".txt",
        ".md",
        ".py",
        ".js",
        ".java",
        ".cpp",
        ".c",
        ".h",
    }
    # Max file size in MB
    MAX_FILE_SIZE_MB = 50

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.store = ChromaStore()

    def run(self, folder_path: str) -> Dict[str, Any]:
        """
        Runs the complete ingestion pipeline.
        Returns statistics about the ingestion process.
        """
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return {"success": False, "error": "Folder not found"}

        logger.info(f"Starting Ingestion for {folder_path}...")

        # Track statistics
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "total_chunks": 0,
            "success": True,
            "error": None,
            "files_processed": [],
            "files_failed": [],
        }

        all_chunks = []

        # 1. Processing Phase
        for root, _, files in os.walk(folder_path):
            for file in files:
                fpath = os.path.join(root, file)

                # Check file extension
                if not self._is_supported_file(fpath):
                    logger.debug(f"Skipping unsupported file: {file}")
                    continue

                # Check file size
                if not self._check_file_size(fpath):
                    logger.warning(f"File too large, skipping: {file}")
                    stats["files_failed"].append(
                        {"file": file, "reason": "File too large"}
                    )
                    stats["failed_files"] += 1
                    continue

                stats["total_files"] += 1

                try:
                    logger.debug(f"Processing: {file}")
                    file_chunks = self.preprocessor.process_file(fpath)

                    # Validate chunks
                    if not file_chunks:
                        logger.warning(f"File produced no chunks: {file}")
                        stats["files_failed"].append(
                            {"file": file, "reason": "No chunks produced"}
                        )
                        stats["failed_files"] += 1
                        continue

                    # Validate each chunk has metadata and content
                    valid_chunks = [
                        c
                        for c in file_chunks
                        if c.metadata and len(c.page_content.strip()) > 0
                    ]
                    if len(valid_chunks) < len(file_chunks):
                        logger.warning(
                            f"File {file}: {len(file_chunks) - len(valid_chunks)} invalid chunks removed"
                        )

                    if valid_chunks:
                        all_chunks.extend(valid_chunks)
                        stats["processed_files"] += 1
                        stats["files_processed"].append(
                            {"file": file, "chunks": len(valid_chunks)}
                        )
                        logger.info(f"Processed {file}: {len(valid_chunks)} chunks")

                except Exception as e:
                    logger.error(f"Skipping file {file} due to error: {e}")
                    stats["files_failed"].append({"file": file, "reason": str(e)})
                    stats["failed_files"] += 1
                    continue

        # 2. Storage Phase
        if all_chunks:
            logger.info(f"Total chunks to ingest: {len(all_chunks)}")
            try:
                self.store.add_documents(all_chunks)
                stats["total_chunks"] = len(all_chunks)
                logger.info("Ingestion complete.")
            except Exception as e:
                logger.critical(f"Failed to store documents in ChromaDB: {e}")
                stats["success"] = False
                stats["error"] = str(e)
                raise IngestionError("Storage Phase Failed", detail=str(e))
        else:
            logger.warning("No valid chunks processed. ChromaDB update skipped.")
            stats["success"] = False
            stats["error"] = "No valid chunks to ingest"

        # Log final statistics
        logger.info(
            f"Ingestion Summary: {stats['processed_files']}/{stats['total_files']} files processed, {stats['total_chunks']} chunks stored"
        )
        return stats

    def _is_supported_file(self, file_path: str) -> bool:
        """Checks if file extension is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def _check_file_size(self, file_path: str) -> bool:
        """Checks if file size is within limits."""
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            logger.warning(
                f"File {file_path} is {size_mb:.1f}MB (max: {self.MAX_FILE_SIZE_MB}MB)"
            )
            return False
        return True

    # Test
    data_path = "d:/rag/data/docs"
    # Ensure dummy data exists for test
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        with open(os.path.join(data_path, "quick_test.txt"), "w") as f:
            f.write("A quick brown fox jumps over the lazy dog.")

    try:
        pipeline = IngestionPipeline()
        result = pipeline.run(data_path)
        print(f"\nIngestion Result: {result}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
