import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List
from src.core.settings import Settings
from src.core.services import VectorStoreService
from src.ingestion.preprocess import Preprocessor
from src.utils.exception import IngestionError

logger = logging.getLogger(__name__)

class IngestionPipeline:
    # File extensions to process
    SUPPORTED_EXTENSIONS = {
        ".pdf", ".txt", ".md", ".py", ".js", ".java", ".cpp", ".c", ".h",
    }
    # Max file size in MB
    MAX_FILE_SIZE_MB = 50

    def __init__(self, vs_service: VectorStoreService = None):
        """
        Initialize with optional shared services.
        Uses singleton pattern via ModelRegistry inside services.
        """
        self.preprocessor = Preprocessor()
        self.vs_service = vs_service or VectorStoreService()
        logger.info("✅ Ingestion Pipeline initialized with shared services.")

    def run(self, folder_path: str) -> Dict[str, Any]:
        """
        Runs the complete ingestion pipeline.
        Returns statistics about the ingestion process.
        """
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return {"success": False, "error": "Folder not found"}

        logger.info(f"🚀 Starting Ingestion for {folder_path}...")

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

                if not self._is_supported_file(fpath):
                    continue

                if not self._check_file_size(fpath):
                    stats["files_failed"].append({"file": file, "reason": "File too large"})
                    stats["failed_files"] += 1
                    continue

                stats["total_files"] += 1

                try:
                    logger.debug(f"Processing: {file}")
                    file_chunks = self.preprocessor.process_file(fpath)

                    if not file_chunks:
                        stats["files_failed"].append({"file": file, "reason": "No chunks produced"})
                        stats["failed_files"] += 1
                        continue

                    valid_chunks = [c for c in file_chunks if c.metadata and len(c.page_content.strip()) > 0]
                    
                    if valid_chunks:
                        all_chunks.extend(valid_chunks)
                        stats["processed_files"] += 1
                        stats["files_processed"].append({"file": file, "chunks": len(valid_chunks)})
                        logger.info(f"Processed {file}: {len(valid_chunks)} chunks")

                except Exception as e:
                    logger.error(f"Skipping file {file} due to error: {e}")
                    stats["files_failed"].append({"file": file, "reason": str(e)})
                    stats["failed_files"] += 1
                    continue

        # 2. Storage Phase
        if all_chunks:
            logger.info(f"📥 Total chunks to ingest: {len(all_chunks)}")
            try:
                self.vs_service.add_documents(all_chunks)
                stats["total_chunks"] = len(all_chunks)
                logger.info("✅ Ingestion and storage complete.")
            except Exception as e:
                logger.critical(f"❌ Storage Phase Failed: {e}")
                stats["success"] = False
                stats["error"] = str(e)
                raise IngestionError(f"Storage Phase Failed: {str(e)}", sys)
        else:
            logger.warning("⚠️ No valid chunks processed.")
            stats["success"] = False
            stats["error"] = "No valid chunks to ingest"

        return stats

    def _is_supported_file(self, file_path: str) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS

    def _check_file_size(self, file_path: str) -> bool:
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return size_mb <= self.MAX_FILE_SIZE_MB

if __name__ == "__main__":
    # Test script for ingestion
    import json
    import traceback
    pipeline = IngestionPipeline()
    user_file_dir = os.path.join(os.getcwd(), "file")
    if os.path.exists(user_file_dir):
        try:
            results = pipeline.run(user_file_dir)
            print(json.dumps(results, indent=2))
        except Exception:
            traceback.print_exc()
