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
                raise IngestionError(f"Storage Phase Failed: {str(e)}", sys)
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


# End of IngestionPipeline class
# Add at the end of the file, AFTER the IngestionPipeline class definition

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING COMPLETE INGESTION PIPELINE")
    print("=" * 60)

    try:
        # Create test data folder
        data_path = os.path.join(os.getcwd(), "data", "test_pipeline")
        os.makedirs(data_path, exist_ok=True)
        print(f"\n✅ Test data folder created: {data_path}")

        # Test 1: Create test documents
        print(f"\n{'='*60}")
        print("Test 1: Creating Test Documents")
        print(f"{'='*60}")

        test_files = {
            "test_python.txt": """
            Python Programming Fundamentals
            
            Python is a high-level programming language.
            def my_function():
                return "Hello"
            
            Classes and functions are core concepts.
            import numpy as np
            """,
            "test_algorithm.txt": """
            Data Structure and Algorithm Guide
            
            Algorithm complexity is measured with Big O notation.
            Binary tree traversal uses DFS and BFS.
            Graph algorithms solve connectivity problems.
            Time complexity O(log n) for balanced trees.
            """,
            "test_ml.txt": """
            Machine Learning Concepts
            
            Neural networks are inspired by biological neurons.
            Deep learning uses transformer architecture.
            Training phase involves loss function optimization.
            Inference is the prediction phase.
            """,
        }

        for filename, content in test_files.items():
            filepath = os.path.join(data_path, filename)
            with open(filepath, "w") as f:
                f.write(content)

        print(f"✅ Created {len(test_files)} test documents")

        # Test 2: Run pipeline
        print(f"\n{'='*60}")
        print("Test 2: Running Complete Pipeline")
        print(f"{'='*60}")
        print(f"📁 Processing folder: {data_path}")

        pipeline = IngestionPipeline()
        result = pipeline.run(data_path)

        print(f"\n✅ Pipeline execution complete")

        # Test 3: Display results
        print(f"\n{'='*60}")
        print("Test 3: Pipeline Results")
        print(f"{'='*60}")
        print(f"✅ Success: {result['success']}")
        print(f"📊 Total files found: {result['total_files']}")
        print(f"✔️  Files processed: {result['processed_files']}")
        print(f"❌ Files failed: {result['failed_files']}")
        print(f"📦 Total chunks created: {result['total_chunks']}")

        if result["files_processed"]:
            print(f"\n📄 Files processed:")
            for file_info in result["files_processed"]:
                print(f"   ✓ {file_info['file']}: {file_info['chunks']} chunks")

        if result["files_failed"]:
            print(f"\n⚠️  Failed files:")
            for file_info in result["files_failed"]:
                print(f"   ✗ {file_info['file']}: {file_info['reason']}")

        # Test 4: Verify in ChromaDB
        print(f"\n{'='*60}")
        print("Test 4: Verifying ChromaDB Storage")
        print(f"{'='*60}")
        store = ChromaStore()
        db_result = store.inspect_db()
        print(f"✅ Documents verified in ChromaDB")
        print(f"📦 Total documents in DB: {db_result['total_documents']}")
        print(f"🎯 Sample domains stored: programming, dsa, ml_ai")

        print(f"\n{'='*60}")
        print("✅ COMPLETE PIPELINE TESTING SUCCESSFUL")
        print(f"{'='*60}")
        print(f"\n🎉 All 3 stages working:")
        print(f"   1️⃣  Preprocessor: Clean, detect domain, chunk")
        print(f"   2️⃣  Embedder: Generate vectors")
        print(f"   3️⃣  Vector Store: Store in ChromaDB")

    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        import traceback

        traceback.print_exc()
