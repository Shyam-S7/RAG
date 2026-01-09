import os
import sys

# Ensure root import if run directly
try:
    from src.ingestion.preprocess import Preprocessor
    from src.ingestion.vector_store import ChromaStore
    from src.utils.logging import get_logger
    from src.utils.exception import IngestionError
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.ingestion.preprocess import Preprocessor
    from src.ingestion.vector_store import ChromaStore
    from src.utils.logging import get_logger
    from src.utils.exception import IngestionError

logger = get_logger(__name__)

class IngestionPipeline:
    def __init__(self):
        self.preprocessor = Preprocessor()
        self.store = ChromaStore()

    def run(self, folder_path: str):
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return
            
        logger.info(f"Starting Ingestion for {folder_path}...")
        all_chunks = []
        
        # 1. Processing Phase
        for root, _, files in os.walk(folder_path):
            for file in files:
                fpath = os.path.join(root, file)
                try:
                    file_chunks = self.preprocessor.process_file(fpath)
                    if file_chunks:
                        all_chunks.extend(file_chunks)
                        logger.info(f"Processed {file}: {len(file_chunks)} chunks")
                except Exception as e:
                    logger.error(f"Skipping file {file} due to error: {e}")
        
        # 2. Storage Phase
        if all_chunks:
            logger.info(f"Total chunks to ingest: {len(all_chunks)}")
            try:
                self.store.add_documents(all_chunks)
                logger.info("Ingestion complete.")
            except Exception as e:
                logger.critical(f"Failed to store documents in ChromaDB: {e}")
                raise IngestionError("Storage Phase Failed", detail=str(e))
        else:
            logger.warning("No valid chunks processed. ChromaDB update skipped.")

if __name__ == "__main__":
    # Test
    data_path = "d:/rag/data/docs"
    # Ensure dummy data exists for test
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
        with open(os.path.join(data_path, "quick_test.txt"), "w") as f:
            f.write("A quick brown fox jumps over the lazy dog.")
            
    try:
        pipeline = IngestionPipeline()
        pipeline.run(data_path)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
