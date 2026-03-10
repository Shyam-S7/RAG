import os
import re
from typing import List, Dict, Any
from enum import Enum
import time

try:
    from src.utils.logging import get_logger
    from src.utils.exception import IngestionError
except ModuleNotFoundError:
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.utils.logging import get_logger
    from src.utils.exception import IngestionError

logger = get_logger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
)
from langchain_core.documents import Document as LangchainDocument


class Domain(Enum):
    PROGRAMMING = "programming"
    DSA = "dsa"
    SYSTEM_DESIGN = "system_design"
    IOT = "iot"
    WEB_DEV = "web_development"
    ML_AI = "ml_ai"
    GEN_AI = "gen_ai"
    DATA_SCIENCE = "data_science"
    GENERAL = "general"


class TextCleaner:
    """Handles text cleaning and noise removal."""

    @staticmethod
    def clean(text: str) -> str:
        # Remove null bytes
        text = text.replace("\x00", "")
        # Remove email addresses
        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "", text)
        # Remove phone numbers
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", text)
        # Remove URLs
        text = re.sub(r"https?://\S+", "", text)
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text


class DomainDetector:
    """Detects technical domain using keywords with word boundaries."""

    KEYWORDS = {
        Domain.PROGRAMMING: [
            "python",
            "java",
            "c\\+\\+",
            "function",
            "class",
            "import",
            "def",
            "return",
        ],
        Domain.DSA: [
            "algorithm",
            "complexity",
            "big o",
            "tree",
            "graph",
            "sorting",
            "dfs",
            "bfs",
        ],
        Domain.SYSTEM_DESIGN: [
            "scalability",
            "load balancer",
            "database",
            "sharding",
            "cap theorem",
            "microservices",
        ],
        Domain.IOT: [
            "sensor",
            "arduino",
            "raspberry pi",
            "mqtt",
            "esp32",
            "gpio",
            "voltage",
        ],
        Domain.WEB_DEV: [
            "http",
            "api",
            "rest",
            "react",
            "html",
            "css",
            "json",
            "endpoint",
        ],
        Domain.ML_AI: [
            "neural network",
            "transformer",
            "pytorch",
            "training",
            "inference",
            "loss function",
        ],
        Domain.GEN_AI: [
            "llm",
            "generative",
            "gpt",
            "bert",
            "diffusion",
            "rag",
            "prompt engineering",
            "hallucination",
        ],
        Domain.DATA_SCIENCE: [
            "dataframe",
            "pandas",
            "visualization",
            "statistics",
            "outlier",
            "regression",
        ],
    }

    @staticmethod
    def detect(text: str) -> Domain:
        text_lower = text.lower()
        scores = {domain: 0 for domain in Domain}

        for domain, keywords in DomainDetector.KEYWORDS.items():
            for kw in keywords:
                # Use word boundaries for more accurate matching
                pattern = r"\b" + kw + r"\b"
                matches = re.findall(pattern, text_lower)
                scores[domain] += len(matches)

        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else Domain.GENERAL


class Chunker:
    """Splits text based on domain rules."""

    @staticmethod
    def split(doc: LangchainDocument, domain: Domain) -> List[LangchainDocument]:
        # Domain-specific config
        config = {
            Domain.PROGRAMMING: {"chunk_size": 1200, "chunk_overlap": 200},
            Domain.DSA: {"chunk_size": 1200, "chunk_overlap": 200},
            Domain.SYSTEM_DESIGN: {"chunk_size": 1200, "chunk_overlap": 200},
            Domain.IOT: {"chunk_size": 1200, "chunk_overlap": 200},
            Domain.WEB_DEV: {"chunk_size": 1200, "chunk_overlap": 200},
            Domain.ML_AI: {"chunk_size": 1200, "chunk_overlap": 200},
            Domain.GEN_AI: {"chunk_size": 1200, "chunk_overlap": 200},
            Domain.DATA_SCIENCE: {"chunk_size": 1200, "chunk_overlap": 200},
            Domain.GENERAL: {"chunk_size": 1200, "chunk_overlap": 200},
        }

        params = config.get(domain, config[Domain.GENERAL])

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=params["chunk_size"],
            chunk_overlap=params["chunk_overlap"],
            separators=["\n\n", "\n", ".", " ", ""],
        )

        cols = splitter.split_documents([doc])
        return cols


class MetadataMerger:
    """Merges and manages metadata updates."""

    @staticmethod
    def merge(
        doc_metadata: Dict[str, Any], extra_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merges existing document metadata with new fields.
        Ensures strict typing for key fields.
        """
        combined = doc_metadata.copy()
        combined.update(extra_metadata)

        # Ensure timestamp exists
        if "processed_at" not in combined:
            combined["processed_at"] = time.time()

        return combined


class FileLoader:
    """Universal File Loader."""

    @staticmethod
    def load(file_path: str) -> List[LangchainDocument]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                loader = PDFPlumberLoader(file_path)
            elif ext == ".md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif ext in [".txt", ".py", ".js", ".java", ".cpp", ".c", ".h"]:
                loader = TextLoader(file_path, autodetect_encoding=True)
            else:
                # Fallback for other types
                loader = UnstructuredFileLoader(file_path)

            return loader.load()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []


class Preprocessor:
    """Facade for the preprocessing pipeline."""

    def __init__(self):
        self.loader = FileLoader()
        self.cleaner = TextCleaner()
        self.detector = DomainDetector()
        self.chunker = Chunker()
        self.merger = MetadataMerger()

    def process_file(self, file_path: str) -> List[LangchainDocument]:
        logger.info(f"Processing file: {file_path}")
        try:
            # 1. Load
            raw_docs = self.loader.load(file_path)
            if not raw_docs:
                logger.warning(f"No content loaded from {file_path}")
                return []

            final_chunks = []

            for doc in raw_docs:
                try:
                    # 2. Clean
                    cleaned_text = self.cleaner.clean(doc.page_content)
                    doc.page_content = cleaned_text

                    # 3. Detect Domain
                    domain = self.detector.detect(cleaned_text)
                    logger.debug(f"Detected domain '{domain.value}' for doc segment.")

                    # 4. Chunk
                    chunks = self.chunker.split(doc, domain)

                    # 5. Merge Metadata & Post-process Chunks
                    for chunk in chunks:
                        extra_meta = {
                            "domain": domain.value,
                            "source": file_path,
                            "char_count": len(chunk.page_content),
                        }
                        chunk.metadata = self.merger.merge(chunk.metadata, extra_meta)
                        final_chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Error processing document chunk in {file_path}: {e}")
                    # Continue to next doc/chunk rather than failing entire file?
                    # For now, let's continue.
                    continue

            logger.info(
                f"Successfully processed {file_path}: {len(final_chunks)} chunks created."
            )
            return final_chunks

        except Exception as e:
            logger.error(f"Critical error processing file {file_path}: {e}")
            raise IngestionError(f"Failed to process file {file_path}: {str(e)}", sys)


if __name__ == "__main__":
    # Test Block
    print("=" * 60)
    print("TESTING PREPROCESSOR")
    print("=" * 60)

    processor = Preprocessor()
    
    # 1. Check if 'file' folder has content
    user_file_dir = os.path.join(os.getcwd(), "file")
    test_files = []
    
    if os.path.exists(user_file_dir):
        files_in_dir = [os.path.join(user_file_dir, f) for f in os.listdir(user_file_dir) 
                        if os.path.isfile(os.path.join(user_file_dir, f))]
        if files_in_dir:
            print(f"📂 Found {len(files_in_dir)} user files in /file folder.")
            test_files = files_in_dir

    # 2. If no user files, create dummy test files
    if not test_files:
        print("ℹ️ No user files found in /file. Creating dummy test files...")
        test_dir = os.path.join(os.getcwd(), "data", "test_ingestion")
        os.makedirs(test_dir, exist_ok=True)
        
        test_file_path = os.path.join(test_dir, "test_python.txt")
        with open(test_file_path, "w") as f:
            f.write("def test(): print('hello world')\nimport os")
        test_files = [test_file_path]

    # 3. Process and Store
    import json
    all_test_chunks = []
    
    for test_file in test_files:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(test_file)}")
            print(f"{'='*60}")

            chunks = processor.process_file(test_file)

            if chunks:
                print(f"✅ Successfully processed '{os.path.basename(test_file)}'")
                print(f"📦 Total Chunks: {len(chunks)}")
                
                # Collect for saving
                for c in chunks:
                    all_test_chunks.append({
                        "file": os.path.basename(test_file),
                        "domain": c.metadata['domain'],
                        "content": c.page_content
                    })
                
                # Show sample chunk
                print(f"\n📋 Sample Chunk Content (First 150 chars):")
                print(f"   {chunks[0].page_content[:150]}...")
            else:
                print(f"❌ No content extracted from {os.path.basename(test_file)}")

        except Exception as e:
            print(f"❌ Error processing {os.path.basename(test_file)}: {e}")

    # 4. Save to Test Folder
    if all_test_chunks:
        output_path = os.path.join(os.getcwd(), "test")
        os.makedirs(output_path, exist_ok=True)
        file_name = os.path.join(output_path, "preprocessed_chunks.json")
        
        # Limit to 10 chunks as requested by user
        export_data = all_test_chunks[:10]
        
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4)
        
        print(f"\n💾 Saved first {len(export_data)} chunks to: {file_name}")

    print(f"\n{'='*60}")
    print("✅ PREPROCESSOR TESTING COMPLETE")
    print(f"{'='*60}")
