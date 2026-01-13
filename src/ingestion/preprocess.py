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
            Domain.PROGRAMMING: {"chunk_size": 350, "chunk_overlap": 50},
            Domain.DSA: {"chunk_size": 600, "chunk_overlap": 100},
            Domain.SYSTEM_DESIGN: {"chunk_size": 900, "chunk_overlap": 150},
            Domain.IOT: {"chunk_size": 500, "chunk_overlap": 100},
            Domain.WEB_DEV: {"chunk_size": 400, "chunk_overlap": 80},
            Domain.ML_AI: {"chunk_size": 400, "chunk_overlap": 50},
            Domain.GEN_AI: {"chunk_size": 450, "chunk_overlap": 60},
            Domain.DATA_SCIENCE: {"chunk_size": 400, "chunk_overlap": 50},
            Domain.GENERAL: {"chunk_size": 500, "chunk_overlap": 100},
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
            raise IngestionError(f"Failed to process file {file_path}", detail=str(e))


if __name__ == "__main__":
    # Test Block
    print("Testing Preprocessor...")

    # 1. Create a dummy file
    test_dir = "d:/rag/data/test"
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, "test_algo.txt")

    with open(test_file, "w") as f:
        f.write("fine tune")

    # 2. Run Processor
    processor = Preprocessor()
    try:
        chunks = processor.process_file(test_file)
        print(f"\nSuccessfully processed '{test_file}'.")
        print(f"Detected Domain: {chunks[0].metadata['domain']}")
        print(f"Total Chunks: {len(chunks)}")
        print(f"Sample Chunk Content: {chunks[0].page_content}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Test Block
    print("=" * 60)
    print("TESTING PREPROCESSOR")
    print("=" * 60)

    # 1. Create test files
    test_dir = "d:/rag/data/test"
    os.makedirs(test_dir, exist_ok=True)

    # Test 1: Python file (Programming domain)
    test_file_1 = os.path.join(test_dir, "test_python.txt")
    with open(test_file_1, "w") as f:
        f.write(
            """
        Python Programming Guide
        
        def hello_world():
            print('Hello World')
        
        This is a Python function that demonstrates basic syntax.
        Functions are reusable blocks of code.
        
        Class is also an important concept.
        import os
        """
        )

    # Test 2: Algorithm file (DSA domain)
    test_file_2 = os.path.join(test_dir, "test_algorithm.txt")
    with open(test_file_2, "w") as f:
        f.write(
            """
        Data Structure and Algorithm
        
        Binary Search Tree is a common data structure.
        Time Complexity: O(log n) for balanced tree.
        Space Complexity: O(n) for storing nodes.
        
        DFS and BFS are graph traversal algorithms.
        Big O notation helps analyze algorithm efficiency.
        """
        )

    # Test 3: Web development file
    test_file_3 = os.path.join(test_dir, "test_web.txt")
    with open(test_file_3, "w") as f:
        f.write(
            """
        Web Development Basics
        
        REST API is the standard for web services.
        HTTP methods: GET, POST, PUT, DELETE.
        JSON format is widely used for data exchange.
        React framework simplifies frontend development.
        HTML and CSS are fundamental web technologies.
        """
        )

    # 2. Run Processor on all test files
    processor = Preprocessor()
    test_files = [test_file_1, test_file_2, test_file_3]

    for test_file in test_files:
        try:
            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(test_file)}")
            print(f"{'='*60}")

            chunks = processor.process_file(test_file)

            if chunks:
                print(f"✅ Successfully processed '{os.path.basename(test_file)}'")
                print(f"📦 Total Chunks: {len(chunks)}")
                print(f"🎯 Detected Domain: {chunks[0].metadata['domain']}")
                print(f"📊 Chunk Size: {len(chunks[0].page_content)} chars")
                print(f"📄 Source: {chunks[0].metadata['source']}")
                print(f"⏱️  Processed At: {chunks[0].metadata['processed_at']}")

                # Show sample chunk
                print(f"\n📋 Sample Chunk Content:")
                print(f"   {chunks[0].page_content[:100]}...")
            else:
                print(f"❌ No chunks created for {os.path.basename(test_file)}")

        except Exception as e:
            print(f"❌ Error processing {os.path.basename(test_file)}: {e}")

    print(f"\n{'='*60}")
    print("✅ PREPROCESSOR TESTING COMPLETE")
    print(f"{'='*60}")
