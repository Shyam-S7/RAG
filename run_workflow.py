import os
import sys

sys.path.insert(0, os.getcwd())

from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid_search import HybridSearch

print("=" * 70)
print("STEP 1: INGESTING PDF")
print("=" * 70)

try:
    pipeline = IngestionPipeline()
    result = pipeline.run("d:/rag/data/docs")

    print(f"\nProcessed: {result['processed_files']} files")
    print(f"Total chunks: {result['total_chunks']}")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 2: HYBRID SEARCH")
print("=" * 70)

try:
    searcher = HybridSearch()
    queries = ["python programming", "algorithms", "functions"]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = searcher.search(query, k=2)
        for i, (doc, meta) in enumerate(results, 1):
            print(
                f"  #{i} (Score: {meta.get('rrf_score', 0):.4f}): {doc.page_content[:60]}..."
            )

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("WORKFLOW COMPLETE")
print("=" * 70)
