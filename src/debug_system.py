import os
import sys
import shutil

# Setup Paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.vector_store import ChromaStore
from src.retrieval.hybrid_search import HybridSearch

def debug_run():
    print("--- 1. Setup Debug Data ---")
    debug_dir = os.path.join("d:/rag/data/debug_temp")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    os.makedirs(debug_dir)
    
    file_path = os.path.join(debug_dir, "space_exploration.txt")
    with open(file_path, "w") as f:
        f.write("SpaceX uses the Starship rocket for Mars colonization missions. It is fully reusable.")
    print(f"Created file: {file_path}")

    print("\n--- 2. Run Ingestion ---")
    pipeline = IngestionPipeline()
    pipeline.run(debug_dir)
    print("Ingestion run complete.")

    print("\n--- 3. Verify Persistence ---")
    store = ChromaStore()
    chroma = store.get_vectorstore()
    data = chroma.get()
    count = len(data['ids'])
    print(f"Total Documents in DB: {count}")
    
    found = False
    for doc in data['documents']:
        if "Starship" in doc:
            print("✅ SUCCESS: Found 'Starship' chunk in DB.")
            found = True
            break
    
    if not found:
        print("❌ FAILURE: 'Starship' chunk NOT found in DB.")
        return

    print("\n--- 4. Verify Hybrid Search ---")
    searcher = HybridSearch()
    # Force refresh if needed
    searcher.refresh()
    
    results = searcher.search("Mars colonization", k=3)
    print(f"Search found {len(results)} results.")
    
    if results and "Starship" in results[0].page_content:
         print("✅ SUCCESS: Search returned correct document.")
    else:
         print(f"❌ FAILURE: Top result was: {results[0].page_content if results else 'None'}")

if __name__ == "__main__":
    debug_run()
