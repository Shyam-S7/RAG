from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import uuid

# Core Modules
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid_search import HybridSearch
from src.generation.llm import LLMClient
from src.generation.prompts import PromptManager
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize Singletons
try:
    ingest_pipeline = IngestionPipeline()
    search_engine = HybridSearch()
    llm_client = LLMClient()
    logger.info("API Services Initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize API services: {e}")
    raise e

class QueryRequest(BaseModel):
    question: str
    k: int = 5

@router.post("/ingest/")
async def ingest_file(file: UploadFile = File(...)):
    """
    Uploads a file and runs the ingestion pipeline on it.
    """
    logger.info(f"Received file upload: {file.filename}")
    try:
        # Create temp unique directory to avoid collisions
        session_id = str(uuid.uuid4())
        temp_dir = os.path.join(os.getcwd(), "data", "temp", session_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        file_path = os.path.join(temp_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"File saved to {file_path}. Starting ingestion...")
        
        # Run ingestion
        ingest_pipeline.run(temp_dir)
        
        # Refresh Search Index (Rebuild BM25)
        search_engine.refresh()
        
        # Cleanup (Optional: Keep for debug, or remove)
        # shutil.rmtree(temp_dir)
        
        return {
            "message": f"Successfully ingested {file.filename}",
            "session_id": session_id
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search/")
async def search_documents(request: QueryRequest):
    """
    Performs Hybrid Search -> Rerank -> Generation.
    """
    logger.info(f"Search request: '{request.question}'")
    try:
        # 1. Hybrid Search
        candidates = search_engine.search(request.question, k=request.k * 2) # Fetch more for reranking
        
        # 2. Rerank (Placeholder/Simple for now, can enable full Reranker if dependencies ready)
        # For now, let's take top k from candidates
        final_docs = candidates[:request.k]
        
        response_data = []
        context_parts = []
        
        for doc in final_docs:
            response_data.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "domain": doc.metadata.get('domain', 'unknown')
            })
            context_parts.append(doc.page_content)
            
        # 3. Generate Answer
        logger.info("Generating answer with LLM...")
        context_str = "\n\n".join(context_parts)
        # Detect domain from first doc or default
        domain = final_docs[0].metadata.get('domain', 'general') if final_docs else "general"
        
        system_prompt = PromptManager.build_prompt(context_str, domain=domain)
        answer = llm_client.generate(system_prompt, request.question)
            
        return {
            "answer": answer,
            "count": len(final_docs),
            "results": response_data
        }
    except Exception as e:
        logger.error(f"Search API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
