from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import shutil
import os
import uuid

# Core Modules
from src.pipeline.pipeline import IngestionPipeline
from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.pipeline.generation_pipeline import GenerationPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize Singletons
try:
    ingest_pipeline = IngestionPipeline()
    retrieval_pipeline = RetrievalPipeline()
    generation_pipeline = GenerationPipeline()
    logger.info("API Services Initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize API services: {e}")
    raise e

class QueryRequest(BaseModel):
    question: str
    k: int = 5
    session_id: Optional[str] = None

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
        
        # Refresh Retrieval Pipeline (e.g., Rebuild BM25 index with new docs)
        retrieval_pipeline.refresh()
        
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
        # Load history for query rewriting if session_id provided
        history = []
        if request.session_id:
            history = generation_pipeline.memory.get_history(request.session_id)
            
        # 1. Execute Retrieval Pipeline (Now with Query Rewriting!)
        final_docs = retrieval_pipeline.run(request.question, k=request.k, history=history)
        
        response_data = []
        context_parts = []
        
        for doc in final_docs:
            response_data.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "domain": doc.metadata.get('domain', 'unknown')
            })
            context_parts.append(doc.page_content)
            
        # 2. Execute Generation Pipeline
        logger.info("Generating answer with LLM...")
        answer = generation_pipeline.run(request.question, final_docs, session_id=request.session_id)
            
        return {
            "answer": answer,
            "count": len(final_docs),
            "results": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "domain": doc.metadata.get('domain', 'unknown')
                } for doc in final_docs
            ]
        }
    except Exception as e:
        logger.error(f"Search API failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
