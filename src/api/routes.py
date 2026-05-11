from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import shutil
import os
import uuid

# Core Modules
from src.pipeline.ingestion_pipeline import IngestionPipeline
from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.pipeline.generation_pipeline import GenerationPipeline
from src.evaluation.evaluator import RAGEvaluator
from src.utils.logging import get_logger
import csv
import json
from datetime import datetime

logger = get_logger(__name__)
router = APIRouter()

# Initialize Singletons
try:
    ingest_pipeline = IngestionPipeline()
    retrieval_pipeline = RetrievalPipeline()
    generation_pipeline = GenerationPipeline()
    evaluator = RAGEvaluator()
    logger.info("API Services Initialized.")
except Exception as e:
    logger.critical(f"Failed to initialize API services: {e}")
    raise e

class QueryRequest(BaseModel):
    question: str
    k: int = 5
    session_id: Optional[str] = None

class EvalRequest(BaseModel):
    test_cases: Optional[List[Dict[str, str]]] = None

@router.post("/evaluate/")
async def run_evaluation(request: Optional[EvalRequest] = None):
    """
    Runs Ragas evaluation on the provided test cases or default set.
    """
    logger.info("Evaluation request received.")
    try:
        # Default evaluation set: Try to load our high-quality 50-sample Agent dataset
        eval_file = os.path.join(os.getcwd(), "test", "ground_truth.json")
        
        if not request or not request.test_cases:
            if os.path.exists(eval_file):
                logger.info(f"Loading Golden Dataset from {eval_file}")
                with open(eval_file, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    eval_set = [{"question": d["question"], "ground_truth": d["ground_truth"]} for d in raw_data]
            else:
                logger.warning("No ground_truth.json found. Using minimal fallback.")
                eval_set = [
                    {
                        "question": "What is RAG methodology?",
                        "ground_truth": "RAG is a methodology that combines retrieval and generation in large language models to provide factually correct content."
                    }
                ]
        else:
            eval_set = request.test_cases
        
        results = evaluator.run_evaluation(eval_set)
        
        return {
            "status": "success",
            "scores": results,
            "results_file": "test/ragas_evaluation_results.csv"
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
        
        # Cleanup temp directory
        shutil.rmtree(temp_dir)
        
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
        
        # 2. Execute Generation Pipeline
        logger.info("Generating answer with LLM...")
        answer = generation_pipeline.run(request.question, final_docs, session_id=request.session_id)
        
        # 3. Log to History CSV
        try:
            log_dir = os.path.join(os.getcwd(), "test")
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "chat_history.csv")
            
            file_exists = os.path.isfile(log_file)
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, quoting=csv.QUOTE_ALL)
                if not file_exists:
                    writer.writerow(["timestamp", "question", "answer", "context_count"])
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    request.question,
                    answer,
                    len(final_docs)
                ])
        except Exception as log_err:
            logger.error(f"Failed to log chat to CSV: {log_err}")
            
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
