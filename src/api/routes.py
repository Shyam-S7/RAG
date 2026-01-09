from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os

from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.rerank import Reranker
from src.retrieval.post_processing import PostProcessor
from src.generation.llm import LLMClient
from src.generation.prompts import PromptManager

router = APIRouter()

# Initialize Singletons (In prod, use dependency injection)
ingest_pipeline = IngestionPipeline()
search_engine = HybridSearch()
reranker = Reranker()
llm_client = LLMClient()

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"

@router.post("/ingest/")
async def ingest_file(file: UploadFile = File(...)):
    temp_dir = "d:/rag/data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Run ingestion on this file's folder (simplified for demo)
    # Ideally pipeline accepts single file too
    ingest_pipeline.run(temp_dir)
    return {"message": f"Ingested {file.filename}"}

@router.post("/ask/")
async def ask_question(request: QueryRequest):
    # 1. Retrieval
    try:
        candidates = search_engine.search(request.question, k=10)
        
        # 2. Rerank
        ranked_docs = reranker.rerank(request.question, candidates, top_n=5)
        
        # 3. Post Process (Reorder)
        final_docs = PostProcessor.reorder(ranked_docs)
        
        # 4. Generate
        context_str = "\n\n".join([d.page_content for d in final_docs])
        system_prompt = PromptManager.build_prompt(context_str)
        answer = llm_client.generate(system_prompt, request.question)
        
        return {
            "answer": answer,
            "sources": [d.metadata.get("source") for d in final_docs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
