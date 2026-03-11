import os
import sys
import pandas as pd
from typing import List, Dict
from datasets import Dataset

# Ensure root import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.pipeline.generation_pipeline import GenerationPipeline
from src.pipeline.retrieval_pipeline import RetrievalPipeline
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from langchain_groq import ChatGroq
from src.ingestion.embedding import Embedder
from src.config import Config

class RAGEvaluator:
    """
    RAG Evaluation using Ragas framework with Groq (No OpenAI required).
    Evaluates: Faithfulness, Answer Relevancy, Context Precision, and Context Recall.
    """
    
    def __init__(self):
        self.gen_pipeline = GenerationPipeline()
        self.ret_pipeline = RetrievalPipeline()
        
        # Initialize Groq as the Evaluator LLM
        self.evaluator_llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=Config.GROQ_API_KEY,
            temperature=0
        )
        
        # Initialize our project's Embedder for evaluation metrics
        self.evaluator_embeddings = Embedder().get_function()
        
        print("🤖 RAGEvaluator initialized with Groq as the critic model.")

    def prepare_evaluation_batch(self, test_cases: List[Dict[str, str]]) -> Dataset:
        """
        Runs the RAG pipeline on a list of test questions to build a Ragas dataset.
        test_cases: List of {"question": "...", "ground_truth": "..."}
        """
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        print(f"🔄 Generating RAG outputs for {len(test_cases)} test cases...")
        
        for i, case in enumerate(test_cases):
            question = case["question"]
            print(f"  [{i+1}/{len(test_cases)}] Evaluating: {question[:50]}...")
            
            # 1. Retrieval
            context_docs = self.ret_pipeline.run(question, k=5)
            contexts = [doc.page_content for doc in context_docs]
            
            # 2. Generation
            answer = self.gen_pipeline.run(question, context_docs)
            
            # 3. Store
            data["question"].append(question)
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(case["ground_truth"])

        return Dataset.from_dict(data)

    def run_evaluation(self, test_cases: List[Dict[str, str]]):
        """
        Runs Ragas evaluation and returns scores.
        """
        # 1. Prepare data
        dataset = self.prepare_evaluation_batch(test_cases)
        
        # 2. Run Ragas
        print("\n🧪 Running Ragas Evaluation Metrics...")
        # Metrics:
        # - Faithfulness: Is the answer derived solely from context?
        # - Answer Relevancy: How relevant is the answer to the question?
        # - Context Precision: Is the ground truth rank high in retrieval?
        # - Context Recall: Did retrieval find the necessary info?
        
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            ],
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings
        )
        
        # 3. Save & Display
        df = result.to_pandas()
        output_dir = os.path.join(os.getcwd(), "test")
        os.makedirs(output_dir, exist_ok=True)
        
        csv_path = os.path.join(output_dir, "ragas_evaluation_results.csv")
        import csv
        df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL)
        
        print("\n✅ Evaluation Results:")
        print(result)
        
        # Calculate summary scores (averages) from the dataframe
        # This is the most reliable way to get a serializable dict across Ragas versions
        summary_scores = df.mean(numeric_only=True).to_dict()
        
        print(f"\n📊 Detailed results saved to: {csv_path}")
        return summary_scores

if __name__ == "__main__":
    # Define your evaluation set (Golden Dataset)
    # These should be based on your PDF content (rag.pdf)
    eval_set = [
        {
            "question": "What is RAG methodology?",
            "ground_truth": "RAG is a methodology that combines retrieval and generation in large language models to provide factually correct content."
        },
        {
            "question": "What are the common tasks RAG is being expanded into?",
            "ground_truth": "RAG is being expanded into tasks like Information Extraction (IE), dialogue generation, and code search."
        }
    ]

    evaluator = RAGEvaluator()
    evaluator.run_evaluation(eval_set)
