import os
import logging
import pandas as pd
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from src.core.settings import Settings
from src.core.models import ModelRegistry
from src.pipeline.retrieval_pipeline import RetrievalPipeline

logger = logging.getLogger(__name__)

class RetrievalEvaluator:
    """
    Evaluates ONLY retrieval quality using Ragas metrics.
    Reuses shared models from ModelRegistry.
    """

    def __init__(self, ret_pipeline: RetrievalPipeline = None):
        self.ret_pipeline = ret_pipeline or RetrievalPipeline()
        self.evaluator_llm = ModelRegistry.get_llm()
        self.evaluator_embeddings = ModelRegistry.get_embedding_model()
        logger.info("✅ Retrieval Evaluator initialized with shared models.")

    def prepare_dataset(self, test_cases: List[Dict[str, str]]):
        data = {"question": [], "contexts": [], "ground_truth": []}
        for case in test_cases:
            question = case["question"]
            docs = self.ret_pipeline.run(question, k=5)
            contexts = [doc.page_content for doc in docs]
            data["question"].append(question)
            data["contexts"].append(contexts)
            data["ground_truth"].append(case["ground_truth"])
        return Dataset.from_dict(data)

    def run(self, test_cases):
        logger.info(f"🧪 Running retrieval evaluation on {len(test_cases)} cases...")
        dataset = self.prepare_dataset(test_cases)
        result = evaluate(
            dataset,
            metrics=[context_precision, context_recall],
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
        )
        df = result.to_pandas()
        output_dir = Settings.BASE_DIR / "test"
        os.makedirs(output_dir, exist_ok=True)
        save_path = output_dir / "retrieval_evaluation.csv"
        df.to_csv(save_path, index=False)
        logger.info(f"✅ Evaluation complete. Results saved to {save_path}")
        return df.mean(numeric_only=True).to_dict()
