import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from src.core.settings import Settings
from src.core.services import VectorStoreService, RetrievalService, GenerationService
from src.evaluation.retrieval_eval import RetrievalEvaluator
from src.evaluation.generation_eval import GenerationEvaluator
from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.pipeline.generation_pipeline import GenerationPipeline

logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """
    Complete Evaluation Pipeline
    Orchestrates:
    1. Retrieval Evaluation (Context Precision/Recall)
    2. Generation Evaluation (Faithfulness/Relevancy)
    
    Uses shared core services to ensure evaluation is consistent with production.
    """

    def __init__(self, vs_service: VectorStoreService = None):
        logger.info("🚀 Initializing Evaluation Pipeline...")
        
        # Share services across evaluators
        self.vs_service = vs_service or VectorStoreService()
        self.ret_service = RetrievalService(self.vs_service)
        self.gen_service = GenerationService()
        
        # Pipelines
        self.ret_pipeline = RetrievalPipeline(vs_service=self.vs_service, ret_service=self.ret_service)
        self.gen_pipeline = GenerationPipeline(gen_service=self.gen_service)
        
        # Evaluators
        self.retrieval_evaluator = RetrievalEvaluator(ret_pipeline=self.ret_pipeline)
        self.generation_evaluator = GenerationEvaluator(ret_pipeline=self.ret_pipeline, gen_pipeline=self.gen_pipeline)

    def run(self, test_cases: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Runs the full evaluation suite.
        """
        if not test_cases:
            test_cases = self._load_default_test_cases()

        if not test_cases:
            logger.error("❌ No test cases found for evaluation.")
            return {}

        logger.info(f"🧪 Starting full evaluation on {len(test_cases)} cases...")
        
        # 1. Evaluate Retrieval
        ret_scores = self.retrieval_evaluator.run(test_cases)
        
        # 2. Evaluate Generation
        gen_scores = self.generation_evaluator.run(test_cases)
        
        # Combined Results
        summary = {
            "timestamp": datetime.now().isoformat(),
            "case_count": len(test_cases),
            "retrieval_scores": ret_scores,
            "generation_scores": gen_scores,
            "overall_average": (sum(ret_scores.values()) + sum(gen_scores.values())) / (len(ret_scores) + len(gen_scores))
        }
        
        self._save_summary(summary)
        logger.info(f"✅ Full Evaluation Complete. Overall Score: {summary['overall_average']:.4f}")
        
        return summary

    def _load_default_test_cases(self) -> List[Dict[str, str]]:
        eval_file = Settings.EVAL_DATASET_PATH
        if os.path.exists(eval_file):
            with open(eval_file, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                return [{"question": d["question"], "ground_truth": d["ground_truth"]} for d in raw_data]
        logger.warning(f"⚠️ Evaluation file not found at: {eval_file}")
        return []

    def _save_summary(self, summary: Dict[str, Any]):
        output_path = Settings.BASE_DIR / "test" / "evaluation_summary.json"
        os.makedirs(output_path.parent, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

if __name__ == "__main__":
    # Setup basic logging to see results in console
    logging.basicConfig(level=logging.INFO)
    pipeline = EvaluationPipeline()
    pipeline.run()
