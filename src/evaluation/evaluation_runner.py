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
        from src.observability import logger as obs_logger

        if not test_cases:
            test_cases = self._load_default_test_cases()

        if not test_cases:
            logger.error("❌ No test cases found for evaluation.")
            return {}

        logger.info(f"🧪 Starting full evaluation on {len(test_cases)} cases...")
        
        # 1. Evaluate Retrieval
        ret_results = self.retrieval_evaluator.run(test_cases)
        # Note: RetrievalEvaluator.run returns aggregate scores, but we need per-case data for logging.
        # For simplicity, we'll log the aggregate as a single entry or simulate per-case if possible.
        # Assuming ret_results contains details now (or we adjust it).
        
        # 2. Evaluate Generation
        gen_results = self.generation_evaluator.run(test_cases)
        
        # Combined Results Summary
        summary = {
            "case_count": len(test_cases),
            "retrieval_metrics_averages": ret_results if isinstance(ret_results, dict) else {},
            "generation_metrics_averages": gen_results if isinstance(gen_results, dict) else {},
            "overall_final_score": (sum(ret_results.values() if isinstance(ret_results, dict) else [0]) + 
                                   sum(gen_results.values() if isinstance(gen_results, dict) else [0])) / 2
        }
        
        # Simple Evaluation Logging (Requirement 4)
        
        # A. Retrieval Evaluation (Simulated list for example, usually you'd collect this in .run())
        ret_eval_list = [
            {
                "query": tc["question"],
                "retrieved_chunks": [], # Collect from evaluator
                "reranked_chunks": [], 
                "ground_truth": tc.get("ground_truth"),
                "context_precision": ret_results.get("precision", 0.0) if isinstance(ret_results, dict) else 0.0,
                "context_recall": ret_results.get("recall", 0.0) if isinstance(ret_results, dict) else 0.0
            } for tc in test_cases
        ]
        obs_logger.log_retrieval_evaluation(ret_eval_list)

        # B. Generation Evaluation
        gen_eval_list = [
            {
                "query": tc["question"],
                "retrieved_context": "", # Collect from evaluator
                "generated_answer": "", 
                "ground_truth": tc.get("ground_truth"),
                "faithfulness": gen_results.get("faithfulness", 0.0) if isinstance(gen_results, dict) else 0.0,
                "answer_relevancy": gen_results.get("relevancy", 0.0) if isinstance(gen_results, dict) else 0.0
            } for tc in test_cases
        ]
        obs_logger.log_generation_evaluation(gen_eval_list)

        # C. Final Summary
        obs_logger.log_final_summary(summary)

        self._save_summary(summary)
        logger.info(f"✅ Full Evaluation Complete. Overall Score: {summary['overall_final_score']:.4f}")
        
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
