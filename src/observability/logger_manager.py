import os
import json
import csv
from datetime import datetime
from typing import Any, Dict, List, Optional

class SimpleLogger:
    """
    A simple, modular logging system for RAG pipelines.
    Handles directory creation and saves logs in JSON and CSV formats.
    """
    
    BASE_OBSERVABILITY_DIR = "observability"
    BASE_EVALUATION_DIR = "evaluation_logs"

    def __init__(self):
        self._setup_directories()

    def _setup_directories(self):
        """Creates the necessary folder structure."""
        folders = [
            os.path.join(self.BASE_OBSERVABILITY_DIR, "retrieval_logs"),
            os.path.join(self.BASE_OBSERVABILITY_DIR, "generation_logs"),
            os.path.join(self.BASE_EVALUATION_DIR, "retrieval_evaluation"),
            os.path.join(self.BASE_EVALUATION_DIR, "generation_evaluation"),
            os.path.join(self.BASE_EVALUATION_DIR, "final_summary"),
        ]
        for folder in folders:
            os.makedirs(folder, exist_ok=True)

    def _get_timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _save_json(self, data: Dict, folder: str, filename: str):
        filepath = os.path.join(folder, f"{filename}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def _save_csv(self, data: List[Dict], folder: str, filename: str):
        if not data:
            return
        filepath = os.path.join(folder, f"{filename}.csv")
        keys = data[0].keys()
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

    # --- Observability Logging ---

    def log_retrieval(self, query: str, retrieved_chunks: List[Any], reranked_chunks: List[Any], 
                      retrieval_scores: List[float], rerank_scores: List[float], latency: float):
        """Saves retrieval pipeline logs to JSON."""
        log_entry = {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "reranked_chunks": reranked_chunks,
            "retrieval_scores": retrieval_scores,
            "rerank_scores": rerank_scores,
            "retrieval_latency": latency,
            "timestamp": self._get_timestamp()
        }
        filename = f"retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self._save_json(log_entry, os.path.join(self.BASE_OBSERVABILITY_DIR, "retrieval_logs"), filename)

    def log_generation(self, query: str, retrieved_context: str, generated_answer: str, latency: float):
        """Saves generation pipeline logs to JSON."""
        log_entry = {
            "query": query,
            "retrieved_context": retrieved_context,
            "generated_answer": generated_answer,
            "generation_latency": latency,
            "timestamp": self._get_timestamp()
        }
        filename = f"generation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self._save_json(log_entry, os.path.join(self.BASE_OBSERVABILITY_DIR, "generation_logs"), filename)

    # --- Evaluation Logging ---

    def log_retrieval_evaluation(self, eval_data: List[Dict]):
        """Saves retrieval evaluation results as JSON and CSV."""
        filename = f"retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        folder = os.path.join(self.BASE_EVALUATION_DIR, "retrieval_evaluation")
        
        # Save complete JSON
        self._save_json({"results": eval_data}, folder, filename)
        
        # Save flat CSV (flattening chunks for readability if needed, but here keeping it simple)
        self._save_csv(eval_data, folder, filename)

    def log_generation_evaluation(self, eval_data: List[Dict]):
        """Saves generation evaluation results as JSON and CSV."""
        filename = f"generation_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        folder = os.path.join(self.BASE_EVALUATION_DIR, "generation_evaluation")
        
        self._save_json({"results": eval_data}, folder, filename)
        self._save_csv(eval_data, folder, filename)

    def log_final_summary(self, summary: Dict):
        """Saves the final evaluation summary as JSON."""
        filename = f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        folder = os.path.join(self.BASE_EVALUATION_DIR, "final_summary")
        summary["timestamp"] = self._get_timestamp()
        self._save_json(summary, folder, filename)

# Singleton instance for easy import
logger = SimpleLogger()
