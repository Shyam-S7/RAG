# src/pipeline/evaluation_pipeline.py

import os
import sys
import json

# Root import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.evaluation.retrieval_evaluator import RetrievalEvaluator
from src.evaluation.generation_evaluator import GenerationEvaluator


class EvaluationPipeline:
    """
    Complete Evaluation Pipeline

    Runs:
    1. Retrieval Evaluation
    2. Generation Evaluation
    """

    def __init__(self):

        print("🚀 Initializing Evaluation Pipeline...")

        self.retrieval_evaluator = RetrievalEvaluator()

        self.generation_evaluator = GenerationEvaluator()

        print("✅ Evaluation Pipeline Ready.")

    def load_dataset(self, eval_file):

        if not os.path.exists(eval_file):

            raise FileNotFoundError(f"Evaluation file not found: {eval_file}")

        with open(eval_file, "r", encoding="utf-8") as f:

            raw_data = json.load(f)

        eval_set = [
            {"question": d["question"], "ground_truth": d["ground_truth"]}
            for d in raw_data
        ]

        return eval_set

    def run(self, eval_file):

        print("\n📦 Loading Evaluation Dataset...")

        eval_set = self.load_dataset(eval_file)

        print(f"✅ Loaded {len(eval_set)} test samples.")

        # ====================================================
        # Retrieval Evaluation
        # ====================================================

        print("\n" + "=" * 60)
        print("🔍 RUNNING RETRIEVAL EVALUATION")
        print("=" * 60)

        retrieval_scores = self.retrieval_evaluator.run(eval_set)

        # ====================================================
        # Generation Evaluation
        # ====================================================

        print("\n" + "=" * 60)
        print("🧠 RUNNING GENERATION EVALUATION")
        print("=" * 60)

        generation_scores = self.generation_evaluator.run(eval_set)

        # ====================================================
        # Final Summary
        # ====================================================

        final_results = {
            "retrieval_evaluation": retrieval_scores,
            "generation_evaluation": generation_scores,
        }

        print("\n" + "=" * 60)
        print("📊 FINAL EVALUATION SUMMARY")
        print("=" * 60)

        print("\n🔍 Retrieval Scores:")
        for k, v in retrieval_scores.items():
            print(f"{k}: {round(v, 4)}")

        print("\n🧠 Generation Scores:")
        for k, v in generation_scores.items():
            print(f"{k}: {round(v, 4)}")

        # Save Final Summary
        output_dir = os.path.join(os.getcwd(), "test")

        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, "evaluation_summary.json")

        with open(save_path, "w", encoding="utf-8") as f:

            json.dump(final_results, f, indent=4)

        print(f"\n💾 Final summary saved to:\n{save_path}")

        return final_results


if __name__ == "__main__":

    print("=" * 60)
    print("🚀 TESTING COMPLETE EVALUATION PIPELINE")
    print("=" * 60)

    eval_file = os.path.join(os.getcwd(), "test", "ground_truth.json")

    pipeline = EvaluationPipeline()

    results = pipeline.run(eval_file)

    print("\n" + "=" * 60)
    print("✅ EVALUATION PIPELINE COMPLETE")
    print("=" * 60)
