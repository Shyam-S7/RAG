import os
import sys
import json
import pandas as pd
from typing import List, Dict
from datasets import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
)

from langchain_groq import ChatGroq

from src.pipeline.retrieval_pipeline import RetrievalPipeline
from src.ingestion.embedding import Embedder
from src.config import Config


class RetrievalEvaluator:
    """
    Evaluates ONLY retrieval quality.
    """

    def __init__(self):

        self.ret_pipeline = RetrievalPipeline()

        self.evaluator_llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            groq_api_key=Config.GROQ_API_KEY,
            temperature=0,
        )

        self.evaluator_embeddings = Embedder().get_function()

        print("✅ Retrieval Evaluator initialized.")

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

        dataset = self.prepare_dataset(test_cases)

        result = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
            ],
            llm=self.evaluator_llm,
            embeddings=self.evaluator_embeddings,
        )

        df = result.to_pandas()

        output_dir = os.path.join(os.getcwd(), "test")
        os.makedirs(output_dir, exist_ok=True)

        save_path = os.path.join(output_dir, "retrieval_evaluation.csv")

        df.to_csv(save_path, index=False)

        print(result)

        return df.mean(numeric_only=True).to_dict()


if __name__ == "__main__":

    eval_file = os.path.join(os.getcwd(), "test", "ground_truth.json")

    with open(eval_file, "r", encoding="utf-8") as f:

        raw_data = json.load(f)

    eval_set = [
        {"question": d["question"], "ground_truth": d["ground_truth"]} for d in raw_data
    ]

    evaluator = RetrievalEvaluator()

    scores = evaluator.run(eval_set)

    print("\n📊 Retrieval Scores:")
    print(scores)
