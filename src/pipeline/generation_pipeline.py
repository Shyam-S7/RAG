import logging
import time
from typing import List, Optional
from langchain_core.documents import Document
from src.core.settings import Settings
from src.core.services import GenerationService
from src.generation.prompts import PromptManager
from src.generation.memory import MemoryManager


logger = logging.getLogger(__name__)


class GenerationPipeline:
    """
    Facade for Answer Generation:
    1. Memory Handling (Session-based)
    2. Prompt Engineering (Domain-aware context building)
    3. LLM Interaction (via core.services)
    """

    def __init__(self, gen_service: GenerationService = None):
        self.gen_service = gen_service or GenerationService()
        self.memory = MemoryManager()
        logger.info("✅ Generation Pipeline initialized with shared services.")

    def run(
        self, query: str, context_docs: List[Document], session_id: Optional[str] = None
    ) -> str:
        """
        Generates an answer using retrieved context and previous conversation history.
        """
        start_time = time.time()
        try:
            # 1. Prepare Context
            context_str = "\n\n".join([doc.page_content for doc in context_docs])

            # 2. Detect Domain (for prompt styling)
            domain = "general"
            if (
                context_docs
                and hasattr(context_docs[0], "metadata")
                and "domain" in context_docs[0].metadata
            ):
                domain = context_docs[0].metadata["domain"]

            # 3. Handle Memory (Inject recent history)
            history_str = ""
            if session_id:
                past_msgs = self.memory.get_history(session_id)
                if past_msgs:
                    for msg in past_msgs[-10:]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        history_str += f"{role}: {msg['content']}\n"

            # 4. Generate Answer using shared service
            answer = self.gen_service.generate(query, context_docs)

            # 5. Update Memory
            if session_id:
                self.memory.add_message(session_id, "user", query)
                self.memory.add_message(session_id, "assistant", answer)

            latency = time.time() - start_time

            # Simple Observability Logging
            from src.observability import logger as obs_logger

            obs_logger.log_generation(
                query=query,
                retrieved_context=context_str,
                generated_answer=answer,
                latency=latency,
            )

            return answer

        except Exception as e:
            import traceback

            logger.error("❌ Generation failed.")
            logger.error(str(e))
            logger.error(traceback.format_exc())

            return f"Generation Error: {str(e)}"


if __name__ == "__main__":
    # Test script for generation
    pipeline = GenerationPipeline()
    print("Generation Pipeline Ready.")
