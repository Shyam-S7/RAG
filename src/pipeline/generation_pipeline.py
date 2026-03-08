import os
import sys
from typing import List, Optional

try:
    from src.generation.llm import LLMClient
    from src.generation.prompts import PromptManager
    from src.generation.memory import MemoryManager
    from src.utils.logging import get_logger
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.generation.llm import LLMClient
    from src.generation.prompts import PromptManager
    from src.generation.memory import MemoryManager
    from src.utils.logging import get_logger

logger = get_logger(__name__)

class GenerationPipeline:
    """
    Facade for Answer Generation:
    1. Memory Handling (Session-based)
    2. Prompt Engineering (Domain-aware context building)
    3. LLM Interaction (Groq LLaMA-3)
    """

    def __init__(self):
        try:
            self.llm = LLMClient()
            self.memory = MemoryManager()
            logger.info("Generation Pipeline initialized.")
        except Exception as e:
            logger.error(f"Generation Pipeline Init Failed: {e}")
            raise e

    def run(self, query: str, context_docs: List, session_id: Optional[str] = None) -> str:
        """
        Generates an answer using retrieved context and previous conversation history.
        """
        try:
            # 1. Prepare Context
            context_str = "\n\n".join([doc.page_content for doc in context_docs])
            
            # 2. Detect Domain (for prompt styling)
            domain = "general"
            if context_docs and hasattr(context_docs[0], 'metadata') and "domain" in context_docs[0].metadata:
                domain = context_docs[0].metadata["domain"]

            # 3. Handle Memory (Inject recent history)
            history_str = ""
            if session_id:
                past_msgs = self.memory.get_history(session_id)
                if past_msgs:
                    # Capture up to last 10 messages for deeper context
                    for msg in past_msgs[-10:]:
                        role = "User" if msg["role"] == "user" else "Assistant"
                        history_str += f"{role}: {msg['content']}\n"

            # 4. Build Detailed System Prompt
            system_prompt = PromptManager.build_prompt(
                context_str=context_str, 
                history_str=history_str, 
                domain=domain
            )
            
            # 5. Generate Answer
            logger.info(f"Generating answer for query: '{query[:50]}...'")
            answer = self.llm.generate(system_prompt, query)
            
            # 6. Update Memory
            if session_id:
                self.memory.add_message(session_id, "user", query)
                self.memory.add_message(session_id, "assistant", answer)

            return answer

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return "I'm sorry, I encountered an error while generating the response."
