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

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING COMPLETE GENERATION PIPELINE")
    print("=" * 60)
    
    try:
        import json
        from langchain_core.documents import Document
        
        # 1. Load results from Retrieval Pipeline
        input_file = os.path.join(os.getcwd(), "test", "retrieval_pipeline_results.json")
        if not os.path.exists(input_file):
            print(f"❌ Error: {input_file} not found. Run retrieval_pipeline.py first.")
            sys.exit(1)
            
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            
        test_query = input_data["query"]
        context_data = input_data["results"]
        
        # Convert back to Documents
        context_docs = [
            Document(page_content=item["content"], metadata=item["metadata"])
            for item in context_data
        ]
        
        print(f"✅ Loaded {len(context_docs)} context documents for query: '{test_query}'")

        # 2. Run Generation
        print("\n🤖 Generating final response with LLaMA-3.3...")
        pipeline = GenerationPipeline()
        session_id = "test_session_001"
        answer = pipeline.run(test_query, context_docs, session_id=session_id)
        
        # 3. Save Results
        output_data = {
            "query": test_query,
            "session_id": session_id,
            "answer": answer
        }
        
        test_dir = os.path.join(os.getcwd(), "test")
        os.makedirs(test_dir, exist_ok=True)
        
        # Save JSON package
        output_file = os.path.join(test_dir, "generation_pipeline_results.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
            
        # Save plain text answer
        answer_file = os.path.join(test_dir, "final_pipeline_answer.txt")
        with open(answer_file, "w", encoding="utf-8") as f:
            f.write(answer)
            
        print(f"\n✅ Generation complete.")
        print(f"💾 Results saved to: {output_file}")
        print(f"💾 Clean answer saved to: {answer_file}")
        
        print(f"\n{'='*60}")
        print("PIPELINE ANSWER")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}")

    except Exception as e:
        print(f"❌ Generation Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
