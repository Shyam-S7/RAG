import os
import sys

# Ensure root import if run directly
try:
    from src.config import Config
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from src.config import Config

from groq import Groq

class LLMClient:
    def __init__(self):
        self.api_key = Config.GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in Config")
            
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.1-8b-instant" 

    def generate(self, system_prompt: str, user_query: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0,
            max_tokens=1024
        )
        return completion.choices[0].message.content

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING LLM GENERATION (TECHNICAL ASSISTANT)")
    print("=" * 60)
    
    try:
        import json
        from src.generation.prompts import PromptManager
        
        # 1. Load Chained Context
        input_file = os.path.join(os.getcwd(), "test", "final_context.json")
        if not os.path.exists(input_file):
            print(f"❌ Error: {input_file} not found. Run post_processing.py first.")
            sys.exit(1)
            
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
            
        test_query = input_data["query"]
        context_items = input_data["final_context"]
        
        # Format context for LLM
        context_str = "\n\n".join([f"Source: {item['metadata']['source']}\n{item['content']}" for item in context_items])
        
        print(f"✅ Loaded context for query: '{test_query}'")

        # 2. Build Detailed System Prompt
        print("\n🏗️ Building Prompt...")
        system_prompt = PromptManager.build_prompt(
            context_str=context_str,
            history_str="No previous history (First turn).",
            domain="gen_ai"
        )

        # 3. Call LLM
        print("\n🤖 Calling Groq (llama-3.3-70b-versatile)...")
        llm = LLMClient()
        answer = llm.generate(system_prompt, test_query)
        
        # 4. Save and Show Result
        output_data = {
            "query": test_query,
            "system_prompt": system_prompt,
            "answer": answer
        }
        
        output_file = os.path.join(os.getcwd(), "test", "generation_result.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)
            
        # Also save plain text answer as requested
        answer_file = os.path.join(os.getcwd(), "test", "final_answer.txt")
        with open(answer_file, "w", encoding="utf-8") as f:
            f.write(answer)
            
        print(f"💾 Generation result saved to: {output_file}")
        print(f"💾 Plain text answer saved to: {answer_file}")
        
        print(f"\n{'='*60}")
        print("FINAL ANSWER FROM LLM")
        print(f"{'='*60}")
        print(answer)
        print(f"{'='*60}")

    except Exception as e:
        print(f"❌ Generation Error: {e}")
        import traceback
        traceback.print_exc()
