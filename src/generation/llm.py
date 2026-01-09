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
        self.model = "llama3-70b-8192" 

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
