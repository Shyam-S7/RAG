from typing import List, Dict
import time

class MemoryManager:
    """Simple in-memory conversation history manager."""
    
    def __init__(self, history_limit: int = 10):
        self.history: Dict[str, List[Dict[str, str]]] = {}
        self.limit = history_limit

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self.history.get(session_id, [])

    def add_message(self, session_id: str, role: str, content: str):
        if session_id not in self.history:
            self.history[session_id] = []
        
        self.history[session_id].append({
            "role": role,
            "content": content,
            "timestamp": time.time()
        })
        
        # Trim history
        if len(self.history[session_id]) > self.limit:
            self.history[session_id] = self.history[session_id][-self.limit:]

    def clear_history(self, session_id: str):
        if session_id in self.history:
            del self.history[session_id]
