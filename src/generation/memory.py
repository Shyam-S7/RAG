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

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MEMORY MANAGER (SESSION HISTORY)")
    print("=" * 60)
    
    try:
        import json
        import os
        
        manager = MemoryManager(history_limit=3)
        session = "test_user_123"
        
        print(f"\n📝 Adding messages to session: {session}")
        manager.add_message(session, "user", "What is RAG?")
        manager.add_message(session, "assistant", "RAG is Retrieval-Augmented Generation.")
        manager.add_message(session, "user", "Explain embeddings.")
        manager.add_message(session, "assistant", "Embeddings are vector representations.")
        
        # This 5th message should trigger the limit (limit=3)
        manager.add_message(session, "user", "Test trimming.")
        
        history = manager.get_history(session)
        print(f"✅ Total messages in memory: {len(history)}")
        print(f"📊 (Limit was 3, so old messages should be gone)")
        
        # Save to test folder
        test_dir = os.path.join(os.getcwd(), "test")
        os.makedirs(test_dir, exist_ok=True)
        output_file = os.path.join(test_dir, "memory_test.json")
        
        with open(output_file, "w") as f:
            json.dump(history, f, indent=4)
            
        print(f"💾 Memory test results saved to: {output_file}")
        
        for i, msg in enumerate(history):
            print(f"   [{i+1}] {msg['role'].upper()}: {msg['content']}")

    except Exception as e:
        print(f"❌ Memory Test Error: {e}")
