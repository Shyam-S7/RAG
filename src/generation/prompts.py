class PromptManager:
    
    SYSTEM_TEMPLATE = """
    You are TechDocAI, a production-grade RAG assistant.
    Answer the user's question based ONLY on the following context.
    If the answer is not in the context, say "I don't know".
    
    Domain: {domain}
    
    Context:
    {context}
    """
    
    @staticmethod
    def build_prompt(context_str: str, domain: str = "general") -> str:
        return PromptManager.SYSTEM_TEMPLATE.format(context=context_str, domain=domain)
