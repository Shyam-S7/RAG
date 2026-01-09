class TechDocException(Exception):
    """Base exception for TechDocAI application."""
    def __init__(self, message: str, detail: str = None):
        self.message = message
        self.detail = detail
        super().__init__(self.message)

class IngestionError(TechDocException):
    """Raised when data ingestion fails."""
    pass

class EmbeddingError(TechDocException):
    """Raised when embedding generation fails."""
    pass

class VectorStoreError(TechDocException):
    """Raised when ChromaDB operations fail."""
    pass

class RetrievalError(TechDocException):
    """Raised when search returns no results or fails."""
    pass

class GenerationError(TechDocException):
    """Raised when LLM fails to generate response."""
    pass

class ConfigurationError(TechDocException):
    """Raised when config/environment variables are missing."""
    pass
