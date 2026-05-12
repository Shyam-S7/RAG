import logging
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.core.settings import Settings

from src.utils.logging import get_logger

logger = get_logger(__name__)

class ModelRegistry:
    """
    Singleton-like registry to initialize and store shared model instances.
    Prevents repeated loading of heavy models across different pipelines.
    """
    _embedding_model: Optional[HuggingFaceEmbeddings] = None
    _reranker_model: Optional[CrossEncoder] = None
    _llm: Optional[ChatGroq] = None
    
    _device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def get_embedding_model(cls) -> HuggingFaceEmbeddings:
        """Returns the shared LangChain-compatible Embedding Model instance."""
        if cls._embedding_model is None:
            logger.info(f"🚀 Loading Embedding Model: {Settings.EMBEDDING_MODEL_NAME} on {cls._device}...")
            cls._embedding_model = HuggingFaceEmbeddings(
                model_name=Settings.EMBEDDING_MODEL_NAME,
                model_kwargs={"device": cls._device},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("✅ Embedding Model loaded successfully.")
        return cls._embedding_model

    @classmethod
    def get_reranker_model(cls) -> CrossEncoder:
        """Returns the shared Reranker instance."""
        if cls._reranker_model is None:
            logger.info(f"🚀 Loading Reranker Model: {Settings.RERANKER_MODEL_NAME} on {cls._device}...")
            cls._reranker_model = CrossEncoder(
                Settings.RERANKER_MODEL_NAME,
                device=cls._device
            )
            logger.info("✅ Reranker Model loaded successfully.")
        return cls._reranker_model

    @classmethod
    def get_llm(cls) -> ChatGroq:
        """Returns the shared LLM instance."""
        if cls._llm is None:
            logger.info(f"🚀 Initializing LLM: {Settings.LLM_MODEL_NAME}")
            if Settings.LLM_PROVIDER == "groq":
                cls._llm = ChatGroq(
                    model_name=Settings.LLM_MODEL_NAME,
                    groq_api_key=Settings.GROQ_API_KEY,
                    temperature=0
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {Settings.LLM_PROVIDER}")
            logger.info(f"✅ LLM ({Settings.LLM_PROVIDER}) initialized.")
        return cls._llm
