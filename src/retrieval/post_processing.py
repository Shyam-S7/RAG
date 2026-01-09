from typing import List
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder

class PostProcessor:
    @staticmethod
    def reorder(documents: List[Document]) -> List[Document]:
        """Lost in the Middle reordering"""
        return LongContextReorder().transform_documents(documents)
        
    @staticmethod
    def compress(documents: List[Document]) -> List[Document]:
        # Placeholder for Context Compression
        # If needed, can use LLMChainExtractor here
        return documents
