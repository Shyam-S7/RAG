from typing import List
from langchain_core.documents import Document
from langchain_community.document_transformers import LongContextReorder

class PostProcessor:
    """
    Modular Post-Processing for Retrieval:
    - Reorder: Handles 'Lost in the Middle' bias.
    - Diversity (MMR-style): Reduces redundancy.
    - Compression: Trims context to save tokens.
    """

    @staticmethod
    def reorder(documents: List[Document]) -> List[Document]:
        """
        Reorders documents so the most relevant are at the start and end.
        LLMs tend to pay more attention to the beginning and end of context.
        """
        if not documents:
            return []
        reorder = LongContextReorder()
        return reorder.transform_documents(documents)

    @staticmethod
    def filter_redundant(documents: List[Document], threshold: float = 0.95) -> List[Document]:
        """
        Simple text-based redundancy filter.
        Ensures we don't send identical or extremely similar chunks to the LLM.
        """
        unique_docs = []
        seen_contents = set()

        for doc in documents:
            # Simple content cleaning for comparison
            content_key = " ".join(doc.page_content.lower().split())
            if content_key not in seen_contents:
                unique_docs.append(doc)
                seen_contents.add(content_key)
        
        return unique_docs

    @staticmethod
    def compress_context(documents: List[Document], max_chunk_chars: int = 800) -> List[Document]:
        """
        Basic context compression.
        Trims individual documents to a maximum length to ensure space for more sources.
        """
        for doc in documents:
            if len(doc.page_content) > max_chunk_chars:
                doc.page_content = doc.page_content[:max_chunk_chars] + "... [trimmed]"
        return documents

    @staticmethod
    def optimize(documents: List[Document], k: int = 5) -> List[Document]:
        """
        The Final Pipeline: Redundancy Filter -> Trimming -> Reordering.
        """
        # 1. Remove duplicates/redundant chunks
        docs = PostProcessor.filter_redundant(documents)
        
        # 2. Limit to top K
        docs = docs[:k]
        
        # 3. Compress if chunks are too bulky
        docs = PostProcessor.compress_context(docs)
        
        # 4. Reorder for LLM attention
        return PostProcessor.reorder(docs)
