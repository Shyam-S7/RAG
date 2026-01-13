import sys
import logging

logger = logging.getLogger(__name__)


def error_message_detail(error, error_detail: sys):
    """Format error message with file name and line number."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename

    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )

    return error_message


class CustomException(Exception):
    """Custom exception with detailed error information."""

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail
        )

    def __str__(self):
        return self.error_message


# Add specific exceptions for RAG pipeline
class IngestionError(CustomException):
    """Exception raised during document ingestion."""

    pass


class EmbeddingError(CustomException):
    """Exception raised during embedding generation."""

    pass


class VectorStoreError(CustomException):
    """Exception raised during vector store operations."""

    pass


class RetrievalError(CustomException):
    """Exception raised during retrieval operations."""

    pass
