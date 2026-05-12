import uuid
import contextvars
from typing import Optional

# Context variable to store the trace_id for the current execution context
_trace_id_ctx = contextvars.ContextVar("trace_id", default=None)

class TraceManager:
    """
    Manages unique trace IDs for the RAG pipeline.
    Uses contextvars to ensure the trace ID is accessible across the current execution context.
    """

    @staticmethod
    def start_trace(trace_id: Optional[str] = None, force: bool = False) -> str:
        """
        Starts a new trace. If a trace already exists and force is False, returns existing.
        """
        existing = _trace_id_ctx.get()
        if existing and not force and not trace_id:
            return existing
            
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        _trace_id_ctx.set(trace_id)
        return trace_id

    @staticmethod
    def get_current_trace_id() -> str:
        """
        Retrieves the trace ID for the current context.
        Generates one if it doesn't exist.
        """
        trace_id = _trace_id_ctx.get()
        if trace_id is None:
            trace_id = TraceManager.start_trace()
        return trace_id

    @staticmethod
    def clear_trace():
        """
        Clears the trace ID for the current context.
        """
        _trace_id_ctx.set(None)
