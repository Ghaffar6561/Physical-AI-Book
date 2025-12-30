from pydantic import BaseModel, Field
from typing import List, Optional
from .chat_models import Answer, SourceCitation


class ChatRequest(BaseModel):
    """
    Request model for the chat endpoint
    """
    message: str = Field(..., min_length=1, max_length=1000, description="The user's question text")
    selected_text: Optional[str] = Field(None, min_length=1, max_length=5000, description="Optional text selected by the user on the page (for selected-text mode)")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of top results to retrieve from the RAG system")


class ChatResponse(BaseModel):
    """
    Response model for the chat endpoint
    """
    answer: str = Field(..., min_length=1, max_length=10000, description="The AI-generated answer to the user's question")
    sources: List[SourceCitation] = Field(default=[], max_length=10, description="List of sources used to generate the answer")
    question_id: Optional[str] = Field(None, description="Unique identifier for the question (for potential future use)")


class ErrorResponse(BaseModel):
    """
    Response model for error cases
    """
    error: str = Field(..., description="Error message describing what went wrong")
    code: str = Field(..., description="Error code for the specific error condition")


class HealthResponse(BaseModel):
    """
    Response model for health check endpoint
    """
    status: str = Field(..., description="Service health status")
    timestamp: str = Field(..., description="Timestamp of the health check")