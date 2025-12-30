from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SourceCitation(BaseModel):
    """
    Reference to the specific location in the book where the answer information was found
    """
    content: str = Field(..., min_length=1, max_length=2000, description="The text content from the source")
    page_number: Optional[int] = Field(None, ge=1, description="Page number where the content appears")
    section_title: Optional[str] = Field(None, description="Title of the section containing the content")
    url: Optional[str] = Field(None, description="URL to the specific location in the online book")
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence score of the citation relevance")


class Question(BaseModel):
    """
    A text query submitted by the user, with optional context about selected text
    """
    message: str = Field(..., min_length=1, max_length=1000, description="The user's question text")
    selected_text: Optional[str] = Field(None, min_length=1, max_length=5000, description="Text selected by the user on the page (for selected-text mode)")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of top results to retrieve from the RAG system")


class Answer(BaseModel):
    """
    A response to the user's question, including the answer text and source citations
    """
    answer: str = Field(..., min_length=1, max_length=10000, description="The AI-generated answer to the user's question")
    sources: List[SourceCitation] = Field(..., min_length=1, max_length=10, description="List of sources used to generate the answer")
    question_id: Optional[str] = Field(None, description="Unique identifier for the question (for potential future use)")


class ChatSession(BaseModel):
    """
    A sequence of interactions between the user and the system (though no persistence required for this implementation)
    """
    session_id: str = Field(..., description="Unique identifier for the session")
    created_at: datetime = Field(..., description="Timestamp when the session was created")
    messages: Optional[List[object]] = Field(None, description="List of question-answer pairs in the session")