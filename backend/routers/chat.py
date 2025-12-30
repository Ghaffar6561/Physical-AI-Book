from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from ..models.response_models import ChatResponse
from ..models.chat_models import Question
from ..services.rag_service import RAGService
from ..logging_config import logger
from ..utils.sanitization import sanitize_input

# Initialize the limiter
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

rag_service = RAGService()

@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")  # Limit to 20 requests per minute per IP
async def chat_endpoint(request: Request, question: Question):
    """
    Submit a question about book content and receive an answer with source citations
    """
    # Sanitize inputs
    sanitized_message = sanitize_input(question.message)
    sanitized_selected_text = sanitize_input(question.selected_text)

    # Log the incoming request
    logger.info(f"Received chat request: {sanitized_message[:50]}...")

    try:
        # Validate the question
        if not sanitized_message or len(sanitized_message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Question message cannot be empty")

        if sanitized_selected_text and len(sanitized_selected_text) > 5000:
            raise HTTPException(status_code=400, detail="Selected text is too long. Maximum allowed length is 5000 characters.")

        if question.top_k and (question.top_k < 1 or question.top_k > 20):
            raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

        # Process the question using the RAG service
        result = await rag_service.process_question(
            message=sanitized_message,
            selected_text=sanitized_selected_text,
            top_k=question.top_k or 5
        )

        # Log the successful response
        logger.info(f"Successfully processed chat request, response length: {len(result['answer'])}")

        return ChatResponse(
            answer=result['answer'],
            sources=result['sources'],
            question_id=result.get('question_id')
        )
    except HTTPException as e:
        # Log the HTTP error
        logger.error(f"HTTP error in chat endpoint: {e.detail}")
        raise
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing your request"
        )