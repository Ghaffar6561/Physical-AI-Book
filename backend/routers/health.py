from fastapi import APIRouter, HTTPException
from datetime import datetime
from ..models.response_models import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint that returns the health status of the service
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat()
    )