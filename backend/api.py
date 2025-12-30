from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os

# Import routers
from backend.routers import health, chat
from backend.rate_limiter import add_rate_limiting

# Create the FastAPI app instance
app = FastAPI(
    title="Book Content Chat API",
    description="API for interacting with the RAG system to ask questions about book content",
    version="1.0.0"
)

# Add rate limiting
add_rate_limiting(app)

# Configure CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Docusaurus default
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    # In production, you would want to be more specific about allowed origins
)

# Include routers
app.include_router(health.router, prefix="", tags=["health"])
app.include_router(chat.router, prefix="", tags=["chat"])