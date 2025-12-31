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

# Configure CORS middleware
# Allow both localhost for development and Vercel for production
allowed_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "https://ghaffar-physical-ai-book.vercel.app",
    "https://physical-ai-humanoid-robotics-orpin.vercel.app",
]

# Also allow origins from environment variable
if os.getenv("ALLOWED_ORIGINS"):
    allowed_origins.extend(os.getenv("ALLOWED_ORIGINS").split(","))

# Regex pattern to allow all Vercel preview deployments
# Matches: https://*.vercel.app and https://*-ghaffar-ahmeds-projects.vercel.app
vercel_origin_regex = r"https://.*\.vercel\.app"

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=vercel_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="", tags=["health"])
app.include_router(chat.router, prefix="", tags=["chat"])
