import pytest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint():
    """Test the chat endpoint with a basic question"""
    test_data = {
        "message": "What is this book about?",
        "top_k": 5
    }
    
    response = client.post("/chat", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0

def test_chat_endpoint_with_selected_text():
    """Test the chat endpoint with selected text"""
    test_data = {
        "message": "What does this selected text mean?",
        "selected_text": "This is a sample selected text for testing purposes.",
        "top_k": 3
    }
    
    response = client.post("/chat", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0

def test_chat_endpoint_empty_message():
    """Test the chat endpoint with an empty message"""
    test_data = {
        "message": "",
        "top_k": 5
    }

    response = client.post("/chat", json=test_data)
    assert response.status_code == 422  # FastAPI/Pydantic validation error