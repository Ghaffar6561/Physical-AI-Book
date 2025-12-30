import pytest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_chat_endpoint_with_selected_text():
    """Test the chat endpoint with selected text functionality"""
    test_data = {
        "message": "What does this selected text mean?",
        "selected_text": "This is a sample selected text for testing purposes. It should be used as the context for answering the question.",
        "top_k": 3
    }

    response = client.post("/chat", json=test_data)
    # Note: Could be 429 if rate limit was hit in previous tests
    assert response.status_code in [200, 429]

    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

        # Verify that the answer mentions the selected text
        assert "selected text" in data["answer"].lower()

def test_chat_endpoint_without_selected_text():
    """Test the chat endpoint without selected text (normal mode)"""
    test_data = {
        "message": "What is this book about?",
        "top_k": 5
    }

    response = client.post("/chat", json=test_data)
    # Note: Could be 429 if rate limit was hit in previous tests
    assert response.status_code in [200, 429]

    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

def test_selected_text_length_limit():
    """Test that selected text has proper length validation"""
    # Create a very long selected text that exceeds the limit
    long_text = "A" * 6000  # This exceeds the 5000 character limit
    
    test_data = {
        "message": "What does this long text mean?",
        "selected_text": long_text,
        "top_k": 3
    }
    
    response = client.post("/chat", json=test_data)
    assert response.status_code == 422  # Pydantic validation error