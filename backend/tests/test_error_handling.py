import pytest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_error_handling_empty_message():
    """Test error handling for empty message"""
    test_data = {
        "message": "",
        "top_k": 5
    }

    response = client.post("/chat", json=test_data)
    assert response.status_code == 422  # Pydantic validation error

    data = response.json()
    assert "detail" in data
    # Pydantic returns detail as a list of error objects
    assert isinstance(data["detail"], list)
    assert any("message" in str(err.get("loc", [])) for err in data["detail"])

def test_error_handling_long_selected_text():
    """Test error handling for selected text that is too long"""
    test_data = {
        "message": "What does this mean?",
        "selected_text": "A" * 6000,  # This exceeds the 5000 character limit
        "top_k": 5
    }

    response = client.post("/chat", json=test_data)
    assert response.status_code == 422  # Pydantic validation error

    data = response.json()
    assert "detail" in data
    # Pydantic returns detail as a list of error objects
    assert isinstance(data["detail"], list)
    assert any("selected_text" in str(err.get("loc", [])) for err in data["detail"])

def test_error_handling_invalid_top_k():
    """Test error handling for invalid top_k value"""
    test_data = {
        "message": "What is this about?",
        "top_k": 25  # This exceeds the maximum value of 20
    }

    response = client.post("/chat", json=test_data)
    assert response.status_code == 422  # Pydantic validation error

    data = response.json()
    assert "detail" in data
    # Pydantic returns detail as a list of error objects
    assert isinstance(data["detail"], list)
    assert any("top_k" in str(err.get("loc", [])) for err in data["detail"])

def test_health_endpoint():
    """Test the health endpoint still works correctly"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"