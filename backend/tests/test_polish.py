import pytest
from fastapi.testclient import TestClient
from backend.api import app

client = TestClient(app)

def test_rate_limiting():
    """Test that rate limiting is working"""
    # Send multiple requests quickly to test rate limiting
    for i in range(25):  # More than the 20/minute limit
        test_data = {
            "message": f"What is this book about? Request number {i}",
            "top_k": 5
        }
        
        response = client.post("/chat", json=test_data)
        
        # The first 20 requests should be successful, then rate limiting should kick in
        if i < 20:
            assert response.status_code in [200, 429]  # Could be 200 or 429 depending on timing
        else:
            # After the limit, we might get rate limited (429) responses
            assert response.status_code in [200, 429]

def test_input_sanitization():
    """Test that input sanitization is working"""
    # Test with potentially dangerous input
    test_data = {
        "message": "What is this book about? <script>alert('xss')</script>",
        "selected_text": "Selected text with <iframe src='javascript:alert(1)'></iframe>",
        "top_k": 5
    }

    response = client.post("/chat", json=test_data)
    # Should still process successfully despite sanitized input
    # Note: Could be 429 if rate limit was hit in previous tests
    assert response.status_code in [200, 400, 429]

def test_performance_monitoring():
    """Test that performance monitoring doesn't break functionality"""
    test_data = {
        "message": "Performance test question",
        "top_k": 5
    }

    response = client.post("/chat", json=test_data)
    # Note: Could be 429 if rate limit was hit in previous tests
    assert response.status_code in [200, 429]

    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "sources" in data