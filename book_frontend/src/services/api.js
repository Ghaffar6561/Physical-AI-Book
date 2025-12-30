// API service for backend communication
// This file will be used by the frontend to communicate with the backend API

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

class ApiService {
  // Health check endpoint
  static async healthCheck() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      throw error;
    }
  }

  // Chat endpoint
  static async chat(message, selectedText = null, topK = 5) {
    try {
      const requestBody = {
        message: message,
        top_k: topK
      };

      if (selectedText) {
        requestBody.selected_text = selectedText;
      }

      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get response from chat API');
      }

      return await response.json();
    } catch (error) {
      console.error('Chat API call failed:', error);
      throw error;
    }
  }
}

export default ApiService;