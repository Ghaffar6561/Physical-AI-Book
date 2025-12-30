import React, { useState, useEffect } from 'react';
import ApiService from '../services/api';
import config from '../services/config';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');

  // Function to handle sending a message
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    // Add user message to the chat
    const userMessage = { id: Date.now(), text: inputValue, sender: 'user' };
    setMessages(prev => [...prev, userMessage]);
    
    setIsLoading(true);
    
    try {
      // Send the message to the backend
      const response = await ApiService.chat(inputValue, selectedText);
      
      // Add bot response to the chat
      const botMessage = {
        id: Date.now() + 1,
        text: response.answer,
        sender: 'bot',
        sources: response.sources
      };
      
      setMessages(prev => [...prev, botMessage]);
      setInputValue('');
    } catch (error) {
      // Add error message to the chat
      const errorMessage = {
        id: Date.now() + 1,
        text: `Error: ${error.message || 'An error occurred while processing your request'}`,
        sender: 'system'
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  // Function to handle key press (Enter to send)
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Function to capture selected text
  useEffect(() => {
    const handleSelection = () => {
      const selectedText = window.getSelection().toString().trim();
      setSelectedText(selectedText);
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  return (
    <div className="chat-interface">
      <div className="chat-header">
        <h2>Book Content Assistant</h2>
        {selectedText && (
          <div className="selected-text-preview">
            <p><strong>Selected text:</strong> "{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"</p>
          </div>
        )}
      </div>
      
      <div className="chat-messages">
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.sender}`}>
            <div className="message-text">{message.text}</div>
            {message.sources && message.sources.length > 0 && (
              <div className="sources">
                <h4>Sources:</h4>
                <ul>
                  {message.sources.map((source, index) => (
                    <li key={index}>
                      <a href={source.url} target="_blank" rel="noopener noreferrer">
                        {source.section_title || `Source ${index + 1}`}
                      </a>
                      {source.page_number && <span> (Page {source.page_number})</span>}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
        
        {isLoading && (
          <div className="message bot">
            <div className="message-text">Thinking...</div>
          </div>
        )}
      </div>
      
      <div className="chat-input-area">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask a question about the book content..."
          rows="3"
          disabled={isLoading}
        />
        <button 
          onClick={handleSendMessage} 
          disabled={!inputValue.trim() || isLoading}
          className="send-button"
        >
          {isLoading ? 'Sending...' : 'Send'}
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;