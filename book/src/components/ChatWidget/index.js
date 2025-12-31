import React, { useState, useRef, useEffect } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './styles.module.css';

export default function ChatWidget() {
  const { siteConfig } = useDocusaurusContext();
  const API_BASE_URL = siteConfig.customFields?.apiBaseUrl || 'http://localhost:8001';

  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Capture selected text from the page
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection().toString().trim();
      if (selection && selection.length > 0) {
        setSelectedText(selection);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => document.removeEventListener('mouseup', handleSelection);
  }, []);

  // Generate a unique session ID for this browser session
  const [sessionId] = useState(() => {
    const stored = typeof window !== 'undefined' ? sessionStorage.getItem('chat_session_id') : null;
    if (stored) return stored;
    const newId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    if (typeof window !== 'undefined') sessionStorage.setItem('chat_session_id', newId);
    return newId;
  });

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user',
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const requestBody = {
        query: inputValue,
        session_id: sessionId,
      };

      if (selectedText) {
        requestBody.selected_text = selectedText;
      }

      const response = await fetch(`${API_BASE_URL}/api/chat/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to get response');
      }

      // Handle streaming response (SSE format)
      const text = await response.text();
      const lines = text.split('\n').filter(line => line.startsWith('data: '));

      let answer = '';
      let sources = [];

      for (const line of lines) {
        try {
          const data = JSON.parse(line.replace('data: ', ''));
          if (data.type === 'answer') {
            answer = data.content;
          } else if (data.type === 'sources') {
            sources = data.sources || [];
          }
        } catch (e) {
          // Skip malformed lines
        }
      }

      const botMessage = {
        id: Date.now() + 1,
        text: answer || 'I received your question but could not generate a response.',
        sender: 'bot',
        sources: sources,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: `Error: ${error.message || 'Failed to connect to the assistant'}`,
        sender: 'error',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
      setSelectedText('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const clearSelectedText = () => {
    setSelectedText('');
  };

  return (
    <div className={styles.chatWidgetContainer}>
      {/* Floating Toggle Button */}
      <button
        className={styles.toggleButton}
        onClick={() => setIsOpen(!isOpen)}
        aria-label={isOpen ? 'Close chat' : 'Open chat'}
      >
        {isOpen ? (
          <span className={styles.closeIcon}>×</span>
        ) : (
          <span className={styles.chatIcon}>💬</span>
        )}
      </button>

      {/* Chat Window */}
      {isOpen && (
        <div className={styles.chatWindow}>
          <div className={styles.chatHeader}>
            <h3>Book Assistant</h3>
            <span className={styles.headerSubtext}>Ask questions about the book content</span>
          </div>

          {/* Selected Text Preview */}
          {selectedText && (
            <div className={styles.selectedTextPreview}>
              <div className={styles.selectedTextHeader}>
                <span>Selected text:</span>
                <button onClick={clearSelectedText} className={styles.clearButton}>
                  Clear
                </button>
              </div>
              <p>"{selectedText.substring(0, 100)}{selectedText.length > 100 ? '...' : ''}"</p>
            </div>
          )}

          {/* Messages Area */}
          <div className={styles.messagesContainer}>
            {messages.length === 0 && (
              <div className={styles.welcomeMessage}>
                <p>Welcome! Ask me anything about the book content.</p>
                <p className={styles.tip}>Tip: Select text on the page to ask questions about specific content.</p>
              </div>
            )}
            {messages.map((message) => (
              <div
                key={message.id}
                className={`${styles.message} ${styles[message.sender]}`}
              >
                <div className={styles.messageText}>{message.text}</div>
              </div>
            ))}
            {isLoading && (
              <div className={`${styles.message} ${styles.bot}`}>
                <div className={styles.typingIndicator}>
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className={styles.inputArea}>
            <textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question..."
              rows="2"
              disabled={isLoading}
              className={styles.input}
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className={styles.sendButton}
            >
              {isLoading ? '...' : 'Send'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
