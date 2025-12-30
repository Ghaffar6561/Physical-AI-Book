// Environment configuration for frontend
// This file handles environment-specific configurations

const config = {
  // API base URL - can be overridden by environment variable
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  
  // Development mode
  IS_DEV: import.meta.env.DEV,
  
  // Default settings
  DEFAULT_TOP_K: 5,
  DEFAULT_TIMEOUT: 30000, // 30 seconds
};

export default config;