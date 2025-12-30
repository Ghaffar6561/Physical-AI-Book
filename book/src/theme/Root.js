import React from 'react';
import ChatWidget from '@site/src/components/ChatWidget';

// This component wraps the entire Docusaurus app
// The ChatWidget will appear on every page
export default function Root({ children }) {
  return (
    <>
      {children}
      <ChatWidget />
    </>
  );
}
