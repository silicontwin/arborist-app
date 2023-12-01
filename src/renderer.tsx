// /src/renderer.tsx

import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import '../public/styles/global.css';

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container); // create a root
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>,
  );
} else {
  console.error('Failed to find the root element');
}
