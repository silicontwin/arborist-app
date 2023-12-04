import React from 'react';
import * as ReactDOM from 'react-dom/client';

const App = () => (
  <div className="flex items-center justify-center min-h-screen bg-gray-100">
    <div className="text-center mx-10">Hello world.</div>
  </div>
);

function render() {
  const root = ReactDOM.createRoot(document.getElementById('app'));
  root.render(<App />);
}

render();
