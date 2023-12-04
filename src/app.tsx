import React from 'react';
import { HashRouter as Router, Route, Routes } from 'react-router-dom';
import * as ReactDOM from 'react-dom/client';
import Header from './components/Header';
import About from './pages/About';
import Homepage from './pages/Homepage';

const App = () => (
  <Router>
    <div className="bg-[#1D1D1D] w-full text-white">
      <Header />

      <div className="h-[calc(800px_-_50px)]">
        <Routes>
          <Route path="/" element={<Homepage />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </div>
  </Router>
);

function render() {
  const root = ReactDOM.createRoot(document.getElementById('app'));
  root.render(<App />);
}

render();
