import React from 'react';
import { HashRouter as Router, Route, Routes } from 'react-router-dom';
import * as ReactDOM from 'react-dom/client';
import Header from './components/Header';
import About from './pages/About';
import Workspace from './pages/Workspace';

const App = () => (
  <Router>
    <div className="bg-[#1D1D1D] w-full text-white h-full flex flex-col">
      <Header />

      <div className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Workspace />} />
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
