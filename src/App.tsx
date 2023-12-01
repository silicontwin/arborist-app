// /src/App.tsx

import React from 'react';
import { HashRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import Homepage from './pages/Homepage';
import About from './pages/About';

const App = () => {
  return (
    <Router>
      <div className="w-full">
        <Header />

        <div className="bg-gray-100 h-[calc(800px_-_50px)]">
          <Routes>
            <Route path="/" element={<Homepage />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
};

export default App;
