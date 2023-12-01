// /src/App.tsx

import React from 'react';
import { HashRouter as Router, Route, Routes } from 'react-router-dom';
import Header from './components/Header';
import Homepage from './pages/Homepage';
import About from './pages/About';

const App = () => {
  return (
    <Router>
      <div className="bg-gray-300 w-screen h-screen">
        <Header />
        <Routes>
          <Route path="/" element={<Homepage />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
