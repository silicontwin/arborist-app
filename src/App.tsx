// /src/App.tsx

import React from 'react';
import { BrowserRouter as Router, Route } from 'react-router-dom';
import Header from './components/Header';
import Homepage from './pages/Homepage';
import About from './pages/About';

const App = () => {
  return (
    <div className="bg-gray-300 w-screen h-screen">
      <Header />
      <Homepage />
    </div>
  );
};

export default App;
