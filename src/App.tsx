import { HashRouter as Router, Route, Routes } from 'react-router-dom';
import * as ReactDOM from 'react-dom/client';
import Header from './components/Header';
import About from './pages/About';
import Upload from './pages/Upload';
import Workspace from './pages/Workspace';

const App = () => (
  <Router>
    <div className="w-full h-full flex flex-col">
      <Header />

      <div className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Workspace />} />
          <Route path="/upload" element={<Upload />} />
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
