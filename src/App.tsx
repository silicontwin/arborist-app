import { HashRouter as Router, Route, Routes } from 'react-router-dom';
import * as ReactDOM from 'react-dom/client';
import Header from './components/Header';
import Plots from './pages/Plots';
import Models from './pages/Models';
import Workspace from './pages/Workspace';

const App = () => (
  <Router>
    <div className="w-full h-full flex flex-col">
      <Header />

      <div className="flex-1 overflow-auto">
        <Routes>
          <Route path="/" element={<Workspace />} />
          <Route path="/models" element={<Models />} />
          <Route path="/plots" element={<Plots />} />
        </Routes>
      </div>
    </div>
  </Router>
);

function render() {
  const rootElement = document.getElementById('app');
  if (!rootElement) throw new Error('Root element not found');

  const root = ReactDOM.createRoot(rootElement);
  root.render(<App />);
}

render();
