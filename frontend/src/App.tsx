import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import PredictionPage from './pages/Prediction';
import ModelComparison from './pages/ModelComparison';
import ConfusionMatrixPage from './pages/ConfusionMatrix';
import HistoryPage from './pages/History';

export default function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<PredictionPage />} />
            <Route path="/comparison" element={<ModelComparison />} />
            <Route path="/confusion" element={<ConfusionMatrixPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
