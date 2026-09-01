import { useEffect, useState } from 'react';
import { getModels } from '../api/client';
import type { ModelsData } from '../types';
import { AlertTriangle, Loader2 } from 'lucide-react';

const MODEL_NAMES = ['Logistic Regression', 'KNN', 'Decision Tree'];
const CLASS_LABELS = ['LOW', 'MEDIUM', 'HIGH'];

function ConfusionMatrixGrid({ matrix, labels }: { matrix: number[][]; labels: string[] }) {
  const maxVal = Math.max(...matrix.flat());

  const cellColor = (row: number, col: number, value: number) => {
    if (row === col) {
      const intensity = maxVal > 0 ? value / maxVal : 0;
      return `rgba(99, 102, 241, ${0.15 + intensity * 0.7})`;
    }
    const intensity = maxVal > 0 ? value / maxVal : 0;
    return `rgba(239, 68, 68, ${intensity * 0.5})`;
  };

  return (
    <div className="overflow-x-auto">
      <table className="mx-auto border-collapse">
        <thead>
          <tr>
            <th className="p-3 text-xs text-slate-500 font-normal">Actual ↓ / Predicted →</th>
            {labels.map((l) => (
              <th key={l} className="p-3 text-xs text-slate-400 font-semibold text-center min-w-[90px]">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, ri) => (
            <tr key={ri}>
              <td className="p-3 text-xs text-slate-400 font-semibold text-right pr-4">
                {labels[ri]}
              </td>
              {row.map((val, ci) => (
                <td
                  key={ci}
                  className="p-3 text-center rounded-lg border border-slate-700/30"
                  style={{ backgroundColor: cellColor(ri, ci, val), minWidth: 90 }}
                >
                  <div className="text-lg font-bold text-white">{val}</div>
                  <div className="text-[10px] text-slate-400 mt-0.5">
                    {ri === ci ? '✓ Correct' : '✗ Wrong'}
                  </div>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mt-4 flex items-center justify-center gap-6 text-xs text-slate-400">
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ background: 'rgba(99,102,241,0.7)' }} />
          Correct prediction (diagonal)
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded" style={{ background: 'rgba(239,68,68,0.4)' }} />
          Misclassification
        </div>
      </div>
    </div>
  );
}

export default function ConfusionMatrixPage() {
  const [data, setData] = useState<ModelsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string>('Logistic Regression');

  useEffect(() => {
    const load = async () => {
      try {
        const d = await getModels();
        setData(d);
        setSelected(d.best_model);
      } catch (err: any) {
        setError(err?.response?.data?.detail ?? 'Failed to load metrics.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return (
    <div className="p-8 flex items-center gap-3 text-slate-400">
      <Loader2 className="w-5 h-5 animate-spin" /> Loading confusion matrices…
    </div>
  );

  if (error) return (
    <div className="p-8">
      <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
        <p className="text-red-300 text-sm">{error}</p>
      </div>
    </div>
  );

  if (!data) return null;

  const currentMetrics = data.models[selected];

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Confusion Matrix</h1>
        <p className="text-slate-400 mt-1 text-sm">
          Visualise true vs predicted classifications for each model
        </p>
      </div>

      {/* Model selector */}
      <div className="flex gap-3 flex-wrap">
        {MODEL_NAMES.map((name) => (
          <button
            key={name}
            onClick={() => setSelected(name)}
            className={`px-4 py-2 rounded-xl text-sm font-medium border transition-all duration-150 ${
              selected === name
                ? 'bg-indigo-600/20 border-indigo-500/50 text-indigo-300'
                : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200 hover:bg-slate-700'
            }`}
          >
            {name}
            {name === data.best_model && (
              <span className="ml-2 text-[10px] text-indigo-400">★ Best</span>
            )}
          </button>
        ))}
      </div>

      {/* Matrix */}
      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-base font-semibold text-white">{selected} – Confusion Matrix</h2>
          <div className="flex gap-4 text-xs text-slate-400">
            <span>Accuracy: <strong className="text-white">{(currentMetrics.accuracy * 100).toFixed(2)}%</strong></span>
            <span>F1: <strong className="text-white">{(currentMetrics.f1_score * 100).toFixed(2)}%</strong></span>
          </div>
        </div>
        <ConfusionMatrixGrid
          matrix={currentMetrics.confusion_matrix}
          labels={CLASS_LABELS}
        />
      </div>

      {/* Explanation */}
      <div className="card bg-slate-900/40">
        <h3 className="text-sm font-semibold text-white mb-2">What is a Confusion Matrix?</h3>
        <p className="text-slate-400 text-sm leading-relaxed">
          A confusion matrix is a table that shows how many predictions the model got right and wrong,
          broken down by class. Each <strong className="text-indigo-300">row</strong> represents the actual
          class, and each <strong className="text-indigo-300">column</strong> represents the predicted class.
          Values along the <strong className="text-indigo-300">diagonal</strong> are correct predictions
          (true positives for each class). Off-diagonal values are misclassifications.
          A perfect classifier would have all values on the diagonal.
        </p>
      </div>
    </div>
  );
}
