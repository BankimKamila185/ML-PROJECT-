import { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Trophy, AlertTriangle, Loader2 } from 'lucide-react';
import { getModels } from '../api/client';
import type { ModelsData } from '../types';

const MODEL_COLORS: Record<string, string> = {
  'Logistic Regression': '#6366f1',
  KNN: '#06b6d4',
  'Decision Tree': '#a78bfa',
};

const METRICS: Array<{ key: string; label: string; color: string }> = [
  { key: 'accuracy', label: 'Accuracy', color: '#6366f1' },
  { key: 'precision', label: 'Precision', color: '#06b6d4' },
  { key: 'recall', label: 'Recall', color: '#10b981' },
  { key: 'f1_score', label: 'F1 Score', color: '#f59e0b' },
];

function MetricChart({
  title,
  dataKey,
  color,
  chartData,
}: {
  title: string;
  dataKey: string;
  color: string;
  chartData: any[];
}) {
  return (
    <div className="card">
      <h3 className="text-sm font-semibold text-slate-300 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={chartData} margin={{ top: 0, right: 8, left: -20, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
          <XAxis dataKey="model" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <Tooltip
            formatter={(v) => [`${(Number(v) * 100).toFixed(2)}%`, title]}
            contentStyle={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 8, color: '#e2e8f0' }}
          />
          <Bar dataKey={dataKey} fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function ModelComparison() {
  const [data, setData] = useState<ModelsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const load = async () => {
      setLoading(true);
      try {
        setData(await getModels());
      } catch (err: any) {
        setError(err?.response?.data?.detail ?? 'Failed to load model metrics. Ensure models are trained.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  if (loading) return (
    <div className="p-8 flex items-center gap-3 text-slate-400">
      <Loader2 className="w-5 h-5 animate-spin" /> Loading model metrics…
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

  const modelNames = Object.keys(data.models);
  const chartData = modelNames.map((name) => ({
    model: name === 'Logistic Regression' ? 'LR' : name,
    accuracy: data.models[name].accuracy,
    precision: data.models[name].precision,
    recall: data.models[name].recall,
    f1_score: data.models[name].f1_score,
  }));

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-white">Model Comparison</h1>
        <p className="text-slate-400 mt-1 text-sm">
          Performance metrics for all three classification algorithms
        </p>
      </div>

      {/* Best model banner */}
      <div className="card bg-indigo-600/10 border-indigo-500/30 flex items-start gap-4">
        <div className="w-10 h-10 bg-indigo-600/20 rounded-xl flex items-center justify-center shrink-0">
          <Trophy className="w-5 h-5 text-indigo-400" />
        </div>
        <div>
          <p className="text-indigo-300 text-xs font-semibold uppercase tracking-wider">Best Model Selected</p>
          <p className="text-white text-lg font-bold mt-0.5">{data.best_model}</p>
          <p className="text-slate-400 text-sm mt-1">
            The {data.best_model} model achieved the highest F1 score and has been selected for prediction.
            F1 score:{' '}
            <strong className="text-indigo-300">
              {(data.models[data.best_model].f1_score * 100).toFixed(2)}%
            </strong>
          </p>
        </div>
      </div>

      {/* Metric charts */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
        {METRICS.map(({ key, label, color }) => (
          <MetricChart key={key} title={label} dataKey={key} color={color} chartData={chartData} />
        ))}
      </div>

      {/* Comparison table */}
      <div className="card overflow-x-auto">
        <h2 className="text-base font-semibold text-white mb-4">Model Comparison Table</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700">
              <th className="text-left py-3 px-4 text-slate-400 font-medium">Model</th>
              <th className="text-right py-3 px-4 text-slate-400 font-medium">Accuracy</th>
              <th className="text-right py-3 px-4 text-slate-400 font-medium">Precision</th>
              <th className="text-right py-3 px-4 text-slate-400 font-medium">Recall</th>
              <th className="text-right py-3 px-4 text-slate-400 font-medium">F1 Score</th>
              <th className="text-center py-3 px-4 text-slate-400 font-medium">Status</th>
            </tr>
          </thead>
          <tbody>
            {modelNames.map((name) => {
              const m = data.models[name];
              const isBest = name === data.best_model;
              return (
                <tr key={name} className={`border-b border-slate-800 ${isBest ? 'bg-indigo-600/5' : ''}`}>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2.5 h-2.5 rounded-full"
                        style={{ background: MODEL_COLORS[name] ?? '#6366f1' }}
                      />
                      <span className="text-slate-200 font-medium">{name}</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 text-right text-slate-300">{(m.accuracy * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-right text-slate-300">{(m.precision * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-right text-slate-300">{(m.recall * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-right text-slate-300">{(m.f1_score * 100).toFixed(2)}%</td>
                  <td className="py-3 px-4 text-center">
                    {isBest ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold
                                       bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                        <Trophy className="w-3 h-3" /> Best
                      </span>
                    ) : (
                      <span className="text-slate-600 text-xs">–</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
