import { useEffect, useState } from 'react';
import {
  Users,
  CheckCircle,
  MinusCircle,
  AlertTriangle,
  Trophy,
  Activity,
  Cpu,
  TrendingUp,
} from 'lucide-react';
import { getDashboard } from '../api/client';
import type { DashboardData } from '../types';

function KpiCard({
  icon: Icon,
  label,
  value,
  sub,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: string | number;
  sub?: string;
  color: string;
}) {
  return (
    <div className="card flex items-start gap-4">
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center shrink-0 ${color}`}>
        <Icon className="w-6 h-6" />
      </div>
      <div className="min-w-0">
        <p className="text-slate-400 text-sm font-medium">{label}</p>
        <p className="text-2xl font-bold text-white mt-0.5">{value}</p>
        {sub && <p className="text-xs text-slate-500 mt-0.5">{sub}</p>}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await getDashboard();
      setData(d);
    } catch {
      setError('Could not connect to backend. Make sure the server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white">Dashboard</h1>
        <p className="text-slate-400 mt-1 text-sm">
          Student Academic Risk Overview · Module V – Classification
        </p>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <p className="text-red-300 font-medium text-sm">Backend unavailable</p>
            <p className="text-red-400/80 text-xs mt-1">{error}</p>
          </div>
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
          {[...Array(8)].map((_, i) => (
            <div key={i} className="card animate-pulse h-28 bg-slate-800/40" />
          ))}
        </div>
      ) : data ? (
        <>
          {/* KPI row 1 */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-5">
            <KpiCard
              icon={Users}
              label="Total Students Analyzed"
              value={data.total}
              color="bg-indigo-600/20 text-indigo-400"
            />
            <KpiCard
              icon={CheckCircle}
              label="Low Risk Students"
              value={data.low}
              sub={data.total > 0 ? `${((data.low / data.total) * 100).toFixed(1)}% of total` : undefined}
              color="bg-emerald-600/20 text-emerald-400"
            />
            <KpiCard
              icon={MinusCircle}
              label="Medium Risk Students"
              value={data.medium}
              sub={data.total > 0 ? `${((data.medium / data.total) * 100).toFixed(1)}% of total` : undefined}
              color="bg-amber-600/20 text-amber-400"
            />
            <KpiCard
              icon={AlertTriangle}
              label="High Risk Students"
              value={data.high}
              sub={data.total > 0 ? `${((data.high / data.total) * 100).toFixed(1)}% of total` : undefined}
              color="bg-red-600/20 text-red-400"
            />
          </div>

          {/* KPI row 2 */}
          <div className="grid grid-cols-2 lg:grid-cols-3 gap-5">
            <KpiCard
              icon={Trophy}
              label="Best ML Model"
              value={data.best_model ?? '–'}
              color="bg-violet-600/20 text-violet-400"
            />
            <KpiCard
              icon={Activity}
              label="Model Accuracy"
              value={data.best_model_accuracy != null ? `${(data.best_model_accuracy * 100).toFixed(1)}%` : '–'}
              color="bg-cyan-600/20 text-cyan-400"
            />
            <KpiCard
              icon={TrendingUp}
              label="Model F1 Score"
              value={data.best_model_f1 != null ? `${(data.best_model_f1 * 100).toFixed(1)}%` : '–'}
              color="bg-pink-600/20 text-pink-400"
            />
          </div>

          {/* Info banner */}
          <div className="card">
            <div className="flex items-start gap-4">
              <Cpu className="w-6 h-6 text-indigo-400 shrink-0 mt-0.5" />
              <div>
                <h3 className="text-white font-semibold">About This System</h3>
                <p className="text-slate-400 text-sm mt-1 leading-relaxed">
                  This application implements <strong className="text-slate-300">Module V – Supervised Learning: Classification</strong>.
                  Three algorithms are trained and compared: <strong className="text-slate-300">Logistic Regression</strong>,{' '}
                  <strong className="text-slate-300">K-Nearest Neighbors (KNN)</strong>, and{' '}
                  <strong className="text-slate-300">Decision Tree</strong>.
                  The best-performing model (by F1 score) is automatically selected for predictions.
                  Student data is classified into <em className="text-emerald-400">LOW</em>,{' '}
                  <em className="text-amber-400">MEDIUM</em>, or{' '}
                  <em className="text-red-400">HIGH</em> academic risk categories.
                </p>
              </div>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
