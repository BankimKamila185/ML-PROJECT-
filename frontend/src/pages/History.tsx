import { useEffect, useState, useCallback } from 'react';
import { Search, ChevronUp, ChevronDown, Loader2, AlertTriangle, Filter } from 'lucide-react';
import { getHistory } from '../api/client';
import type { HistoryRecord, HistoryResponse } from '../types';
import RiskBadge from '../components/RiskBadge';

const PAGE_SIZE = 20;

function SortButton({
  active,
  direction,
  onClick,
}: {
  active: boolean;
  direction: string;
  onClick: () => void;
}) {
  return (
    <button onClick={onClick} className="ml-1 opacity-60 hover:opacity-100">
      {active && direction === 'asc' ? (
        <ChevronUp className="w-3.5 h-3.5" />
      ) : (
        <ChevronDown className="w-3.5 h-3.5" />
      )}
    </button>
  );
}

export default function HistoryPage() {
  const [data, setData] = useState<HistoryResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState('');
  const [sortBy, setSortBy] = useState('created_at');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [page, setPage] = useState(0);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getHistory({
        skip: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        search: search || undefined,
        risk_filter: riskFilter || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
      });
      setData(res);
    } catch (err: any) {
      setError('Failed to load prediction history.');
    } finally {
      setLoading(false);
    }
  }, [search, riskFilter, sortBy, sortOrder, page]);

  useEffect(() => { load(); }, [load]);

  const handleSort = (col: string) => {
    if (sortBy === col) {
      setSortOrder((o) => (o === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(col);
      setSortOrder('desc');
    }
    setPage(0);
  };

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;

  const cols: Array<{ key: string; label: string; sortable?: boolean }> = [
    { key: 'student_id', label: 'Student ID', sortable: true },
    { key: 'attendance', label: 'Attend %', sortable: true },
    { key: 'internal_marks', label: 'Int. Marks', sortable: true },
    { key: 'assignment_score', label: 'Assign.', sortable: true },
    { key: 'previous_gpa', label: 'GPA', sortable: true },
    { key: 'study_hours', label: 'Study Hrs', sortable: true },
    { key: 'backlogs', label: 'Backlogs', sortable: true },
    { key: 'risk_level', label: 'Risk Level', sortable: true },
    { key: 'confidence', label: 'Confidence' },
    { key: 'model_used', label: 'Model' },
    { key: 'created_at', label: 'Date', sortable: true },
  ];

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white">Prediction History</h1>
        <p className="text-slate-400 mt-1 text-sm">
          {data ? `${data.total} total predictions stored` : 'Loading…'}
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3">
        <div className="relative flex-1 min-w-48">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="text"
            value={search}
            onChange={(e) => { setSearch(e.target.value); setPage(0); }}
            placeholder="Search student ID or risk level…"
            className="input-field pl-10"
          />
        </div>
        <div className="relative">
          <Filter className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <select
            value={riskFilter}
            onChange={(e) => { setRiskFilter(e.target.value); setPage(0); }}
            className="input-field pl-9 pr-8 appearance-none"
          >
            <option value="">All Risk Levels</option>
            <option value="LOW">LOW RISK</option>
            <option value="MEDIUM">MEDIUM RISK</option>
            <option value="HIGH">HIGH RISK</option>
          </select>
        </div>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <p className="text-red-300 text-sm">{error}</p>
        </div>
      )}

      {/* Table */}
      <div className="card overflow-x-auto p-0">
        <table className="w-full text-sm min-w-[900px]">
          <thead>
            <tr className="border-b border-slate-700">
              {cols.map(({ key, label, sortable }) => (
                <th
                  key={key}
                  className="text-left py-3 px-4 text-slate-400 font-medium whitespace-nowrap"
                >
                  <span className="flex items-center">
                    {label}
                    {sortable && (
                      <SortButton
                        active={sortBy === key}
                        direction={sortOrder}
                        onClick={() => handleSort(key)}
                      />
                    )}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={cols.length} className="py-12 text-center">
                  <Loader2 className="w-6 h-6 animate-spin mx-auto text-slate-500" />
                </td>
              </tr>
            ) : data?.records.length === 0 ? (
              <tr>
                <td colSpan={cols.length} className="py-12 text-center text-slate-500">
                  No prediction records found.
                </td>
              </tr>
            ) : (
              data?.records.map((r: HistoryRecord) => (
                <tr key={r.id} className="border-b border-slate-800/80 hover:bg-slate-800/30 transition-colors">
                  <td className="py-3 px-4 font-mono text-indigo-300 text-xs">{r.student_id}</td>
                  <td className="py-3 px-4 text-slate-300">{r.attendance.toFixed(1)}</td>
                  <td className="py-3 px-4 text-slate-300">{r.internal_marks.toFixed(1)}</td>
                  <td className="py-3 px-4 text-slate-300">{r.assignment_score.toFixed(1)}</td>
                  <td className="py-3 px-4 text-slate-300">{r.previous_gpa.toFixed(2)}</td>
                  <td className="py-3 px-4 text-slate-300">{r.study_hours.toFixed(1)}</td>
                  <td className="py-3 px-4 text-slate-300">{r.backlogs}</td>
                  <td className="py-3 px-4">
                    <RiskBadge level={r.risk_level} size="sm" />
                  </td>
                  <td className="py-3 px-4 text-slate-400 text-xs">
                    {r.confidence != null ? `${(r.confidence * 100).toFixed(1)}%` : 'N/A'}
                  </td>
                  <td className="py-3 px-4 text-slate-400 text-xs whitespace-nowrap">{r.model_used}</td>
                  <td className="py-3 px-4 text-slate-500 text-xs whitespace-nowrap">
                    {new Date(r.created_at).toLocaleString()}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between text-sm text-slate-400">
          <span>
            Page {page + 1} of {totalPages}
          </span>
          <div className="flex gap-2">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg hover:bg-slate-700
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Previous
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg hover:bg-slate-700
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
