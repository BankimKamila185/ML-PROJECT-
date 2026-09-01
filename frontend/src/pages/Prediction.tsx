import { useState } from 'react';
import { AlertTriangle, CheckCircle, MinusCircle, Loader2, BrainCircuit } from 'lucide-react';
import { predict } from '../api/client';
import type { PredictionRequest, PredictionResponse } from '../types';
import RiskBadge from '../components/RiskBadge';

const RISK_CONFIG = {
  LOW: {
    bg: 'bg-emerald-500/10 border-emerald-500/30',
    title: 'text-emerald-400',
    icon: CheckCircle,
  },
  MEDIUM: {
    bg: 'bg-amber-500/10 border-amber-500/30',
    title: 'text-amber-400',
    icon: MinusCircle,
  },
  HIGH: {
    bg: 'bg-red-500/10 border-red-500/30',
    title: 'text-red-400',
    icon: AlertTriangle,
  },
};

type FormData = Record<keyof PredictionRequest, string>;

const DEFAULT_FORM: FormData = {
  attendance: '',
  internal_marks: '',
  assignment_score: '',
  previous_gpa: '',
  study_hours: '',
  backlogs: '',
  class_participation: '',
};

function FormField({
  id,
  label,
  value,
  onChange,
  min,
  max,
  step = '0.1',
  placeholder,
  error,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  min: number;
  max: number;
  step?: string;
  placeholder: string;
  error?: string;
}) {
  return (
    <div>
      <label htmlFor={id} className="block text-sm font-medium text-slate-300 mb-1.5">
        {label}
        <span className="text-slate-500 text-xs ml-1">({min}–{max})</span>
      </label>
      <input
        id={id}
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={`input-field ${error ? 'border-red-500/60 focus:ring-red-500/30' : ''}`}
        aria-describedby={error ? `${id}-error` : undefined}
      />
      {error && (
        <p id={`${id}-error`} className="text-xs text-red-400 mt-1">
          {error}
        </p>
      )}
    </div>
  );
}

export default function PredictionPage() {
  const [form, setForm] = useState<FormData>(DEFAULT_FORM);
  const [errors, setErrors] = useState<Partial<FormData>>({});
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);

  const setField = (k: keyof FormData) => (v: string) => {
    setForm((f) => ({ ...f, [k]: v }));
    setErrors((e) => ({ ...e, [k]: undefined }));
  };

  const validate = (): boolean => {
    const errs: Partial<FormData> = {};
    const checks: Array<[keyof FormData, number, number, string]> = [
      ['attendance', 0, 100, 'Attendance (0–100)'],
      ['internal_marks', 0, 100, 'Internal Marks (0–100)'],
      ['assignment_score', 0, 100, 'Assignment Score (0–100)'],
      ['previous_gpa', 0, 10, 'Previous GPA (0–10)'],
      ['study_hours', 0, 24, 'Study Hours (0–24)'],
      ['backlogs', 0, 100, 'Backlogs (0 or more)'],
      ['class_participation', 0, 100, 'Class Participation (0–100)'],
    ];
    for (const [key, min, max, label] of checks) {
      const v = form[key];
      if (v === '' || v === undefined) {
        errs[key] = `${label} is required` as any;
      } else {
        const n = parseFloat(v);
        if (isNaN(n) || n < min || n > max) {
          errs[key] = `Must be between ${min} and ${max}` as any;
        }
      }
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!validate()) return;
    setLoading(true);
    setApiError(null);
    setResult(null);
    try {
      const req: PredictionRequest = {
        attendance: parseFloat(form.attendance),
        internal_marks: parseFloat(form.internal_marks),
        assignment_score: parseFloat(form.assignment_score),
        previous_gpa: parseFloat(form.previous_gpa),
        study_hours: parseFloat(form.study_hours),
        backlogs: parseInt(form.backlogs, 10),
        class_participation: parseFloat(form.class_participation),
      };
      const res = await predict(req);
      setResult(res);
    } catch (err: any) {
      const detail = err?.response?.data?.detail;
      setApiError(
        typeof detail === 'string'
          ? detail
          : 'Prediction failed. Ensure the backend is running and models are trained.'
      );
    } finally {
      setLoading(false);
    }
  };

  const riskCfg = result ? RISK_CONFIG[result.risk_level] : null;

  return (
    <div className="p-8 space-y-8 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-white">Predict Student Risk</h1>
        <p className="text-slate-400 mt-1 text-sm">
          Enter student academic data to classify performance risk level
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6" noValidate>
        {/* Academic Information */}
        <div className="card space-y-5">
          <h2 className="text-base font-semibold text-white border-b border-slate-700 pb-3">
            Academic Information
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <FormField id="attendance" label="Attendance Percentage" value={form.attendance}
              onChange={setField('attendance')} min={0} max={100} placeholder="e.g. 85" error={errors.attendance} />
            <FormField id="internal_marks" label="Internal Examination Marks" value={form.internal_marks}
              onChange={setField('internal_marks')} min={0} max={100} placeholder="e.g. 72" error={errors.internal_marks} />
            <FormField id="assignment_score" label="Assignment Score" value={form.assignment_score}
              onChange={setField('assignment_score')} min={0} max={100} placeholder="e.g. 80" error={errors.assignment_score} />
            <FormField id="previous_gpa" label="Previous Semester GPA" value={form.previous_gpa}
              onChange={setField('previous_gpa')} min={0} max={10} step="0.01" placeholder="e.g. 7.5" error={errors.previous_gpa} />
            <FormField id="backlogs" label="Number of Backlogs" value={form.backlogs}
              onChange={setField('backlogs')} min={0} max={20} step="1" placeholder="e.g. 0" error={errors.backlogs} />
          </div>
        </div>

        {/* Study Information */}
        <div className="card space-y-5">
          <h2 className="text-base font-semibold text-white border-b border-slate-700 pb-3">
            Study Information
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <FormField id="study_hours" label="Daily Study Hours" value={form.study_hours}
              onChange={setField('study_hours')} min={0} max={24} placeholder="e.g. 4" error={errors.study_hours} />
            <FormField id="class_participation" label="Class Participation Percentage" value={form.class_participation}
              onChange={setField('class_participation')} min={0} max={100} placeholder="e.g. 65" error={errors.class_participation} />
          </div>
        </div>

        {apiError && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <p className="text-red-300 text-sm">{apiError}</p>
          </div>
        )}

        <button type="submit" disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Predicting…
            </>
          ) : (
            <>
              <BrainCircuit className="w-4 h-4" />
              PREDICT RISK
            </>
          )}
        </button>
      </form>

      {/* Result Card */}
      {result && riskCfg && (
        <div className={`card border-2 ${riskCfg.bg} space-y-4`}>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-bold text-white">Performance Risk Result</h2>
            <RiskBadge level={result.risk_level} size="lg" />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="bg-slate-900/60 rounded-xl p-4">
              <p className="text-slate-400 text-xs font-medium mb-1">Confidence</p>
              <p className="text-2xl font-bold text-white">
                {result.confidence != null ? `${(result.confidence * 100).toFixed(1)}%` : 'N/A'}
              </p>
            </div>
            <div className="bg-slate-900/60 rounded-xl p-4">
              <p className="text-slate-400 text-xs font-medium mb-1">Model Used</p>
              <p className="text-sm font-semibold text-white">{result.model}</p>
            </div>
            <div className="bg-slate-900/60 rounded-xl p-4">
              <p className="text-slate-400 text-xs font-medium mb-1">Algorithm Type</p>
              <p className="text-sm font-semibold text-indigo-300">Classification</p>
            </div>
          </div>

          <div className="bg-slate-900/60 rounded-xl p-4">
            <p className="text-slate-400 text-xs font-medium mb-2">⚠ Recommendation</p>
            <p className="text-slate-200 text-sm leading-relaxed">{result.recommendation}</p>
            <p className="text-slate-500 text-xs mt-2 italic">
              Note: Recommendations are rule-based and separate from the ML prediction.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
