import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  BrainCircuit,
  BarChart3,
  Grid3x3,
  History,
  GraduationCap,
} from 'lucide-react';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/predict', icon: BrainCircuit, label: 'Predict Risk' },
  { to: '/comparison', icon: BarChart3, label: 'Model Comparison' },
  { to: '/confusion', icon: Grid3x3, label: 'Confusion Matrix' },
  { to: '/history', icon: History, label: 'Prediction History' },
];

export default function Sidebar() {
  return (
    <aside className="w-64 min-h-screen bg-slate-900/80 border-r border-slate-700/50 flex flex-col shrink-0">
      {/* Logo */}
      <div className="p-6 border-b border-slate-700/50">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-indigo-600 flex items-center justify-center shrink-0">
            <GraduationCap className="w-5 h-5 text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-white leading-tight">StudentRisk</p>
            <p className="text-xs text-slate-500">ML Classification</p>
          </div>
        </div>
      </div>

      {/* Module badge */}
      <div className="px-4 py-3">
        <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-lg px-3 py-2">
          <p className="text-[10px] text-indigo-400 font-semibold uppercase tracking-wider">Module V</p>
          <p className="text-xs text-slate-300 mt-0.5">Supervised Learning</p>
          <p className="text-xs text-slate-400">Classification</p>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-2 space-y-1">
        {navItems.map(({ to, icon: Icon, label }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all duration-150 ${
                isActive
                  ? 'bg-indigo-600/20 text-indigo-300 border border-indigo-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
              }`
            }
          >
            <Icon className="w-4 h-4 shrink-0" />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700/50">
        <p className="text-xs text-slate-600 text-center">
          CO3 · Scikit-learn · FastAPI
        </p>
      </div>
    </aside>
  );
}
