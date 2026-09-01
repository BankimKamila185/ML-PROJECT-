import { AlertTriangle, CheckCircle, MinusCircle } from 'lucide-react';

interface Props {
  level: 'LOW' | 'MEDIUM' | 'HIGH';
  size?: 'sm' | 'md' | 'lg';
}

const config = {
  LOW: {
    className: 'risk-badge-low',
    icon: CheckCircle,
    label: 'LOW RISK',
  },
  MEDIUM: {
    className: 'risk-badge-medium',
    icon: MinusCircle,
    label: 'MEDIUM RISK',
  },
  HIGH: {
    className: 'risk-badge-high',
    icon: AlertTriangle,
    label: 'HIGH RISK',
  },
};

export default function RiskBadge({ level, size = 'md' }: Props) {
  const { className, icon: Icon, label } = config[level];
  const iconSize = size === 'lg' ? 'w-5 h-5' : 'w-3.5 h-3.5';
  const textSize = size === 'lg' ? 'text-sm' : size === 'sm' ? 'text-[10px]' : 'text-xs';
  const padding = size === 'lg' ? 'px-4 py-2' : 'px-3 py-1';

  return (
    <span className={`${className} ${textSize} ${padding}`}>
      <Icon className={iconSize} aria-hidden="true" />
      {label}
    </span>
  );
}
