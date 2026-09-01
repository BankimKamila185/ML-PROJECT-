// API types for the Student Performance Risk Prediction system

export interface PredictionRequest {
  attendance: number;
  internal_marks: number;
  assignment_score: number;
  previous_gpa: number;
  study_hours: number;
  backlogs: number;
  class_participation: number;
}

export interface PredictionResponse {
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  confidence: number | null;
  model: string;
  recommendation: string;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: number[][];
  class_labels: string[];
}

export interface ModelsData {
  best_model: string;
  models: Record<string, ModelMetrics>;
}

export interface HistoryRecord {
  id: number;
  student_id: string;
  attendance: number;
  internal_marks: number;
  assignment_score: number;
  previous_gpa: number;
  study_hours: number;
  backlogs: number;
  class_participation: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  confidence: number | null;
  model_used: string;
  created_at: string;
}

export interface HistoryResponse {
  total: number;
  records: HistoryRecord[];
}

export interface DashboardData {
  total: number;
  low: number;
  medium: number;
  high: number;
  best_model?: string;
  best_model_accuracy?: number;
  best_model_f1?: number;
}

export interface HealthResponse {
  status: string;
  trained: boolean;
}
