import axios from 'axios';
import type {
  PredictionRequest,
  PredictionResponse,
  ModelsData,
  HistoryResponse,
  DashboardData,
  HealthResponse,
} from '../types';

const BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Health ─────────────────────────────────────────────────────────────────
export const checkHealth = async (): Promise<HealthResponse> => {
  const { data } = await api.get('/api/health');
  return data;
};

// ── Predict ────────────────────────────────────────────────────────────────
export const predict = async (req: PredictionRequest): Promise<PredictionResponse> => {
  const { data } = await api.post('/api/predict', req);
  return data;
};

// ── Model metrics ──────────────────────────────────────────────────────────
export const getModels = async (): Promise<ModelsData> => {
  const { data } = await api.get('/api/models');
  return data;
};

// ── History ────────────────────────────────────────────────────────────────
export const getHistory = async (params: {
  skip?: number;
  limit?: number;
  search?: string;
  risk_filter?: string;
  sort_by?: string;
  sort_order?: string;
}): Promise<HistoryResponse> => {
  const { data } = await api.get('/api/history', { params });
  return data;
};

// ── Dashboard ──────────────────────────────────────────────────────────────
export const getDashboard = async (): Promise<DashboardData> => {
  const { data } = await api.get('/api/dashboard');
  return data;
};
