// Types mirroring the FastAPI backend Pydantic schemas exactly

// ─── Tickers ────────────────────────────────────────────────────────────────

export interface Ticker {
  id: number;
  symbol: string;
  name: string | null;
  is_active: boolean;
  created_at: string;
}

// ─── Forecast ───────────────────────────────────────────────────────────────

export type ForecastSignal = 'long' | 'flat';

export type ForecastModel =
  | 'lstm_baseline'
  | 'cnn_transformer'
  | 'cnn_transformer_multimodal'
  | 'mamba_ssm'
  | 'prophet';

export type ForecastHorizon = 1 | 5 | 10;

export interface DailyForecastRequest {
  asset_id: string;
  horizon_days: ForecastHorizon;
  model: ForecastModel;
  as_of_date?: string | null;
}

export interface DailyForecastPoint {
  timestamp: string;
  predicted_return: number;
  predicted_price: number | null;
  signal: ForecastSignal;
  confidence: number;
}

export interface DailyForecastResponse {
  asset_id: string;
  model: ForecastModel;
  horizon_days: number;
  generated_at: string;
  predictions: DailyForecastPoint[];
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

export interface MetricsSummaryResponse {
  mae: number;
  rmse: number;
  directional_accuracy: number;
  sharpe: number;
  sortino: number;
  max_drawdown: number;
  model: string | null;
  run_id: string | null;
}

// ─── Auth ────────────────────────────────────────────────────────────────────

export interface AuthResponse {
  user_id: string | null;
  access_token: string | null;
  message: string;
}

// ─── Alpha ───────────────────────────────────────────────────────────────────

export interface AlphaResponse {
  asset_id: string;
  date: string;
  alpha_values: Record<string, number>;
  alpha_version: string;
  cached: boolean;
  generated_at: string;
  rationale: string | null;
}

// ─── UI helpers ─────────────────────────────────────────────────────────────

export type UIMode = 'retail' | 'quant';
