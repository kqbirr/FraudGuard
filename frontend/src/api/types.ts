/**
 * Mirrors of the FastAPI response models in `app/schemas.py`.
 *
 * These are written by hand rather than generated so that a backend contract
 * change shows up as a compile error here, and so field names are copied
 * exactly: snake_case, no client-side prettifying. Nothing in this file is
 * guessed — each interface matches one Pydantic model field for field.
 */

/**
 * The 30 model features, in the order the saved artifact expects
 * (`feature_names` in `artifacts/models/fraud_model.joblib`).
 *
 * The API rejects any other field — including `Class` — with HTTP 422, so this
 * tuple is the complete, closed set of keys a `/predict` body may contain.
 */
export const FEATURE_KEYS = [
  "Time",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
  "V7",
  "V8",
  "V9",
  "V10",
  "V11",
  "V12",
  "V13",
  "V14",
  "V15",
  "V16",
  "V17",
  "V18",
  "V19",
  "V20",
  "V21",
  "V22",
  "V23",
  "V24",
  "V25",
  "V26",
  "V27",
  "V28",
  "Amount",
] as const;

export type FeatureKey = (typeof FEATURE_KEYS)[number];

/** The exact body `POST /predict` accepts. */
export type TransactionFeatures = Record<FeatureKey, number>;

/** `GET /health` — `degraded` is served with HTTP 503. */
export interface HealthResponse {
  status: "ok" | "degraded";
  model_loaded: boolean;
  model_name: string | null;
  /**
   * Decision threshold from the loaded model artifact, or `null` when no model
   * is loaded. Never hardcoded on this side: it is rendered as reported.
   */
  threshold: number | null;
  model_artifact_available: boolean;
  model_artifact_path: string | null;
  database_configured: boolean;
}

/**
 * Which real API response supplied the threshold on screen. `/health` is the
 * primary source; a stored alert is the fallback while health is still loading.
 */
export interface ThresholdInfo {
  value: number;
  source: "health" | "alert";
}

/** `GET /metrics-lite` — process-local counters, not aggregated. */
export interface MetricsLiteResponse {
  total_requests: number;
  fraud_predictions: number;
  legitimate_predictions: number;
  average_latency_ms: number | null;
  min_latency_ms: number | null;
  max_latency_ms: number | null;
  note: string;
}

/** `POST /predict` — `"fraud"` when `fraud_probability >= threshold`. */
export interface PredictionResponse {
  fraud_probability: number;
  decision: "fraud" | "legitimate";
  threshold: number;
  latency_ms: number;
}

/** One stored alert from `GET /flagged`. Deliberately has no model features. */
export interface FlaggedTransactionRead {
  id: string;
  transaction_id: string | null;
  fraud_probability: number;
  /** The backend types this as `str`; today only `"fraud"` is ever written. */
  decision: string;
  threshold: number;
  model_name: string;
  prediction_latency_ms: number | null;
  transaction_time: number | null;
  transaction_amount: number | null;
  /** ISO 8601 timestamp, always UTC. */
  created_at: string;
}

/** `GET /flagged` — one page of alerts, newest first. */
export interface FlaggedListResponse {
  /** Total stored alerts across all pages, not just this page. */
  total: number;
  limit: number;
  offset: number;
  items: FlaggedTransactionRead[];
}

/** One sanitised demo row, as written by `scripts/export_demo_transactions.py`. */
export interface DemoRow {
  id: string;
  features: TransactionFeatures;
}

/**
 * The bundled replay set. It contains the 30 features only: the Kaggle `Class`
 * label is stripped at export time and is never present here or in a request.
 */
export interface DemoBundle {
  source: string;
  featureCount: number;
  note: string;
  rows: DemoRow[];
}
