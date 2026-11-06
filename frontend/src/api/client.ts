/**
 * The only module that talks HTTP.
 *
 * Components never call `fetch` directly: every request goes through here so the
 * base URL, timeout, error shape and JSON handling are defined once. Errors are
 * always thrown as `ApiError`, which carries enough structure for the UI to tell
 * "backend is down" apart from "the backend rejected this payload".
 */

import type {
  FlaggedListResponse,
  HealthResponse,
  MetricsLiteResponse,
  PredictionResponse,
  TransactionFeatures,
} from "./types";

const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

/**
 * Configured through `VITE_API_BASE_URL` (see `frontend/.env.example`). This is
 * public configuration only — Vite inlines it into the bundle, so no secret may
 * live here. The database URL and model artifact stay entirely server-side.
 */
const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const API_BASE_URL = (
  configuredBaseUrl && configuredBaseUrl.length > 0 ? configuredBaseUrl : DEFAULT_BASE_URL
).replace(/\/+$/, "");

const REQUEST_TIMEOUT_MS = 12_000;

export type ApiErrorKind = "network" | "timeout" | "http" | "malformed";

/** A failed API call, with the HTTP status and the backend's own detail text. */
export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  readonly detail: string;

  constructor(
    kind: ApiErrorKind,
    message: string,
    options: { status?: number; detail?: string } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = options.status ?? null;
    this.detail = options.detail ?? "";
  }
}

interface RequestOptions {
  method?: "GET" | "POST";
  body?: unknown;
  /**
   * Statuses whose body is still meaningful. `/health` uses this: a degraded
   * service answers 503 with a valid body, which is information, not a failure.
   */
  acceptStatuses?: readonly number[];
}

function isAbortError(error: unknown): boolean {
  return (
    (error instanceof DOMException && error.name === "AbortError") ||
    (typeof error === "object" && error !== null && (error as { name?: string }).name === "AbortError")
  );
}

/** Pull the human-readable part out of FastAPI's `{"detail": ...}` bodies. */
function extractDetail(payload: unknown): string {
  if (typeof payload !== "object" || payload === null || !("detail" in payload)) {
    return "";
  }

  const detail = (payload as { detail: unknown }).detail;

  if (typeof detail === "string") {
    return detail;
  }

  // Pydantic validation errors: a list of {loc, msg, type}.
  if (Array.isArray(detail) && detail.length > 0) {
    const first = detail[0] as { loc?: unknown; msg?: unknown };
    const location = Array.isArray(first.loc)
      ? first.loc.filter((part) => typeof part === "string" || typeof part === "number").join(".")
      : "";
    const message = typeof first.msg === "string" ? first.msg : "invalid value";
    return location ? `${location}: ${message}` : message;
  }

  return "";
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method: options.method ?? "GET",
      headers: options.body === undefined ? undefined : { "Content-Type": "application/json" },
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
      signal: controller.signal,
    });
  } catch (error) {
    if (isAbortError(error)) {
      throw new ApiError("timeout", `The API did not respond within ${REQUEST_TIMEOUT_MS / 1000} s.`);
    }
    throw new ApiError("network", `Cannot reach the API at ${API_BASE_URL}.`);
  } finally {
    window.clearTimeout(timer);
  }

  const raw = await response.text();
  let payload: unknown = null;

  if (raw.length > 0) {
    try {
      payload = JSON.parse(raw);
    } catch {
      throw new ApiError("malformed", "The API returned a response that was not JSON.", {
        status: response.status,
      });
    }
  }

  const accepted = options.acceptStatuses?.includes(response.status) ?? false;

  if (!response.ok && !accepted) {
    const detail = extractDetail(payload);
    throw new ApiError("http", detail || `Request failed with HTTP ${response.status}.`, {
      status: response.status,
      detail,
    });
  }

  if (payload === null) {
    throw new ApiError("malformed", "The API returned an empty response body.", {
      status: response.status,
    });
  }

  return payload as T;
}

/** Health of the API and model. A 503 (`degraded`) is returned, not thrown. */
export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { acceptStatuses: [503] });
}

/** In-memory, per-process counters. They reset when the backend restarts. */
export function fetchMetrics(): Promise<MetricsLiteResponse> {
  return request<MetricsLiteResponse>("/metrics-lite");
}

/** One page of stored fraud alerts, newest first. */
export function fetchFlagged(limit: number, offset: number): Promise<FlaggedListResponse> {
  return request<FlaggedListResponse>(`/flagged?limit=${limit}&offset=${offset}`);
}

/** Score exactly one transaction. Fraud results are stored server-side. */
export function predict(features: TransactionFeatures): Promise<PredictionResponse> {
  return request<PredictionResponse>("/predict", { method: "POST", body: features });
}

/** Turn any thrown value into a sentence worth showing an operator. */
export function describeFailure(error: unknown): string {
  if (error instanceof ApiError) {
    switch (error.kind) {
      case "network":
        return `Cannot reach the FraudGuard API at ${API_BASE_URL}. Is uvicorn running?`;
      case "timeout":
        return error.message;
      case "http":
        if (error.status === 503) {
          return `Service unavailable (503): ${error.message}`;
        }
        if (error.status === 422) {
          return `Rejected by the API (422): ${error.message}`;
        }
        return `Request failed (HTTP ${error.status ?? "?"}): ${error.message}`;
      case "malformed":
        return error.message;
    }
  }

  return error instanceof Error ? error.message : "Unexpected error.";
}
