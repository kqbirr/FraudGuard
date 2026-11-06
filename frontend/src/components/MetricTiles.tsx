import { Activity, Cpu, Percent, ShieldCheck, ShieldX } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

import type { HealthResponse, MetricsLiteResponse, ThresholdInfo } from "../api/types";
import { formatCount, formatProbability } from "../lib/format";
import { Skeleton } from "./States";

interface MetricTilesProps {
  metrics: MetricsLiteResponse | null;
  health: HealthResponse | null;
  threshold: ThresholdInfo | null;
  loading: boolean;
}

function Tile({
  label,
  icon: Icon,
  value,
  hint,
  tone,
  title,
}: {
  label: string;
  icon: LucideIcon;
  value: ReactNode;
  hint: ReactNode;
  tone?: "fraud" | "legit" | "warn";
  title?: string;
}) {
  return (
    <article className={tone ? `tile tile-${tone}` : "tile"} title={title}>
      <div className="tile-label">
        <Icon size={12} strokeWidth={1.9} aria-hidden="true" />
        <span>{label}</span>
      </div>
      <div className="tile-value">{value}</div>
      <div className="tile-hint">{hint}</div>
    </article>
  );
}

/**
 * Session counters from `/metrics-lite` plus the model identity from `/health`.
 *
 * The counters are process-local: they belong to the uvicorn worker that answered
 * the request, so every tile says so instead of implying fleet-wide totals. The
 * threshold is never hardcoded — it is whatever `/health` reports from the loaded
 * artifact, or the newest stored alert's value until that response arrives.
 */
export function MetricTiles({ metrics, health, threshold, loading }: MetricTilesProps) {
  const total = metrics?.total_requests ?? null;
  const fraud = metrics?.fraud_predictions ?? null;
  const legitimate = metrics?.legitimate_predictions ?? null;
  const rate = total !== null && total > 0 && fraud !== null ? fraud / total : null;

  const modelName = health?.model_name ?? null;
  const modelNotLoaded = health !== null && !health.model_loaded;

  return (
    <section className="metric-strip" aria-label="Session metrics">
      <div className="strip-head">
        <h2>Session metrics</h2>
        <p>Process-local counters from this API worker · reset when the service restarts</p>
      </div>

      <div className="kpi-grid">
        <Tile
          label="Predictions"
          icon={Activity}
          value={loading ? <Skeleton className="sk-value" /> : formatCount(total)}
          hint="total /predict calls served"
          title="total_requests from GET /metrics-lite"
        />
        <Tile
          label="Fraud"
          icon={ShieldX}
          tone="fraud"
          value={loading ? <Skeleton className="sk-value" /> : formatCount(fraud)}
          hint="decision = fraud · persisted"
          title="fraud_predictions from GET /metrics-lite"
        />
        <Tile
          label="Legitimate"
          icon={ShieldCheck}
          tone="legit"
          value={loading ? <Skeleton className="sk-value" /> : formatCount(legitimate)}
          hint="decision = legitimate · not stored"
          title="legitimate_predictions from GET /metrics-lite"
        />
        <Tile
          label="Fraud rate"
          icon={Percent}
          value={loading ? <Skeleton className="sk-value" /> : formatProbability(rate)}
          hint={rate === null ? "no requests yet" : `of ${formatCount(total)} scores`}
          title="fraud_predictions / total_requests from GET /metrics-lite"
        />
        <Tile
          label="Model"
          icon={Cpu}
          tone={modelNotLoaded ? "warn" : undefined}
          value={loading ? <Skeleton className="sk-value" /> : (modelName ?? "—")}
          hint={
            threshold
              ? `threshold ${formatProbability(threshold.value)} · alert at or above`
              : modelNotLoaded
                ? "artifact not loaded"
                : "threshold not reported"
          }
          title={
            threshold
              ? threshold.source === "health"
                ? "Threshold stored in the loaded model artifact, as reported by GET /health"
                : "Threshold reported by the newest stored alert from GET /flagged"
              : "GET /health reports no threshold because no model is loaded"
          }
        />
      </div>
    </section>
  );
}
