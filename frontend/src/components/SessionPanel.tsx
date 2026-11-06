import { Activity, Gauge, Timer } from "lucide-react";
import type { ReactNode } from "react";

import type { MetricsLiteResponse } from "../api/types";
import { formatCount, formatLatency, formatProbability } from "../lib/format";
import { Chip } from "./Badges";
import { Panel } from "./Panel";
import { EmptyState, InlineError, Skeleton } from "./States";

interface SessionPanelProps {
  metrics: MetricsLiteResponse | null;
  loading: boolean;
  error: string | null;
  onRetry: () => void;
}

/**
 * The one visualization in the dashboard: a split bar of this process's real
 * fraud/legitimate counts, plus the real latency range. Both come straight from
 * `/metrics-lite`; there is no time axis because the API keeps no history.
 */
export function SessionPanel({ metrics, loading, error, onRetry }: SessionPanelProps) {
  const total = metrics?.total_requests ?? 0;
  const fraud = metrics?.fraud_predictions ?? 0;
  const legitimate = metrics?.legitimate_predictions ?? 0;

  let body: ReactNode;

  if (loading && !metrics) {
    body = (
      <div className="stack">
        <Skeleton className="sk-bar" />
        <Skeleton className="sk-line" />
        <Skeleton className="sk-line short" />
      </div>
    );
  } else if (!metrics) {
    body = (
      <InlineError
        title="Metrics unavailable"
        message={error ?? "No metrics response from the API."}
        onRetry={onRetry}
      />
    );
  } else if (total === 0) {
    body = (
      <EmptyState
        icon={Activity}
        title="No scores in this process yet"
        detail="Run the demo replay, or send a transaction to POST /predict. These counters are in memory, so they start again from zero after an API restart."
      />
    );
  } else {
    const fraudShare = (fraud / total) * 100;
    const legitimateShare = (legitimate / total) * 100;

    body = (
      <div className="stack">
        <div
          className="mix"
          role="img"
          aria-label={`${formatCount(fraud)} fraud and ${formatCount(legitimate)} legitimate decisions out of ${formatCount(total)}`}
        >
          <span className="mix-fraud" style={{ width: `${fraudShare}%` }} />
          <span className="mix-legit" style={{ width: `${legitimateShare}%` }} />
        </div>

        <div className="legend">
          <div className="legend-row">
            <span className="dot dot-fraud" aria-hidden="true" />
            <span className="legend-label">Fraud</span>
            <span className="legend-count">{formatCount(fraud)}</span>
            <span className="legend-pct">{formatProbability(fraud / total)}</span>
          </div>
          <div className="legend-row">
            <span className="dot dot-legit" aria-hidden="true" />
            <span className="legend-label">Legitimate</span>
            <span className="legend-count">{formatCount(legitimate)}</span>
            <span className="legend-pct">{formatProbability(legitimate / total)}</span>
          </div>
        </div>

        <div className="rule" />

        <p className="sub-label">
          <Timer size={12} strokeWidth={1.9} aria-hidden="true" />
          Scoring latency
        </p>
        <div className="latency-grid">
          <div className="latency-cell">
            <span className="latency-value">{formatLatency(metrics.average_latency_ms)}</span>
            <span className="latency-label">average</span>
          </div>
          <div className="latency-cell">
            <span className="latency-value">{formatLatency(metrics.min_latency_ms)}</span>
            <span className="latency-label">min</span>
          </div>
          <div className="latency-cell">
            <span className="latency-value">{formatLatency(metrics.max_latency_ms)}</span>
            <span className="latency-label">max</span>
          </div>
        </div>

        <p className="note">{metrics.note}</p>
      </div>
    );
  }

  return (
    <Panel
      title="Decision mix"
      icon={Gauge}
      tools={metrics ? <Chip>{formatCount(total)} scored</Chip> : null}
    >
      {body}
    </Panel>
  );
}
