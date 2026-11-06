import { ServerCrash, WifiOff } from "lucide-react";
import { useCallback } from "react";

import { API_BASE_URL } from "./api/client";
import type { HealthResponse } from "./api/types";
import type { ConnectionState } from "./components/Badges";
import { FlaggedTable } from "./components/FlaggedTable";
import { Header } from "./components/Header";
import { MetricTiles } from "./components/MetricTiles";
import { ReplayPanel } from "./components/ReplayPanel";
import { SessionPanel } from "./components/SessionPanel";
import { PAGE_SIZES, useDashboard } from "./hooks/useDashboard";

function connectionState(health: HealthResponse | null, error: string | null): ConnectionState {
  if (health?.status === "degraded") return "degraded";
  if (health) return "online";
  if (error) return "offline";
  return "checking";
}

/** Why the API cannot score right now, phrased from `/health`'s own fields. */
function degradations(health: HealthResponse): string[] {
  const reasons: string[] = [];
  if (!health.model_loaded) {
    reasons.push(
      health.model_artifact_available
        ? "the model artifact exists but failed to load, so POST /predict answers 503"
        : "the model artifact is missing, so POST /predict answers 503",
    );
  }
  if (!health.database_configured) {
    reasons.push(
      "no DATABASE_URL is configured, so fraud alerts cannot be stored and GET /flagged answers 503",
    );
  }
  return reasons;
}

export default function App() {
  const dashboard = useDashboard();
  const { health, metrics, flagged } = dashboard;

  const connection = connectionState(health.data, health.error);
  const canScore = health.data?.status === "ok";

  const handleRunCompleted = useCallback(() => {
    metrics.reload();
    flagged.reload();
    health.reload();
  }, [metrics.reload, flagged.reload, health.reload]);

  const healthFailure = health.error !== null;
  const reasons = health.data ? degradations(health.data) : [];
  const showDegraded = health.data?.status === "degraded" && reasons.length > 0;

  return (
    <>
      <Header
        connection={connection}
        health={health.data}
        updatedAt={health.updatedAt}
        refreshing={health.refreshing || metrics.refreshing || flagged.refreshing}
        onRefresh={dashboard.refreshAll}
      />

      <main className="shell">
        {connection === "offline" ? (
          <div className="banner banner-error" role="alert">
            <WifiOff size={15} strokeWidth={1.9} aria-hidden="true" />
            <div className="banner-text">
              <strong>API unreachable</strong>
              <p>
                Nothing below is live. Start the backend (<code>uvicorn app.main:app --reload</code>
                ) and retry, or point <code>VITE_API_BASE_URL</code> at a different host. Current
                base URL: <code>{API_BASE_URL}</code>.
              </p>
            </div>
            <button type="button" className="btn btn-primary btn-sm" onClick={dashboard.refreshAll}>
              Retry
            </button>
          </div>
        ) : null}

        {showDegraded && health.data ? (
          <div className="banner banner-warn" role="alert">
            <ServerCrash size={15} strokeWidth={1.9} aria-hidden="true" />
            <div className="banner-text">
              <strong>Degraded service</strong>
              <p>The API is responding but cannot fully serve requests because {reasons.join("; and ")}.</p>
            </div>
          </div>
        ) : null}

        {healthFailure && connection !== "offline" ? (
          <div className="banner banner-warn" role="status">
            <ServerCrash size={15} strokeWidth={1.9} aria-hidden="true" />
            <div className="banner-text">
              <strong>Health check unavailable</strong>
              <p>{health.error}</p>
            </div>
          </div>
        ) : null}

        <MetricTiles
          metrics={metrics.data}
          health={health.data}
          threshold={dashboard.threshold}
          loading={metrics.loading}
        />

        <div className="workspace">
          <ReplayPanel
            disabled={!canScore}
            disabledReason={
              canScore ? null : "Waiting for a healthy API: the model must be loaded to score"
            }
            onRunCompleted={handleRunCompleted}
          />

          <SessionPanel
            metrics={metrics.data}
            loading={metrics.loading}
            error={metrics.error}
            onRetry={metrics.reload}
          />
        </div>

        <FlaggedTable
          flagged={flagged}
          limit={dashboard.limit}
          offset={dashboard.offset}
          limitOptions={PAGE_SIZES}
          onLimitChange={dashboard.setLimit}
          onOffsetChange={dashboard.setOffset}
        />

        <footer className="foot">
          <span>
            FraudGuard · Phase 6 dashboard · decisions come from the deployed model artifact, never
            from dataset labels
          </span>
          <span className="foot-meta">
            API <code>{API_BASE_URL}</code> · counters in-memory per process
          </span>
        </footer>
      </main>
    </>
  );
}
