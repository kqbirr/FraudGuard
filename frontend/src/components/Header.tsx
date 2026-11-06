import { Clock, Database, RefreshCw, ShieldExclamationPoint } from "lucide-react";

import { API_BASE_URL } from "../api/client";
import type { HealthResponse } from "../api/types";
import { formatClock } from "../lib/format";
import { ConnectionBadge, type ConnectionState } from "./Badges";

interface HeaderProps {
  connection: ConnectionState;
  health: HealthResponse | null;
  updatedAt: number | null;
  refreshing: boolean;
  onRefresh: () => void;
}

export function Header({ connection, health, updatedAt, refreshing, onRefresh }: HeaderProps) {
  const databaseConfigured = health?.database_configured ?? null;

  return (
    <header className="topbar">
      <div className="topbar-inner">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            <ShieldExclamationPoint size={17} strokeWidth={1.8} />
          </span>
          <div className="brand-text">
            <h1>FraudGuard</h1>
            <p>Fraud detection operations · card-not-present scoring</p>
          </div>
        </div>

        <div className="topbar-tools">
          <ConnectionBadge state={connection} />

          <span
            className={`pill ${databaseConfigured === false ? "pill-warn" : "pill-quiet"}`}
            title={
              databaseConfigured === null
                ? "Database status not reported yet"
                : databaseConfigured
                  ? "Health reports database_configured=true: fraud alerts are being persisted. The engine is not disclosed by the API."
                  : "No database configured: fraud alerts cannot be stored"
            }
          >
            <Database size={12} strokeWidth={1.9} aria-hidden="true" />
            <span className="pill-text">
              {databaseConfigured === null ? "DB unknown" : databaseConfigured ? "DB connected" : "No DB"}
            </span>
          </span>

          <span className="updated" title={`Last successful poll · API ${API_BASE_URL}`}>
            <Clock size={12} strokeWidth={1.9} aria-hidden="true" />
            <span className="updated-time">{updatedAt === null ? "—" : formatClock(updatedAt)}</span>
          </span>

          <button
            type="button"
            className="btn btn-icon"
            onClick={onRefresh}
            disabled={refreshing}
            aria-label="Refresh health, metrics and alerts"
            title="Refresh all panels"
          >
            <RefreshCw size={14} strokeWidth={2} aria-hidden="true" className={refreshing ? "spin" : undefined} />
          </button>
        </div>
      </div>
    </header>
  );
}
