import { Activity, LoaderCircle, ShieldCheck, ShieldExclamationPoint, ShieldX, WifiOff } from "lucide-react";
import type { ReactNode } from "react";

export type ConnectionState = "checking" | "online" | "degraded" | "offline";

const CONNECTION_COPY: Record<ConnectionState, string> = {
  checking: "Checking API",
  online: "API operational",
  degraded: "API degraded",
  offline: "API offline",
};

/**
 * Health indicator driven by `GET /health`.
 *
 * A degraded service answers 503 with a real body, so `degraded` is a distinct
 * state from `offline`: the API is reachable but cannot score right now.
 */
export function ConnectionBadge({ state }: { state: ConnectionState }) {
  const Icon =
    state === "online"
      ? Activity
      : state === "degraded"
        ? ShieldExclamationPoint
        : state === "offline"
          ? WifiOff
          : LoaderCircle;

  return (
    <span className={`conn conn-${state}`} role="status" aria-live="polite">
      {state === "checking" ? (
        <Icon size={13} className="spin" aria-hidden="true" />
      ) : (
        <span className="conn-dot" aria-hidden="true" />
      )}
      <span>{CONNECTION_COPY[state]}</span>
    </span>
  );
}

/** The model's own verdict, never the dataset label. */
export function DecisionBadge({ decision }: { decision: string }) {
  const isFraud = decision === "fraud";
  const isLegit = decision === "legitimate";
  const Icon = isFraud ? ShieldX : isLegit ? ShieldCheck : ShieldExclamationPoint;

  return (
    <span className={`badge ${isFraud ? "badge-fraud" : isLegit ? "badge-legit" : "badge-neutral"}`}>
      <Icon size={11} strokeWidth={2.1} aria-hidden="true" />
      <span>{isFraud ? "Fraud" : isLegit ? "Legitimate" : decision}</span>
    </span>
  );
}

/** Small, quiet label used for provenance and "demo" markers. */
export function Chip({
  children,
  tone = "neutral",
  title,
}: {
  children: ReactNode;
  tone?: "neutral" | "fraud" | "warn";
  title?: string;
}) {
  return (
    <span className={`chip chip-${tone}`} title={title}>
      {children}
    </span>
  );
}
