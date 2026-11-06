import { Pause, Play, ShieldExclamationPoint, Square } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { describeFailure, predict } from "../api/client";
import type { DemoBundle, PredictionResponse } from "../api/types";
import demoBundle from "../data/demoTransactions.json";
import { formatLatency, formatProbability } from "../lib/format";
import { Chip, DecisionBadge } from "./Badges";
import { Panel } from "./Panel";

/**
 * Compile-time contract check: the bundled JSON must expose exactly the 30
 * `TransactionFeatures` keys. If the export script or the model's feature list
 * drifts, the build fails here rather than sending a malformed body.
 */
const demo: DemoBundle = demoBundle;

const TOTAL = demo.rows.length;
const ROW_DELAY_MS = 420;
const PAUSE_POLL_MS = 180;

type RunStatus = "idle" | "running" | "paused" | "stopped" | "done";

const STATUS_LABEL: Record<RunStatus, string> = {
  idle: "Ready",
  running: "Replaying",
  paused: "Paused",
  stopped: "Stopped",
  done: "Complete",
};

interface RecentItem {
  id: string;
  prediction: PredictionResponse;
  at: number;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

/**
 * Read the ref through a function call.
 *
 * TypeScript keeps the narrowing from an early `status !== "running"` guard on a
 * direct property read, which would wrongly make every later comparison look
 * impossible. A call result is never narrowed, so this returns the live value.
 */
function readStatus(ref: { current: RunStatus }): RunStatus {
  return ref.current;
}

interface ReplayPanelProps {
  /** True when the API cannot score (unreachable, or model not loaded). */
  disabled: boolean;
  disabledReason: string | null;
  /** Fired once a full pass ends, so alerts and counters can be refreshed. */
  onRunCompleted: () => void;
}

/**
 * Button-driven replay of the sanitised demo sample.
 *
 * Deliberately manual: nothing is sent until the operator presses start, so no
 * dashboard load silently writes alert rows. Rows go one at a time through
 * `POST /predict`, and the panel shows the model's real decisions — including
 * the false negatives this sample is known to contain.
 */
export function ReplayPanel({
  disabled,
  disabledReason,
  onRunCompleted,
}: ReplayPanelProps) {
  const [status, setStatus] = useState<RunStatus>("idle");
  const [completed, setCompleted] = useState(0);
  const [fraudCount, setFraudCount] = useState(0);
  const [legitCount, setLegitCount] = useState(0);
  const [errorCount, setErrorCount] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentItem[]>([]);

  const statusRef = useRef<RunStatus>("idle");
  const runToken = useRef(0);

  const transition = useCallback((next: RunStatus) => {
    statusRef.current = next;
    setStatus(next);
  }, []);

  const start = useCallback(async () => {
    if (statusRef.current === "running" || statusRef.current === "paused") return;

    const token = runToken.current + 1;
    runToken.current = token;

    transition("running");
    setCompleted(0);
    setFraudCount(0);
    setLegitCount(0);
    setErrorCount(0);
    setLastError(null);
    setRecent([]);

    let done = 0;
    let fraud = 0;
    let legitimate = 0;
    let errors = 0;

    for (const row of demo.rows) {
      while (readStatus(statusRef) === "paused") {
        if (runToken.current !== token) return;
        await sleep(PAUSE_POLL_MS);
      }
      if (runToken.current !== token || readStatus(statusRef) !== "running") return;

      try {
        const prediction = await predict(row.features);
        if (runToken.current !== token) return;
        done += 1;
        if (prediction.decision === "fraud") {
          fraud += 1;
        } else {
          legitimate += 1;
        }
        setRecent((previous) =>
          [{ id: row.id, prediction, at: Date.now() }, ...previous].slice(0, 5),
        );
      } catch (error) {
        if (runToken.current !== token) return;
        errors += 1;
        setLastError(describeFailure(error));
      }

      setCompleted(done + errors);
      setFraudCount(fraud);
      setLegitCount(legitimate);
      setErrorCount(errors);

      await sleep(ROW_DELAY_MS);
    }

    if (runToken.current !== token) return;

    // Stay paused if the operator paused during the final row's delay: the
    // summary must not claim completion while the run is held.
    if (readStatus(statusRef) === "running") {
      transition("done");
      onRunCompleted();
    }
  }, [onRunCompleted, transition]);

  const pause = useCallback(() => transition("paused"), [transition]);

  const resume = useCallback(() => {
    if (completed >= TOTAL) {
      transition("done");
      onRunCompleted();
      return;
    }
    transition("running");
  }, [completed, onRunCompleted, transition]);

  const stop = useCallback(() => {
    // Invalidates the running loop; the in-flight request may still finish
    // server-side, which is why the counts stop at what the UI has seen.
    runToken.current += 1;
    transition("stopped");
  }, [transition]);

  const active = status === "running" || status === "paused";
  const progress = TOTAL > 0 ? (completed / TOTAL) * 100 : 0;
  const primaryLabel = status === "done" || status === "stopped" ? "Run again" : "Run replay";

  return (
    <Panel
      title="Transaction replay"
      icon={Play}
      tools={<Chip tone="warn" title="Sample rows sent to POST /predict on request">Demo</Chip>}
    >
      <p className="replay-intro">
        Sends {TOTAL} sanitised sample transactions to <code>POST /predict</code>, one at a time.
        Only the 30 model features travel in the body — the dataset&rsquo;s <code>Class</code> label is
        stripped at export time and never reaches the browser or the API, so every verdict below is
        the model&rsquo;s own.
      </p>

      <div className="replay-progress">
        <div className="replay-status-row">
          <span className={`run-state run-${status}`}>{STATUS_LABEL[status]}</span>
          <span className="replay-count">
            {completed} / {TOTAL}
          </span>
        </div>
        <div
          className="progress"
          role="progressbar"
          aria-label="Replay progress"
          aria-valuemin={0}
          aria-valuemax={TOTAL}
          aria-valuenow={completed}
        >
          <span className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>

      <div className="replay-stats">
        <div className="replay-stat">
          <span className="replay-stat-value tone-fraud">{fraudCount}</span>
          <span className="replay-stat-label">Fraud detected</span>
        </div>
        <div className="replay-stat">
          <span className="replay-stat-value">{legitCount}</span>
          <span className="replay-stat-label">Legitimate</span>
        </div>
        <div className="replay-stat">
          <span className={`replay-stat-value${errorCount > 0 ? " tone-warn" : ""}`}>{errorCount}</span>
          <span className="replay-stat-label">Errors</span>
        </div>
      </div>

      <div className="replay-controls">
        {status === "running" ? (
          <button type="button" className="btn btn-ghost" onClick={pause}>
            <Pause size={13} strokeWidth={2.2} aria-hidden="true" />
            Pause
          </button>
        ) : status === "paused" ? (
          <button type="button" className="btn btn-primary" onClick={resume}>
            <Play size={13} strokeWidth={2.2} aria-hidden="true" />
            Resume
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => void start()}
            disabled={disabled}
            title={disabled ? (disabledReason ?? undefined) : `Send ${TOTAL} demo transactions`}
          >
            <Play size={13} strokeWidth={2.2} aria-hidden="true" />
            {primaryLabel}
          </button>
        )}

        {active ? (
          <button type="button" className="btn btn-ghost" onClick={stop}>
            <Square size={12} strokeWidth={2.4} aria-hidden="true" />
            Stop
          </button>
        ) : null}

        <span className="replay-hint">
          {disabled && disabledReason
            ? disabledReason
            : `${TOTAL} rows · ${demo.featureCount} features each · ~${ROW_DELAY_MS} ms apart`}
        </span>
      </div>

      {recent.length > 0 ? (
        <div className="recent">
          <p className="sub-label">Latest decisions</p>
          <ul className="recent-list">
            {recent.map((item) => (
              <li key={`${item.id}-${item.at}`} className="recent-row">
                <span className="recent-id">{item.id}</span>
                <DecisionBadge decision={item.prediction.decision} />
                <span className="recent-prob">{formatProbability(item.prediction.fraud_probability)}</span>
                <span className="recent-lat">{formatLatency(item.prediction.latency_ms)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {lastError ? (
        <p className="replay-error" role="alert">
          <ShieldExclamationPoint size={13} strokeWidth={2} aria-hidden="true" />
          {lastError}
        </p>
      ) : null}

      <p className="replay-source">
        Source <code>{demo.source}</code> · labels excluded
      </p>
    </Panel>
  );
}
