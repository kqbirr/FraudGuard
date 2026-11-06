import { ChevronLeft, ChevronRight, Inbox, RefreshCw } from "lucide-react";
import type { ReactNode } from "react";

import type { FlaggedListResponse } from "../api/types";
import type { Resource } from "../hooks/useResource";
import {
  formatAmount,
  formatCount,
  formatDatasetTime,
  formatLatency,
  formatProbability,
  formatTimestamp,
  shortId,
} from "../lib/format";
import { Chip, DecisionBadge } from "./Badges";
import { Panel } from "./Panel";
import { EmptyState, InlineError, Skeleton } from "./States";

interface FlaggedTableProps {
  flagged: Resource<FlaggedListResponse>;
  limit: number;
  offset: number;
  limitOptions: readonly number[];
  onLimitChange: (limit: number) => void;
  onOffsetChange: (offset: number) => void;
}

function SkeletonTable() {
  return (
    <div className="table-wrap" aria-hidden="true">
      <table>
        <thead>
          <tr>
            <th scope="col">Alert</th>
            <th scope="col">Created</th>
            <th scope="col">Decision</th>
            <th scope="col">Probability</th>
            <th scope="col">Threshold</th>
            <th scope="col" className="col-model">Model</th>
            <th scope="col" className="col-latency">Latency</th>
            <th scope="col" className="col-amount">Amount</th>
          </tr>
        </thead>
        <tbody>
          {Array.from({ length: 4 }, (_, index) => (
            <tr key={index} className="row-skeleton">
              <td><Skeleton className="sk-cell short" /></td>
              <td><Skeleton className="sk-cell" /></td>
              <td><Skeleton className="sk-cell short" /></td>
              <td><Skeleton className="sk-cell short" /></td>
              <td><Skeleton className="sk-cell short" /></td>
              <td className="col-model"><Skeleton className="sk-cell short" /></td>
              <td className="col-latency"><Skeleton className="sk-cell short" /></td>
              <td className="col-amount"><Skeleton className="sk-cell short" /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * Stored fraud alerts from `GET /flagged`, newest first.
 *
 * Only the fields the API returns are shown — no model features, no transaction
 * identifiers beyond the alert's own short reference. Legitimate predictions are
 * never inserted by the backend, so an empty table is the normal first state.
 */
export function FlaggedTable({
  flagged,
  limit,
  offset,
  limitOptions,
  onLimitChange,
  onOffsetChange,
}: FlaggedTableProps) {
  const { data, error, loading, refreshing, reload } = flagged;
  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const hasNextPage = offset + items.length < total;

  let body: ReactNode;

  if (loading) {
    body = <SkeletonTable />;
  } else if (!data) {
    body = (
      <div className="panel-pad">
        <InlineError
          title="Alerts unavailable"
          message={error ?? "No response from GET /flagged."}
          onRetry={reload}
        />
      </div>
    );
  } else if (total === 0) {
    body = (
      <div className="panel-pad">
        <EmptyState
          icon={Inbox}
          title="No fraud alerts recorded yet"
          detail="Only transactions the model scores as fraud are stored. Run the demo replay above, or send one to POST /predict — the sample includes rows the model misses, so a replay may add nothing."
        />
      </div>
    );
  } else {
    body = (
      <>
        {error ? (
          <div className="panel-pad">
            <InlineError title="Latest refresh failed" message={error} onRetry={reload} />
          </div>
        ) : null}

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Alert</th>
                <th scope="col">Created</th>
                <th scope="col">Decision</th>
                <th scope="col">Probability</th>
                <th scope="col">Threshold</th>
                <th scope="col" className="col-model">Model</th>
                <th scope="col" className="col-latency">Latency</th>
                <th scope="col" className="col-amount">Amount</th>
              </tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={item.id}>
                  <td className="cell-id">
                    <code title={item.id}>{shortId(item.id)}</code>
                  </td>
                  <td className="cell-created">{formatTimestamp(item.created_at)}</td>
                  <td>
                    <DecisionBadge decision={item.decision} />
                  </td>
                  <td className="cell-prob">
                    <span className="prob-value">{formatProbability(item.fraud_probability)}</span>
                    <span className="prob-bar" aria-hidden="true">
                      <span
                        style={{
                          width: `${Math.min(100, Math.max(3, item.fraud_probability * 100))}%`,
                        }}
                      />
                    </span>
                  </td>
                  <td className="cell-num">{formatProbability(item.threshold)}</td>
                  <td className="cell-model col-model">{item.model_name}</td>
                  <td className="cell-num col-latency">{formatLatency(item.prediction_latency_ms)}</td>
                  <td
                    className="cell-num col-amount"
                    title={`dataset time ${formatDatasetTime(item.transaction_time)}`}
                  >
                    {formatAmount(item.transaction_amount)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="table-foot">
          <span className="foot-range">
            {offset + 1}–{offset + items.length} of {formatCount(total)}
          </span>
          <div className="pager">
            <label className="pager-size">
              <span>Rows</span>
              <select
                value={limit}
                onChange={(event) => onLimitChange(Number(event.target.value))}
                aria-label="Rows per page"
              >
                {limitOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="button"
              className="btn btn-icon btn-sm"
              onClick={() => onOffsetChange(Math.max(0, offset - limit))}
              disabled={offset === 0}
              aria-label="Previous page"
            >
              <ChevronLeft size={14} strokeWidth={2.2} aria-hidden="true" />
            </button>
            <button
              type="button"
              className="btn btn-icon btn-sm"
              onClick={() => onOffsetChange(offset + limit)}
              disabled={!hasNextPage}
              aria-label="Next page"
            >
              <ChevronRight size={14} strokeWidth={2.2} aria-hidden="true" />
            </button>
          </div>
        </div>
      </>
    );
  }

  return (
    <Panel
      title="Flagged transactions"
      icon={Inbox}
      className="flagged"
      bodyClassName="flush"
      tools={
        <>
          <Chip>{data ? `${formatCount(total)} alerts` : "loading"}</Chip>
          <button
            type="button"
            className="btn btn-ghost btn-sm"
            onClick={reload}
            disabled={refreshing}
            title="Reload the current page of alerts"
          >
            <RefreshCw
              size={13}
              strokeWidth={2}
              aria-hidden="true"
              className={refreshing ? "spin" : undefined}
            />
            Refresh
          </button>
        </>
      }
    >
      {body}
    </Panel>
  );
}
