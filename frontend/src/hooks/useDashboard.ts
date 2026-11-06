import { useCallback, useState } from "react";

import { fetchFlagged, fetchHealth, fetchMetrics } from "../api/client";
import type {
  FlaggedListResponse,
  HealthResponse,
  MetricsLiteResponse,
  ThresholdInfo,
} from "../api/types";
import { useResource, type Resource } from "./useResource";

const HEALTH_POLL_MS = 15_000;
const METRICS_POLL_MS = 10_000;
const FLAGGED_POLL_MS = 20_000;

export const PAGE_SIZES = [10, 20, 50, 100] as const;
export const DEFAULT_PAGE_SIZE = 20;

export interface Dashboard {
  health: Resource<HealthResponse>;
  metrics: Resource<MetricsLiteResponse>;
  flagged: Resource<FlaggedListResponse>;
  /** The model's real cutoff, from `/health`; falls back to a stored alert. */
  threshold: ThresholdInfo | null;
  limit: number;
  offset: number;
  setLimit: (limit: number) => void;
  setOffset: (offset: number) => void;
  refreshAll: () => void;
}

/**
 * The dashboard's read side: three polled GET endpoints plus alert pagination.
 *
 * Polling is the only push mechanism in this phase — no WebSockets. Intervals
 * are deliberately slow (10-20 s): the backend's counters are process-local, so
 * the numbers here move only when someone scores a transaction.
 */
export function useDashboard(): Dashboard {
  const health = useResource(fetchHealth, { intervalMs: HEALTH_POLL_MS });
  const metrics = useResource(fetchMetrics, { intervalMs: METRICS_POLL_MS });

  const [limit, setLimitState] = useState<number>(DEFAULT_PAGE_SIZE);
  const [offset, setOffset] = useState<number>(0);

  const flagged = useResource(() => fetchFlagged(limit, offset), {
    intervalMs: FLAGGED_POLL_MS,
    deps: [limit, offset],
  });

  // Changing the page size rewinds to the first page: keeping the old offset can
  // land past the end of the new page count and show a needlessly empty table.
  const setLimit = useCallback((nextLimit: number) => {
    setLimitState(nextLimit);
    setOffset(0);
  }, []);

  const refreshAll = useCallback(() => {
    health.reload();
    metrics.reload();
    flagged.reload();
  }, [health.reload, metrics.reload, flagged.reload]);

  // `/health` now reports the loaded artifact's threshold, so the dashboard can
  // show it before the first score. The newest stored alert carries the same
  // value and covers the moment before the health response arrives; both are
  // server-reported, so nothing here is a client-side constant.
  const healthThreshold = health.data?.threshold ?? null;
  const alertThreshold = flagged.data?.items[0]?.threshold ?? null;
  const threshold: ThresholdInfo | null =
    healthThreshold !== null
      ? { value: healthThreshold, source: "health" }
      : alertThreshold !== null
        ? { value: alertThreshold, source: "alert" }
        : null;

  return {
    health,
    metrics,
    flagged,
    threshold,
    limit,
    offset,
    setLimit,
    setOffset,
    refreshAll,
  };
}
