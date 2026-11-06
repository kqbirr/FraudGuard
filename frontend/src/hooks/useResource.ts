import { useCallback, useEffect, useRef, useState } from "react";

import { describeFailure } from "../api/client";

export interface ResourceState<T> {
  data: T | null;
  error: string | null;
  /** True only before the first successful load, so the UI can show a skeleton. */
  loading: boolean;
  /** True while a new value is in flight but older data is already on screen. */
  refreshing: boolean;
  updatedAt: number | null;
}

export interface Resource<T> extends ResourceState<T> {
  reload: () => void;
}

const INITIAL_STATE = {
  data: null,
  error: null,
  loading: true,
  refreshing: false,
  updatedAt: null,
} as const;

interface Options {
  /** Poll interval in milliseconds. Omitted means fetch once per dependency change. */
  intervalMs?: number;
  /** Values that should trigger a refetch (page number, page size, ...). */
  deps?: readonly unknown[];
}

/**
 * One GET endpoint's state: data, error, loading flags, and optional polling.
 *
 * Kept generic and separate from `useEffect` calls in components so every
 * request behaves the same way — stale data is kept on screen when a refresh
 * fails, an error never blanks a table that already has rows, and out-of-order
 * responses can't overwrite a newer one.
 */
export function useResource<T>(fetcher: () => Promise<T>, options: Options = {}): Resource<T> {
  const { intervalMs, deps = [] } = options;

  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  const [state, setState] = useState<ResourceState<T>>(INITIAL_STATE);

  const latestRequest = useRef(0);
  const mounted = useRef(true);

  const load = useCallback(async () => {
    const ticket = latestRequest.current + 1;
    latestRequest.current = ticket;

    setState((previous) =>
      previous.data !== null
        ? { ...previous, refreshing: true, error: null }
        : { ...previous, loading: true, error: null },
    );

    try {
      const data = await fetcherRef.current();
      if (!mounted.current || ticket !== latestRequest.current) return;
      setState({ data, error: null, loading: false, refreshing: false, updatedAt: Date.now() });
    } catch (error) {
      if (!mounted.current || ticket !== latestRequest.current) return;
      setState((previous) => ({
        data: previous.data,
        error: describeFailure(error),
        loading: false,
        refreshing: false,
        updatedAt: previous.updatedAt,
      }));
    }
  }, []);

  useEffect(() => {
    mounted.current = true;
    void load();

    if (intervalMs === undefined) {
      return () => {
        mounted.current = false;
      };
    }

    const timer = window.setInterval(() => void load(), intervalMs);
    return () => {
      mounted.current = false;
      window.clearInterval(timer);
    };
    // `deps` is intentionally spread: it is the caller-declared refetch trigger.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intervalMs, load, ...deps]);

  const reload = useCallback(() => {
    void load();
  }, [load]);

  return { ...state, reload };
}
