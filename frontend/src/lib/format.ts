/**
 * Presentation helpers. Everything here formats values that came from the API;
 * nothing invents a value, and an absent field renders as an em dash rather than
 * a plausible-looking zero.
 */

const EM_DASH = "—";

const timestampFormatter = new Intl.DateTimeFormat(undefined, {
  day: "2-digit",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

const clockFormatter = new Intl.DateTimeFormat(undefined, {
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
});

const amountFormatter = new Intl.NumberFormat(undefined, {
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

/** "05 Nov, 14:03:22" — an ISO timestamp from the API, in the viewer's locale. */
export function formatTimestamp(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return EM_DASH;
  }
  return timestampFormatter.format(date);
}

/** "14:03:22" — for the "last updated" line in the header. */
export function formatClock(epochMs: number): string {
  return clockFormatter.format(new Date(epochMs));
}

/** Probabilities are always rendered with two decimals: 0.9871 -> "98.71%". */
export function formatProbability(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return EM_DASH;
  }
  return `${(value * 100).toFixed(2)}%`;
}

/** "1.83 ms", or "—" when the backend reported no latency. */
export function formatLatency(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return EM_DASH;
  }
  return `${value.toFixed(value < 10 ? 2 : 1)} ms`;
}

/** Transaction amounts are USD floats in the source data. */
export function formatAmount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return EM_DASH;
  }
  return `$${amountFormatter.format(value)}`;
}

/** The raw `Time` column is seconds since the first transaction in the dataset. */
export function formatDatasetTime(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return EM_DASH;
  }
  return `t+${Math.round(value).toLocaleString()} s`;
}

export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return EM_DASH;
  }
  return value.toLocaleString();
}

/** Alert UUIDs are long; the first segment is enough to reference one. */
export function shortId(id: string): string {
  const [first] = id.split("-");
  return first ? first.slice(0, 8) : id;
}
