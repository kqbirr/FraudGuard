import { ServerCrash } from "lucide-react";
import type { LucideIcon } from "lucide-react";

/** A failed fetch, shown in place of the content it belongs to. */
export function InlineError({
  title = "Could not load",
  message,
  onRetry,
}: {
  title?: string;
  message: string;
  onRetry?: () => void;
}) {
  return (
    <div className="state state-error" role="alert">
      <ServerCrash size={16} strokeWidth={1.8} aria-hidden="true" />
      <div className="state-text">
        <p className="state-title">{title}</p>
        <p className="state-detail">{message}</p>
      </div>
      {onRetry ? (
        <button type="button" className="btn btn-ghost btn-sm" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

/** "Nothing here yet" — always explains what would create content. */
export function EmptyState({
  icon: Icon,
  title,
  detail,
}: {
  icon: LucideIcon;
  title: string;
  detail: string;
}) {
  return (
    <div className="state state-empty">
      <span className="state-glyph" aria-hidden="true">
        <Icon size={18} strokeWidth={1.6} />
      </span>
      <p className="state-title">{title}</p>
      <p className="state-detail">{detail}</p>
    </div>
  );
}

/** Grey placeholder lines, sized by className so each caller can fit its box. */
export function Skeleton({ className }: { className?: string }) {
  return <span className={`skeleton ${className ?? ""}`} aria-hidden="true" />;
}
