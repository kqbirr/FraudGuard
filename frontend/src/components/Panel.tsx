import type { LucideIcon } from "lucide-react";
import type { ReactNode } from "react";

interface PanelProps {
  title: string;
  icon: LucideIcon;
  /** Right-aligned header content: chips, counts, refresh buttons. */
  tools?: ReactNode;
  children: ReactNode;
  className?: string;
  /** Applied to the body, for panels that need edge-to-edge tables. */
  bodyClassName?: string;
}

export function Panel({ title, icon: Icon, tools, children, className, bodyClassName }: PanelProps) {
  return (
    <section className={className ? `panel ${className}` : "panel"}>
      <header className="panel-head">
        <div className="panel-title">
          <Icon size={14} strokeWidth={1.9} aria-hidden="true" className="panel-icon" />
          <h2>{title}</h2>
        </div>
        {tools ? <div className="panel-tools">{tools}</div> : null}
      </header>
      <div className={bodyClassName ? `panel-body ${bodyClassName}` : "panel-body"}>{children}</div>
    </section>
  );
}
