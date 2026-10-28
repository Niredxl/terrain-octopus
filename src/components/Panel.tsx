import type { PropsWithChildren, ReactNode } from "react";

export function Panel({
  title,
  eyebrow,
  action,
  className = "",
  children,
}: PropsWithChildren<{ title: string; eyebrow?: string; action?: ReactNode; className?: string }>) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-header">
        <div>
          {eyebrow && <span className="eyebrow">{eyebrow}</span>}
          <h2>{title}</h2>
        </div>
        {action}
      </header>
      {children}
    </section>
  );
}
