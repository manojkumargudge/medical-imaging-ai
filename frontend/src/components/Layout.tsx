import type { ReactNode } from "react";
import type { Badge } from "../status";

export const NOTICE = "For education only. Not for clinical use.";

export function Layout({ children }: { children: ReactNode }) {
  return (
    <>
      <div className="banner" role="note">
        {NOTICE}
      </div>
      <header className="header">
        <a href="#/" className="brand">
          AIML-02 viewer
        </a>
        <nav>
          <a href="#/">Worklist</a> <a href="#/upload">Upload</a> <a href="#/audit">Audit log</a>
        </nav>
      </header>
      <main className="main">{children}</main>
    </>
  );
}

export function BadgeView({ badge }: { badge: Badge }) {
  return <span className={`badge badge-${badge.tone}`}>{badge.text}</span>;
}

export function ErrorBox({ title, lines }: { title: string; lines: string[] }) {
  return (
    <div className="error" role="alert">
      <strong>{title}</strong>
      {lines.map((line) => (
        <div key={line}>{line}</div>
      ))}
    </div>
  );
}
