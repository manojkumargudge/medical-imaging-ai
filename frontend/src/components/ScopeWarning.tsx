import { SCOPE_WARNING } from "../upload";

/** Shown for studies that are not one of the demo studies (uploads, or unknown origin). */
export function ScopeWarning({ text = SCOPE_WARNING }: { text?: string }) {
  return (
    <aside className="warning" aria-label="Model scope warning">
      <strong>Model scope:</strong> {text}
    </aside>
  );
}
