import { SCOPE_WARNING } from "../upload";

export function ScopeWarning() {
  return (
    <aside className="warning" aria-label="Model scope warning">
      <strong>Model scope:</strong> {SCOPE_WARNING}
    </aside>
  );
}
