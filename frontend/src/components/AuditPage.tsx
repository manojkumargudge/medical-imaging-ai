import { useEffect, useState } from "react";
import { type AuditEntry, type AuditPage as Page, getJson } from "../api";
import { formatLocal } from "../time";
import { ErrorBox } from "./Layout";

export const PAGE_SIZE = 25;

const EVENT_TEXT: Record<AuditEntry["event"], string> = {
  upload: "Upload",
  inference_done: "Inference done",
  inference_failed: "Inference failed",
  review: "Review",
};

export function describe(entry: AuditEntry): string {
  const d = entry.details;
  const text = (key: string) => (d[key] === undefined || d[key] === null ? "" : String(d[key]));
  switch (entry.event) {
    case "review": {
      const verb = d.decision === "approve" ? "Approved" : "Rejected";
      const comment = text("comment");
      return `${verb} result ${text("inference_result_id")}${comment ? `: ${comment}` : ""}`;
    }
    case "inference_done":
      return `${text("model")}, ${text("seconds")} s (result ${text("inference_result_id")})`;
    case "inference_failed":
      return `${text("model")}: ${text("error")}`;
    case "upload":
      return text("filename");
    default:
      return JSON.stringify(d);
  }
}

export function AuditPage() {
  const [page, setPage] = useState(1);
  const [data, setData] = useState<Page | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setError(null);
    getJson<Page>(`/api/audit?page=${page}&page_size=${PAGE_SIZE}`)
      .then(setData)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [page]);

  const pages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;
  return (
    <section>
      <h1>Audit log</h1>
      <p className="muted">
        Every upload, inference result and review, newest first. The log is append-only. Names
        are not verified (education demo).
      </p>
      {error && <ErrorBox title="Could not load the audit log" lines={[error]} />}
      {data && (
        <>
          <table className="table">
            <thead>
              <tr>
                <th>When</th>
                <th>Event</th>
                <th>By</th>
                <th>Study UID</th>
                <th>Details</th>
              </tr>
            </thead>
            <tbody>
              {data.entries.map((e) => (
                <tr key={e.id}>
                  <td>{formatLocal(e.created_at)}</td>
                  <td>{EVENT_TEXT[e.event] ?? e.event}</td>
                  <td>{e.actor}</td>
                  <td className="uid">{e.study_uid ?? "—"}</td>
                  <td>{describe(e)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="pager">
            <button type="button" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              Newer
            </button>{" "}
            Page {data.page} of {pages} ({data.total} entries){" "}
            <button type="button" disabled={page >= pages} onClick={() => setPage(page + 1)}>
              Older
            </button>
          </p>
        </>
      )}
      {!data && !error && <p>Loading…</p>}
    </section>
  );
}
