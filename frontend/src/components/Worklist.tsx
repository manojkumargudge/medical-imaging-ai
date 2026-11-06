import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getHealth, getJson, healthProblems, type StudySummary } from "../api";
import { POLL_MS, reviewBadge, shouldPoll, statusBadge } from "../status";
import { formatLocal } from "../time";
import { BadgeView, ErrorBox } from "./Layout";

/** The whole row opens the study; the Open link stays for keyboard users. */
export function openStudy(id: string): void {
  window.location.hash = `#/studies/${id}`;
}

interface Problem {
  title: string;
  lines: string[];
}

async function explain(error: unknown): Promise<Problem> {
  const message = error instanceof Error ? error.message : String(error);
  try {
    const problems = healthProblems(await getHealth());
    if (problems.length) return { title: "The viewer API reports a service is down", lines: problems };
  } catch {
    return { title: "The viewer API is not reachable", lines: [message] };
  }
  const status = error instanceof ApiError ? ` (HTTP ${error.status})` : "";
  return { title: `Could not load the worklist${status}`, lines: [message] };
}

export function Worklist() {
  const [studies, setStudies] = useState<StudySummary[] | null>(null);
  const [problem, setProblem] = useState<Problem | null>(null);
  const [polling, setPolling] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const load = useCallback(async () => {
    clearTimeout(timer.current);
    try {
      const body = await getJson<{ studies: StudySummary[] }>("/api/studies");
      setStudies(body.studies);
      setProblem(null);
      const again = shouldPoll(body.studies);
      setPolling(again);
      if (again) timer.current = setTimeout(() => void load(), POLL_MS);
    } catch (error) {
      setPolling(false);
      setProblem(await explain(error));
    }
  }, []);

  useEffect(() => {
    void load();
    return () => clearTimeout(timer.current);
  }, [load]);

  return (
    <section>
      <h1>Worklist</h1>
      {problem && (
        <>
          <ErrorBox title={problem.title} lines={problem.lines} />
          <button type="button" onClick={() => void load()}>
            Try again
          </button>
        </>
      )}
      {studies && (
        <>
          <p className="muted">
            {studies.length} {studies.length === 1 ? "study" : "studies"}.{" "}
            {polling
              ? "Refreshing every 3 s while a study is being received or processed."
              : "Nothing is being processed; the list is not refreshing."}
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Patient ID</th>
                <th>Study</th>
                <th>Modality</th>
                <th>Status</th>
                <th>Summary</th>
                <th>Review</th>
                <th>Source</th>
                <th>Updated</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {studies.map((s) => (
                <tr key={s.id} className="clickable" onClick={() => openStudy(s.id)}>
                  <td>{s.patient_id}</td>
                  <td>{s.description || "—"}</td>
                  <td>{s.modality || "—"}</td>
                  <td>
                    <BadgeView badge={statusBadge(s.status)} />
                  </td>
                  <td>{s.summary ?? "—"}</td>
                  <td>
                    <BadgeView badge={reviewBadge(s.review_status)} />
                  </td>
                  <td>{s.source}</td>
                  <td>{formatLocal(s.updated_at)}</td>
                  <td>
                    <a href={`#/studies/${s.id}`}>Open</a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
      {!studies && !problem && <p>Loading…</p>}
    </section>
  );
}
