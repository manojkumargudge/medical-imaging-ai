import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, type Decision, getJson, newToken, postJson, type Review } from "../api";
import { formatLocal } from "../time";

export const NAMES_NOTE = "Names are not verified (education demo).";

export function isValid(decision: Decision | null, reviewer: string, comment: string): boolean {
  if (decision === null || reviewer.trim() === "") return false;
  return decision === "approve" || comment.trim() !== "";
}

/** Approve or reject the result the reviewer sees. There is no <form>: Enter never submits,
 * only a click on Save does. Neither choice is selected at first. A new submit token is made
 * each time the panel opens (and after each save), so a repeated request is saved once. */
export function ReviewPanel({ studyId, resultId }: { studyId: string; resultId: number | null }) {
  const [token, setToken] = useState(newToken);
  const [decision, setDecision] = useState<Decision | null>(null);
  const [reviewer, setReviewer] = useState("");
  const [comment, setComment] = useState("");
  const [saving, setSaving] = useState(false);
  const busy = useRef(false); // blocks a second click before React re-renders
  const [message, setMessage] = useState<{ ok: boolean; text: string } | null>(null);
  const [history, setHistory] = useState<Review[] | null>(null);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const path = `/api/studies/${encodeURIComponent(studyId)}/reviews`;

  const loadHistory = useCallback(async () => {
    try {
      setHistory((await getJson<{ reviews: Review[] }>(path)).reviews);
      setHistoryError(null);
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : String(error));
    }
  }, [path]);

  useEffect(() => {
    void loadHistory();
  }, [loadHistory]);

  const valid = isValid(decision, reviewer, comment);

  async function save() {
    if (!valid || busy.current || resultId === null || decision === null) return;
    busy.current = true;
    setSaving(true);
    setMessage(null);
    try {
      const { review } = await postJson<{ review: Review }>(path, {
        decision,
        reviewer: reviewer.trim(),
        comment: comment.trim() || null,
        submit_token: token,
        inference_result_id: resultId,
      });
      const verb = review.decision === "approve" ? "Approved" : "Rejected";
      setMessage({ ok: true, text: `Saved: ${verb} by ${review.reviewer}.` });
      setDecision(null);
      setComment("");
      setToken(newToken());
      await loadHistory();
    } catch (error) {
      const status = error instanceof ApiError ? ` (${error.status})` : "";
      const text = error instanceof Error ? error.message : String(error);
      setMessage({ ok: false, text: `Not saved${status}: ${text}` });
    } finally {
      busy.current = false;
      setSaving(false);
    }
  }

  const group = `decision-${studyId}`;
  return (
    <section className="panel review" aria-label="Review">
      <h2>Review</h2>
      <p className="muted">{NAMES_NOTE}</p>
      {resultId === null ? (
        <p>No result to review yet.</p>
      ) : (
        <div className="review-fields">
          <fieldset>
            <legend>Decision for result {resultId}</legend>
            <label>
              <input
                type="radio"
                name={group}
                value="approve"
                checked={decision === "approve"}
                onChange={() => setDecision("approve")}
              />{" "}
              Approve
            </label>
            <label>
              <input
                type="radio"
                name={group}
                value="reject"
                checked={decision === "reject"}
                onChange={() => setDecision("reject")}
              />{" "}
              Reject
            </label>
          </fieldset>
          <label className="field">
            Your name
            <input
              type="text"
              value={reviewer}
              maxLength={100}
              autoComplete="off"
              onChange={(e) => setReviewer(e.target.value)}
            />
          </label>
          <label className="field">
            Comment {decision === "reject" ? "(required to reject)" : "(optional)"}
            <textarea
              value={comment}
              maxLength={2000}
              rows={3}
              onChange={(e) => setComment(e.target.value)}
            />
          </label>
          <button type="button" onClick={() => void save()} disabled={!valid || saving}>
            {saving ? "Saving…" : "Save review"}
          </button>
          {message && (
            <p className={message.ok ? "saved" : "error"} role={message.ok ? "status" : "alert"}>
              {message.text}
            </p>
          )}
        </div>
      )}
      <h3>History</h3>
      {historyError && <p className="error">Could not load the history: {historyError}</p>}
      {history && history.length === 0 && <p className="muted">No reviews yet.</p>}
      {history && history.length > 0 && (
        <table className="table">
          <thead>
            <tr>
              <th>When</th>
              <th>Decision</th>
              <th>Reviewer (not verified)</th>
              <th>Comment</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {history.map((r, i) => (
              <tr key={r.id}>
                <td>{formatLocal(r.created_at)}</td>
                <td>
                  {r.decision === "approve" ? "Approved" : "Rejected"}
                  {i === 0 ? " (current)" : ""}
                </td>
                <td>{r.reviewer}</td>
                <td>{r.comment ?? ""}</td>
                <td>{r.inference_result_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}
