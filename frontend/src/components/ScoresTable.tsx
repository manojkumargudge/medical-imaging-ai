import { formatScore, type ScoreRow, VERDICT_TEXT } from "../scores";

export function ScoresTable({
  rows,
  selected,
  note,
}: {
  rows: ScoreRow[];
  selected?: string;
  note: string;
}) {
  return (
    <>
      <table className="table scores">
        <thead>
          <tr>
            <th>Label</th>
            <th>Score</th>
            <th>Threshold</th>
            <th>Result</th>
            <th>Reliability</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label} className={row.label === selected ? "selected" : undefined}>
              <td>{row.label}</td>
              <td>{formatScore(row.score)}</td>
              <td>{row.verdict === "score_only" ? "none" : formatScore(row.threshold)}</td>
              <td>{VERDICT_TEXT[row.verdict]}</td>
              <td>
                {row.lowReliability ? (
                  <span className="badge badge-warn">Low reliability</span>
                ) : (
                  ""
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">{note}</p>
    </>
  );
}
