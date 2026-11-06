import type { LabelResult, ThresholdEntry } from "./api";

export type Verdict = "above" | "below" | "score_only" | "missing";

export interface ScoreRow {
  label: string;
  score: number | null;
  threshold: number | null;
  verdict: Verdict;
  lowReliability: boolean;
}

export const VERDICT_TEXT: Record<Verdict, string> = {
  above: "Above threshold",
  below: "Below threshold",
  score_only: "Score only (no threshold)",
  missing: "No score",
};

/** One row per label in the thresholds file (all 14), in its order. */
export function scoreRows(
  labels: Record<string, LabelResult>,
  thresholds: Record<string, ThresholdEntry>,
): ScoreRow[] {
  return Object.entries(thresholds).map(([label, entry]) => {
    const score = labels[label]?.score ?? null;
    let verdict: Verdict;
    if (score === null) verdict = "missing";
    else if (entry.set === "score_only" || entry.threshold === null) verdict = "score_only";
    else verdict = score >= entry.threshold ? "above" : "below";
    return {
      label,
      score,
      threshold: entry.set === "score_only" ? null : entry.threshold,
      verdict,
      lowReliability: entry.low_reliability,
    };
  });
}

/** The label to show first: the highest score above its threshold, else the highest score. */
export function defaultLabel(rows: readonly ScoreRow[]): string | undefined {
  const scored = rows.filter((r) => r.score !== null);
  const pool = scored.some((r) => r.verdict === "above")
    ? scored.filter((r) => r.verdict === "above")
    : scored;
  return [...pool].sort((a, b) => (b.score ?? 0) - (a.score ?? 0))[0]?.label;
}

export function formatScore(value: number | null): string {
  return value === null ? "—" : value.toFixed(3);
}
