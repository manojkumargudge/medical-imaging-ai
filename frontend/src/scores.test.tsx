import { render, screen, within } from "@testing-library/react";
import type { ThresholdEntry } from "./api";
import { ScoresTable } from "./components/ScoresTable";
import { defaultLabel, scoreRows } from "./scores";

const THRESHOLDS: Record<string, ThresholdEntry> = {
  Cardiomegaly: { set: "sensitivity_80", threshold: 0.0248, low_reliability: false },
  Effusion: { set: "sensitivity_80", threshold: 0.0805, low_reliability: false },
  Hernia: { set: "score_only", threshold: null, low_reliability: false },
  Nodule: { set: "best_f1", threshold: 0.179, low_reliability: true },
};
const SCORES = {
  Cardiomegaly: { score: 0.734 },
  Effusion: { score: 0.05 },
  Hernia: { score: 0.9 },
  Nodule: { score: 0.179 },
};
const NOTE = "Scores are not calibrated probabilities: a threshold is a cut-off on the score.";

test("rows: above/below by threshold, Hernia score only, flags from the config", () => {
  const rows = scoreRows(SCORES, THRESHOLDS);
  expect(rows.map((r) => [r.label, r.verdict, r.lowReliability])).toEqual([
    ["Cardiomegaly", "above", false],
    ["Effusion", "below", false],
    ["Hernia", "score_only", false],
    ["Nodule", "above", true], // equal to the threshold counts as above, as in the worker
  ]);
  // A high Hernia score is not above any threshold, so Hernia is not the default label.
  expect(defaultLabel(rows)).toBe("Cardiomegaly");
  expect(scoreRows({}, THRESHOLDS).every((r) => r.verdict === "missing")).toBe(true);
});

test("the table shows text for every result, the reliability badge and the note", () => {
  render(<ScoresTable rows={scoreRows(SCORES, THRESHOLDS)} note={NOTE} />);
  const row = (label: string) => within(screen.getByText(label).closest("tr")!);
  expect(row("Cardiomegaly").getByText("Above threshold")).toBeTruthy();
  expect(row("Effusion").getByText("Below threshold")).toBeTruthy();
  expect(row("Hernia").getByText("Score only (no threshold)")).toBeTruthy();
  expect(row("Hernia").getByText("none")).toBeTruthy();
  expect(row("Hernia").getByText("0.900")).toBeTruthy();
  expect(row("Nodule").getByText("Low reliability")).toBeTruthy();
  expect(screen.getAllByText("Low reliability")).toHaveLength(1);
  expect(screen.getByText(NOTE)).toBeTruthy();
});
