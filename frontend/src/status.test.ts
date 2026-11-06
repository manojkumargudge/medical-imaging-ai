import type { ProcessingStatus, ReviewStatus, StudySummary } from "./api";
import { reviewBadge, shouldPoll, statusBadge, viewerKind } from "./status";

const study = (status: ProcessingStatus): StudySummary => ({
  id: status,
  study_uid: "1",
  patient_id: "p",
  description: "",
  summary: null,
  modality: "DX",
  status,
  review_status: "none",
  source: "demo",
  updated_at: null,
});

test("every status and review state has its own text, not just a colour", () => {
  const statuses: ProcessingStatus[] = ["receiving", "processing", "done", "failed", "no_model"];
  const reviews: ReviewStatus[] = ["none", "approved", "rejected"];
  const texts = statuses.map((s) => statusBadge(s).text);
  expect(new Set(texts).size).toBe(statuses.length);
  expect(texts).toEqual([
    "Receiving",
    "Processing",
    "Done",
    "Failed",
    "No model for this modality",
  ]);
  expect(reviews.map((r) => reviewBadge(r).text)).toEqual(["Not reviewed", "Approved", "Rejected"]);
  expect(statusBadge("failed").tone).toBe("error");
  expect(statusBadge("bogus" as ProcessingStatus).text).toBe("Unknown (bogus)");
});

test("poll only while a study is being received or processed", () => {
  expect(shouldPoll([study("done"), study("receiving")])).toBe(true);
  expect(shouldPoll([study("processing")])).toBe(true);
  expect(shouldPoll([study("done"), study("failed"), study("no_model")])).toBe(false);
  expect(shouldPoll([])).toBe(false);
});

test("the viewer follows the modality", () => {
  expect(viewerKind("DX")).toBe("dx");
  expect(viewerKind("CR")).toBe("dx");
  expect(viewerKind("CT")).toBe("ct");
  expect(viewerKind("CT,DX")).toBe("dx");
  expect(viewerKind("MR")).toBe("none");
  expect(viewerKind("")).toBe("none");
});
