import type { ProcessingStatus, ReviewStatus, StudySummary } from "./api";

export type Tone = "info" | "ok" | "warn" | "error" | "muted";

export interface Badge {
  text: string; // always shown: the colour only repeats it
  tone: Tone;
}

export const POLL_MS = 3000;

const STATUS: Record<ProcessingStatus, Badge> = {
  receiving: { text: "Receiving", tone: "info" },
  processing: { text: "Processing", tone: "info" },
  done: { text: "Done", tone: "ok" },
  failed: { text: "Failed", tone: "error" },
  no_model: { text: "No model for this modality", tone: "muted" },
};

const REVIEW: Record<ReviewStatus, Badge> = {
  none: { text: "Not reviewed", tone: "muted" },
  approved: { text: "Approved", tone: "ok" },
  rejected: { text: "Rejected", tone: "warn" },
};

export function statusBadge(status: ProcessingStatus): Badge {
  return STATUS[status] ?? { text: `Unknown (${status})`, tone: "muted" };
}

export function reviewBadge(status: ReviewStatus): Badge {
  return REVIEW[status] ?? { text: `Unknown (${status})`, tone: "muted" };
}

/** Poll again only while a study is still on its way (receiving or processing). */
export function shouldPoll(studies: readonly StudySummary[]): boolean {
  return studies.some((s) => s.status === "receiving" || s.status === "processing");
}

export type ViewerKind = "dx" | "ct" | "none";

export function viewerKind(modality: string): ViewerKind {
  const modalities = modality.split(",");
  if (modalities.some((m) => m === "DX" || m === "CR")) return "dx";
  if (modalities.includes("CT")) return "ct";
  return "none";
}
