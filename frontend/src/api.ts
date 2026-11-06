// The viewer API (src/api). Same origin: nginx (or the Vite dev server) proxies /api.

export type ProcessingStatus = "receiving" | "processing" | "done" | "failed" | "no_model";
export type ReviewStatus = "none" | "approved" | "rejected";

export interface StudySummary {
  id: string;
  study_uid: string;
  patient_id: string;
  modality: string;
  status: ProcessingStatus;
  review_status: ReviewStatus;
  source: "demo" | "upload";
  updated_at: string | null;
}

export interface LabelResult {
  score: number;
  cam?: string;
}

export interface ClassifierResults {
  instances: { sop_instance_uid?: string; labels: Record<string, LabelResult> }[];
}

export interface SegmentationResults {
  series: { series_uid: string; slices: number; spleen_voxels: number; spleen_ml: number }[];
}

export interface ResultRow {
  id: number;
  model: string;
  status: "ok" | "error";
  checkpoint_name: string;
  created_at: string;
  results: ClassifierResults | SegmentationResults | { error: string };
}

export interface StudyDetail {
  study: StudySummary;
  results: ResultRow[];
  score_note: string;
}

export interface ThresholdEntry {
  set: "sensitivity_80" | "best_f1" | "score_only";
  threshold: number | null;
  low_reliability: boolean;
}

export interface Thresholds {
  labels: Record<string, ThresholdEntry>;
  note: string;
}

export interface Health {
  status: "ok" | "unavailable";
  database: { ok: boolean; name?: string; error?: string };
  orthanc: { ok: boolean; name?: string; stack?: string; error?: string };
}

export const CLASSIFIER = "densenet121-chestxray14";
export const SEGMENTATION = "monai-unet-spleen";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // not JSON: keep the status text
    }
    throw new ApiError(response.status, detail || `HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

/** Health, including the 503 body (which says which service is down and why). */
export async function getHealth(): Promise<Health> {
  const response = await fetch("/api/health", { headers: { Accept: "application/json" } });
  return (await response.json()) as Health;
}

export function healthProblems(health: Health): string[] {
  const problems = [];
  if (!health.orthanc.ok) problems.push(`Orthanc: ${health.orthanc.error ?? "unavailable"}`);
  if (!health.database.ok) problems.push(`Results database: ${health.database.error ?? "unavailable"}`);
  return problems;
}
