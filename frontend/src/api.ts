// The viewer API (src/api). Same origin: nginx (or the Vite dev server) proxies /api.

export type ProcessingStatus = "receiving" | "processing" | "done" | "failed" | "no_model";
export type ReviewStatus = "none" | "approved" | "rejected";

export interface StudySummary {
  id: string;
  study_uid: string;
  patient_id: string;
  description: string;
  modality: string;
  summary: string | null;
  status: ProcessingStatus;
  review_status: ReviewStatus;
  source: "demo" | "upload" | "unknown"; // demo: one of the demo studies, by UID
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

export type Decision = "approve" | "reject";

export interface Review {
  id: number;
  study_uid: string;
  inference_result_id: number;
  decision: Decision;
  reviewer: string;
  comment: string | null;
  created_at: string;
}

export interface NewReview {
  decision: Decision;
  reviewer: string;
  comment: string | null;
  submit_token: string;
  inference_result_id: number;
}

export interface AuditEntry {
  id: number;
  created_at: string;
  actor: string;
  event: "upload" | "inference_done" | "inference_failed" | "review";
  study_uid: string | null;
  details: Record<string, unknown>;
}

export interface AuditPage {
  entries: AuditEntry[];
  page: number;
  page_size: number;
  total: number;
}

export interface CtMeta {
  slices: number;
  columns: number;
  rows: number;
  window: { center: number; width: number };
  mask: boolean;
  first_mask_slice: number | null; // 0-based
  last_mask_slice: number | null;
  spleen_ml: number | null;
  spleen_ml_dicom_grid: number | null;
  result_id: number | null;
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
  return parse<T>(await fetch(path, { headers: { Accept: "application/json" } }));
}

export async function postJson<T>(path: string, body: unknown): Promise<T> {
  return parse<T>(
    await fetch(path, {
      method: "POST",
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

/** A new submit token: one per opened review panel (the server saves a token once). */
export function newToken(): string {
  return crypto.randomUUID();
}

async function parse<T>(response: Response): Promise<T> {
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
