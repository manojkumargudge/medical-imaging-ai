import { useRef, useState } from "react";
import { ApiError } from "../api";
import { type Check, checkFile } from "../upload";
import { ScopeWarning } from "./ScopeWarning";

interface UploadAnswer {
  duplicate: boolean;
  upload_id: number;
  study_id: string;
  message?: string;
}

async function send(file: File, uploader: string): Promise<UploadAnswer> {
  const body = new FormData();
  body.append("file", file);
  body.append("uploader", uploader);
  const response = await fetch("/api/uploads", { method: "POST", body });
  const json = (await response.json().catch(() => ({}))) as { detail?: unknown };
  if (!response.ok) {
    const detail = typeof json.detail === "string" ? json.detail : response.statusText;
    throw new ApiError(response.status, detail);
  }
  return json as UploadAnswer;
}

/** One PNG or DX/CR DICOM file. No <form>: only a click on Upload sends. After an upload the
 * worklist opens, where the new study refreshes until it is processed. */
export function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [check, setCheck] = useState<Check | null>(null);
  const [uploader, setUploader] = useState("");
  const [sending, setSending] = useState(false);
  const busy = useRef(false);
  const [error, setError] = useState<string | null>(null);
  const [duplicate, setDuplicate] = useState<UploadAnswer | null>(null);

  async function choose(chosen: File | null) {
    setFile(chosen);
    setCheck(null);
    setError(null);
    setDuplicate(null);
    if (chosen) setCheck(await checkFile(chosen));
  }

  const ready = file !== null && check?.ok === true && uploader.trim() !== "";

  async function upload() {
    if (!ready || busy.current || !file) return;
    busy.current = true;
    setSending(true);
    setError(null);
    try {
      const answer = await send(file, uploader.trim());
      if (answer.duplicate) setDuplicate(answer);
      else window.location.hash = "#/";
    } catch (e) {
      const status = e instanceof ApiError ? ` (${e.status})` : "";
      setError(`Not uploaded${status}: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      busy.current = false;
      setSending(false);
    }
  }

  return (
    <section>
      <h1>Upload</h1>
      <ScopeWarning />
      <div className="panel">
        <p>
          One file: a PNG chest X-ray, or a DICOM DX or CR image marked de-identified
          (PatientIdentityRemoved = YES). At most 20 MB; each side 256 to 4096 pixels. A CT needs a
          full series, which cannot be uploaded here. Use public or synthetic data only.
        </p>
        <div className="review-fields">
          <label className="field">
            File (PNG or DICOM)
            <input
              type="file"
              accept=".png,.dcm,image/png,application/dicom"
              onChange={(e) => void choose(e.target.files?.[0] ?? null)}
            />
          </label>
          {check && !check.ok && (
            <p className="error" role="alert">
              {check.message}
            </p>
          )}
          {check?.ok && <p className="muted">Looks like a {check.kind.toUpperCase()} file.</p>}
          <label className="field">
            Your name
            <input
              type="text"
              value={uploader}
              maxLength={100}
              autoComplete="off"
              onChange={(e) => setUploader(e.target.value)}
            />
          </label>
          <p className="muted">Names are not verified (education demo).</p>
          <button type="button" onClick={() => void upload()} disabled={!ready || sending}>
            {sending ? "Uploading…" : "Upload"}
          </button>
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          {duplicate && (
            <p role="status">
              This file was uploaded before; nothing new was added.{" "}
              <a href={`#/studies/${duplicate.study_id}`}>Open the earlier study</a>
            </p>
          )}
        </div>
      </div>
    </section>
  );
}
