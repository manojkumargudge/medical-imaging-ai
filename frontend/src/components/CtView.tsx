import { useCallback, useEffect, useState } from "react";
import { type CtMeta, getJson, SEGMENTATION, type StudyDetail } from "../api";
import { statusBadge } from "../status";
import { BadgeView, ErrorBox } from "./Layout";
import { ReviewPanel } from "./ReviewPanel";
import { ScopeWarning } from "./ScopeWarning";

export const CT_SCOPE_WARNING =
  "Trained on MSD Task09_Spleen CT volumes. Masks for other scans have no meaning.";

/** The slices next to k (to load before they are shown). */
export function neighbours(k: number, count: number): number[] {
  return [k - 1, k + 1].filter((n) => n >= 0 && n < count);
}

export function clampSlice(k: number, count: number): number {
  return Math.min(Math.max(k, 0), Math.max(count - 1, 0));
}

function typingIn(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  if (target instanceof HTMLInputElement) return target.type !== "checkbox";
  return target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement;
}

/** CT slices in patient order with the spleen mask on the DICOM grid. Slice numbers on screen
 * are 1-based; the API's are 0-based. Arrow keys step through the slices (not while typing). */
export function CtView({ detail, initialSlice }: { detail: StudyDetail; initialSlice?: number }) {
  const id = detail.study.id;
  const [meta, setMeta] = useState<CtMeta | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [k, setK] = useState(0);
  const [showMask, setShowMask] = useState(true);
  const [opacity, setOpacity] = useState(0.5);
  const base = `/api/studies/${encodeURIComponent(id)}/ct`;

  useEffect(() => {
    getJson<CtMeta>(`${base}/meta`)
      .then((m) => {
        setMeta(m);
        const start = initialSlice !== undefined ? initialSlice - 1 : Math.floor(m.slices / 2);
        setK(clampSlice(start, m.slices));
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, [base, initialSlice]);

  const count = meta?.slices ?? 0;
  const step = useCallback((by: number) => setK((now) => clampSlice(now + by, count)), [count]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (typingIn(event.target)) return; // the slider and text fields handle their own keys
      if (event.key === "ArrowLeft" || event.key === "ArrowDown") step(-1);
      else if (event.key === "ArrowRight" || event.key === "ArrowUp") step(1);
      else return;
      event.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step]);

  useEffect(() => {
    if (!meta) return;
    for (const n of neighbours(k, meta.slices)) {
      new Image().src = `${base}/slice/${n}`;
      if (meta.mask) new Image().src = `${base}/mask/${n}`;
    }
  }, [base, k, meta]);

  const result = detail.results.find((r) => r.model === SEGMENTATION);
  return (
    <>
      <h1>
        CT study, patient {detail.study.patient_id}{" "}
        <BadgeView badge={statusBadge(detail.study.status)} />
      </h1>
      {detail.study.source !== "demo" && <ScopeWarning text={CT_SCOPE_WARNING} />}
      {error && <ErrorBox title="Could not load the slices" lines={[error]} />}
      {!meta && !error && <p>Loading…</p>}
      {error && <ReviewPanel studyId={id} resultId={result?.id ?? null} />}
      {meta && (
        <div className="dx">
          <div className="dx-image">
            <div className="stack">
              <img
                src={`${base}/slice/${k}`}
                alt={`CT slice ${k + 1} of ${meta.slices}`}
                className="stack-image"
              />
              {meta.mask && showMask && (
                <img
                  src={`${base}/mask/${k}`}
                  alt=""
                  aria-hidden="true"
                  className="stack-overlay"
                  style={{ opacity }}
                  data-testid="mask"
                />
              )}
            </div>
            <div className="controls">
              <label>
                Slice{" "}
                <input
                  type="range"
                  min={1}
                  max={meta.slices}
                  value={k + 1}
                  aria-label="Slice"
                  onChange={(e) => setK(clampSlice(Number(e.target.value) - 1, meta.slices))}
                />
              </label>
              <span data-testid="position">
                {k + 1} / {meta.slices}
              </span>
              <button type="button" onClick={() => step(-1)} disabled={k === 0}>
                Previous
              </button>
              <button type="button" onClick={() => step(1)} disabled={k === meta.slices - 1}>
                Next
              </button>
              <button
                type="button"
                disabled={meta.first_mask_slice === null}
                onClick={() => meta.first_mask_slice !== null && setK(meta.first_mask_slice)}
              >
                Jump to spleen
              </button>
            </div>
            <div className="controls">
              <label>
                <input
                  type="checkbox"
                  checked={showMask}
                  disabled={!meta.mask}
                  onChange={(e) => setShowMask(e.target.checked)}
                />{" "}
                Spleen mask
              </label>
              <label>
                Opacity{" "}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={opacity}
                  disabled={!meta.mask || !showMask}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                />{" "}
                {opacity.toFixed(2)}
              </label>
            </div>
            <p className="muted">
              Window 40/400 (HU {-160} to 240). Arrow keys step through the slices.{" "}
              {meta.mask && meta.first_mask_slice !== null && meta.last_mask_slice !== null
                ? `The mask is on slices ${meta.first_mask_slice + 1} to ${meta.last_mask_slice + 1}.`
                : "No mask for this study."}
            </p>
          </div>
          <div className="dx-side">
            <section className="panel">
              <h2>Spleen volume</h2>
              {meta.spleen_ml !== null ? (
                <>
                  <p className="big">{meta.spleen_ml.toFixed(1)} mL</p>
                  <p className="muted">
                    From the 3D U-Net's mask (1.5 mm grid), trained on MSD Task09_Spleen. A model
                    output, not a measurement.
                    {meta.spleen_ml_dicom_grid !== null &&
                      ` The mask on the slices shown here (resampled to the DICOM grid by nearest neighbour) is ${meta.spleen_ml_dicom_grid.toFixed(1)} mL.`}
                  </p>
                </>
              ) : (
                <p>No segmentation result for this study yet.</p>
              )}
            </section>
            <ReviewPanel studyId={id} resultId={result?.id ?? null} />
          </div>
        </div>
      )}
    </>
  );
}
