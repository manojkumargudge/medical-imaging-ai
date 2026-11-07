import { useMemo, useState } from "react";
import {
  CLASSIFIER,
  type ClassifierResults,
  type StudyDetail,
  type Thresholds,
} from "../api";
import { DEFAULT_OPACITY } from "../blend";
import { defaultLabel, formatScore, scoreRows, VERDICT_TEXT } from "../scores";
import { statusBadge } from "../status";
import { CamOverlay } from "./CamOverlay";
import { BadgeView } from "./Layout";
import { ReviewPanel } from "./ReviewPanel";
import { ScopeWarning } from "./ScopeWarning";
import { ScoresTable } from "./ScoresTable";

export function KnownLimits() {
  return (
    <section className="panel">
      <h2>Known limits</h2>
      <p className="muted">
        From the Grad-CAM demo on 12 NIH ChestX-ray14 test images (docs/figures/gradcam) and the
        test-split metrics (docs/results_v2.md). A few images show how maps look; they are not a
        rate.
      </p>
      <ul>
        <li>
          In the Grad-CAM demo, all 4 missed positives had maps outside the lungs: at the image
          edge, or just below the heart.
        </li>
        <li>
          A pacemaker was visible in 4 of the first 6 demo images, including both Cardiomegaly
          images. The map sat on the heart, not the device, but a pacemaker and Cardiomegaly
          confound is possible.
        </li>
        <li>
          Maps may not be label-specific: one patient's Effusion and Pneumothorax maps covered
          nearly the same region (correlation 0.79).
        </li>
        <li>
          For Infiltration, Pneumonia and Nodule, the threshold reaching val sensitivity ≥ 0.80
          gave test specificity below 0.40 (0.234, 0.271, 0.363). The viewer uses the highest-F1
          threshold for them instead and marks them low reliability.
        </li>
        <li>
          The Cardiomegaly threshold did not transfer: chosen for val sensitivity 0.80, it gave
          test sensitivity 0.749.
        </li>
        <li>A map shows which regions raised the score, not where a finding is.</li>
      </ul>
    </section>
  );
}

export function ModelScope() {
  return (
    <section className="panel">
      <h2>Model scope</h2>
      <p>
        Trained on NIH ChestX-ray14 frontal X-rays only, with labels mined from reports by NLP
        (not checked by radiologists). Other views, scanners, populations or image types are
        outside what it was trained on.
      </p>
    </section>
  );
}

export function DxView({
  detail,
  thresholds,
  initialLabel,
}: {
  detail: StudyDetail;
  thresholds: Thresholds;
  initialLabel?: string;
}) {
  const result = detail.results.find((r) => r.model === CLASSIFIER && r.status === "ok");
  const labels = result ? (result.results as ClassifierResults).instances[0]?.labels : undefined;
  const rows = useMemo(
    () => (labels ? scoreRows(labels, thresholds.labels) : []),
    [labels, thresholds.labels],
  );
  const firstLabel =
    initialLabel && initialLabel in thresholds.labels ? initialLabel : defaultLabel(rows);
  const [label, setLabel] = useState(firstLabel ?? Object.keys(thresholds.labels)[0] ?? "");
  const [visible, setVisible] = useState(true);
  const [opacity, setOpacity] = useState(DEFAULT_OPACITY);
  const id = detail.study.id;
  const row = rows.find((r) => r.label === label);

  return (
    <>
      <h1>
        {detail.study.modality || "DX"} study, patient {detail.study.patient_id}{" "}
        <BadgeView badge={statusBadge(detail.study.status)} />
      </h1>
      {detail.study.source !== "demo" && <ScopeWarning />}
      {!result && (
        <p>No classifier result for this study yet. The image is shown without an overlay.</p>
      )}
      <div className="dx">
        <div className="dx-image">
          <CamOverlay
            imageUrl={`/api/studies/${id}/image`}
            camUrl={`/api/studies/${id}/cam/${encodeURIComponent(label)}`}
            visible={visible && Boolean(result)}
            opacity={opacity}
            alt={`DX image of patient ${detail.study.patient_id}`}
          />
          {result && (
            <div className="controls">
              <label>
                <input
                  type="checkbox"
                  checked={visible}
                  onChange={(e) => setVisible(e.target.checked)}
                />{" "}
                Grad-CAM overlay
              </label>
              <label>
                Label{" "}
                <select value={label} onChange={(e) => setLabel(e.target.value)}>
                  {rows.map((r) => (
                    <option key={r.label} value={r.label}>
                      {r.label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Opacity{" "}
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={opacity}
                  disabled={!visible}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                />{" "}
                {opacity.toFixed(2)}
              </label>
              {row && (
                <p className="muted">
                  {label}: score {formatScore(row.score)}, {VERDICT_TEXT[row.verdict].toLowerCase()}
                  {row.lowReliability ? ", low reliability" : ""}. Red marks the regions that
                  raised this label's score most, blue the least.
                </p>
              )}
            </div>
          )}
        </div>
        <div className="dx-side">
          {result && <ScoresTable rows={rows} selected={label} note={thresholds.note} />}
          <ReviewPanel
            studyId={id}
            resultId={detail.results.find((r) => r.model === CLASSIFIER)?.id ?? null}
          />
          <ModelScope />
          <KnownLimits />
        </div>
      </div>
    </>
  );
}
