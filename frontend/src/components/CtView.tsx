import { SEGMENTATION, type SegmentationResults, type StudyDetail } from "../api";
import { statusBadge } from "../status";
import { BadgeView } from "./Layout";
import { ReviewPanel } from "./ReviewPanel";

export function CtView({ detail }: { detail: StudyDetail }) {
  const result = detail.results.find((r) => r.model === SEGMENTATION && r.status === "ok");
  const series = result ? (result.results as SegmentationResults).series : [];
  return (
    <>
      <h1>
        CT study, patient {detail.study.patient_id}{" "}
        <BadgeView badge={statusBadge(detail.study.status)} />
      </h1>
      {series.length === 0 && <p>No segmentation result for this study yet.</p>}
      {series.map((s) => (
        <section className="panel" key={s.series_uid}>
          <h2>Spleen volume</h2>
          <p className="big">{s.spleen_ml.toFixed(1)} mL</p>
          <p className="muted">
            From the 3D U-Net's mask ({s.spleen_voxels.toLocaleString("en")} voxels of 1.5 mm, on{" "}
            {s.slices} slices), trained on MSD Task09_Spleen. A model output, not a measurement.
          </p>
        </section>
      ))}
      <p>The slice view with the spleen mask comes later.</p>
      <ReviewPanel
        studyId={detail.study.id}
        resultId={detail.results.find((r) => r.model === SEGMENTATION)?.id ?? null}
      />
    </>
  );
}
