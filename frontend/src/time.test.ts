import { describe as describeAudit } from "./components/AuditPage";
import { formatLocal } from "./time";

test("times are shown in the browser's zone with its name", () => {
  const iso = "2026-11-06T05:10:21.931668+00:00";
  expect(formatLocal(iso, "Asia/Kolkata", "en-IN")).toBe("06 Nov 2026, 10:40 IST");
  expect(formatLocal(iso, "UTC", "en-GB")).toBe("06 Nov 2026, 05:10 UTC");
  expect(formatLocal(iso, "Asia/Kolkata", "en-US")).toBe("06 Nov 2026, 10:40 GMT+5:30");
  expect(formatLocal(null)).toBe("—");
  expect(formatLocal("not a time")).toBe("not a time");
});

test("audit details are described in words", () => {
  const entry = {
    id: 1,
    created_at: "2026-11-06T05:00:00+00:00",
    actor: "A. Reviewer",
    study_uid: "1.2",
  };
  expect(
    describeAudit({
      ...entry,
      event: "review",
      details: { decision: "reject", inference_result_id: 19, comment: "off the lungs" },
    }),
  ).toBe("Rejected result 19: off the lungs");
  expect(
    describeAudit({
      ...entry,
      event: "inference_done",
      details: { model: "monai-unet-spleen", seconds: 58.5, inference_result_id: 19 },
    }),
  ).toBe("monai-unet-spleen, 58.5 s (result 19)");
});
