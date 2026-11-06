import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { CtMeta, StudyDetail } from "./api";
import { clampSlice, CT_SCOPE_WARNING, CtView, neighbours } from "./components/CtView";

const META: CtMeta = {
  slices: 54,
  columns: 512,
  rows: 512,
  window: { center: 40, width: 400 },
  mask: true,
  first_mask_slice: 22,
  last_mask_slice: 42,
  spleen_ml: 440.764875,
  spleen_ml_dicom_grid: 441.9,
  result_id: 22,
};

function detail(source: "demo" | "upload" | "unknown" = "demo"): StudyDetail {
  return {
    study: {
      id: "ct1",
      study_uid: "1.2",
      patient_id: "spleen_14",
      description: "MSD Task09 spleen_14",
      modality: "CT",
      status: "done",
      review_status: "none",
      source,
      summary: "Spleen 440.8 mL",
      updated_at: null,
    },
    results: [
      {
        id: 22,
        model: "monai-unet-spleen",
        status: "ok",
        checkpoint_name: "segmentation.pt",
        created_at: "2026-11-06T07:40:00+00:00",
        results: { series: [] },
      },
    ],
    score_note: "",
  };
}

const loaded: string[] = [];

beforeEach(() => {
  loaded.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) =>
      new Response(JSON.stringify(url.endsWith("/ct/meta") ? META : { reviews: [] })),
    ),
  );
  vi.stubGlobal(
    "Image",
    class {
      set src(value: string) {
        loaded.push(value);
      }
    },
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

async function open(source: "demo" | "upload" | "unknown" = "demo", initialSlice?: number) {
  render(<CtView detail={detail(source)} initialSlice={initialSlice} />);
  await act(async () => {});
}

const position = () => screen.getByTestId("position").textContent;
const sliceImage = () => screen.getByAltText(/^CT slice/) as HTMLImageElement;

test("the slider shows k / N, starts in the middle, and the image follows it", async () => {
  await open();
  expect(position()).toBe("28 / 54");
  expect(sliceImage().getAttribute("src")).toBe("/api/studies/ct1/ct/slice/27");
  fireEvent.change(screen.getByLabelText("Slice"), { target: { value: "1" } });
  expect(position()).toBe("1 / 54");
  expect(sliceImage().getAttribute("src")).toBe("/api/studies/ct1/ct/slice/0");
  expect(screen.getByTestId("mask").getAttribute("src")).toBe("/api/studies/ct1/ct/mask/0");
});

test("arrow keys step through the slices, but not while typing a name", async () => {
  await open("demo", 10);
  expect(position()).toBe("10 / 54");
  fireEvent.keyDown(window, { key: "ArrowRight" });
  fireEvent.keyDown(window, { key: "ArrowUp" });
  expect(position()).toBe("12 / 54");
  fireEvent.keyDown(window, { key: "ArrowLeft" });
  expect(position()).toBe("11 / 54");
  fireEvent.keyDown(screen.getByLabelText("Your name"), { key: "ArrowLeft" });
  expect(position()).toBe("11 / 54");
});

test("jump to spleen goes to the first mask slice; the mask can be hidden", async () => {
  await open();
  fireEvent.click(screen.getByRole("button", { name: "Jump to spleen" }));
  expect(position()).toBe("23 / 54");
  expect(screen.getByText(/The mask is on slices 23 to 43/)).toBeTruthy();
  expect((screen.getByTestId("mask") as HTMLImageElement).style.opacity).toBe("0.5");
  fireEvent.change(screen.getAllByRole("slider")[1]!, { target: { value: "0.8" } });
  expect((screen.getByTestId("mask") as HTMLImageElement).style.opacity).toBe("0.8");
  fireEvent.click(screen.getByLabelText("Spleen mask"));
  expect(screen.queryByTestId("mask")).toBeNull();
  expect(screen.getByText("440.8 mL")).toBeTruthy();
});

test("the next and previous slices are loaded ahead", async () => {
  await open("demo", 30);
  expect(loaded).toEqual(
    expect.arrayContaining([
      "/api/studies/ct1/ct/slice/28",
      "/api/studies/ct1/ct/slice/30",
      "/api/studies/ct1/ct/mask/28",
      "/api/studies/ct1/ct/mask/30",
    ]),
  );
  expect(neighbours(0, 54)).toEqual([1]);
  expect(neighbours(53, 54)).toEqual([52]);
  expect(clampSlice(99, 54)).toBe(53);
  expect(clampSlice(-3, 54)).toBe(0);
});

test("a CT that is not a demo study shows the model-scope warning", async () => {
  await open("unknown");
  expect(screen.getByLabelText("Model scope warning").textContent).toContain(CT_SCOPE_WARNING);
  cleanup();
  await open("demo");
  expect(screen.queryByLabelText("Model scope warning")).toBeNull();
});
