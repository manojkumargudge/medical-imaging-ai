import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import type { StudyDetail, Thresholds } from "./api";
import { DxView } from "./components/DxView";
import { UploadPage } from "./components/UploadPage";
import { checkFile, kindFromBytes, MAX_BYTES, SCOPE_WARNING } from "./upload";

const PNG_HEAD = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function pngFile(name = "x.png", extra = 100): File {
  return new File([new Uint8Array([...PNG_HEAD, ...new Array(extra).fill(1)])], name);
}

function dicomFile(name = "x.dcm"): File {
  const bytes = new Uint8Array(300);
  bytes.set([0x44, 0x49, 0x43, 0x4d], 128);
  return new File([bytes], name);
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("the kind comes from the content, not the name", async () => {
  expect(kindFromBytes(new Uint8Array(PNG_HEAD))).toBe("png");
  expect(await checkFile(pngFile("scan.dcm"))).toEqual({ ok: true, kind: "png" });
  expect(await checkFile(dicomFile("image.png"))).toEqual({ ok: true, kind: "dicom" });
  const text = await checkFile(new File(["hello, not an image"], "fake.png"));
  expect(text).toEqual({
    ok: false,
    message: "Only PNG or DICOM files (checked by content, not by name).",
  });
  expect(await checkFile(new File([], "empty.png"))).toEqual({
    ok: false,
    message: "The file is empty.",
  });
});

test("files over 20 MB are refused before sending", async () => {
  const big = new File([new Uint8Array(MAX_BYTES + 1)], "big.png");
  expect(await checkFile(big)).toEqual({ ok: false, message: "The file is larger than 20 MB." });
  const exact = new File([new Uint8Array(PNG_HEAD), new Uint8Array(MAX_BYTES - 8)], "ok.png");
  expect(await checkFile(exact)).toEqual({ ok: true, kind: "png" });
});

async function choose(file: File) {
  const input = screen.getByLabelText("File (PNG or DICOM)") as HTMLInputElement;
  await act(async () => {
    fireEvent.change(input, { target: { files: [file] } });
  });
  await act(async () => {});
}

const uploadButton = () => screen.getByRole("button", { name: /Upload/ }) as HTMLButtonElement;

test("the upload page shows the model-scope warning and checks the file first", async () => {
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  render(<UploadPage />);
  expect(screen.getByLabelText("Model scope warning").textContent).toContain(SCOPE_WARNING);
  expect(uploadButton().disabled).toBe(true);
  await choose(new File(["not an image"], "x.png"));
  expect(screen.getByRole("alert").textContent).toContain("Only PNG or DICOM files");
  fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "A. Uploader" } });
  expect(uploadButton().disabled).toBe(true); // a bad file is never sent
  await choose(pngFile());
  expect(uploadButton().disabled).toBe(false);
  fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "  " } });
  expect(uploadButton().disabled).toBe(true); // a name is required
  expect(fetch).not.toHaveBeenCalled();
});

test("an upload sends the file and the name once, then opens the worklist", async () => {
  const fetch = vi.fn(
    async () =>
      new Response(JSON.stringify({ duplicate: false, upload_id: 3, study_id: "abc" }), {
        status: 201,
      }),
  );
  vi.stubGlobal("fetch", fetch);
  window.location.hash = "#/upload";
  render(<UploadPage />);
  await choose(pngFile());
  fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "A. Uploader" } });
  fireEvent.click(uploadButton());
  fireEvent.click(uploadButton());
  await act(async () => {});
  expect(fetch).toHaveBeenCalledTimes(1);
  const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
  const body = init.body as FormData;
  expect(url).toBe("/api/uploads");
  expect(body.get("uploader")).toBe("A. Uploader");
  expect((body.get("file") as File).name).toBe("x.png");
  expect(window.location.hash).toBe("#/");
});

test("API errors and duplicates are shown", async () => {
  const answers = [
    new Response(JSON.stringify({ detail: "CT needs a full series; not supported for upload" }), {
      status: 422,
    }),
    new Response(JSON.stringify({ duplicate: true, upload_id: 3, study_id: "abc" }), {
      status: 200,
    }),
  ];
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => answers.shift()),
  );
  render(<UploadPage />);
  await choose(dicomFile());
  fireEvent.change(screen.getByLabelText("Your name"), { target: { value: "A. Uploader" } });
  fireEvent.click(uploadButton());
  await act(async () => {});
  expect(screen.getByRole("alert").textContent).toBe(
    "Not uploaded (422): CT needs a full series; not supported for upload",
  );
  fireEvent.click(uploadButton());
  await act(async () => {});
  expect(screen.getByRole("status").textContent).toContain("uploaded before");
  expect(screen.getByRole("link", { name: "Open the earlier study" }).getAttribute("href")).toBe(
    "#/studies/abc",
  );
});

test("the DX view warns about model scope for every study that is not a demo study", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ reviews: [] }))),
  );
  const thresholds: Thresholds = { labels: {}, note: "" };
  for (const source of ["upload", "unknown", "demo"] as const) {
    const detail = {
      study: {
        id: "s",
        study_uid: "1",
        patient_id: "UPLOAD-3",
        description: "Upload 3",
        modality: "DX",
        status: "processing",
        review_status: "none",
        source,
        summary: null,
        updated_at: null,
      },
      results: [],
      score_note: "",
    } satisfies StudyDetail;
    render(<DxView detail={detail} thresholds={thresholds} />);
    const warning = screen.queryByLabelText("Model scope warning");
    expect(warning?.textContent?.includes(SCOPE_WARNING) ?? false).toBe(source !== "demo");
    cleanup();
  }
});
