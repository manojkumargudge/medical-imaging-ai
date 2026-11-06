import { act, cleanup, render, screen } from "@testing-library/react";
import type { ProcessingStatus } from "./api";
import { App } from "./App";
import { Worklist } from "./components/Worklist";

function studies(...statuses: ProcessingStatus[]) {
  return {
    studies: statuses.map((status, i) => ({
      id: `s${i}`,
      study_uid: `1.${i}`,
      patient_id: `p${i}`,
      modality: "DX",
      status,
      review_status: "none",
      source: "demo",
      updated_at: "2026-11-06T05:00:00+00:00",
    })),
  };
}

function reply(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const flush = () => act(() => vi.advanceTimersByTimeAsync(0));

beforeEach(() => {
  vi.useFakeTimers();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

test("polls every 3 s while a study is processing, and stops when all are done", async () => {
  const answers = [studies("processing", "done"), studies("receiving"), studies("done", "done")];
  const fetch = vi.fn(async (url: string) => {
    expect(url).toBe("/api/studies");
    return reply(answers.shift() ?? studies("done", "done"));
  });
  vi.stubGlobal("fetch", fetch);
  render(<Worklist />);
  await flush();
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(screen.getByText("Processing")).toBeTruthy();

  await act(() => vi.advanceTimersByTimeAsync(2999));
  expect(fetch).toHaveBeenCalledTimes(1);
  await act(() => vi.advanceTimersByTimeAsync(1));
  expect(fetch).toHaveBeenCalledTimes(2);
  expect(screen.getByText("Receiving")).toBeTruthy();

  await act(() => vi.advanceTimersByTimeAsync(3000));
  expect(fetch).toHaveBeenCalledTimes(3);
  expect(screen.getAllByText("Done")).toHaveLength(2);
  expect(screen.getByText(/not refreshing/)).toBeTruthy();

  await act(() => vi.advanceTimersByTimeAsync(30000));
  expect(fetch).toHaveBeenCalledTimes(3); // all done: no more requests
});

test("does not poll when nothing is processing", async () => {
  const fetch = vi.fn(async () => reply(studies("done", "failed", "no_model")));
  vi.stubGlobal("fetch", fetch);
  render(<Worklist />);
  await flush();
  await act(() => vi.advanceTimersByTimeAsync(10000));
  expect(fetch).toHaveBeenCalledTimes(1);
  expect(screen.getByText("Failed")).toBeTruthy();
  expect(screen.getByText("No model for this modality")).toBeTruthy();
});

test("a 503 from the API shows which service is down", async () => {
  const down = "Orthanc is not reachable at http://orthanc:8042: refused";
  const fetch = vi.fn(async (url: string) =>
    url === "/api/health"
      ? reply(
          {
            status: "unavailable",
            database: { ok: true, name: "aiml02_results" },
            orthanc: { ok: false, error: down },
          },
          503,
        )
      : reply({ detail: down }, 503),
  );
  vi.stubGlobal("fetch", fetch);
  render(<Worklist />);
  await flush();
  await flush();
  expect(screen.getByRole("alert").textContent).toContain(`Orthanc: ${down}`);
  expect(screen.getByText("The viewer API reports a service is down")).toBeTruthy();
  await act(() => vi.advanceTimersByTimeAsync(10000));
  expect(fetch).toHaveBeenCalledTimes(2); // no polling while in error
});

test("the banner is on every page", async () => {
  const detail = { study: studies("done").studies[0], results: [], score_note: "" };
  const fetch = vi.fn(async (url: string) =>
    reply(url === "/api/studies" ? studies("done") : url.endsWith("thresholds") ? { labels: {}, note: "" } : detail),
  );
  vi.stubGlobal("fetch", fetch);
  for (const hash of ["#/", "#/studies/s0"]) {
    window.location.hash = hash;
    render(<App />);
    await flush();
    expect(screen.getByRole("note").textContent).toBe("For education only. Not for clinical use.");
    cleanup();
  }
});
