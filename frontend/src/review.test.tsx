import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { isValid, NAMES_NOTE, ReviewPanel } from "./components/ReviewPanel";

const STUDY = "3f1d117e-f6f727f2-b354e775-0d964b50-3d9ade55";
const PATH = `/api/studies/${STUDY}/reviews`;

function reply(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

/** A fake API: GET returns the history; POST answers with `post` (or waits until released). */
function api(post: (body: Record<string, unknown>) => Promise<Response>) {
  const posts: Record<string, unknown>[] = [];
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    expect(url).toBe(PATH);
    if (init?.method === "POST") {
      const body = JSON.parse(String(init.body)) as Record<string, unknown>;
      posts.push(body);
      return post(body);
    }
    return reply({ reviews: [] });
  });
  vi.stubGlobal("fetch", fetch);
  return posts;
}

const saved = (body: Record<string, unknown>) =>
  reply(
    {
      review: {
        id: 1,
        study_uid: "1.2",
        inference_result_id: 19,
        decision: body.decision,
        reviewer: body.reviewer,
        comment: body.comment,
        created_at: "2026-11-06T05:00:00+00:00",
      },
    },
    201,
  );

async function open() {
  render(<ReviewPanel studyId={STUDY} resultId={19} />);
  await act(async () => {});
}

const radio = (name: "Approve" | "Reject") => screen.getByLabelText(name) as HTMLInputElement;
const nameField = () => screen.getByLabelText("Your name") as HTMLInputElement;
const commentField = () => screen.getByLabelText(/^Comment/) as HTMLTextAreaElement;
const saveButton = () => screen.getByRole("button", { name: /Save review|Saving/ }) as HTMLButtonElement;

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("nothing is selected at first, and the names note is shown", async () => {
  api(async (b) => saved(b));
  await open();
  expect(radio("Approve").checked).toBe(false);
  expect(radio("Reject").checked).toBe(false);
  expect(saveButton().disabled).toBe(true);
  expect(screen.getByText(NAMES_NOTE)).toBeTruthy();
  expect(document.querySelector("form")).toBeNull(); // nothing for Enter to submit
});

test("Save is disabled until the review is valid", async () => {
  api(async (b) => saved(b));
  await open();
  fireEvent.click(radio("Approve"));
  expect(saveButton().disabled).toBe(true); // no name
  fireEvent.change(nameField(), { target: { value: "   " } });
  expect(saveButton().disabled).toBe(true); // blank name
  fireEvent.change(nameField(), { target: { value: "A. Reviewer" } });
  expect(saveButton().disabled).toBe(false);
  fireEvent.click(radio("Reject"));
  expect(saveButton().disabled).toBe(true); // reject needs a comment
  fireEvent.change(commentField(), { target: { value: "map outside the lungs" } });
  expect(saveButton().disabled).toBe(false);
  expect(isValid("reject", "A", "  ")).toBe(false);
  expect(isValid(null, "A", "x")).toBe(false);
});

test("Enter in any field never submits", async () => {
  const posts = api(async (b) => saved(b));
  await open();
  fireEvent.click(radio("Reject"));
  fireEvent.change(nameField(), { target: { value: "A. Reviewer" } });
  fireEvent.change(commentField(), { target: { value: "comment" } });
  for (const field of [nameField(), commentField(), radio("Approve"), radio("Reject")]) {
    fireEvent.keyDown(field, { key: "Enter", code: "Enter" });
    fireEvent.keyPress(field, { key: "Enter", code: "Enter", charCode: 13 });
    fireEvent.keyUp(field, { key: "Enter", code: "Enter" });
  }
  await act(async () => {});
  expect(posts).toHaveLength(0);
});

test("a double click sends one request, and the button is disabled while saving", async () => {
  let release: (r: Response) => void = () => {};
  const posts = api((b) => new Promise<Response>((done) => (release = () => done(saved(b)))));
  await open();
  fireEvent.click(radio("Approve"));
  fireEvent.change(nameField(), { target: { value: "A. Reviewer" } });
  fireEvent.click(saveButton());
  fireEvent.click(saveButton());
  await act(async () => {});
  expect(posts).toHaveLength(1);
  expect(saveButton().disabled).toBe(true);
  expect(saveButton().textContent).toBe("Saving…");
  await act(async () => release(new Response()));
  expect(screen.getByRole("status").textContent).toBe("Saved: Approved by A. Reviewer.");
  expect(posts[0]).toMatchObject({ decision: "approve", reviewer: "A. Reviewer", comment: null });
  expect(posts[0]?.inference_result_id).toBe(19);
  // After a save the form starts again: no choice, and a new token for the next review.
  expect(radio("Approve").checked).toBe(false);
});

test("each opened panel sends its own token", async () => {
  const posts = api(async (b) => saved(b));
  for (let i = 0; i < 2; i++) {
    await open();
    fireEvent.click(radio("Approve"));
    fireEvent.change(nameField(), { target: { value: "A. Reviewer" } });
    fireEvent.click(saveButton());
    await act(async () => {});
    cleanup();
  }
  const tokens = posts.map((p) => p.submit_token);
  expect(tokens).toHaveLength(2);
  expect(tokens[0]).not.toBe(tokens[1]);
  expect(String(tokens[0])).toMatch(/^[0-9a-f-]{36}$/);
});

test.each([
  [409, "already saved: this review was submitted before (same submit token)"],
  [409, "a newer result exists for this study; reload the page and review the newest one"],
  [422, "comment: a comment is required to reject"],
])("a %i is shown with the API's message", async (status, detail) => {
  api(async () => reply({ detail }, status));
  await open();
  fireEvent.click(radio("Approve"));
  fireEvent.change(nameField(), { target: { value: "A. Reviewer" } });
  fireEvent.click(saveButton());
  await act(async () => {});
  expect(screen.getByRole("alert").textContent).toBe(`Not saved (${status}): ${detail}`);
  expect(saveButton().disabled).toBe(false); // the reviewer can correct and try again
});

test("without a result there is nothing to review", async () => {
  api(async (b) => saved(b));
  render(<ReviewPanel studyId={STUDY} resultId={null} />);
  await act(async () => {});
  expect(screen.getByText("No result to review yet.")).toBeTruthy();
  expect(screen.queryByRole("button", { name: /Save/ })).toBeNull();
});
