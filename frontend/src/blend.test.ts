// @vitest-environment node
// The browser's colouring and blending against the worker's stored overlay (gradcam.py, alpha
// 0.45) for DX 00000116_000 / Cardiomegaly: fixtures from scripts/export_blend_fixture.py.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import { colourCam, composite, DEFAULT_OPACITY, jet, type Pixels } from "./blend";

function read(name: string): Pixels {
  const png = PNG.sync.read(readFileSync(join(process.cwd(), "src", "__fixtures__", name)));
  return { width: png.width, height: png.height, data: new Uint8ClampedArray(png.data) };
}

test("jet matches numpy's colours at the ends and the middle", () => {
  expect(jet(0)).toEqual([0, 0, 127]);
  expect(jet(0.5)).toEqual([127, 255, 127]);
  expect(jet(1)).toEqual([127, 0, 0]);
});

test("opacity sets the alpha of every pixel", () => {
  const cam = { width: 1, height: 1, data: new Uint8ClampedArray([255, 255, 255, 255]) };
  expect(colourCam(cam, 0).data[3]).toBe(0);
  expect(colourCam(cam, DEFAULT_OPACITY).data[3]).toBe(115);
  expect(colourCam(cam, 1).data[3]).toBe(255);
});

test("colouring the raw map at 0.45 gives the worker's stored overlay (<= 2 grey levels)", () => {
  const seen = read("seen.png");
  const cam = read("cam.png");
  const stored = read("overlay.png");
  expect([seen.width, seen.height, cam.width, stored.width]).toEqual([224, 224, 224, 224]);
  const ours = composite(seen, colourCam(cam, DEFAULT_OPACITY));
  let worst = 0;
  const counts = [0, 0, 0, 0];
  for (let i = 0; i < ours.data.length; i += 4) {
    for (let c = 0; c < 3; c++) {
      const diff = Math.abs((ours.data[i + c] ?? 0) - (stored.data[i + c] ?? 0));
      worst = Math.max(worst, diff);
      counts[Math.min(diff, 3)]! += 1;
    }
  }
  console.log(`224x224x3 channels: diff 0/1/2/3+ = ${counts.join("/")}, max ${worst}`);
  expect(worst).toBeLessThanOrEqual(2);
});
