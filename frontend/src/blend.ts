// Colour a raw Grad-CAM map (8-bit grey, 0-255 for 0-1) in the browser, the same way the worker
// colours its stored overlay (src/explain/gradcam.py: `jet`, then a blend with OVERLAY_ALPHA).

export const DEFAULT_OPACITY = 0.45;

export interface Pixels {
  width: number;
  height: number;
  data: Uint8ClampedArray | Uint8Array; // RGBA, as in ImageData
}

/** Blue (0) to red (1); as numpy's jet() in gradcam.py, including the truncation to uint8. */
export function jet(value: number): [number, number, number] {
  const channel = (centre: number) =>
    Math.floor(Math.min(Math.max(1.5 - Math.abs(4 * value - centre), 0), 1) * 255);
  return [channel(3), channel(2), channel(1)];
}

/** The map as RGBA: jet colours, alpha = opacity. The grey value is read from the R channel. */
export function colourCam(cam: Pixels, opacity: number): Pixels {
  const alpha = Math.round(Math.min(Math.max(opacity, 0), 1) * 255);
  const out = new Uint8ClampedArray(cam.width * cam.height * 4);
  for (let i = 0; i < cam.width * cam.height; i++) {
    const [r, g, b] = jet((cam.data[i * 4] ?? 0) / 255);
    out[i * 4] = r;
    out[i * 4 + 1] = g;
    out[i * 4 + 2] = b;
    out[i * 4 + 3] = alpha;
  }
  return { width: cam.width, height: cam.height, data: out };
}

/** Source-over compositing of `top` on an opaque `base` of the same size (what the browser
 * draws when the coloured map lies over the image). */
export function composite(base: Pixels, top: Pixels): Pixels {
  if (base.width !== top.width || base.height !== top.height) {
    throw new Error(`sizes differ: ${base.width}x${base.height} vs ${top.width}x${top.height}`);
  }
  const out = new Uint8ClampedArray(base.data.length);
  for (let i = 0; i < base.data.length; i += 4) {
    const a = (top.data[i + 3] ?? 0) / 255;
    for (let c = 0; c < 3; c++) {
      out[i + c] = Math.round((1 - a) * (base.data[i + c] ?? 0) + a * (top.data[i + c] ?? 0));
    }
    out[i + 3] = 255;
  }
  return { width: base.width, height: base.height, data: out };
}
