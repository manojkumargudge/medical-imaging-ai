import { useEffect, useRef, useState } from "react";
import { colourCam, type Pixels } from "../blend";

/** The DX image with the raw Grad-CAM map coloured on a canvas laid over it. The canvas keeps
 * the map's 224x224 pixels and is stretched by CSS to the image's box, so the map covers the
 * whole image without cropping, as the model saw it (resized, not cropped). */
export function CamOverlay({
  imageUrl,
  camUrl,
  visible,
  opacity,
  alt,
}: {
  imageUrl: string;
  camUrl: string;
  visible: boolean;
  opacity: number;
  alt: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const [cam, setCam] = useState<Pixels | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setCam(null);
    setError(null);
    const image = new Image();
    image.onload = () => {
      if (cancelled) return;
      const work = document.createElement("canvas");
      work.width = image.naturalWidth;
      work.height = image.naturalHeight;
      const context = work.getContext("2d");
      if (!context) return setError("this browser cannot draw the map");
      context.drawImage(image, 0, 0);
      setCam(context.getImageData(0, 0, work.width, work.height));
    };
    image.onerror = () => !cancelled && setError("no raw map for this label");
    image.src = camUrl;
    return () => {
      cancelled = true;
    };
  }, [camUrl]);

  useEffect(() => {
    const target = canvas.current;
    const context = target?.getContext("2d");
    if (!target || !context || !cam) return;
    target.width = cam.width;
    target.height = cam.height;
    const coloured = colourCam(cam, opacity);
    context.putImageData(new ImageData(new Uint8ClampedArray(coloured.data), cam.width), 0, 0);
  }, [cam, opacity]);

  return (
    <div className="stack">
      <img src={imageUrl} alt={alt} className="stack-image" />
      <canvas ref={canvas} className="stack-overlay" hidden={!visible || !cam} aria-hidden="true" />
      {error && visible && <p className="muted">Overlay: {error}.</p>}
    </div>
  );
}
