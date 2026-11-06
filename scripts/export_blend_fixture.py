"""Export the frontend's blend test fixture from one demo study the worker processed.

Writes to frontend/src/__fixtures__/ (never overwrites):
- seen.png: the 224px grey image the model saw (gradcam.preprocess of the DX pixels)
- cam.png: the worker's raw Grad-CAM map for the label
- overlay.png: the worker's stored blended overlay (OVERLAY_ALPHA 0.45)

frontend/src/blend.test.ts colours cam.png and blends it over seen.png in TypeScript and compares
with overlay.png. The image is NIH ChestX-ray14 00000116_000 (public, de-identified), whose
overlay is already published in docs/figures/gradcam/. For education only, not for clinical use.

Run (after the worker processed the demo studies): python -m scripts.export_blend_fixture
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

from scripts.upload_demo import DATA_ENTRY, DEMO, dx_from_png
from src.data.preprocess import read_params
from src.dicom.render import to_png
from src.explain import gradcam
from src.service.analyzers import ClassifierAnalyzer

OUT = Path("frontend/src/__fixtures__")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, default=DEMO / "00000116_000.png")
    parser.add_argument("--label", default="Cardiomegaly")
    parser.add_argument("--results-dir", type=Path, default=Path("results/worker"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    ds = dx_from_png(args.png, DATA_ENTRY)
    folder = args.results_dir / str(ds.StudyInstanceUID) / ClassifierAnalyzer.name
    prefix = f"{ds.SOPInstanceUID}_{args.label}"
    sources = {"cam.png": folder / f"{prefix}_cam.png", "overlay.png": folder / f"{prefix}.png"}
    targets = [args.out / name for name in ("seen.png", *sources)]
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"{existing} exist; not overwriting")
    args.out.mkdir(parents=True, exist_ok=True)
    seen, _ = gradcam.preprocess(
        to_png(ds), read_params(Path("params.yaml")), Path("configs/classifier.yaml")
    )
    seen.save(args.out / "seen.png", format="PNG")
    for name, source in sources.items():
        shutil.copyfile(source, args.out / name)
    print(f"wrote {[str(path) for path in targets]} ({args.png.name}, {args.label})")


if __name__ == "__main__":
    main()
