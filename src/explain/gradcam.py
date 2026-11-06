"""Grad-CAM for the DenseNet121 classifier: which image regions raised one label's score.

The target layer is the last dense block (`features.denseblock4`, 1024 channels at 7x7 for a
224px input). The map is the ReLU of the activations weighted by their spatial-mean gradients,
upsampled to the input size and scaled to [0, 1] (all zeros when nothing raises the score).

An image goes through exactly the evaluation path: `resize_png` from the preprocess stage, then
the test-split transform from src/models/data.py, with the mean and std in params.yaml. There is
no separate resize or normalize code here.

Writes `<image>_<label>.png` (the map over the 224px image the model saw) and `<image>_<label>.json`
(score, and the threshold and reliability from configs/viewer_thresholds.yaml). Existing files
are never overwritten. For education only, not for clinical use.

Run: python -m src.explain.gradcam --checkpoint epoch_003.pt --image 00000001_000.png \
         --label Cardiomegaly --out-dir docs/figures/gradcam
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from numpy.typing import NDArray
from PIL import Image
from torch import nn

from src.data.preprocess import read_params, resize_png
from src.models.checkpoints import load_checkpoint, save_new, write_text
from src.models.classifier import build_model
from src.models.config import load_config
from src.models.data import transform_for_split

NOTICE = "For education only, not for clinical use."
TARGET_LAYER = "features.denseblock4"
OVERLAY_ALPHA = 0.45


@dataclass(frozen=True)
class Explanation:
    cam: NDArray[np.float32]  # input height x width, values in [0, 1]
    score: float  # sigmoid of the label's logit; not a calibrated probability


def preprocess(
    png: bytes, params: dict[str, Any], config_path: Path
) -> tuple[Image.Image, torch.Tensor]:
    """The 224px grayscale image and the normalized 3-channel tensor, as in evaluation."""
    with Image.open(io.BytesIO(resize_png(png, params["image_size"]))) as image:
        image.load()
        seen = image.copy()
    augment = load_config(config_path).augment  # not used for the test split
    transform = transform_for_split("test", params["mean"], params["std"], augment)
    return seen, transform(seen).as_subclass(torch.Tensor)


def target_layer(model: nn.Module) -> nn.Module:
    return model.get_submodule(TARGET_LAYER)


def grad_cam(model: nn.Module, pixels: torch.Tensor, label_index: int) -> Explanation:
    """Grad-CAM of one label for one image tensor (3 x H x W)."""
    model.eval()
    store: dict[str, torch.Tensor] = {}

    def keep(_module: nn.Module, _inputs: object, output: torch.Tensor) -> None:
        store["activations"] = output
        output.register_hook(partial(store.__setitem__, "gradients"))  # type: ignore[no-untyped-call]

    handle = target_layer(model).register_forward_hook(keep)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logit = model(pixels.unsqueeze(0))[0, label_index]
            logit.backward()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
    activations, gradients = store["activations"].detach(), store["gradients"].detach()
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
    cam = nn.functional.interpolate(cam, size=pixels.shape[1:], mode="bilinear")[0, 0]
    peak = cam.max()
    cam = cam / peak if peak > 0 else torch.zeros_like(cam)
    return Explanation(cam.numpy().astype(np.float32), float(torch.sigmoid(logit.detach())))


def jet(values: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Blue (0) to red (1) colours for a map in [0, 1]."""
    channels = [np.clip(1.5 - np.abs(4 * values - centre), 0, 1) for centre in (3, 2, 1)]
    return (np.stack(channels, axis=-1) * 255).astype(np.uint8)


def overlay(image: Image.Image, cam: NDArray[np.float32]) -> Image.Image:
    gray = np.asarray(image.convert("RGB"), dtype=np.float32)
    if gray.shape[:2] != cam.shape:
        raise ValueError(f"image {gray.shape[:2]} and map {cam.shape} differ in size")
    blended = (1 - OVERLAY_ALPHA) * gray + OVERLAY_ALPHA * jet(cam).astype(np.float32)
    return Image.fromarray(blended.round().astype(np.uint8))


def cam_image(cam: NDArray[np.float32]) -> Image.Image:
    """The raw map as an 8-bit grey image (0-255 for 0-1), for a viewer to colour itself.

    Re-blending it with `overlay` gives the stored overlay within 1 grey level."""
    return Image.fromarray(np.round(np.clip(cam, 0, 1) * 255).astype(np.uint8), mode="L")


def cam_from_image(image: Image.Image) -> NDArray[np.float32]:
    values: NDArray[np.float32] = np.asarray(image.convert("L"), dtype=np.float32) / np.float32(255)
    return values


def viewer_entry(path: Path, label: str) -> dict[str, Any]:
    entry: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))["labels"][label]
    return entry


def explain_file(
    checkpoint: Path,
    image: Path,
    label: str,
    out_dir: Path,
    config_path: Path = Path("configs/classifier.yaml"),
    params_path: Path = Path("params.yaml"),
    viewer_path: Path = Path("configs/viewer_thresholds.yaml"),
) -> dict[str, Any]:
    state = load_checkpoint(checkpoint)
    labels: Sequence[str] = state["config"]["labels"]
    if label not in labels:
        raise ValueError(f"unknown label {label!r}; expected one of {list(labels)}")
    stem = f"{image.stem}_{label}"
    png_path, json_path = out_dir / f"{stem}.png", out_dir / f"{stem}.json"
    for path in (png_path, json_path):
        if path.exists():
            raise FileExistsError(f"{path} exists; not overwriting")

    model = build_model(len(labels), None)
    model.load_state_dict(state["model"])
    seen, pixels = preprocess(image.read_bytes(), read_params(params_path), config_path)
    result = grad_cam(model, pixels, list(labels).index(label))

    viewer = viewer_entry(viewer_path, label)
    threshold = viewer["threshold"]
    record = {
        "notice": NOTICE,
        "image": image.name,
        "label": label,
        "checkpoint": checkpoint.name,
        "checkpoint_epoch": state.get("epoch"),
        "score": result.score,
        "score_note": "sigmoid of the model output; not a calibrated probability",
        "threshold": threshold,
        "threshold_set": viewer["set"],
        "low_reliability": viewer["low_reliability"],
        "above_threshold": None if threshold is None else result.score >= threshold,
        "target_layer": TARGET_LAYER,
        "overlay": png_path.name,
    }
    save_new(png_path, lambda target: overlay(seen, result.cam).save(target, format="PNG"))
    save_new(json_path, partial(write_text, text=json.dumps(record, indent=2) + "\n"))
    print(f"wrote {png_path} and {json_path}")
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="an NIH PNG")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/gradcam"))
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    parser.add_argument("--viewer", type=Path, default=Path("configs/viewer_thresholds.yaml"))
    args = parser.parse_args(argv)
    explain_file(
        args.checkpoint, args.image, args.label, args.out_dir, args.config, args.params, args.viewer
    )


if __name__ == "__main__":
    main()
