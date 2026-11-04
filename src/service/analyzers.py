"""Models the worker runs on a study: DenseNet121 + Grad-CAM for DX/CR, the 3D U-Net for CT.

Both reuse the evaluation code: the classifier's `resize_png` + test-split transform (through
src/explain/gradcam.py), and the U-Net's evaluation transforms with sliding-window inference.
CPU only. Output files go under `<results_dir>/<study UID>/<model>/` and are stored as paths
relative to results_dir. For education only, not for clinical use.
"""

from __future__ import annotations

import io
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import nibabel as nib
import numpy as np
import torch
import yaml
from monai.inferers import sliding_window_inference
from numpy.typing import NDArray
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydicom.pixels.processing import apply_voi_lut
from torch import nn

from src.data.preprocess import read_params, sha256_file
from src.dicom.convert import read_ct_series
from src.explain import gradcam
from src.models.checkpoints import load_checkpoint
from src.models.classifier import build_model
from src.models.segmentation import SegConfig, build_transforms, build_unet, load_seg_config


class Reader(Protocol):
    def get_json(self, path: str) -> Any: ...

    def get_bytes(self, path: str) -> bytes: ...


class Analyzer(Protocol):
    name: str
    modalities: tuple[str, ...]
    checkpoint_name: str
    checkpoint_sha256: str

    def analyze(
        self, orthanc: Reader, series: Sequence[Mapping[str, Any]], out_dir: Path, root: Path
    ) -> dict[str, Any]: ...


def to_png(ds: Dataset) -> bytes:
    """A grayscale DX/CR image as an 8-bit PNG, the format the classifier was trained on.

    8-bit MONOCHROME2 pixels (NIH images converted by src/dicom) are passed through unchanged.
    Other images get their VOI LUT or window, are scaled to 0-255 and inverted if MONOCHROME1;
    such images differ from the NIH training data.
    """
    pixels = ds.pixel_array
    if pixels.ndim != 2:
        raise ValueError(f"expected one grayscale frame, got shape {pixels.shape}")
    photometric = str(ds.PhotometricInterpretation)
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise ValueError(f"expected a grayscale image, got {photometric}")
    if pixels.dtype != np.uint8 or photometric == "MONOCHROME1":
        values = np.asarray(apply_voi_lut(pixels, ds), dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        scaled = (values - low) / (high - low) if high > low else np.zeros_like(values)
        if photometric == "MONOCHROME1":
            scaled = 1 - scaled
        pixels = np.round(scaled * 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class ClassifierAnalyzer:
    model: nn.Module
    labels: Sequence[str]
    viewer: Mapping[str, Mapping[str, Any]]
    params: dict[str, Any]
    config_path: Path
    checkpoint_name: str
    checkpoint_sha256: str
    name: str = "densenet121-chestxray14"
    modalities: tuple[str, ...] = ("DX", "CR")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        config_path: Path = Path("configs/classifier.yaml"),
        params_path: Path = Path("params.yaml"),
        viewer_path: Path = Path("configs/viewer_thresholds.yaml"),
    ) -> ClassifierAnalyzer:
        state = load_checkpoint(checkpoint)
        labels = list(state["config"]["labels"])
        model = build_model(len(labels), None)
        model.load_state_dict(state["model"])
        viewer = yaml.safe_load(viewer_path.read_text(encoding="utf-8"))["labels"]
        if sorted(viewer) != sorted(labels):
            raise ValueError("viewer_thresholds.yaml and the checkpoint have different labels")
        return cls(
            model.eval(),
            labels,
            viewer,
            read_params(params_path),
            config_path,
            checkpoint.name,
            sha256_file(checkpoint),
        )

    def analyze(
        self, orthanc: Reader, series: Sequence[Mapping[str, Any]], out_dir: Path, root: Path
    ) -> dict[str, Any]:
        instances = []
        for one in series:
            for instance in one["Instances"]:
                ds = dcmread(io.BytesIO(orthanc.get_bytes(f"/instances/{instance}/file")))
                instances.append(self._instance(ds, out_dir, root))
        return {"instances": instances}

    def _instance(self, ds: Dataset, out_dir: Path, root: Path) -> dict[str, Any]:
        seen, pixels = gradcam.preprocess(to_png(ds), self.params, self.config_path)
        sop = str(ds.SOPInstanceUID)
        labels: dict[str, Any] = {}
        for index, label in enumerate(self.labels):
            result = gradcam.grad_cam(self.model, pixels, index)
            path = out_dir / f"{sop}_{label}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            gradcam.overlay(seen, result.cam).save(path, format="PNG")
            entry = self.viewer[label]
            threshold = entry["threshold"]
            labels[label] = {
                "score": result.score,
                "threshold": threshold,
                "threshold_set": entry["set"],
                "low_reliability": entry["low_reliability"],
                "above_threshold": None if threshold is None else result.score >= threshold,
                "overlay": path.relative_to(root).as_posix(),
            }
        return {
            "sop_instance_uid": sop,
            "series_uid": str(ds.SeriesInstanceUID),
            "labels": labels,
            "score_note": "sigmoid of the model output; not a calibrated probability",
        }


@dataclass
class SegmentationAnalyzer:
    model: nn.Module
    config: SegConfig
    checkpoint_name: str
    checkpoint_sha256: str
    name: str = "monai-unet-spleen"
    modalities: tuple[str, ...] = ("CT",)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: Path, config_path: Path = Path("configs/segmentation.yaml")
    ) -> SegmentationAnalyzer:
        config = load_seg_config(config_path)
        state = load_checkpoint(checkpoint)
        if state["fingerprint"] != config.fingerprint():
            raise ValueError("the checkpoint was trained with another segmentation config")
        model = build_unet(config)
        model.load_state_dict(state["model"])
        return cls(model.eval(), config, checkpoint.name, sha256_file(checkpoint))

    def segment(self, nifti: Path) -> tuple[NDArray[np.uint8], NDArray[np.float64]]:
        """Spleen mask (1) and its affine, in the model's space: RAS, pixdim, body crop."""
        image = build_transforms(self.config, train=False, keys=("image",))({"image": str(nifti)})[
            "image"
        ]
        with torch.no_grad():
            logits = sliding_window_inference(
                image.unsqueeze(0),
                self.config.roi_size,
                self.config.sw_batch_size,
                self.model,
                overlap=self.config.overlap,
            )
        assert isinstance(logits, torch.Tensor)
        mask = logits.argmax(dim=1)[0].to(torch.uint8).numpy()
        return mask, np.asarray(image.affine, dtype=np.float64)

    def analyze(
        self, orthanc: Reader, series: Sequence[Mapping[str, Any]], out_dir: Path, root: Path
    ) -> dict[str, Any]:
        results = []
        for one in series:
            with tempfile.TemporaryDirectory() as folder:
                paths = []
                for number, instance in enumerate(one["Instances"]):
                    path = Path(folder) / f"{number:05d}.dcm"
                    path.write_bytes(orthanc.get_bytes(f"/instances/{instance}/file"))
                    paths.append(path)
                volume, affine = read_ct_series(paths)
                nifti = Path(folder) / "ct.nii.gz"
                nib.save(nib.Nifti1Image(volume.astype(np.float32), affine), nifti)
                mask, mask_affine = self.segment(nifti)
            series_uid = str(one["MainDicomTags"]["SeriesInstanceUID"])
            path = out_dir / f"{series_uid}_spleen_mask.nii.gz"
            path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(mask, mask_affine), path)
            voxels = int(mask.sum())
            results.append(
                {
                    "series_uid": series_uid,
                    "slices": len(paths),
                    "mask": path.relative_to(root).as_posix(),
                    "mask_space": (
                        f"RAS, {self.config.pixdim} mm, cropped to the body (the model's input "
                        "space); the NIfTI affine maps it to patient coordinates"
                    ),
                    "spleen_voxels": voxels,
                    "spleen_ml": voxels * float(np.prod(self.config.pixdim)) / 1000,
                }
            )
        return {"series": results}
