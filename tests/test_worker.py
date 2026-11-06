"""The inference worker with fake models and an in-memory Orthanc (CPU, offline)."""

import io
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import nibabel as nib
import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset
from torch import nn

from src.data.eda import FINDINGS
from src.data.preprocess import read_params
from src.dicom import convert
from src.dicom.render import to_png
from src.explain import gradcam
from src.models.segmentation import load_seg_config
from src.service.analyzers import ClassifierAnalyzer, SegmentationAnalyzer
from src.service.orthanc import OrthancError
from src.service.store import MemoryStore
from src.service.worker import Worker

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "classifier.yaml"
VIEWER = REPO_ROOT / "configs" / "viewer_thresholds.yaml"


class FakeClassifier(nn.Module):
    """Has `features.denseblock4` like DenseNet121, so Grad-CAM finds its target layer."""

    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.features = nn.Sequential()
        self.features.add_module("denseblock4", nn.Conv2d(3, 4, 3, stride=8, padding=1))
        self.head = nn.Linear(4, len(FINDINGS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features: torch.Tensor = torch.relu(self.features(x))
        logits: torch.Tensor = self.head(features.mean(dim=(2, 3)))
        return logits


def classifier() -> ClassifierAnalyzer:
    viewer = yaml.safe_load(VIEWER.read_text(encoding="utf-8"))["labels"]
    params = read_params(REPO_ROOT / "params.yaml")
    return ClassifierAnalyzer(
        FakeClassifier().eval(), FINDINGS, viewer, params, CONFIG, "fake.pt", "sha-classifier"
    )


def segmenter() -> SegmentationAnalyzer:
    torch.manual_seed(0)
    config = load_seg_config(REPO_ROOT / "configs" / "segmentation.yaml", roi_size=(16, 16, 16))
    return SegmentationAnalyzer(nn.Conv3d(1, 2, 1).eval(), config, "fake-seg.pt", "sha-seg")


@dataclass
class FakeOrthanc:
    """The part of Orthanc's REST API the worker reads."""

    files: dict[str, bytes] = field(default_factory=dict)
    studies: dict[str, list[str]] = field(default_factory=dict)  # study -> instance ids
    changes: list[dict[str, Any]] = field(default_factory=list)
    requests: list[str] = field(default_factory=list)

    def add_study(self, study_id: str, datasets: Sequence[Dataset]) -> None:
        for number, ds in enumerate(datasets):
            buffer = io.BytesIO()
            ds.save_as(buffer, enforce_file_format=True)
            self.files[f"{study_id}-{number}"] = buffer.getvalue()
        self.studies[study_id] = [f"{study_id}-{n}" for n in range(len(datasets))]
        self.change("StableStudy", study_id)

    def change(self, kind: str, resource: str) -> None:
        self.changes.append({"Seq": len(self.changes) + 1, "ChangeType": kind, "ID": resource})

    def _ds(self, instance: str) -> Dataset:
        return dcmread(io.BytesIO(self.files[instance]))

    def get_json(self, path: str) -> Any:
        self.requests.append(path)
        url = urlparse(path)
        parts = url.path.strip("/").split("/")
        if parts == ["changes"]:
            query = parse_qs(url.query)
            since, limit = int(query["since"][0]), int(query["limit"][0])
            batch = [c for c in self.changes if c["Seq"] > since][:limit]
            return {"Changes": batch, "Done": True, "Last": batch[-1]["Seq"] if batch else since}
        if parts[1] not in self.studies:
            raise OrthancError(404, path, "Unknown resource")
        first = self._ds(self.studies[parts[1]][0])
        if len(parts) == 2:
            return {"MainDicomTags": {"StudyInstanceUID": str(first.StudyInstanceUID)}}
        series: dict[str, dict[str, Any]] = {}
        for instance in self.studies[parts[1]]:
            ds = self._ds(instance)
            entry = series.setdefault(
                str(ds.SeriesInstanceUID),
                {
                    "MainDicomTags": {
                        "Modality": str(ds.Modality),
                        "SeriesInstanceUID": str(ds.SeriesInstanceUID),
                    },
                    "Instances": [],
                },
            )
            entry["Instances"].append(instance)
        return list(series.values())

    def get_bytes(self, path: str) -> bytes:
        return self.files[path.split("/")[2]]


def png(seed: int = 0, size: int = 64) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 256, (size, size), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def dx(seed: int = 0) -> Dataset:
    return convert.png_to_dx(png(seed), f"p{seed}", source=f"img{seed}.png")


def ct_series(tmp_path: Path, name: str = "c1") -> list[Dataset]:
    volume = np.full((24, 20, 6), -1000, dtype=np.int16)
    volume[6:18, 5:15, 1:5] = 60  # a soft-tissue block inside air
    nib.save(nib.Nifti1Image(volume, np.diag([2.0, 2.0, 3.0, 1.0])), tmp_path / f"{name}.nii.gz")
    paths = convert.nifti_to_ct_series(tmp_path / f"{name}.nii.gz", tmp_path / name, name)
    return [dcmread(path) for path in paths]


def make_worker(orthanc: FakeOrthanc, store: MemoryStore, root: Path) -> Worker:
    return Worker(orthanc, store, [classifier(), segmenter()], root, log=lambda _: None)


def test_dx_study_gets_scores_thresholds_and_overlays(tmp_path: Path) -> None:
    orthanc, store = FakeOrthanc(), MemoryStore()
    orthanc.add_study("s1", [dx(1)])
    worker = make_worker(orthanc, store, tmp_path / "results")
    assert worker.poll_once() == 1
    [row] = store.rows
    assert (row["model"], row["status"], row["modality"]) == ("densenet121-chestxray14", "ok", "DX")
    assert row["checkpoint_sha256"] == "sha-classifier" and row["created_at"]
    [instance] = row["results"]["instances"]
    viewer = yaml.safe_load(VIEWER.read_text(encoding="utf-8"))["labels"]
    assert list(instance["labels"]) == list(FINDINGS)
    for label, entry in instance["labels"].items():
        assert 0 < entry["score"] < 1
        assert entry["threshold"] == viewer[label]["threshold"]
        assert entry["low_reliability"] == viewer[label]["low_reliability"]
        if entry["threshold"] is None:
            assert entry["above_threshold"] is None
        else:
            assert entry["above_threshold"] == (entry["score"] >= entry["threshold"])
        with Image.open(tmp_path / "results" / entry["overlay"]) as overlay:
            assert overlay.size == (224, 224)
        with Image.open(tmp_path / "results" / entry["cam"]) as cam:
            assert (cam.size, cam.mode) == ((224, 224), "L")
        assert not Path(entry["overlay"]).is_absolute() and not Path(entry["cam"]).is_absolute()


def test_reblending_the_raw_cam_gives_the_stored_overlay(tmp_path: Path) -> None:
    orthanc, store = FakeOrthanc(), MemoryStore()
    ds = dx(13)
    orthanc.add_study("s1", [ds])
    analyzer = classifier()
    Worker(orthanc, store, [analyzer], tmp_path, log=lambda _: None).poll_once()
    seen, _ = gradcam.preprocess(to_png(ds), analyzer.params, CONFIG)
    labels = store.rows[0]["results"]["instances"][0]["labels"]
    worst = 0
    for entry in labels.values():
        with (
            Image.open(tmp_path / entry["cam"]) as cam,
            Image.open(tmp_path / entry["overlay"]) as stored,
        ):
            reblended = gradcam.overlay(seen, gradcam.cam_from_image(cam))
            difference = np.abs(np.asarray(reblended, int) - np.asarray(stored, int)).max()
        worst = max(worst, int(difference))
    print(f"{len(labels)} labels, re-blended vs stored overlay: max pixel diff {worst}")
    assert len(labels) == len(FINDINGS) and worst <= 1


def test_the_raw_cam_keeps_the_map_within_one_grey_level() -> None:
    cam = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(224, 224)
    back = gradcam.cam_from_image(gradcam.cam_image(cam))
    assert np.abs(back - cam).max() <= 0.5 / 255 + 1e-7
    assert gradcam.cam_image(np.full((2, 2), 1.5, np.float32)).getextrema() == (255, 255)


def test_dx_scores_equal_the_gradcam_cli_path(tmp_path: Path) -> None:
    orthanc, store = FakeOrthanc(), MemoryStore()
    orthanc.add_study("s1", [dx(2)])
    analyzer = classifier()
    Worker(orthanc, store, [analyzer], tmp_path, log=lambda _: None).poll_once()
    scores = store.rows[0]["results"]["instances"][0]["labels"]
    _, pixels = gradcam.preprocess(png(2), analyzer.params, CONFIG)
    for index, label in enumerate(FINDINGS):
        assert scores[label]["score"] == gradcam.grad_cam(analyzer.model, pixels, index).score


def test_ct_study_gets_a_mask_in_model_space(tmp_path: Path) -> None:
    orthanc, store = FakeOrthanc(), MemoryStore()
    orthanc.add_study("ct", ct_series(tmp_path))
    make_worker(orthanc, store, tmp_path / "results").poll_once()
    [row] = store.rows
    assert (row["model"], row["status"], row["modality"]) == ("monai-unet-spleen", "ok", "CT")
    [series] = row["results"]["series"]
    mask = nib.load(tmp_path / "results" / series["mask"])
    assert series["slices"] == 6
    assert np.allclose(np.linalg.norm(mask.affine[:3, :3], axis=0), 1.5, atol=1e-4)
    assert set(np.unique(np.asarray(mask.dataobj))) <= {0, 1}
    assert series["spleen_voxels"] == int(np.asarray(mask.dataobj).sum())


def test_each_modality_goes_to_its_model_and_others_are_ignored(tmp_path: Path) -> None:
    orthanc, store = FakeOrthanc(), MemoryStore()
    orthanc.add_study("both", [dx(3), *ct_series(tmp_path)])
    mr = dx(4)
    mr.Modality = "MR"
    orthanc.add_study("mr", [mr])
    make_worker(orthanc, store, tmp_path / "results").poll_once()
    assert sorted((row["model"], row["modality"]) for row in store.rows) == [
        ("densenet121-chestxray14", "DX"),
        ("monai-unet-spleen", "CT"),
    ]


def test_reruns_do_not_duplicate_and_the_seq_is_kept(tmp_path: Path) -> None:
    orthanc, store = FakeOrthanc(), MemoryStore()
    orthanc.add_study("s1", [dx(5)])
    orthanc.change("NewInstance", "x")
    worker = make_worker(orthanc, store, tmp_path)
    worker.poll_once()
    first = [dict(row) for row in store.rows]
    assert store.last_seq() == 2

    # Orthanc reports the same study stable again; a restarted worker replays the whole log.
    orthanc.change("StableStudy", "s1")
    assert make_worker(orthanc, store, tmp_path).process_study("s1") == ["skipped"]
    store.set_last_seq(0)
    make_worker(orthanc, store, tmp_path).poll_once()
    print(f"rows before {len(first)}, after {len(store.rows)}")
    assert first and store.rows and store.rows == first
    assert store.last_seq() == 3

    # Only new changes are read after that.
    orthanc.add_study("s2", [dx(6)])
    orthanc.requests.clear()
    worker.poll_once()
    assert orthanc.requests[0] == "/changes?since=3&limit=100"
    assert len(store.rows) == 2


def test_a_failing_study_is_stored_as_an_error_and_the_worker_moves_on(tmp_path: Path) -> None:
    orthanc, store = FakeOrthanc(), MemoryStore()
    broken = dx(7)
    broken.PhotometricInterpretation = "RGB"
    orthanc.add_study("bad", [broken])
    orthanc.add_study("good", [dx(8)])
    make_worker(orthanc, store, tmp_path).poll_once()
    statuses = [row["status"] for row in store.rows]
    assert statuses == ["error", "ok"]
    assert "grayscale" in store.rows[0]["results"]["error"]
    assert store.last_seq() == 2


def test_8_bit_monochrome2_pixels_pass_through_unchanged() -> None:
    ds = dx(9)
    with Image.open(io.BytesIO(to_png(ds))) as image:
        assert np.array_equal(np.asarray(image), ds.pixel_array)


def test_16_bit_monochrome1_is_scaled_and_inverted() -> None:
    ds = dx(10)
    pixels = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    ds.Rows, ds.Columns = pixels.shape
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.PixelData = pixels.tobytes()
    with Image.open(io.BytesIO(to_png(ds))) as image:
        assert np.asarray(image).tolist() == [[255, 191], [128, 0]]


def test_worker_needs_a_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no model"):
        Worker(FakeOrthanc(), MemoryStore(), [], tmp_path)


def test_checkpoints_stay_out_of_the_image_and_git() -> None:
    ignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert ignore[0] == "*" and "**/*.pt" in ignore  # allow-list, and never a checkpoint
    copied = [
        line.split()[1]
        for line in (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
        if line.startswith("COPY")
    ]
    assert set(copied) <= {"pyproject.toml", "params.yaml", "src", "configs"}
    compose: Mapping[str, Any] = yaml.safe_load(
        (REPO_ROOT / "docker-compose.worker.yml").read_text(encoding="utf-8")
    )
    volumes = compose["services"]["worker"]["volumes"]
    checkpoints = [volume for volume in volumes if "/checkpoints/" in volume]
    assert len(checkpoints) == 2 and all(volume.endswith(":ro") for volume in checkpoints)
    tracked = subprocess.run(
        ["git", "ls-files", "*.pt", "*.pth"], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert tracked.stdout == ""


def test_a_deleted_study_is_skipped_and_the_seq_advances(tmp_path: Path) -> None:
    orthanc, store = FakeOrthanc(), MemoryStore()
    orthanc.add_study("gone", [dx(11)])
    orthanc.add_study("kept", [dx(12)])
    del orthanc.studies["gone"]  # deleted after it became stable
    worker = make_worker(orthanc, store, tmp_path)
    assert worker.process_study("gone") == ["gone"]
    worker.poll_once()
    assert [row["orthanc_study_id"] for row in store.rows] == ["kept"]
    assert store.last_seq() == 2
