"""A tiny synthetic NIH-like dataset: a zip of PNGs, split CSVs, and the preprocess outputs."""

import io
import random
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from src.data import preprocess, split
from src.data.eda import FINDINGS, NO_FINDING

IMAGE_SIZE = 32
PARAMS = {
    "image_size": IMAGE_SIZE,
    "shard_size": 5,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
# Patients per split, two images each; one test patient has an implausible age.
PATIENTS = {"train": 8, "val": 4, "test": 4}


@dataclass(frozen=True)
class Synthetic:
    zip_path: Path
    splits_dir: Path
    processed_dir: Path
    manifest: Path
    splits: dict[str, list[split.Entry]]


def png_bytes(seed: int, size: tuple[int, int] = (48, 40)) -> bytes:
    rng = random.Random(seed)
    image = Image.new("L", size)
    image.putdata([rng.randrange(256) for _ in range(size[0] * size[1])])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_splits() -> dict[str, list[split.Entry]]:
    rng = random.Random(0)
    splits: dict[str, list[split.Entry]] = {name: [] for name in split.SPLITS}
    patient = 0
    for name, count in PATIENTS.items():
        for _ in range(count):
            patient += 1
            age = 150 if (name == "test" and patient == 16) else rng.randrange(5, 95)
            for index in range(2):
                findings = [label for label in FINDINGS if rng.random() < 0.3]
                splits[name].append(
                    split.Entry(
                        image=f"{patient:08d}_{index:03d}.png",
                        patient=str(patient),
                        labels="|".join(findings) or NO_FINDING,
                        age=age,
                        sex="M" if patient % 2 else "F",
                        view="PA",
                    )
                )
    return splits


def write_dataset_zip(path: Path, images: list[str]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("Data_Entry_2017.csv", "not used\n")
        for number, image in enumerate(images):
            archive.writestr(f"images_{number % 2 + 1:03d}/images/{image}", png_bytes(number))


def build_synthetic(root: Path, resume_dir: Path | None = None) -> Synthetic:
    splits = make_splits()
    splits_dir = root / "splits"
    split.write_splits(splits, splits_dir)
    zip_path = root / "data.zip"
    write_dataset_zip(zip_path, [entry.image for name in split.SPLITS for entry in splits[name]])
    processed_dir = root / "processed"
    manifest = root / "manifests" / "nih.csv"
    preprocess.run_stage(
        zip_path, splits_dir, processed_dir, manifest, root / "work", PARAMS, resume_dir
    )
    return Synthetic(zip_path, splits_dir, processed_dir, manifest, splits)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> Synthetic:
    return build_synthetic(tmp_path_factory.mktemp("synthetic"))
