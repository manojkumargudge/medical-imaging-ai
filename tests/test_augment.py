"""Augmentation is for train only: val and test must get the plain, deterministic pipeline."""

from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import IMAGE_SIZE, Synthetic
from PIL import Image
from torchvision.transforms import v2

from src.models import data
from src.models.config import AugmentConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
AUGMENT = load_config(REPO_ROOT / "configs" / "classifier.yaml").augment
RANDOM_TYPES = (v2.RandomAffine, v2.ColorJitter, v2.RandomHorizontalFlip, v2.RandomVerticalFlip)


def steps(transform: v2.Compose) -> list[object]:
    return list(transform.transforms)


def gradient_image() -> Image.Image:
    pixels = np.tile(np.arange(IMAGE_SIZE, dtype=np.uint8) * 7, (IMAGE_SIZE, 1))
    return Image.fromarray(pixels, mode="L")


def expected_plain(image: Image.Image) -> torch.Tensor:
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (pixels.expand(3, -1, -1) - mean) / std


@pytest.mark.parametrize("split", ["val", "test"])
def test_val_and_test_have_no_random_step(split: str) -> None:
    transform = data.transform_for_split(split, MEAN, STD, AUGMENT)
    assert not [step for step in steps(transform) if isinstance(step, RANDOM_TYPES)]


@pytest.mark.parametrize("split", ["val", "test"])
def test_val_and_test_output_is_only_scaled_and_normalized(split: str) -> None:
    transform = data.transform_for_split(split, MEAN, STD, AUGMENT)
    image = gradient_image()
    first, second = transform(image), transform(image)
    assert first.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.equal(first, second)
    assert torch.allclose(first, expected_plain(image), atol=1e-6)


def test_train_has_rotation_shift_and_brightness_contrast_but_no_flip() -> None:
    transform = data.transform_for_split("train", MEAN, STD, AUGMENT)
    kinds = [type(step) for step in steps(transform)]
    assert v2.RandomAffine in kinds and v2.ColorJitter in kinds
    assert v2.RandomHorizontalFlip not in kinds and v2.RandomVerticalFlip not in kinds
    affine = next(step for step in steps(transform) if isinstance(step, v2.RandomAffine))
    assert list(affine.degrees) == [-AUGMENT.rotation_degrees, AUGMENT.rotation_degrees]
    assert list(affine.translate) == [AUGMENT.translate, AUGMENT.translate]


def test_train_output_varies_and_keeps_shape() -> None:
    transform = data.transform_for_split("train", MEAN, STD, AUGMENT)
    torch.manual_seed(0)
    outputs = [transform(gradient_image()) for _ in range(4)]
    assert all(output.shape == (3, IMAGE_SIZE, IMAGE_SIZE) for output in outputs)
    assert not all(torch.equal(outputs[0], output) for output in outputs[1:])


def test_config_rejects_large_augmentation() -> None:
    with pytest.raises(ValueError, match="small"):
        load_config(
            REPO_ROOT / "configs" / "classifier.yaml",
            augment=AugmentConfig(
                rotation_degrees=45, translate=0.05, brightness=0.1, contrast=0.1
            ),
        )


def test_datasets_augment_train_only(synthetic: Synthetic, tmp_path: Path) -> None:
    config = load_config(
        REPO_ROOT / "configs" / "classifier.yaml",
        processed_dir=synthetic.processed_dir,
        manifest=synthetic.manifest,
        checkpoint_dir=tmp_path,
    )
    datasets = data.make_datasets(config)
    for split in ("val", "test"):
        dataset = datasets[split]
        assert len(dataset) > 0
        assert not [step for step in steps(dataset.transform) if isinstance(step, RANDOM_TYPES)]
        image, _ = dataset[0]
        assert torch.allclose(image, expected_plain(dataset.load_image(0)), atol=1e-6)
    train_steps = steps(datasets["train"].transform)
    assert any(isinstance(step, v2.RandomAffine) for step in train_steps)
