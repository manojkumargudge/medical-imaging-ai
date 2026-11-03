"""3D U-Net pipeline on tiny synthetic CT volumes, CPU only."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import make_spleen_task
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.transforms import (
    RandAffined,
    RandCropByPosNegLabeld,
    Randomizable,
    RandShiftIntensityd,
)

from src.data import spleen
from src.models import segmentation
from src.models.checkpoints import best_epoch
from src.models.segmentation import SegConfig, load_seg_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "segmentation.yaml"
ROI = (16, 16, 16)


@pytest.fixture(scope="module")
def task(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("spleen")
    task_dir = make_spleen_task(root, cases=8, shape=(40, 40, 24))
    splits = spleen.split_cases(spleen.load_cases(task_dir, expected=8), 0.25, 0.25, 42)
    spleen.write_split(splits, root / "split.csv")
    return task_dir, root / "split.csv"


def tiny_config(task: tuple[Path, Path], checkpoint_dir: Path, **overrides: object) -> SegConfig:
    values: dict[str, object] = {
        "task_dir": task[0],
        "split": task[1],
        "checkpoint_dir": checkpoint_dir,
        "num_workers": 0,
        "channels": (4, 8, 16),
        "strides": (2, 2),
        "num_res_units": 1,
        "roi_size": ROI,
        "samples_per_volume": 2,
        "max_epochs": 2,
        "val_interval": 1,
        "sw_batch_size": 2,
        **overrides,
    }
    return load_seg_config(CONFIG, **values)


def test_fixed_config() -> None:
    config = load_seg_config(CONFIG)
    assert config.spatial_dims == 3
    assert config.pixdim == (1.5, 1.5, 1.5)
    assert config.hu_window == (-57.0, 164.0)
    assert config.roi_size == (96, 96, 96)
    assert (config.loss, config.selection_metric) == ("dice_ce", "val_mean_dice")
    assert str(config.checkpoint_dir).replace("\\", "/").startswith("/content/drive/MyDrive/")
    assert isinstance(segmentation.build_loss(), DiceCELoss)


def test_config_must_be_3d() -> None:
    with pytest.raises(ValueError, match="3D"):
        load_seg_config(CONFIG, spatial_dims=2)


def test_only_train_has_random_transforms() -> None:
    config = load_seg_config(CONFIG)
    train = segmentation.build_transforms(config, train=True).transforms
    evaluation = segmentation.build_transforms(config, train=False).transforms
    assert not [step for step in evaluation if isinstance(step, Randomizable)]
    kinds = {type(step) for step in train}
    assert {RandCropByPosNegLabeld, RandAffined, RandShiftIntensityd} <= kinds
    assert not [step for step in train if "Flip" in type(step).__name__]
    assert not [step for step in train if "Rotate90" in type(step).__name__]


def test_eval_preprocessing_resamples_windows_and_reorients(task: tuple[Path, Path]) -> None:
    config = tiny_config(task, Path("unused"))
    item = segmentation.data_dicts(config, "val")[0]
    first = segmentation.build_transforms(config, train=False)(item)
    second = segmentation.build_transforms(config, train=False)(item)
    image = first["image"]
    assert torch.equal(image, second["image"])
    affine = image.affine.numpy()
    assert np.allclose(np.linalg.norm(affine[:3, :3], axis=0), 1.5, atol=1e-4)
    assert np.all(np.diag(affine[:3, :3]) > 0)  # RAS
    assert float(image.min()) >= 0 and float(image.max()) <= 1
    assert image.shape[1:] == first["label"].shape[1:]
    assert set(np.unique(first["label"].numpy())) <= {0.0, 1.0}


def test_train_transform_gives_roi_crops(task: tuple[Path, Path]) -> None:
    config = tiny_config(task, Path("unused"))
    item = segmentation.data_dicts(config, "train")[0]
    crops = segmentation.build_transforms(config, train=True)(item)
    assert len(crops) == config.samples_per_volume
    assert all(tuple(crop["image"].shape) == (1, *ROI) for crop in crops)


def test_sliding_window_covers_the_whole_volume(task: tuple[Path, Path]) -> None:
    config = tiny_config(task, Path("unused"))
    model = segmentation.build_unet(config).eval()
    volume = torch.rand(1, 1, 21, 23, 40)
    with torch.no_grad():
        logits = sliding_window_inference(volume, ROI, 2, model, overlap=0.25)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (1, 2, 21, 23, 40)


def test_training_checkpoints_resume_and_evaluation(
    task: tuple[Path, Path], tmp_path: Path
) -> None:
    checkpoints = tmp_path / "checkpoints"
    config = tiny_config(task, checkpoints, max_epochs=1)
    cpu = torch.device("cpu")
    first = segmentation.train_segmentation(config, device=cpu)
    assert [entry["epoch"] for entry in first["history"]] == [1]
    saved = (checkpoints / "epoch_001.pt").read_bytes()

    second = segmentation.train_segmentation(replace(config, max_epochs=2), device=cpu)
    assert [entry["epoch"] for entry in second["history"]] == [1, 2]
    assert (checkpoints / "epoch_001.pt").read_bytes() == saved
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "epoch_001.json",
        "epoch_001.pt",
        "epoch_002.json",
        "epoch_002.pt",
    ]
    entry = second["history"][-1]
    assert entry["val_cases"] == 2
    assert all(0 <= case["dice"] <= 1 for case in entry["val_dice"])

    with pytest.raises(ValueError, match="another config"):
        segmentation.train_segmentation(replace(config, lr=1e-3, max_epochs=3), device=cpu)

    report = segmentation.evaluate_segmentation(replace(config, max_epochs=2), device=cpu)
    assert report["checkpoint_epoch"] == second["best"]["epoch"]
    assert {split: len(result["cases"]) for split, result in report["splits"].items()} == {
        "val": 2,
        "test": 2,
    }


def test_checkpoints_only_at_val_epochs(task: tuple[Path, Path], tmp_path: Path) -> None:
    config = tiny_config(task, tmp_path, max_epochs=3, val_interval=2)
    result = segmentation.train_segmentation(config, device=torch.device("cpu"))
    # Epoch 2 is a val epoch, epoch 3 the last one.
    assert [entry["epoch"] for entry in result["history"]] == [2, 3]
    assert sorted(path.name for path in tmp_path.glob("*.pt")) == ["epoch_002.pt", "epoch_003.pt"]


def test_best_is_highest_val_dice() -> None:
    history = [
        {"epoch": 10, "val_mean_dice": 0.81},
        {"epoch": 20, "val_mean_dice": 0.90},
        {"epoch": 30, "val_mean_dice": 0.88},
    ]
    assert best_epoch(history, "val_mean_dice")["epoch"] == 20


def test_task_dir_on_drive_is_rejected(task: tuple[Path, Path], tmp_path: Path) -> None:
    config = tiny_config(task, tmp_path, task_dir=Path("/content/drive/MyDrive/Task09_Spleen"))
    with pytest.raises(ValueError, match="Google Drive"):
        segmentation.make_loaders(config, ("val",))
