"""Grad-CAM on an untrained DenseNet121 (CPU, offline): shape, range, sanity check, determinism."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.data.eda import FINDINGS
from src.data.preprocess import read_params, resize_png
from src.explain import gradcam
from src.models.classifier import build_model
from src.models.config import load_config
from src.models.data import transform_for_split

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "classifier.yaml"
PARAMS = REPO_ROOT / "params.yaml"
VIEWER = REPO_ROOT / "configs" / "viewer_thresholds.yaml"


def seeded_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return build_model(len(FINDINGS), None).eval()


def png_bytes(size: tuple[int, int] = (300, 260), seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (size[1], size[0]), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def test_preprocessing_is_the_evaluation_path() -> None:
    params = read_params(PARAMS)
    seen, pixels = gradcam.preprocess(png_bytes(), params, CONFIG)
    with Image.open(io.BytesIO(resize_png(png_bytes(), params["image_size"]))) as image:
        expected = transform_for_split(
            "test", params["mean"], params["std"], load_config(CONFIG).augment
        )(image.copy())
    assert seen.size == (224, 224) and seen.mode == "L"
    assert pixels.shape == (3, 224, 224)
    assert torch.equal(pixels, expected.as_subclass(torch.Tensor))


@pytest.mark.parametrize("size", [(64, 96), (224, 224)])
def test_map_has_the_input_size_and_values_in_0_1(size: tuple[int, int]) -> None:
    torch.manual_seed(1)
    result = gradcam.grad_cam(seeded_model(), torch.randn(3, *size), FINDINGS.index("Effusion"))
    assert result.cam.shape == size
    assert result.cam.min() >= 0 and result.cam.max() <= 1
    assert 0 < result.score < 1


def test_map_is_deterministic_and_leaves_no_gradients() -> None:
    model = seeded_model()
    pixels = gradcam.preprocess(png_bytes(), read_params(PARAMS), CONFIG)[1]
    first = gradcam.grad_cam(model, pixels, 1)
    second = gradcam.grad_cam(model, pixels, 1)
    assert np.array_equal(first.cam, second.cam) and first.score == second.score
    assert all(parameter.grad is None for parameter in model.parameters())


def randomize(model: nn.Module, seed: int) -> nn.Module:
    """Cascading randomization from the top: the classifier and the last dense block."""
    torch.manual_seed(seed)
    for name in ("classifier", gradcam.TARGET_LAYER):
        for module in model.get_submodule(name).modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                module.reset_parameters()
                if isinstance(module, nn.BatchNorm2d):
                    nn.init.uniform_(module.weight, 0.5, 1.5)
    return model


def test_map_changes_when_weights_are_randomized() -> None:
    # Sanity check (Adebayo et al., 2018): a map that survives randomizing the weights does not
    # explain the model.
    pixels = gradcam.preprocess(png_bytes(seed=3), read_params(PARAMS), CONFIG)[1]
    label = FINDINGS.index("Cardiomegaly")
    original = gradcam.grad_cam(seeded_model(), pixels, label).cam
    randomized = gradcam.grad_cam(randomize(seeded_model(), 99), pixels, label).cam
    correlation = float(np.corrcoef(original.ravel(), randomized.ravel())[0, 1])
    print(f"pixels {original.size} vs {randomized.size}, correlation {correlation:.3f}")
    assert original.size and randomized.size
    assert not np.allclose(original, randomized, atol=0.05)
    assert correlation < 0.9


def test_flat_map_is_all_zeros() -> None:
    class Constant(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential()
            self.features.add_module("denseblock4", nn.Conv2d(3, 2, 1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features: torch.Tensor = self.features(x)
            return features.mean(dim=(2, 3)) * 0

    result = gradcam.grad_cam(Constant(), torch.randn(3, 8, 8), 0)
    assert result.cam.shape == (8, 8) and not result.cam.any()


def test_overlay_and_colours() -> None:
    cam = np.linspace(0, 1, 224 * 224, dtype=np.float32).reshape(224, 224)
    image = gradcam.overlay(Image.new("L", (224, 224), 128), cam)
    assert image.size == (224, 224) and image.mode == "RGB"
    colours = gradcam.jet(np.array([0.0, 1.0], dtype=np.float32))
    assert colours[0].argmax() == 2 and colours[1].argmax() == 0  # blue, then red
    with pytest.raises(ValueError, match="differ in size"):
        gradcam.overlay(Image.new("L", (100, 100)), cam)


def test_cli_writes_overlay_and_json(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch_001.pt"
    state = {"epoch": 1, "model": seeded_model().state_dict(), "config": {"labels": FINDINGS}}
    torch.save(state, checkpoint)
    image = tmp_path / "00000001_000.png"
    image.write_bytes(png_bytes((1024, 1024)))
    out = tmp_path / "out"
    args = ["--checkpoint", str(checkpoint), "--image", str(image), "--out-dir", str(out)]
    args += ["--config", str(CONFIG), "--params", str(PARAMS), "--viewer", str(VIEWER)]

    gradcam.main([*args, "--label", "Cardiomegaly"])
    record = json.loads((out / "00000001_000_Cardiomegaly.json").read_text())
    with Image.open(out / "00000001_000_Cardiomegaly.png") as overlay:
        assert overlay.size == (224, 224)
    assert record["notice"] == "For education only, not for clinical use."
    assert record["threshold_set"] == "sensitivity_80" and record["low_reliability"] is False
    assert record["above_threshold"] == (record["score"] >= record["threshold"])

    gradcam.main([*args, "--label", "Hernia"])
    hernia = json.loads((out / "00000001_000_Hernia.json").read_text())
    assert hernia["threshold"] is None and hernia["above_threshold"] is None

    gradcam.main([*args, "--label", "Nodule"])
    assert json.loads((out / "00000001_000_Nodule.json").read_text())["low_reliability"] is True

    with pytest.raises(FileExistsError):
        gradcam.main([*args, "--label", "Cardiomegaly"])
    with pytest.raises(ValueError, match="unknown label"):
        gradcam.main([*args, "--label", "Fracture"])
