"""Config, model, training with resume and evaluation on tiny synthetic data (CPU, offline)."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from conftest import Synthetic

from src.data.eda import FINDINGS
from src.models import evaluate, train
from src.models.checkpoints import load_checkpoint, save_new
from src.models.classifier import build_loss, build_model
from src.models.config import ClassifierConfig, load_config
from src.models.data import make_datasets, read_manifest

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "classifier.yaml"


def tiny_config(
    synthetic: Synthetic, checkpoint_dir: Path, **overrides: object
) -> ClassifierConfig:
    values: dict[str, object] = {
        "processed_dir": synthetic.processed_dir,
        "manifest": synthetic.manifest,
        "checkpoint_dir": checkpoint_dir,
        "weights": None,
        "batch_size": 4,
        "num_workers": 0,
        "max_epochs": 2,
        **overrides,
    }
    return load_config(CONFIG, **values)


def test_fixed_config() -> None:
    config = load_config(CONFIG)
    assert config.architecture == "densenet121"
    assert config.weights == "IMAGENET1K_V1"
    assert config.labels == FINDINGS and len(config.labels) == 14
    assert config.loss == "bce_with_logits"
    assert (config.batch_size, config.optimizer, config.lr) == (64, "adamw", 1e-4)
    assert (config.max_epochs, config.amp) == (5, True)
    assert config.selection_metric == "val_mean_auroc"
    assert str(config.checkpoint_dir).replace("\\", "/").startswith("/content/drive/MyDrive/")


def test_config_rejects_other_labels() -> None:
    with pytest.raises(ValueError, match="14 NIH findings"):
        load_config(CONFIG, labels=FINDINGS[:13])


def test_model_has_14_logits_and_bce_loss() -> None:
    model = build_model(14, None).eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 3, 32, 32))
    assert logits.shape == (2, 14)
    assert isinstance(build_loss(), torch.nn.BCEWithLogitsLoss)


def test_manifest_targets_are_14_findings(synthetic: Synthetic) -> None:
    samples = read_manifest(synthetic.manifest, FINDINGS)
    sample = samples["train"][0]
    entry = synthetic.splits["train"][0]
    assert len(sample.targets) == 14
    assert {FINDINGS[i] for i, value in enumerate(sample.targets) if value} == (
        set(entry.labels.split("|")) - {"No Finding"}
    )


def test_manifest_with_a_patient_in_two_splits_fails(synthetic: Synthetic, tmp_path: Path) -> None:
    lines = synthetic.manifest.read_text().splitlines()
    header, rows = lines[0], lines[1:]
    val_row = next(row for row in rows if ",val," in row)
    leaked = val_row.replace(",val,", ",train,").replace(val_row.split(",")[0], "leak.png")
    path = tmp_path / "manifest.csv"
    path.write_text("\n".join([header, *rows, leaked]) + "\n")
    with pytest.raises(ValueError, match="patients are in both"):
        read_manifest(path, FINDINGS)


def test_training_checkpoints_every_epoch_and_resumes(synthetic: Synthetic, tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    config = tiny_config(synthetic, checkpoints, max_epochs=1)
    first = train.train(config, device=torch.device("cpu"))
    assert [entry["epoch"] for entry in first["history"]] == [1]
    assert (checkpoints / "epoch_001.pt").exists() and (checkpoints / "epoch_001.json").exists()
    saved = (checkpoints / "epoch_001.pt").read_bytes()

    # A new session with a higher epoch limit continues from epoch 2 and keeps epoch 1 as it was.
    second = train.train(replace(config, max_epochs=2), device=torch.device("cpu"))
    assert [entry["epoch"] for entry in second["history"]] == [1, 2]
    assert second["history"][0] == first["history"][0]
    assert (checkpoints / "epoch_001.pt").read_bytes() == saved
    assert sorted(path.name for path in checkpoints.iterdir()) == [
        "epoch_001.json",
        "epoch_001.pt",
        "epoch_002.json",
        "epoch_002.pt",
    ]
    state = load_checkpoint(checkpoints / "epoch_002.pt")
    assert state["epoch"] == 2 and state["fingerprint"] == config.fingerprint()
    assert set(state) >= {"model", "optimizer", "scaler", "history", "config"}

    # Nothing left to do: no new files.
    third = train.train(replace(config, max_epochs=2), device=torch.device("cpu"))
    assert len(third["history"]) == 2
    assert len(list(checkpoints.iterdir())) == 4


def test_resume_with_another_config_fails(synthetic: Synthetic, tmp_path: Path) -> None:
    config = tiny_config(synthetic, tmp_path, max_epochs=1)
    train.train(config, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="another config"):
        train.train(replace(config, lr=1e-3, max_epochs=2), device=torch.device("cpu"))


def test_best_epoch_is_highest_val_mean_auroc() -> None:
    history = [
        {"epoch": 1, "val_mean_auroc": 0.70},
        {"epoch": 2, "val_mean_auroc": 0.74},
        {"epoch": 3, "val_mean_auroc": 0.74},
        {"epoch": 4, "val_mean_auroc": 0.72},
    ]
    assert train.best_epoch(history)["epoch"] == 2


def test_existing_checkpoint_is_never_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "epoch_001.pt"
    path.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        save_new(path, lambda target: target.write_bytes(b"new"))
    assert path.read_bytes() == b"keep"


def test_evaluation_writes_reports(synthetic: Synthetic, tmp_path: Path) -> None:
    config = tiny_config(synthetic, tmp_path / "checkpoints")
    train.train(config, device=torch.device("cpu"))
    report = evaluate.evaluate(config, device=torch.device("cpu"), meta={"repo_commit": "test"})

    assert report["dataset"].startswith("NIH ChestX-ray14")
    assert report["thresholds"]["selected_on"] == "val"
    assert report["checkpoint_epoch"] in (1, 2)
    for split in ("val", "test"):
        result = report["splits"][split]
        assert result["images"] == len(synthetic.splits[split])
        assert [row["label"] for row in result["labels"]] == list(FINDINGS)
        assert all(
            row["positives"] + row["negatives"] == result["images"] for row in result["labels"]
        )
        assert set(result["by_sex"]) == {"F", "M"}
    # The test fixture has one patient (two images) aged 150: counted, never scored.
    invalid = report["splits"]["test"]["by_age_band"]["invalid (age > 100)"]
    assert invalid == {"images": 2, "excluded": True}

    metrics, thresholds, results = (tmp_path / name for name in ("m.json", "t.json", "r.md"))
    evaluate.write_outputs(report, metrics, thresholds, results)
    assert json.loads(thresholds.read_text())["selected_on"] == "val"
    text = results.read_text()
    assert "For education only, not for clinical use." in text
    assert "test (official test_list.txt)" in text
    with pytest.raises(FileExistsError):
        evaluate.write_outputs(report, metrics, thresholds, results)
    evaluate.write_outputs(report, metrics, thresholds, results, overwrite=True)


def test_thresholds_come_from_val_only(synthetic: Synthetic) -> None:
    import numpy as np

    samples = read_manifest(synthetic.manifest, FINDINGS)
    rng = np.random.default_rng(0)

    def fake(split: str) -> evaluate.SplitPredictions:
        targets = np.array([sample.targets for sample in samples[split]], dtype=float)
        return evaluate.SplitPredictions(samples[split], targets, rng.random(targets.shape))

    val, test = fake("val"), fake("test")
    report = evaluate.build_report(FINDINGS, {"val": val, "test": test}, 1, {})
    changed = evaluate.build_report(
        FINDINGS, {"val": val, "test": replace(test, scores=1 - test.scores)}, 1, {}
    )
    assert report["thresholds"] == changed["thresholds"]


@pytest.mark.parametrize("field", ["processed_dir", "manifest"])
def test_training_data_must_not_be_on_drive(
    synthetic: Synthetic, tmp_path: Path, field: str
) -> None:
    config = tiny_config(synthetic, tmp_path, **{field: Path("/content/drive/MyDrive/nih_224")})
    with pytest.raises(ValueError, match="Google Drive"):
        make_datasets(config)
