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

    v2 = evaluate.evaluate(config, device=torch.device("cpu"), version=2)
    assert v2["report_version"] == 2 and v2["checkpoint_epoch"] == report["checkpoint_epoch"]
    assert [row["auroc"] for row in v2["splits"]["test"]["labels"]] == [
        row["auroc"] for row in report["splits"]["test"]["labels"]
    ]
    with pytest.raises(ValueError, match="1 or 2"):
        evaluate.evaluate(config, device=torch.device("cpu"), version=3)


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


def fake_predictions(seed: int = 0) -> dict[str, evaluate.SplitPredictions]:
    """Val and test with distinct patients (two images each) and labels of varied frequency."""
    import numpy as np

    from src.models.data import Sample

    rng = np.random.default_rng(seed)
    # Label i has prevalence rising with i, so the first labels have fewer than 30 val positives.
    prevalence = np.linspace(0.01, 0.5, len(FINDINGS))
    result = {}
    for split, patients, first in (("val", 150, 0), ("test", 200, 1000)):
        samples, targets = [], []
        for number in range(first, first + patients):
            age = 150 if number == first else int(rng.integers(5, 95))
            for image in range(2):
                row = (rng.random(len(FINDINGS)) < prevalence).astype(int)
                targets.append(row)
                samples.append(
                    Sample(
                        f"{number}_{image}.png",
                        str(number),
                        split,
                        "s0",
                        age,
                        "F" if number % 2 else "M",
                        tuple(int(v) for v in row),
                    )
                )
        values = np.array(targets, dtype=float)
        scores = np.clip(values * 0.4 + rng.random(values.shape), 0, None)
        result[split] = evaluate.SplitPredictions(samples, values, scores)
    return result


def test_v2_report_thresholds_intervals_and_subgroups() -> None:
    import numpy as np

    from src.models import metrics

    predictions = fake_predictions()
    val_patients = {sample.patient for sample in predictions["val"].samples}
    test_patients = {sample.patient for sample in predictions["test"].samples}
    print(f"patients: val {len(val_patients)}, test {len(test_patients)}")
    assert val_patients and test_patients and not val_patients & test_patients

    report = evaluate.build_report_v2(FINDINGS, predictions, 10, {}, resamples=200)
    sets = report["thresholds"]["sets"]
    assert report["thresholds"]["selected_on"] == "val"
    val, test = report["splits"]["val"], report["splits"]["test"]

    few = [row["label"] for row in val["labels"] if row["positives"] < 30]
    many = [row["label"] for row in val["labels"] if row["positives"] >= 30]
    print(f"labels with <30 val positives: {len(few)}, with >=30: {len(many)}")
    assert few and many
    assert all(sets["sensitivity_80"]["labels"][label] is None for label in few)
    for index, label in enumerate(FINDINGS):
        chosen = sets["sensitivity_80"]["labels"][label]
        row = test["labels"][index]
        if chosen is None:
            assert row["at_thresholds"]["sensitivity_80"] is None
            continue
        assert chosen["sensitivity"] >= 0.8
        assert (
            chosen["sensitivity"]
            == val["labels"][index]["at_thresholds"]["sensitivity_80"]["sensitivity"]
        )
        expected = metrics.threshold_metrics(
            predictions["test"].targets[:, index],
            predictions["test"].scores[:, index],
            chosen["threshold"],
        )
        assert row["at_thresholds"]["sensitivity_80"] == expected
        assert row["at_thresholds"]["best_f1"] is not None

    # Intervals on test only, per label and for every mean.
    assert all("auroc_ci" not in row for row in val["labels"]) and "mean_auroc_ci" not in val
    for row in test["labels"]:
        assert row["auroc_ci"]["low"] <= row["auroc"] <= row["auroc_ci"]["high"]
    assert test["mean_auroc_ci"]["valid_resamples"] > 0

    # One label set per grouping, used by every group; the invalid age band is not scored.
    for split in (val, test):
        for key in ("by_sex", "by_age_band"):
            shared = split[key]["labels_in_mean"]
            scored = {g: e for g, e in split[key]["groups"].items() if not e.get("excluded")}
            assert shared and len(scored) >= 2
            for entry in scored.values():
                rows = {row["label"]: row for row in entry["labels"]}
                assert all(rows[label]["positives"] >= 10 for label in shared)
                assert entry["mean_auroc"] == pytest.approx(
                    np.mean([rows[label]["auroc"] for label in shared])
                )
                assert ("mean_auroc_ci" in entry) == (split is test)
    assert test["by_age_band"]["groups"]["invalid (age > 100)"] == {"images": 2, "excluded": True}

    # Same seed, same report; other test scores leave both threshold sets unchanged.
    assert evaluate.build_report_v2(FINDINGS, predictions, 10, {}, resamples=200) == report
    flipped = {
        **predictions,
        "test": replace(predictions["test"], scores=1 - predictions["test"].scores),
    }
    changed = evaluate.build_report_v2(FINDINGS, flipped, 10, {}, resamples=20)
    assert changed["thresholds"] == report["thresholds"]


def test_v2_outputs_leave_v1_files_untouched(tmp_path: Path) -> None:
    predictions = fake_predictions()
    meta = {
        "checkpoint_epoch": 1,
        "history": [{"epoch": 1, "train_loss": 0.5, "val_mean_auroc": 0.6}],
    }
    v1 = {name: tmp_path / path for name, path in evaluate.OUTPUTS[1].items()}
    v2 = {name: tmp_path / path for name, path in evaluate.OUTPUTS[2].items()}
    assert not set(v1.values()) & set(v2.values())
    assert {path.name for path in v2.values()} == {
        "classifier_metrics_v2.json",
        "thresholds_v2.json",
        "results_v2.md",
    }

    evaluate.write_outputs(
        evaluate.build_report(FINDINGS, predictions, 10, meta),
        v1["metrics"],
        v1["thresholds"],
        v1["results"],
    )
    before = {name: path.read_bytes() for name, path in v1.items()}
    report = evaluate.build_report_v2(FINDINGS, predictions, 10, meta, resamples=50)
    evaluate.write_outputs(report, v2["metrics"], v2["thresholds"], v2["results"])
    after = {name: path.read_bytes() for name, path in v1.items()}
    for name in v1:
        lines = (before[name].count(b"\n"), after[name].count(b"\n"))
        print(f"{name}: v1 lines before {lines[0]}, after {lines[1]}")
        assert before[name] and after[name] and before[name] == after[name]

    text = v2["results"].read_text()
    assert "For education only, not for clinical use." in text
    assert "score only" in text and "95% CI" in text
    assert json.loads(v2["thresholds"].read_text())["sets"].keys() == {"best_f1", "sensitivity_80"}
    with pytest.raises(FileExistsError):
        evaluate.write_outputs(report, v2["metrics"], v2["thresholds"], v2["results"])
