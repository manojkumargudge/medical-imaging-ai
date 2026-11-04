"""Evaluate the best checkpoint on val and test and write the metrics report.

- Per-label AUROC on val and test, with positive and negative counts.
- One threshold per label, chosen on val only (highest F1), saved to a JSON file and then applied
  unchanged to test.
- AUROC by sex and by age band, per split. Ages above 100 are recording errors: those images are
  counted but left out of the age bands.

Report version 2 (the default) adds:

- a second threshold set: per label, the highest val threshold with val sensitivity >= 0.80;
  labels with fewer than 30 val positives get no threshold (score only). Test sensitivity and
  specificity are reported for both sets.
- 95% bootstrap intervals (1000 patient-level resamples, fixed seed) for per-label test AUROC
  and for the test mean AUROCs (overall, by sex, by age band).
- sex and age-band means over one label set per grouping: the labels with at least
  `subgroup_min_positives` positives in every group of that split.

Version 1 writes reports/classifier_metrics.json, reports/classifier_thresholds.json and
docs/results.md; version 2 writes reports/classifier_metrics_v2.json, reports/thresholds_v2.json
and docs/results_v2.md. An existing output is replaced only with --overwrite.

Run: python -m src.models.evaluate --config configs/classifier.yaml [--version 1]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from src.models.checkpoints import checkpoint_path, latest_checkpoint, load_checkpoint
from src.models.classifier import build_model
from src.models.config import ClassifierConfig, load_config
from src.models.data import Sample, ShardDataset, make_datasets
from src.models.metrics import (
    INVALID_AGE,
    Array,
    age_band,
    best_f1_threshold,
    bootstrap_aurocs,
    common_labels,
    label_aurocs,
    mean_auroc,
    sensitivity_threshold,
    subgroup_aurocs,
    threshold_metrics,
)
from src.models.train import best_epoch
from src.models.train import predict as predict_loader

DATASET = "NIH ChestX-ray14 (Kaggle nih-chest-xrays/data)"
SPLIT_NAMES = {
    "val": "val (10% of train_val_list.txt patients, seed 42)",
    "test": "test (official test_list.txt)",
}
NOTICE = "For education only, not for clinical use."
SENSITIVITY_TARGET = 0.80
SENSITIVITY_MIN_VAL_POSITIVES = 30
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
CI_LEVEL = 0.95
F1_RULE = "highest F1 per label; predict positive when score >= threshold"
SENSITIVITY_RULE = (
    f"highest threshold with val sensitivity >= {SENSITIVITY_TARGET:.2f}; labels with fewer than "
    f"{SENSITIVITY_MIN_VAL_POSITIVES} val positives get no threshold (score only)"
)


@dataclass(frozen=True)
class SplitPredictions:
    samples: Sequence[Sample]
    targets: Array
    scores: Array


def build_report(
    labels: Sequence[str],
    predictions: Mapping[str, SplitPredictions],
    min_positives: int,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Everything in classifier_metrics.json; thresholds come from val only."""
    for split in ("val", "test"):
        print(f"{split}: {len(predictions[split].targets)} images")
        if len(predictions[split].targets) == 0:
            raise ValueError(f"no {split} predictions")

    val = predictions["val"]
    thresholds: dict[str, dict[str, float] | None] = {
        label: best_f1_threshold(val.targets[:, index], val.scores[:, index])
        for index, label in enumerate(labels)
    }
    splits: dict[str, Any] = {}
    for split, result in predictions.items():
        per_label = label_aurocs(result.targets, result.scores, labels)
        for index, row in enumerate(per_label):
            chosen = thresholds[row["label"]]
            row["at_val_threshold"] = (
                None
                if chosen is None
                else threshold_metrics(
                    result.targets[:, index], result.scores[:, index], chosen["threshold"]
                )
            )
        sexes = [sample.sex for sample in result.samples]
        bands = [age_band(sample.age) for sample in result.samples]
        splits[split] = {
            "name": SPLIT_NAMES.get(split, split),
            "images": len(result.samples),
            "patients": len({sample.patient for sample in result.samples}),
            "mean_auroc": mean_auroc(per_label),
            "labels": per_label,
            "by_sex": subgroup_aurocs(result.targets, result.scores, sexes, labels, min_positives),
            "by_age_band": subgroup_aurocs(
                result.targets, result.scores, bands, labels, min_positives, (INVALID_AGE,)
            ),
        }
    return {
        "notice": NOTICE,
        "dataset": DATASET,
        "labels_note": "NLP-mined from radiology reports, not radiologist-verified",
        **meta,
        "thresholds": {
            "selected_on": "val",
            "rule": F1_RULE,
            "labels": thresholds,
        },
        "subgroup_min_positives": min_positives,
        "splits": splits,
    }


def _groups_v2(
    result: SplitPredictions,
    groups: Sequence[str],
    labels: Sequence[str],
    min_positives: int,
    excluded: Sequence[str],
    with_ci: bool,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Mean AUROC per group over one label set shared by every scored group."""
    shared = common_labels(result.targets, groups, labels, min_positives, excluded)
    names = np.asarray(groups)
    entries: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups)):
        mask = names == group
        if group in excluded:
            entries[group] = {"images": int(mask.sum()), "excluded": True}
            continue
        patients = [result.samples[int(i)].patient for i in np.flatnonzero(mask)]
        rows = label_aurocs(result.targets[mask], result.scores[mask], labels)
        entry: dict[str, Any] = {
            "images": int(mask.sum()),
            "patients": len(set(patients)),
            "mean_auroc": mean_auroc([row for row in rows if row["label"] in shared]),
            "labels": rows,
        }
        if with_ci and shared:
            # Only the shared labels enter the mean, so only they are resampled.
            columns = [list(labels).index(label) for label in shared]
            entry["mean_auroc_ci"] = bootstrap_aurocs(
                result.targets[mask][:, columns],
                result.scores[mask][:, columns],
                patients,
                shared,
                shared,
                resamples,
                seed,
                CI_LEVEL,
            )["mean"]
        entries[group] = entry
    return {"labels_in_mean": shared, "groups": entries}


def build_report_v2(
    labels: Sequence[str],
    predictions: Mapping[str, SplitPredictions],
    min_positives: int,
    meta: Mapping[str, Any],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Everything in classifier_metrics_v2.json. Both threshold sets come from val only;
    bootstrap intervals are computed on test only."""
    for split in ("val", "test"):
        print(f"{split}: {len(predictions[split].targets)} images")
        if len(predictions[split].targets) == 0:
            raise ValueError(f"no {split} predictions")

    val = predictions["val"]
    sets: dict[str, dict[str, dict[str, float] | None]] = {
        "best_f1": {
            label: best_f1_threshold(val.targets[:, index], val.scores[:, index])
            for index, label in enumerate(labels)
        },
        "sensitivity_80": {
            label: sensitivity_threshold(
                val.targets[:, index],
                val.scores[:, index],
                SENSITIVITY_TARGET,
                SENSITIVITY_MIN_VAL_POSITIVES,
            )
            for index, label in enumerate(labels)
        },
    }
    splits: dict[str, Any] = {}
    for split, result in predictions.items():
        with_ci = split == "test"
        per_label = label_aurocs(result.targets, result.scores, labels)
        intervals = None
        if with_ci:
            intervals = bootstrap_aurocs(
                result.targets,
                result.scores,
                [sample.patient for sample in result.samples],
                labels,
                [row["label"] for row in per_label if row["auroc"] is not None],
                resamples,
                seed,
                CI_LEVEL,
            )
        for index, row in enumerate(per_label):
            at_thresholds: dict[str, dict[str, Any] | None] = {}
            for name, chosen in sets.items():
                picked = chosen[row["label"]]
                at_thresholds[name] = (
                    None
                    if picked is None
                    else threshold_metrics(
                        result.targets[:, index], result.scores[:, index], picked["threshold"]
                    )
                )
            row["at_thresholds"] = at_thresholds
            if intervals is not None:
                row["auroc_ci"] = (
                    None if row["auroc"] is None else intervals["labels"][row["label"]]
                )
        sexes = [sample.sex for sample in result.samples]
        bands = [age_band(sample.age) for sample in result.samples]
        common = (labels, min_positives)
        splits[split] = {
            "name": SPLIT_NAMES.get(split, split),
            "images": len(result.samples),
            "patients": len({sample.patient for sample in result.samples}),
            "mean_auroc": mean_auroc(per_label),
            **({"mean_auroc_ci": intervals["mean"]} if intervals is not None else {}),
            "labels": per_label,
            "by_sex": _groups_v2(result, sexes, *common, (), with_ci, resamples, seed),
            "by_age_band": _groups_v2(
                result, bands, *common, (INVALID_AGE,), with_ci, resamples, seed
            ),
        }
    return {
        "notice": NOTICE,
        "report_version": 2,
        "dataset": DATASET,
        "labels_note": "NLP-mined from radiology reports, not radiologist-verified",
        **meta,
        "thresholds": {
            "selected_on": "val",
            "sets": {
                "best_f1": {"rule": F1_RULE, "labels": sets["best_f1"]},
                "sensitivity_80": {
                    "rule": SENSITIVITY_RULE,
                    "target_sensitivity": SENSITIVITY_TARGET,
                    "min_val_positives": SENSITIVITY_MIN_VAL_POSITIVES,
                    "labels": sets["sensitivity_80"],
                },
            },
        },
        "bootstrap": {
            "split": "test",
            "resamples": resamples,
            "seed": seed,
            "level": CI_LEVEL,
            "unit": "patient (all images of a drawn patient are kept together)",
            "method": "percentile",
        },
        "subgroup_min_positives": min_positives,
        "splits": splits,
    }


def _number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_markdown(report: Mapping[str, Any]) -> str:
    splits = report["splits"]
    lines = [
        "# Classifier results",
        "",
        f"**{NOTICE}** These numbers describe agreement with NLP-mined labels on one public "
        "dataset. They are not a measure of clinical accuracy.",
        "",
        "Generated by `python -m src.models.evaluate`; do not edit by hand.",
        "",
        f"- Dataset: {report['dataset']}; labels {report['labels_note']}.",
        f"- Model: DenseNet121 (ImageNet weights), checkpoint epoch {report['checkpoint_epoch']} "
        f"(highest val mean AUROC over {len(report['history'])} epochs).",
        f"- Splits: {splits['val']['name']}; {splits['test']['name']}. The official test split "
        "has more findings and more images per patient than val (see eda.md), so val and test "
        "results are not directly comparable.",
        f"- Thresholds: chosen on val only ({report['thresholds']['rule']}), then applied to "
        "test unchanged.",
        "",
        "## AUROC per label",
        "",
        "| label | val AUROC | val positives | test AUROC | test positives | test sensitivity "
        "| test specificity |",
        "|---|---|---|---|---|---|---|",
    ]
    for val_row, test_row in zip(splits["val"]["labels"], splits["test"]["labels"], strict=True):
        at = test_row["at_val_threshold"] or {}
        lines.append(
            f"| {val_row['label']} | {_number(val_row['auroc'])} | {val_row['positives']} "
            f"| {_number(test_row['auroc'])} | {test_row['positives']} "
            f"| {_number(at.get('sensitivity'))} | {_number(at.get('specificity'))} |"
        )
    lines.append(
        f"| **mean** | **{_number(splits['val']['mean_auroc'])}** | "
        f"| **{_number(splits['test']['mean_auroc'])}** | | | |"
    )
    for key, title in (("by_sex", "sex"), ("by_age_band", "age band")):
        lines += [
            "",
            f"## Mean AUROC by {title}",
            "",
            f"Mean over labels with at least {report['subgroup_min_positives']} positives in the "
            "group (labels counted in brackets).",
            "",
            f"| {title} | val images | val mean AUROC | test images | test mean AUROC |",
            "|---|---|---|---|---|",
        ]
        groups = sorted(set(splits["val"][key]) | set(splits["test"][key]))
        for group in groups:
            cells = []
            for split in ("val", "test"):
                entry = splits[split][key].get(group, {"images": 0})
                if entry.get("excluded") or "mean_auroc" not in entry:
                    score = "excluded" if entry.get("excluded") else "n/a"
                else:
                    score = f"{_number(entry['mean_auroc'])} ({entry['labels_in_mean']})"
                cells += [str(entry["images"]), score]
            lines.append(f"| {group} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Epochs",
        "",
        "| epoch | train loss | val mean AUROC |",
        "|---|---|---|",
    ]
    lines += [
        f"| {entry['epoch']} | {entry['train_loss']:.4f} | {_number(entry['val_mean_auroc'])} |"
        for entry in report["history"]
    ]
    return "\n".join(lines) + "\n"


def _with_ci(value: float | None, interval: Mapping[str, Any] | None) -> str:
    if value is None or interval is None:
        return _number(value)
    return f"{_number(value)} ({_number(interval['low'])} to {_number(interval['high'])})"


def render_markdown_v2(report: Mapping[str, Any]) -> str:
    splits, boot = report["splits"], report["bootstrap"]
    sets = report["thresholds"]["sets"]
    sensitivity = sets["sensitivity_80"]
    lines = [
        "# Classifier results (v2)",
        "",
        f"**{NOTICE}** These numbers describe agreement with NLP-mined labels on one public "
        "dataset. They are not a measure of clinical accuracy.",
        "",
        "Generated by `python -m src.models.evaluate --version 2`; do not edit by hand. "
        "The version 1 report ([results.md](results.md)) is kept as it was.",
        "",
        f"- Dataset: {report['dataset']}; labels {report['labels_note']}.",
        f"- Model: DenseNet121 (ImageNet weights), checkpoint epoch {report['checkpoint_epoch']} "
        f"(highest val mean AUROC over {len(report['history'])} epochs).",
        f"- Splits: {splits['val']['name']}; {splits['test']['name']}. The official test split "
        "has more findings and more images per patient than val (see eda.md), so val and test "
        "results are not directly comparable.",
        "- Thresholds: two sets, both chosen on val only and applied to test unchanged. "
        f"Set A: {sets['best_f1']['rule']}. Set B: {sensitivity['rule']}.",
        f"- Intervals: {boot['level']:.0%} percentile bootstrap on {boot['split']}, "
        f"{boot['resamples']} resamples of patients (seed {boot['seed']}); all images of a "
        "drawn patient are kept together.",
        "",
        "## AUROC per label",
        "",
        "| label | val AUROC | val positives | test AUROC (95% CI) | test positives |",
        "|---|---|---|---|---|",
    ]
    for val_row, test_row in zip(splits["val"]["labels"], splits["test"]["labels"], strict=True):
        lines.append(
            f"| {val_row['label']} | {_number(val_row['auroc'])} | {val_row['positives']} "
            f"| {_with_ci(test_row['auroc'], test_row.get('auroc_ci'))} "
            f"| {test_row['positives']} |"
        )
    test_mean = _with_ci(splits["test"]["mean_auroc"], splits["test"].get("mean_auroc_ci"))
    lines.append(f"| **mean** | **{_number(splits['val']['mean_auroc'])}** | | **{test_mean}** | |")
    lines += [
        "",
        "## Thresholds applied to test",
        "",
        "A = highest-F1 threshold; B = highest threshold with val sensitivity >= "
        f'{sensitivity["target_sensitivity"]:.2f}. "score only": fewer than '
        f"{sensitivity['min_val_positives']} val positives, so no B threshold.",
        "",
        "| label | A threshold | A test sensitivity | A test specificity | B threshold "
        "| B val sensitivity | B test sensitivity | B test specificity |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in splits["test"]["labels"]:
        cells = [row["label"]]
        best_f1 = sets["best_f1"]["labels"][row["label"]]
        at = row["at_thresholds"]["best_f1"] or {}
        cells += [
            "n/a" if best_f1 is None else _number(best_f1["threshold"], 4),
            _number(at.get("sensitivity")),
            _number(at.get("specificity")),
        ]
        chosen = sensitivity["labels"][row["label"]]
        at = row["at_thresholds"]["sensitivity_80"] or {}
        cells += (
            ["score only", "n/a"]
            if chosen is None
            else [_number(chosen["threshold"], 4), _number(chosen["sensitivity"])]
        )
        cells += [_number(at.get("sensitivity")), _number(at.get("specificity"))]
        lines.append("| " + " | ".join(cells) + " |")
    for key, title in (("by_sex", "sex"), ("by_age_band", "age band")):
        lines += [
            "",
            f"## Mean AUROC by {title}",
            "",
            f"In each split, every {title} group is averaged over the same labels: those with at "
            f"least {report['subgroup_min_positives']} positives (and a negative) in every group. "
            "Intervals are for test only.",
            "",
        ]
        for split in ("val", "test"):
            shared = splits[split][key]["labels_in_mean"]
            lines.append(f"- {split} ({len(shared)} labels): {', '.join(shared) or 'none'}")
        lines += [
            "",
            f"| {title} | val images | val mean AUROC | test images | test mean AUROC (95% CI) |",
            "|---|---|---|---|---|",
        ]
        groups = sorted(set(splits["val"][key]["groups"]) | set(splits["test"][key]["groups"]))
        for group in groups:
            cells = []
            for split in ("val", "test"):
                entry = splits[split][key]["groups"].get(group, {"images": 0})
                score = (
                    "excluded"
                    if entry.get("excluded")
                    else _with_ci(entry.get("mean_auroc"), entry.get("mean_auroc_ci"))
                )
                cells += [str(entry["images"]), score]
            lines.append(f"| {group} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Epochs",
        "",
        "| epoch | train loss | val mean AUROC |",
        "|---|---|---|",
    ]
    lines += [
        f"| {entry['epoch']} | {entry['train_loss']:.4f} | {_number(entry['val_mean_auroc'])} |"
        for entry in report["history"]
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    report: Mapping[str, Any],
    metrics_path: Path,
    thresholds_path: Path,
    results_path: Path,
    overwrite: bool = False,
) -> None:
    outputs = {
        metrics_path: json.dumps(report, indent=2) + "\n",
        thresholds_path: json.dumps(report["thresholds"], indent=2) + "\n",
        results_path: (
            render_markdown_v2(report)
            if report.get("report_version") == 2
            else render_markdown(report)
        ),
    }
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{existing} exist; pass --overwrite to replace them")
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}")


def evaluate(
    config: ClassifierConfig,
    device: torch.device | None = None,
    meta: Mapping[str, Any] | None = None,
    version: int = 1,
) -> dict[str, Any]:
    """Score val and test with the best epoch's weights and build the report (version 1 or 2)."""
    if version not in (1, 2):
        raise ValueError(f"report version must be 1 or 2, got {version}")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latest = latest_checkpoint(config.checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint in {config.checkpoint_dir}")
    history = load_checkpoint(latest)["history"]
    best = best_epoch(history)
    state = load_checkpoint(checkpoint_path(config.checkpoint_dir, best["epoch"]))
    if state["fingerprint"] != config.fingerprint():
        raise ValueError("the checkpoint was trained with another config")
    model = build_model(len(config.labels), None)
    model.load_state_dict(state["model"])
    model.to(device)

    datasets = make_datasets(config, ("val", "test"))
    predictions = {}
    for split in ("val", "test"):
        dataset: ShardDataset = datasets[split]
        loader = DataLoader(
            dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False
        )
        targets, scores = predict_loader(
            model, loader, device, config.amp and device.type == "cuda"
        )
        predictions[split] = SplitPredictions(dataset.samples, targets, scores)
    build = build_report_v2 if version == 2 else build_report
    return build(
        config.labels,
        predictions,
        config.subgroup_min_positives,
        {
            **(meta or {}),
            "checkpoint_epoch": best["epoch"],
            "history": history,
            "train_versions": state["versions"],
            "eval_versions": {
                "torch": str(torch.__version__),
                "torchvision": torchvision.__version__,
            },
        },
    )


OUTPUTS = {
    1: {
        "metrics": Path("reports/classifier_metrics.json"),
        "thresholds": Path("reports/classifier_thresholds.json"),
        "results": Path("docs/results.md"),
    },
    2: {
        "metrics": Path("reports/classifier_metrics_v2.json"),
        "thresholds": Path("reports/thresholds_v2.json"),
        "results": Path("docs/results_v2.md"),
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/classifier.yaml"))
    parser.add_argument("--checkpoint-dir", type=Path, help="override the config's folder")
    parser.add_argument("--version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--metrics", type=Path, help="default depends on --version")
    parser.add_argument("--thresholds", type=Path, help="default depends on --version")
    parser.add_argument("--results", type=Path, help="default depends on --version")
    parser.add_argument("--repo-commit", default="", help="recorded in the report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    defaults = OUTPUTS[args.version]
    overrides = {"checkpoint_dir": args.checkpoint_dir} if args.checkpoint_dir else {}
    config = load_config(args.config, **overrides)
    report = evaluate(config, meta={"repo_commit": args.repo_commit}, version=args.version)
    write_outputs(
        report,
        args.metrics or defaults["metrics"],
        args.thresholds or defaults["thresholds"],
        args.results or defaults["results"],
        args.overwrite,
    )


if __name__ == "__main__":
    main()
