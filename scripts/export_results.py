"""Export selected, small validation evidence from saved local runs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fer.data import LABELS
from src.fer.models import MODEL_IDS
from src.fer.train import save_training_curves


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_comparison(path: Path, run: dict[str, Any]) -> None:
    fields = [
        "model_id", "role", "parameters", "training_seconds", "validation_loss",
        "validation_accuracy", "validation_macro_f1", "source_test_evaluated",
    ]
    reference = run["majority_validation_reference"]
    rows = [{
        "model_id": "majority_reference",
        "role": "reference",
        "parameters": 0,
        "training_seconds": 0,
        "validation_loss": "",
        "validation_accuracy": reference["accuracy"],
        "validation_macro_f1": reference["macro_f1"],
        "source_test_evaluated": False,
    }]
    for model_id in MODEL_IDS:
        result = run["models"][model_id]
        rows.append({
            "model_id": model_id,
            "role": "neural_baseline",
            "parameters": result["parameters"],
            "training_seconds": result["training_seconds"],
            "validation_loss": result["validation"]["loss"],
            "validation_accuracy": result["validation"]["accuracy"],
            "validation_macro_f1": result["validation"]["macro_f1"],
            "source_test_evaluated": False,
        })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_per_class(path: Path, metrics: dict[str, Any]) -> None:
    fields = ["label", "precision", "recall", "f1-score", "support"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label in LABELS:
            report = metrics["per_class"][label]
            writer.writerow({"label": label, **{field: report[field] for field in fields[1:]}})


def _write_confusion(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row])


def export_run(run_id: str, evidence_type: str, output_root: Path) -> Path:
    source = ROOT / "outputs" / "runs" / run_id
    if not source.is_dir():
        raise FileNotFoundError(f"Saved run not found: {source}")
    run = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
    if run.get("source_test_evaluated"):
        raise ValueError(f"Refusing to export test-scored run as validation evidence: {run_id}")
    if set(run.get("models", {})) != set(MODEL_IDS):
        raise ValueError(f"The saved run does not contain all three neural baselines: {run_id}")

    destination = output_root / evidence_type
    destination.mkdir(parents=True, exist_ok=True)
    package_names = ("torch", "torchvision", "numpy", "Pillow", "scikit-learn", "matplotlib")
    summary = {
        "evidence_type": evidence_type,
        "evaluation_split": "validation",
        "source_test_evaluated": False,
        "saved_run_id": run["run_id"],
        "dataset_source_id": run["dataset_source_id"],
        "dataset_source_url": "https://www.kaggle.com/datasets/msambare/fer2013",
        "dataset_paper": "Goodfellow et al. (2013), Challenges in Representation Learning, arXiv:1307.0414",
        "dataset_paper_url": "https://arxiv.org/abs/1307.0414",
        "dataset_paper_doi": "10.1007/978-3-642-42051-1_16",
        "manifest_sha256": run["manifest_sha256"],
        "source_counts": run["source_counts"],
        "split_counts": run["split_counts"],
        "duplicate_audit_summary": run["duplicate_audit_summary"],
        "training_settings": run["training_settings"],
        "train_records_available": run["train_records_available"],
        "train_records_used": run["train_records_used"],
        "training_subset_sha256": run["training_subset_sha256"],
        "validation_records_available": run["validation_records_available"],
        "validation_records_used": run["validation_records_used"],
        "validation_subset_sha256": run["validation_subset_sha256"],
        "device": run["device"],
        "python": run["python"],
        "package_versions": {name: run["package_versions"][name] for name in package_names if name in run["package_versions"]},
        "majority_validation_reference": {
            "predicted_class": run["majority_validation_reference"]["predicted_class"],
            "accuracy": run["majority_validation_reference"]["accuracy"],
            "macro_f1": run["majority_validation_reference"]["macro_f1"],
        },
        "pilot_run": run["pilot_run"],
        "code_note": "Validation values were rechecked with the frozen checkpoint evaluator; no source-test rows were scored.",
    }
    _write_json(destination / "run_summary.json", summary)
    _write_comparison(destination / "comparison.csv", run)

    for model_id in MODEL_IDS:
        source_model = source / "models" / model_id
        target_model = destination / "models" / model_id
        target_model.mkdir(parents=True, exist_ok=True)
        metrics = run["models"][model_id]["validation"]
        model_summary = {
            "model_id": model_id,
            "evaluation_split": "validation",
            "source_test_evaluated": False,
            "parameters": run["models"][model_id]["parameters"],
            "training_seconds": run["models"][model_id]["training_seconds"],
            "metrics": {
                key: metrics[key]
                for key in ("loss", "accuracy", "macro_f1")
            },
        }
        _write_json(target_model / "metrics.json", model_summary)
        _write_per_class(target_model / "per_class_metrics.csv", metrics)
        _write_confusion(target_model / "confusion_matrix.csv", metrics["confusion_matrix"])
        history_source = source_model / "training_history.csv"
        history_target = target_model / "training_history.csv"
        shutil.copyfile(history_source, history_target)
        with history_target.open(newline="", encoding="utf-8") as handle:
            history = [
                {key: (int(value) if key == "epoch" else float(value)) for key, value in row.items()}
                for row in csv.DictReader(handle)
            ]
        save_training_curves(target_model / "training_curves.png", history)
    return destination


def export_shared_evidence(output_root: Path, full_run_id: str) -> None:
    source = ROOT / "outputs" / "runs" / full_run_id
    run = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
    shared = output_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "class_counts.png", shared / "class_counts.png")
    _write_json(shared / "split_summary.json", {
        "dataset_source_id": run["dataset_source_id"],
        "manifest_sha256": run["manifest_sha256"],
        "label_order": LABELS,
        "source_counts": run["source_counts"],
        "split_counts": run["split_counts"],
        "duplicate_audit_summary": run["duplicate_audit_summary"],
        "source_test_evaluated": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Export selected saved validation evidence.")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--pilot-run", default="20261024-190809-840000")
    parser.add_argument("--full-run", default="20261024-191418-838000")
    args = parser.parse_args()
    output_root = args.output_root.expanduser()
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    pilot = export_run(args.pilot_run, "pilot_validation", output_root)
    full = export_run(args.full_run, "full_budget_validation", output_root)
    export_shared_evidence(output_root, args.full_run)
    final_test = output_root / "final_test"
    final_test.mkdir(parents=True, exist_ok=True)
    (final_test / "README.md").write_text(
        "# Final test results\n\nNo FER2013 source-test metrics are saved here. After the team freezes the model and settings, run the separate evaluator with `--split test --confirm-final-test`, review the output, and copy only the selected final evidence here.\n",
        encoding="utf-8",
    )
    print(json.dumps({"pilot_validation": str(pilot), "full_budget_validation": str(full)}, indent=2))


if __name__ == "__main__":
    main()
