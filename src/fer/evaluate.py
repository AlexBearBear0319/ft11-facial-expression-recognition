"""Evaluate saved FT11 checkpoints without training or changing model weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoints import load_checkpoint
from .data import LABELS, ManifestImageDataset, read_manifest, validate_manifest_config
from .metrics import summarize_predictions
from .models import MODEL_IDS, parameter_count
from .runtime import select_device

ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    """Return shared loss, accuracy, class metrics, and confusion matrix."""
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_records = 0
    actual: list[int] = []
    predicted: list[int] = []
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_records += batch_size
            actual.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
    if not total_records:
        raise ValueError("The evaluation split is empty.")
    return summarize_predictions(actual, predicted, total_loss / total_records)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row])


def evaluate_checkpoints(
    config_path: str | Path,
    checkpoint_dir: str | Path,
    *,
    split: str,
    model_ids: list[str] | None = None,
    train_dir_override: str | Path | None = None,
    test_dir_override: str | Path | None = None,
    manifest_override: str | Path | None = None,
    subset_csv: str | Path | None = None,
    output_dir: str | Path | None = None,
    confirm_final_test: bool = False,
) -> dict[str, Any]:
    """Score existing checkpoints on validation or, with explicit approval, test."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")
    if split == "test" and not confirm_final_test:
        raise ValueError("Final test scoring needs --confirm-final-test after the model is frozen.")

    config_file = _resolve_path(config_path)
    config = json.loads(config_file.read_text(encoding="utf-8"))
    source_root = _resolve_path(os.environ.get("FER2013_ROOT", config["source_root"]))
    train_dir = Path(train_dir_override).expanduser().resolve() if train_dir_override else source_root / "train"
    test_dir = Path(test_dir_override).expanduser().resolve() if test_dir_override else source_root / "test"
    manifest_path = _resolve_path(manifest_override or config["manifest_path"])
    manifest = read_manifest(manifest_path)
    validate_manifest_config(manifest, config)
    expected_hash = manifest["manifest_sha256"]
    rows = [row for row in manifest["records"] if row["split"] == split]
    subset_hash = None
    if subset_csv is not None:
        if split != "validation":
            raise ValueError("A saved subset can only be used with validation; final test always uses all test rows.")
        subset_path = _resolve_path(subset_csv)
        with subset_path.open(newline="", encoding="utf-8") as handle:
            subset_rows = list(csv.DictReader(handle))
        row_lookup = {(row["source_split"], row["relative_path"]): row for row in rows}
        selected_rows = []
        for subset_row in subset_rows:
            key = (subset_row["source_split"], subset_row["relative_path"])
            if key not in row_lookup:
                raise ValueError(f"Subset row is not part of the {split} split: {key[1]}")
            row = row_lookup[key]
            for field in ("label", "class_id", "image_sha256"):
                if str(row[field]) != str(subset_row[field]):
                    raise ValueError(f"Subset {field} does not match the manifest for {key[1]}.")
            selected_rows.append(row)
        if not selected_rows:
            raise ValueError("The validation subset is empty.")
        rows = selected_rows
        subset_content = "\n".join(
            f"{row['image_sha256']}|{row['source_split']}|{row['relative_path']}"
            for row in rows
        )
        subset_hash = hashlib.sha256(subset_content.encode("utf-8")).hexdigest()
    if not rows:
        raise ValueError(f"The manifest has no {split} records.")

    run_path = _resolve_path(checkpoint_dir)
    if subset_hash is not None and (run_path / "run_config.json").is_file():
        run_metadata = json.loads((run_path / "run_config.json").read_text(encoding="utf-8"))
        expected_subset_hash = run_metadata.get("validation_subset_sha256")
        if expected_subset_hash and subset_hash != expected_subset_hash:
            raise ValueError("Validation subset fingerprint does not match the saved training run.")
    selected_models = model_ids or [
        model_id for model_id in MODEL_IDS
        if (run_path / "models" / model_id / "best.pt").is_file()
    ]
    unknown = sorted(set(selected_models) - set(MODEL_IDS))
    if unknown:
        raise ValueError(f"Unknown model ids: {', '.join(unknown)}")
    if not selected_models:
        raise FileNotFoundError(f"No FT11 checkpoints found under {run_path / 'models'}.")

    dataset = ManifestImageDataset(rows, train_dir, test_dir)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    device = select_device()
    evaluation_id = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
    destination = _resolve_path(output_dir) if output_dir else ROOT / "outputs" / "evaluations" / evaluation_id
    destination.mkdir(parents=True, exist_ok=True)
    output: dict[str, Any] = {
        "evaluation_id": evaluation_id,
        "evaluation_split": split,
        "dataset_source_id": manifest["source_id"],
        "manifest_sha256": expected_hash,
        "checkpoint_run_id": run_path.name,
        "device": str(device),
        "record_count": len(rows),
        "subset_csv": str(_resolve_path(subset_csv)) if subset_csv else None,
        "subset_sha256": subset_hash,
        "source_test_evaluated": split == "test",
        "models": {},
    }

    comparison_rows: list[dict[str, Any]] = []
    for model_id in selected_models:
        checkpoint_path = run_path / "models" / model_id / "best.pt"
        model, metadata = load_checkpoint(
            checkpoint_path,
            device=device,
            expected_manifest_sha256=expected_hash,
        )
        if metadata["model_id"] != model_id:
            raise ValueError(f"Checkpoint path says {model_id}, but its metadata names {metadata['model_id']}.")
        started = time.perf_counter()
        scores = evaluate_model(model, loader, device)
        elapsed = time.perf_counter() - started
        model_result = {
            "model_id": model_id,
            "parameters": parameter_count(model),
            "checkpoint": str(checkpoint_path),
            "checkpoint_training_settings": metadata.get("training_settings"),
            "checkpoint_epochs_completed": metadata.get("epochs_completed"),
            "evaluation_seconds": elapsed,
            "metrics": scores,
        }
        output["models"][model_id] = model_result
        model_dir = destination / "models" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        _write_json(model_dir / "metrics.json", model_result)
        _write_confusion_csv(model_dir / f"{split}_confusion_matrix.csv", scores["confusion_matrix"])
        comparison_rows.append({
            "model_id": model_id,
            "parameters": model_result["parameters"],
            "evaluation_seconds": elapsed,
            f"{split}_loss": scores["loss"],
            f"{split}_accuracy": scores["accuracy"],
            f"{split}_macro_f1": scores["macro_f1"],
        })

    output["output_directory"] = str(destination)
    _write_json(destination / "comparison.json", output)
    fields = ["model_id", "parameters", "evaluation_seconds", f"{split}_loss", f"{split}_accuracy", f"{split}_macro_f1"]
    with (destination / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(json.dumps({
        "output_directory": str(destination),
        "evaluation_split": split,
        "source_test_evaluated": output["source_test_evaluated"],
        "models": {
            key: {
                "accuracy": value["metrics"]["accuracy"],
                "macro_f1": value["metrics"]["macro_f1"],
                "source_test_evaluated": split == "test",
            }
            for key, value in output["models"].items()
        },
    }, indent=2))
    return output


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Score saved FT11 checkpoints without training.")
    parser.add_argument("--config", type=Path, default=Path("configs/option1_baselines.json"))
    parser.add_argument("--checkpoint-dir", type=Path, required=True, help="A saved outputs/runs/<run-id> folder.")
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--model", choices=[*MODEL_IDS, "all"], default="all")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--subset-csv", type=Path, help="Use a recorded validation subset, such as a pilot run's validation_subset.csv.")
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confirm-final-test", action="store_true", help="Confirm that model and settings are frozen.")
    args = parser.parse_args()
    model_ids = None if args.model == "all" else [args.model]
    evaluate_checkpoints(
        args.config,
        args.checkpoint_dir,
        split=args.split,
        model_ids=model_ids,
        train_dir_override=args.train_dir,
        test_dir_override=args.test_dir,
        manifest_override=args.manifest,
        subset_csv=args.subset_csv,
        output_dir=args.output_dir,
        confirm_final_test=args.confirm_final_test,
    )


if __name__ == "__main__":
    _cli()
