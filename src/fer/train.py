"""Train and compare the shared Option 1 baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

_CACHE_ROOT = Path(__file__).resolve().parents[2] / "outputs" / ".matplotlib"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .checkpoints import save_checkpoint
from .data import LABELS, ManifestImageDataset, read_manifest, validate_manifest_config
from .evaluate import evaluate_model
from .metrics import majority_reference
from .models import MODEL_IDS, make_model, parameter_count
from .runtime import select_device

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch", "train_loss", "validation_loss", "validation_accuracy", "validation_macro_f1", "epoch_seconds"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_train_subset(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["source_split", "relative_path", "label", "class_id", "image_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _subset_sha256(rows: list[dict[str, Any]]) -> str:
    content = "\n".join(
        f"{row['image_sha256']}|{row['source_split']}|{row['relative_path']}"
        for row in rows
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def save_training_curves(path: str | Path, history: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="Train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="Validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_accuracy"] for row in history], label="Accuracy")
    axes[1].plot(epochs, [row["validation_macro_f1"] for row in history], label="Macro-F1", color="#8f4bd8")
    axes[1].set(title="Validation scores", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _save_class_counts(path: Path, rows: list[dict[str, Any]]) -> None:
    splits = ("train", "validation", "test")
    counts = {
        split: [
            sum(row["split"] == split and row["label"] == label for row in rows)
            for label in LABELS
        ]
        for split in splits
    }
    positions = np.arange(len(LABELS))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9, 4))
    for index, split in enumerate(splits):
        axis.bar(positions + (index - 1) * width, counts[split], width, label=split.capitalize())
    axis.set_xticks(positions, [label.capitalize() for label in LABELS], rotation=30, ha="right")
    axis.set_ylabel("Records")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _write_comparison_csv(path: Path, results: dict[str, Any]) -> None:
    fields = [
        "model_id", "parameters", "training_seconds", "validation_loss", "validation_accuracy",
        "validation_macro_f1",
    ]
    rows = [{
        "model_id": "majority_reference",
        "parameters": 0,
        "training_seconds": 0.0,
        "validation_loss": None,
        "validation_accuracy": results["majority_validation_reference"]["accuracy"],
        "validation_macro_f1": results["majority_validation_reference"]["macro_f1"],
    }]
    for model_id, result in results["models"].items():
        row: dict[str, Any] = {
            "model_id": model_id,
            "parameters": result["parameters"],
            "training_seconds": result["training_seconds"],
            "validation_loss": result["validation"]["loss"],
            "validation_accuracy": result["validation"]["accuracy"],
            "validation_macro_f1": result["validation"]["macro_f1"],
        }
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selection_score(validation: dict[str, Any], metric: str) -> float:
    if metric == "validation_loss":
        return -float(validation["loss"])
    if metric in {"validation_macro_f1", "validation_accuracy"}:
        return float(validation[metric.removeprefix("validation_")])
    raise ValueError(f"Unsupported selection_metric: {metric}")


def _select_pilot_rows(rows: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    if limit < 1:
        raise ValueError("max_train_records must be positive.")

    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (int(row["class_id"]), str(row["image_sha256"]))
        grouped.setdefault(key, []).append(row)
    groups_by_class: dict[int, list[tuple[int, str]]] = {}
    for key in grouped:
        groups_by_class.setdefault(key[0], []).append(key)

    targets: dict[int, int] = {}
    for class_id, class_groups in groups_by_class.items():
        class_records = sum(len(grouped[key]) for key in class_groups)
        targets[class_id] = int(limit * class_records / len(rows))
    for class_id in sorted(groups_by_class, key=lambda key: (targets[key], key), reverse=True):
        if sum(targets.values()) >= limit:
            break
        targets[class_id] += 1

    selected_keys: set[tuple[int, str]] = set()
    selected_count = 0
    deferred: list[tuple[int, str]] = []
    for class_id in sorted(groups_by_class):
        keys = sorted(groups_by_class[class_id])
        rng = np.random.default_rng(np.random.SeedSequence([seed, class_id, 17]))
        order = [keys[index] for index in rng.permutation(len(keys))]
        class_count = 0
        for key in order:
            size = len(grouped[key])
            if class_count + size <= targets[class_id] and selected_count + size <= limit:
                selected_keys.add(key)
                class_count += size
                selected_count += size
            else:
                deferred.append(key)

    rng = np.random.default_rng(np.random.SeedSequence([seed, 911]))
    if deferred:
        order = [deferred[index] for index in rng.permutation(len(deferred))]
        for key in order:
            size = len(grouped[key])
            if selected_count + size <= limit:
                selected_keys.add(key)
                selected_count += size

    selected = [row for key in sorted(selected_keys) for row in grouped[key]]
    return sorted(selected, key=lambda row: (row["class_id"], row["image_sha256"], row["relative_path"]))


def _save_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted", *LABELS])
        for label, row in zip(LABELS, matrix):
            writer.writerow([label, *row])


def _train_one(
    model_id: str,
    manifest: dict[str, Any],
    train_dir: Path,
    test_dir: Path,
    run_dir: Path,
    settings: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    seed = int(settings["seed"])
    seed_everything(seed)
    rows = manifest["records"]
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows or not validation_rows:
        raise ValueError("Both train and validation records are required. Rebuild the split manifest.")

    train_set = ManifestImageDataset(train_rows, train_dir, test_dir)
    validation_set = ManifestImageDataset(validation_rows, train_dir, test_dir)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=int(settings["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(validation_set, batch_size=int(settings["batch_size"]), shuffle=False, num_workers=0)
    model = make_model(model_id, num_classes=len(LABELS)).to(device)
    parameter_total = parameter_count(model)
    loss_function = nn.CrossEntropyLoss()
    optimizer_name = settings.get("optimizer", "Adam").casefold()
    optimizer_args = {
        "lr": float(settings["learning_rate"]),
        "weight_decay": float(settings.get("weight_decay", 0.0)),
    }
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_args)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), momentum=float(settings.get("momentum", 0.0)), **optimizer_args
        )
    else:
        raise ValueError("training.optimizer must be Adam or SGD.")
    model_dir = run_dir / "models" / model_id
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "best.pt"
    history: list[dict[str, Any]] = []
    best_selection_score = float("-inf")
    best_validation_loss = float("inf")
    best_validation_macro_f1 = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, int(settings["epochs"]) + 1):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_records = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            train_loss_sum += float(loss.item()) * batch_size
            train_records += batch_size

        validation = evaluate_model(model, validation_loader, device)
        epoch_seconds = time.perf_counter() - epoch_started
        history_row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_records,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "validation_macro_f1": validation["macro_f1"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(history_row)
        _write_history(model_dir / "training_history.csv", history)

        selection_metric = settings.get("selection_metric", "validation_macro_f1")
        selection_score = _selection_score(validation, selection_metric)
        tie_break = settings.get("tie_break", "lower_validation_loss")
        if tie_break != "lower_validation_loss":
            raise ValueError("training.tie_break must be lower_validation_loss.")
        improved = selection_score > best_selection_score or (
            selection_score == best_selection_score and validation["loss"] < best_validation_loss
        )
        if improved:
            best_selection_score = selection_score
            best_validation_loss = validation["loss"]
            best_validation_macro_f1 = validation["macro_f1"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            metadata = {
                "model_id": model_id,
                "label_order": list(LABELS),
                "preprocessing": manifest["preprocessing"],
                "manifest_sha256": manifest["manifest_sha256"],
                "best_validation_macro_f1": best_validation_macro_f1,
                "best_validation_loss": best_validation_loss,
                "selection_metric": selection_metric,
                "selection_score": best_selection_score,
                "parameters": parameter_total,
                "seed": seed,
                "training_settings": dict(settings),
                "epochs_completed": epoch,
            }
            save_checkpoint(checkpoint_path, best_state, metadata)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"{model_id} epoch {epoch}/{settings['epochs']}: "
            f"train_loss={history_row['train_loss']:.4f} "
            f"val_loss={validation['loss']:.4f} "
            f"val_macro_f1={validation['macro_f1']:.4f} "
            f"{epoch_seconds:.1f}s"
        )
        if epochs_without_improvement >= int(settings["patience"]):
            print(f"{model_id}: early stop after {epoch} epochs without a validation improvement.")
            break

    if best_state is None:
        raise RuntimeError(f"{model_id} did not produce a checkpoint.")
    model.load_state_dict(best_state)
    validation_result = evaluate_model(model, validation_loader, device)
    save_training_curves(model_dir / "training_curves.png", history)
    _save_confusion_csv(model_dir / "validation_confusion_matrix.csv", validation_result["confusion_matrix"])
    result: dict[str, Any] = {
        "model_id": model_id,
        "parameters": parameter_total,
        "training_seconds": time.perf_counter() - started,
        "checkpoint": (
            str(checkpoint_path.relative_to(ROOT))
            if checkpoint_path.is_relative_to(ROOT)
            else str(checkpoint_path)
        ),
        "validation": validation_result,
        "source_test_evaluated": False,
    }

    _write_json(model_dir / "results.json", result)
    return result


def run_experiments(
    config_path: str | Path,
    model_ids: list[str] | None = None,
    epochs_override: int | None = None,
    train_dir_override: str | Path | None = None,
    test_dir_override: str | Path | None = None,
    manifest_override: str | Path | None = None,
    max_train_records: int | None = None,
    max_validation_records: int | None = None,
) -> dict[str, Any]:
    config_path = _resolve_path(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source_root_text = os.environ.get("FER2013_ROOT", config["source_root"])
    source_root = _resolve_path(source_root_text)
    train_dir = Path(train_dir_override).expanduser().resolve() if train_dir_override else source_root / "train"
    test_dir = Path(test_dir_override).expanduser().resolve() if test_dir_override else source_root / "test"
    manifest_path = _resolve_path(manifest_override or config["manifest_path"])
    manifest = read_manifest(manifest_path)
    validate_manifest_config(manifest, config)
    selected_models = model_ids or list(MODEL_IDS)
    unknown = sorted(set(selected_models) - set(MODEL_IDS))
    if unknown:
        raise ValueError(f"Unknown model ids: {', '.join(unknown)}")

    settings = dict(config["training"])
    if epochs_override is not None:
        settings["epochs"] = epochs_override
    device = select_device()
    timestamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
    run_dir = _resolve_path(config.get("output_root", "outputs/runs")) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_class_counts(run_dir / "class_counts.png", manifest["records"])

    validation_rows = [row for row in manifest["records"] if row["split"] == "validation"]
    train_rows = [row for row in manifest["records"] if row["split"] == "train"]
    selected_train_rows = _select_pilot_rows(train_rows, max_train_records, int(settings["seed"]))
    selected_validation_rows = _select_pilot_rows(
        validation_rows, max_validation_records, int(settings["seed"]) + 100_003
    )
    training_manifest = dict(manifest)
    training_manifest["records"] = [
        row for row in manifest["records"] if row["split"] not in {"train", "validation"}
    ] + selected_train_rows + selected_validation_rows
    _write_train_subset(run_dir / "train_subset.csv", selected_train_rows)
    _write_train_subset(run_dir / "validation_subset.csv", selected_validation_rows)
    majority = majority_reference(selected_train_rows, selected_validation_rows)
    majority["loss"] = None
    package_versions = {
        package: importlib.metadata.version(package)
        for package in (
            "torch", "torchvision", "numpy", "Pillow", "scikit-learn", "matplotlib",
            "kagglehub",
        )
    }
    run_config: dict[str, Any] = {
        "run_id": timestamp,
        "dataset_source_id": manifest["source_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "device": str(device),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "package_versions": package_versions,
        "source_counts": manifest["source_counts"],
        "split_counts": manifest["counts"],
        "split_settings": manifest["split_settings"],
        "duplicate_audit_summary": {
            "duplicate_groups": manifest["duplicate_audit"]["duplicate_groups"],
            "train_test_duplicate_groups": len(manifest["duplicate_audit"]["train_test_duplicate_groups"]),
            "conflicting_label_duplicate_groups": len(manifest["duplicate_audit"]["conflicting_label_duplicate_groups"]),
        },
        "training_settings": settings,
        "train_records_available": len(train_rows),
        "train_records_used": len(selected_train_rows),
        "training_subset_sha256": _subset_sha256(selected_train_rows),
        "validation_records_available": len(validation_rows),
        "validation_records_used": len(selected_validation_rows),
        "validation_subset_sha256": _subset_sha256(selected_validation_rows),
        "pilot_run": max_train_records is not None or max_validation_records is not None or epochs_override is not None,
        "source_test_evaluated": False,
    }
    _write_json(run_dir / "run_config.json", run_config)
    results: dict[str, Any] = {
        **run_config,
        "majority_validation_reference": majority,
        "models": {},
    }

    for model_id in selected_models:
        results["models"][model_id] = _train_one(
            model_id,
            training_manifest,
            train_dir,
            test_dir,
            run_dir,
            settings,
            device,
        )

    _write_comparison_csv(run_dir / "comparison.csv", results)
    _write_json(run_dir / "comparison.json", results)
    print(json.dumps({
        "run_directory": str(run_dir),
        "manifest_sha256": manifest["manifest_sha256"],
        "device": str(device),
        "train_records_used": len(selected_train_rows),
        "validation_records_used": len(selected_validation_rows),
        "pilot_run": results["pilot_run"],
        "majority_validation_accuracy": majority["accuracy"],
        "models": {
            key: {
                "parameters": value["parameters"],
                "training_seconds": round(value["training_seconds"], 1),
                "validation_accuracy": value["validation"]["accuracy"],
                "validation_macro_f1": value["validation"]["macro_f1"],
                "source_test_evaluated": value["source_test_evaluated"],
            }
            for key, value in results["models"].items()
        },
    }, indent=2))
    return results


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Train the three shared Option 1 neural baselines.")
    parser.add_argument("--config", type=Path, default=Path("configs/option1_baselines.json"))
    parser.add_argument("--model", choices=[*MODEL_IDS, "all"], default="all")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-validation-records", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--train-dir", type=Path)
    parser.add_argument("--test-dir", type=Path)
    args = parser.parse_args()
    model_ids = list(MODEL_IDS) if args.model == "all" else [args.model]
    run_experiments(
        args.config,
        model_ids=model_ids,
        epochs_override=args.epochs,
        train_dir_override=args.train_dir,
        test_dir_override=args.test_dir,
        manifest_override=args.manifest,
        max_train_records=args.max_train_records,
        max_validation_records=args.max_validation_records,
    )


if __name__ == "__main__":
    _cli()
